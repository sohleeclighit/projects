#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Message CTR@1 evaluator (Spec #5)

- CLI:
    --catalog message_catalog.json
    --scenarios scenarios_5.json
    --answers answers_5.json
    --weights "theme=2.0,tag=1.0,region=1.5,recency=0.5"
    --topk 1
    --output ./output/result.json
    --log ./log/summary.csv
    [--details ./log/recs.ndjson]
    [--test]

- 확정 규격 반영:
  * 문자열 비교: strip만, 대소문자 변경 없음(정확 일치)
  * ThemeMatch: 태그 정확 일치만 (content 매칭 없음)
  * TagOverlap: 중복 제거 후 교집합 크기(상한 없음)
  * RegionMatch: 정확 일치, 누락 시 0점
  * Simple Mode:
      - theme ← 정답 첫 태그, 없으면 content에서 한 단어, 실패 시 ""
      - context.region ← 정답 region
      - context.time ← 정답 datetime의 시간대 버킷
      - userHints.prefer ← 정답 tags 앞에서 2개 (원본 순서, unique)
      - userHints.avoid ← 항상 빈 배열([])로 덮어씀
  * 시간/날짜:
      - 입력 datetime의 Z/오프셋 무시, KST로 해석 (시차 변환 없음)
      - Recency 기준시각 = 실행 시각(now, KST)
      - days = round
      - 시간대 버킷(경계 규칙):
          morning  : [05:00:00, 12:00:00)
          afternoon: [12:00:00, 17:00:00)
          evening  : [17:00:00, 22:00:00)
          night    : [22:00:00, 05:00:00)
  * 정렬: 점수 내림차순, 동점 시 messageId 문자열(lexicographic) 오름차순
  * 오류 처리:
      - answers.json에 없는 scenarioId → 오류 종료
      - answerMessageId가 catalog에 없음 → 오류 종료
  * CTR 분모 = len(scenarios)
  * 출력 시각 문자열: KST 기준이지만 타임존 오프셋 표기 안함 (ISO-like, naive string)
"""

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Set

# =========================
# Config / CLI
# =========================


@dataclass
class Config:
    catalog_path: Path
    scenarios_path: Path
    answers_path: Path
    topk: int
    w_theme: float
    w_tag: float
    w_region: float
    w_recency: float
    output_path: Path  # result.json
    summary_csv: Path  # run summary (ctr 등)
    details_resultjson: Optional[Path]  # recs.ndjson (옵션)


def parse_weights(s: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not s:
        return out
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            print(
                f"[ERROR] Invalid weight token (expected k=v): {part}", file=sys.stderr
            )
            sys.exit(2)
        k, v = part.split("=", 1)
        k = k.strip()
        try:
            out[k] = float(v.strip())
        except ValueError:
            print(f"[ERROR] Invalid float in weights for '{k}': {v}", file=sys.stderr)
            sys.exit(2)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Personalized Message CTR@1 evaluator.")
    # 입력
    p.add_argument(
        "--catalog", dest="catalog_path", type=Path, help="Path to message_catalog.json"
    )
    p.add_argument(
        "--scenarios", dest="scenarios_path", type=Path, help="Path to scenarios_5.json"
    )
    p.add_argument(
        "--answers", dest="answers_path", type=Path, help="Path to answers_5.json"
    )
    # 파라미터
    p.add_argument(
        "--topk",
        dest="topk",
        type=int,
        help="Top-K (CTR@1이지만 내부 후보 정렬엔 사용)",
    )
    p.add_argument(
        "--weights",
        dest="weights",
        type=str,
        help='Weights: "theme=2.0,tag=1.0,region=1.5,recency=0.5"',
    )
    # 출력
    p.add_argument(
        "--output", dest="output_path", type=Path, help="Path to result.json"
    )
    p.add_argument(
        "--log", dest="summary_csv", type=Path, help="Path to summary.csv (run summary)"
    )
    p.add_argument(
        "--details",
        dest="details_resultjson",
        type=Path,
        help="(Optional) Path to recs.ndjson (candidate scores dump)",
    )
    # 테스트 고정값
    p.add_argument(
        "--test",
        action="store_true",
        help="Use fixed test arguments: ./input/*.json → ./output/result.json, ./log/*.csv",
    )
    return p.parse_args()


def build_config_from_args(args: argparse.Namespace) -> Config:
    if args.test:
        # 고정 경로/파라미터
        args.catalog_path = Path("./input/message_catalog.json")
        args.scenarios_path = Path("./input/scenarios_5.json")
        args.answers_path = Path("./input/answers_5.json")
        args.topk = 1
        args.weights = "theme=2.0,tag=1.0,region=1.5,recency=0.5"
        args.output_path = Path("./output/result.json")
        args.summary_csv = Path("./log/summary.csv")
        # details는 지정 시에만 생성
        if args.details_resultjson is None:
            args.details_resultjson = None

    missing = []
    if not args.catalog_path:
        missing.append("--catalog")
    if not args.scenarios_path:
        missing.append("--scenarios")
    if not args.answers_path:
        missing.append("--answers")
    if args.topk is None:
        missing.append("--topk")
    if not args.weights:
        missing.append("--weights")
    if not args.output_path:
        missing.append("--output")
    if not args.summary_csv:
        missing.append("--log")
    if missing:
        print(
            f"[ERROR] Missing required arguments: {' '.join(missing)}", file=sys.stderr
        )
        sys.exit(2)

    w = parse_weights(args.weights)
    # 키 누락 시 0.0
    w_theme = float(w.get("theme", 0.0))
    w_tag = float(w.get("tag", 0.0))
    w_region = float(w.get("region", 0.0))
    w_recency = float(w.get("recency", 0.0))

    return Config(
        catalog_path=args.catalog_path,
        scenarios_path=args.scenarios_path,
        answers_path=args.answers_path,
        topk=int(args.topk),
        w_theme=w_theme,
        w_tag=w_tag,
        w_region=w_region,
        w_recency=w_recency,
        output_path=args.output_path,
        summary_csv=args.summary_csv,
        details_resultjson=args.details_resultjson,
    )


# =========================
# I/O helpers
# =========================


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON parse error in {path}: {e}", file=sys.stderr)
        sys.exit(1)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# =========================
# Normalization / Utilities
# =========================

KST = timezone(timedelta(hours=9))  # 한국 표준시


def norm_str(s: Optional[str]) -> str:
    # strip만, 대소문자 변경 없음
    return (s or "").strip()


def norm_tags(xs: Optional[Iterable[str]]) -> List[str]:
    # 순서 유지 + 중복 제거 + strip만
    seen: Set[str] = set()
    out: List[str] = []
    for x in xs or []:
        if isinstance(x, str):
            t = x.strip()
            if t and t not in seen:
                seen.add(t)
                out.append(t)
    return out


WORD_RE = re.compile(r"[A-Za-z0-9가-힣]+", re.UNICODE)


def extract_one_word(text: str) -> Optional[str]:
    # 영문/숫자/한글로 이뤄진 첫 토큰(원문 그대로, lower 하지 않음)
    if not text:
        return None
    m = WORD_RE.search(text)
    return m.group(0) if m else None


def parse_dt_to_kst(dt_str: str) -> Optional[datetime]:
    """
    날짜 문자열의 Z/오프셋을 '무시'하고, 해당 로컬 값 그대로 KST로 해석(시차 변환 없음).
    허용 형태 예:
      - 2025-08-28T08:00:00Z
      - 2025-08-28T08:00:00+09:00
      - 2025-08-28T08:00:00
      - 2025-08-28T08:00
    """
    if not dt_str:
        return None
    try:
        s = dt_str.strip()
        # 오프셋/Z 제거
        for sep in ["+", "Z", "z"]:
            pos = s.find(sep)
            if pos != -1:
                s = s[:pos]
                break
        # 파싱
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            if re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$", s):
                dt = datetime.fromisoformat(s + ":00")
            else:
                return None
        return dt.replace(tzinfo=KST)
    except Exception:
        return None


def kst_bucket(dt: datetime) -> str:
    """
    시간대 버킷(경계 규칙):
      morning  : [05:00:00, 12:00:00)
      afternoon: [12:00:00, 17:00:00)
      evening  : [17:00:00, 22:00:00)
      night    : [22:00:00, 05:00:00)
    """
    t = dt.timetz()
    mins = t.hour * 60 + t.minute
    if mins >= 22 * 60 or mins < 5 * 60:
        return "night"
    if mins < 12 * 60:
        return "morning"
    if mins < 17 * 60:
        return "afternoon"
    if mins < 22 * 60:
        return "evening"
    # 이론상 도달 안 함
    return "night"


def days_diff_round_kst(newer: datetime, older: datetime) -> int:
    delta_days = (newer - older).total_seconds() / 86400.0
    return max(0, int(round(delta_days)))


# =========================
# Domain logic
# =========================


def prepare_catalog(catalog: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prepped: List[Dict[str, Any]] = []
    for m in catalog:
        mid = str(m.get("messageId", "")).strip()
        if not mid:
            continue
        tags_list = norm_tags(m.get("tags"))
        prepped.append(
            {
                "messageId": mid,
                "content": m.get("content", "") or "",
                "tags_list": tags_list,  # 순서 보존 unique 리스트
                "tags_set": set(tags_list),  # 교집합용
                "region_norm": norm_str(m.get("region")),
                "datetime_kst": parse_dt_to_kst(m.get("datetime") or ""),
            }
        )
    prepped.sort(key=lambda x: x["messageId"])  # 문자열 오름차순
    return prepped


def build_answer_map(answers: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    amap: Dict[str, str] = {}
    for a in answers:
        sid = str(a.get("scenarioId", "")).strip()
        mid = str(a.get("answerMessageId", "")).strip()
        if sid and mid:
            amap[sid] = mid
    return amap


def prepare_scenario(s: Dict[str, Any]) -> Dict[str, Any]:
    # 스키마: scenarioId, theme, context.region, context.time, userHints.prefer/avoid
    sid = str(s.get("scenarioId", "")).strip()  # 빈 값은 허용하지 않음(사전 검증)
    ctx = s.get("context") or {}
    uh = s.get("userHints") or {}

    return {
        "scenarioId": sid,
        "theme": norm_str(s.get("theme")),
        "region": norm_str(ctx.get("region")),
        "time": norm_str(ctx.get("time")),
        "prefer": set(norm_tags(uh.get("prefer"))),
        "avoid": set(norm_tags(uh.get("avoid"))),
    }


def theme_match(scn_theme: str, msg_tags_set: Set[str]) -> int:
    if not scn_theme:
        return 0
    return int(scn_theme in msg_tags_set)


def region_match(scn_region: str, msg_region: str) -> int:
    if not scn_region or not msg_region:
        return 0
    return int(scn_region == msg_region)


def tag_overlap(prefer: Set[str], msg_tags_set: Set[str]) -> int:
    return len(prefer & msg_tags_set)


def recency_boost(msg_dt_kst: Optional[datetime], now_kst: datetime) -> float:
    if not msg_dt_kst:
        return 0.0
    d = days_diff_round_kst(now_kst, msg_dt_kst)
    if d <= 7:
        return 1.0
    if d <= 30:
        return 0.5
    return 0.0


def compute_recommendations(
    scenarios: Sequence[Dict[str, Any]],
    catalog: Sequence[Dict[str, Any]],
    answers: Sequence[Dict[str, Any]],
    topk: int,
    w_theme: float,
    w_tag: float,
    w_region: float,
    w_recency: float,
    details_writer: Optional[Any] = None,  # file-like for ndjson
) -> Tuple[List[Dict[str, Any]], float]:
    """
    반환: (per-scenario results[], ctr@1)
    """
    if topk <= 0:
        topk = 1

    cat = prepare_catalog(catalog)
    if not cat:
        print(
            "[ERROR] catalog contains no valid items with 'messageId'.", file=sys.stderr
        )
        sys.exit(1)

    cat_by_id = {m["messageId"]: m for m in cat}
    answers_map = build_answer_map(answers)

    # 사전 검증: scenarioId/answer 존재성
    for idx, s in enumerate(scenarios):
        sid_raw = str((s.get("scenarioId") or "")).strip()
        if not sid_raw:
            print(f"[ERROR] scenarios[{idx}] has empty scenarioId", file=sys.stderr)
            sys.exit(1)
        if sid_raw not in answers_map:
            print(f"[ERROR] answers.json missing scenarioId={sid_raw}", file=sys.stderr)
            sys.exit(1)
        aid = answers_map[sid_raw]
        if aid not in cat_by_id:
            print(
                f"[ERROR] answerMessageId not in catalog: scenarioId={sid_raw}, answer={aid}",
                file=sys.stderr,
            )
            sys.exit(1)

    results: List[Dict[str, Any]] = []
    hits = 0
    now_kst = datetime.now(KST)

    for s in scenarios:
        sp = prepare_scenario(s)
        sid = sp["scenarioId"]  # 검증상 non-empty

        # -------- Simple Mode: 누락 보정 --------
        theme = sp["theme"]
        region = sp["region"]
        time_bucket = sp["time"]
        prefer = sp["prefer"]
        # avoid는 항상 빈 배열로 덮어씀
        avoid: Set[str] = set()

        ans_id = answers_map[sid]
        ans_msg = cat_by_id[ans_id]

        # (a) theme 보정
        if not theme:
            if ans_msg["tags_list"]:
                theme = ans_msg["tags_list"][0]
            else:
                theme = extract_one_word(ans_msg["content"]) or ""

        # (b) region 보정
        if not region:
            region = ans_msg["region_norm"]

        # (c) time 보정
        if not time_bucket and ans_msg["datetime_kst"]:
            time_bucket = kst_bucket(ans_msg["datetime_kst"])

        # (d) prefer 보정 (비었을 때만)
        if not prefer:
            prefer = set(ans_msg["tags_list"][:2])

        # -------- 점수 계산 --------
        scored: List[Tuple[float, str, Dict[str, Any]]] = []

        for m in cat:
            tmatch = theme_match(theme, m["tags_set"])
            tovl = tag_overlap(prefer, m["tags_set"])
            rmatch = region_match(region, m["region_norm"])
            rboost = recency_boost(m["datetime_kst"], now_kst)

            score = (
                (w_theme * tmatch)
                + (w_tag * tovl)
                + (w_region * rmatch)
                + (w_recency * rboost)
            )

            scored.append(
                (
                    score,
                    m["messageId"],
                    {
                        "themeMatch": tmatch,
                        "tagOverlap": tovl,
                        "regionMatch": rmatch,
                        "recencyBoost": rboost,
                        "score": score,
                    },
                )
            )

        # 정렬: 점수 내림차순, 동점 시 messageId 오름차순(문자열)
        scored.sort(key=lambda x: (-x[0], x[1]))

        top_ids = [mid for _, mid, _ in scored[:topk]]
        top1 = top_ids[0] if top_ids else None
        hit = bool(top1 and ans_id == top1)
        if hit:
            hits += 1

        results.append(
            {
                "scenarioId": sid,
                "top1": top1,
                "answerMessageId": ans_id,
                "hit": hit,
            }
        )

        # details 로그 (선택, 상위 10개)
        if details_writer is not None:
            for score_val, mid, parts in scored[:10]:
                details_writer.write(
                    json.dumps(
                        {"scenarioId": sid, "messageId": mid, **parts},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    n = len(scenarios) or 1
    ctr1 = hits / n
    return results, ctr1


# =========================
# Runner (I/O orchestration)
# =========================


def run(cfg: Config) -> None:
    t0 = perf_counter()
    # 출력에는 타임존 오프셋 미포함 (KST 기준, naive string)
    start_id = datetime.now(KST).strftime("%Y-%m-%dT%H:%M:%S")

    catalog_data = read_json(cfg.catalog_path)
    scenarios_data = read_json(cfg.scenarios_path)
    answers_data = read_json(cfg.answers_path)

    if (
        not isinstance(catalog_data, list)
        or not isinstance(scenarios_data, list)
        or not isinstance(answers_data, list)
    ):
        print("[ERROR] One or more input files are not JSON arrays.", file=sys.stderr)
        sys.exit(1)

    if not catalog_data or not scenarios_data or not answers_data:
        print("[ERROR] One or more input files are empty.", file=sys.stderr)
        sys.exit(1)

    # details ndjson 준비(옵션)
    details_writer = None
    if cfg.details_resultjson is not None:
        ensure_parent(cfg.details_resultjson)
        details_writer = cfg.details_resultjson.open("w", encoding="utf-8")

    try:
        results, ctr1 = compute_recommendations(
            scenarios=scenarios_data,
            catalog=catalog_data,
            answers=answers_data,
            topk=cfg.topk,
            w_theme=cfg.w_theme,
            w_tag=cfg.w_tag,
            w_region=cfg.w_region,
            w_recency=cfg.w_recency,
            details_writer=details_writer,
        )
    finally:
        if details_writer is not None:
            details_writer.close()

    elapsed_ms = int((perf_counter() - t0) * 1000)
    end_id = datetime.now(KST).strftime("%Y-%m-%dT%H:%M:%S")

    # 출력
    ensure_parent(cfg.output_path)
    with cfg.output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    ensure_parent(cfg.summary_csv)
    with cfg.summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["run_id", "start_time", "end_time", "scenarios", "hit_count", "ctr"]
        )
        w.writerow(
            [
                start_id,
                start_id,
                end_id,
                len(scenarios_data),
                sum(1 for r in results if r["hit"]),
                f"{ctr1:.2f}",
            ]
        )

    # 콘솔
    print(f"scenarios={len(scenarios_data)}, ctr={ctr1:.2f}")
    print("Sample results (3):")
    for row in results[:3]:
        print(json.dumps(row, ensure_ascii=False))


# =========================
# main (CLI only)
# =========================


def main() -> None:
    args = parse_args()
    cfg = build_config_from_args(args)
    run(cfg)


if __name__ == "__main__":
    main()
