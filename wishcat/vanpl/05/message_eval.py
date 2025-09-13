#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Message CTR@1 evaluator (Spec #5)
- main(): CLI 파싱만
- run(): 파일 I/O 및 오케스트레이션
- compute_recommendations(): 점수 계산/Top-1/CTR 산출

[ASSUMPTION] 표시는 명세가 모호해 임의로 결정한 부분을 뜻합니다.
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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


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
    details_ndjson: Optional[Path]  # recs.ndjson (옵션)


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
    p = argparse.ArgumentParser(
        description="Personalized Message CTR@1 evaluator (full)."
    )
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
        dest="details_ndjson",
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
        if args.details_ndjson is None:
            args.details_ndjson = None

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
    # [ASSUMPTION] 키 누락 시 0.0으로 봄(명세 미정)
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
        details_ndjson=args.details_ndjson,
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
    # [ASSUMPTION] 간단 정규화: strip + lower (유니코드 정규화/동의어 매핑은 미구현)
    return (s or "").strip().lower()


def norm_tags(xs: Optional[Iterable[str]]) -> List[str]:
    # [ASSUMPTION] 중복 제거 후 사전순 (TagOverlap 정의 모호 → 집합 겹침으로 가정)
    return sorted({norm_str(x) for x in (xs or []) if isinstance(x, str)})


WORD_RE = re.compile(r"[A-Za-z0-9가-힣]+", re.UNICODE)


def extract_one_word(text: str) -> Optional[str]:
    # [ASSUMPTION] content에서 "단어 1개" 추출 규칙: 영문/숫자/한글로 이뤄진 첫 토큰
    # - 불용어 제거 없음
    # - 부분문자열 금지: 토큰 그대로 사용
    if not text:
        return None
    m = WORD_RE.search(text)
    return m.group(0).lower() if m else None


def parse_dt_to_kst(dt_str: str) -> Optional[datetime]:
    # [ASSUMPTION] ISO8601 가정. 'Z'는 UTC로 보고 KST로 변환. 오프셋 없으면 KST로 간주.
    if not dt_str:
        return None
    try:
        # try fromisoformat (Python 3.11+ 괜찮음)
        if dt_str.endswith("Z"):
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00")).astimezone(KST)
        else:
            dt = datetime.fromisoformat(dt_str)
            if dt.tzinfo is None:
                # 오프셋 없으면 KST로 간주
                dt = dt.replace(tzinfo=KST)
            else:
                dt = dt.astimezone(KST)
        return dt
    except Exception:
        return None


def kst_bucket(dt: datetime) -> str:
    # [ASSUMPTION] 시간대 버킷(경계 포함 기준):
    # morning: 06:00–11:59, afternoon: 12:00–17:59, evening: 18:00–21:59, night: 22:00–05:59
    h = dt.hour
    if 6 <= h <= 11:
        return "morning"
    if 12 <= h <= 17:
        return "afternoon"
    if 18 <= h <= 21:
        return "evening"
    return "night"


def days_diff_floor_kst(newest: datetime, older: datetime) -> int:
    # [ASSUMPTION] 일수는 내림(floor)으로 계산
    delta = newest - older
    return max(0, int(delta.total_seconds() // 86400))


# =========================
# Domain logic
# =========================


def prepare_catalog(catalog: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prepped: List[Dict[str, Any]] = []
    for m in catalog:
        mid = str(m.get("messageId", "")).strip()
        if not mid:
            continue
        prepped.append(
            {
                "messageId": mid,
                "content": m.get("content", "") or "",
                "tags_norm": set(norm_tags(m.get("tags"))),
                "region_norm": norm_str(m.get("region")),
                "datetime_kst": parse_dt_to_kst(m.get("datetime") or ""),
            }
        )
    prepped.sort(key=lambda x: x["messageId"])
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
    sid = str(s.get("scenarioId", "")).strip() or None
    theme = norm_str(s.get("theme"))

    ctx = s.get("context") or {}
    region = norm_str(ctx.get("region"))
    time_bucket = norm_str(ctx.get("time"))

    uh = s.get("userHints") or {}
    prefer = set(norm_tags(uh.get("prefer")))
    avoid = set(norm_tags(uh.get("avoid")))

    return {
        "scenarioId": sid,
        "theme": theme,
        "region": region,
        "time": time_bucket,
        "prefer": prefer,
        "avoid": avoid,
    }


def theme_match(scn_theme: str, msg_tags: set, msg_content: str) -> int:
    # 정확히 같은 단어 매칭(대소문자 무시), 부분문자열 불허
    if not scn_theme:
        return 0
    if scn_theme in msg_tags:
        return 1
    # content 단어 토큰화 후 정확 일치 검사
    # [ASSUMPTION] 동일 WORD_RE 사용
    if msg_content:
        for tok in WORD_RE.findall(msg_content.lower()):
            if tok == scn_theme:
                return 1
    return 0


def region_match(scn_region: str, msg_region: str) -> int:
    # [ASSUMPTION] 둘 중 하나라도 비면 0점 (정책: zero)
    if not scn_region or not msg_region:
        return 0
    return int(scn_region == msg_region)


def tag_overlap(prefer: set, msg_tags: set) -> int:
    return len(prefer & msg_tags)


def recency_boost(
    msg_dt_kst: Optional[datetime], newest_kst: Optional[datetime]
) -> float:
    # 버킷: ≤7 → 1.0, 8–30 → 0.5, >30 → 0.0 (KST 기준, 일수 내림)
    if not msg_dt_kst or not newest_kst:
        return 0.0
    d = days_diff_floor_kst(newest_kst, msg_dt_kst)
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

    # 최신시각(KST) 찾기 (Recency 기준)
    newest_kst = None
    for m in cat:
        dt = m["datetime_kst"]
        if dt and (newest_kst is None or dt > newest_kst):
            newest_kst = dt

    answers_map = build_answer_map(answers)

    results: List[Dict[str, Any]] = []
    hits = 0

    for idx, s in enumerate(scenarios):
        sp = prepare_scenario(s)
        sid = sp["scenarioId"] or f"S{idx+1:03d}"

        # -------- Simple Mode: 누락 보정 --------
        # theme
        theme = sp["theme"]
        region = sp["region"]
        time_bucket = sp["time"]
        prefer = sp["prefer"]
        avoid = sp["avoid"]

        ans_id = answers_map.get(sid)
        ans_msg = next((m for m in cat if m["messageId"] == ans_id), None)

        # (a) theme 보정
        if not theme and ans_msg:
            # [ASSUMPTION] "첫 번째 태그": 카탈로그의 태그 **원래 순서** 대신, 정규화·중복제거로 순서가 사라졌으므로
            #             여기서는 사전순 첫 태그를 사용함. (원본 순서 유지 명세 없음)
            tags_sorted = sorted(ans_msg["tags_norm"])
            if tags_sorted:
                theme = tags_sorted[0]
            else:
                # 태그 없으면 content에서 "한 단어"
                theme = extract_one_word(ans_msg["content"]) or ""

        # (b) region 보정
        if not region and ans_msg:
            region = ans_msg["region_norm"]

        # (c) time 보정
        if not time_bucket and ans_msg and ans_msg["datetime_kst"]:
            time_bucket = kst_bucket(ans_msg["datetime_kst"])

        # (d) userHints 보정
        if not prefer and ans_msg:
            tags_sorted = sorted(ans_msg["tags_norm"])
            # [ASSUMPTION] "상위 2개": 정렬 기준이 불명 → 사전순 2개
            prefer = set(tags_sorted[:2])
        if not avoid:
            avoid = set()  # 명세대로 빈 배열

        # -------- 점수 계산 --------
        scored: List[Tuple[float, str, Dict[str, Any]]] = []

        for m in cat:
            tmatch = theme_match(theme, m["tags_norm"], m["content"])
            tovl = tag_overlap(prefer, m["tags_norm"])
            rmatch = region_match(region, m["region_norm"])
            rboost = recency_boost(m["datetime_kst"], newest_kst)

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

        # 정렬: 점수 내림차순, 동점 시 messageId 오름차순
        scored.sort(key=lambda x: (-x[0], x[1]))

        top_ids = [mid for _, mid, _ in scored[:topk]]
        top1 = top_ids[0] if top_ids else None
        hit = bool(ans_id and top1 and ans_id == top1)
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

        # details 로그 (선택)
        if details_writer is not None:
            # 상위 10개만 덤프(가독성) [ASSUMPTION]
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
    start_id = datetime.now(KST).strftime("%Y-%m-%dT%H:%M:%S%z")

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
    if cfg.details_ndjson is not None:
        ensure_parent(cfg.details_ndjson)
        details_writer = cfg.details_ndjson.open("w", encoding="utf-8")

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
    end_id = datetime.now(KST).strftime("%Y-%m-%dT%H:%M:%S%z")

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
