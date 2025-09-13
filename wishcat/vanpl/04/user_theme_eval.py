#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# =========================
# Config / CLI
# =========================

@dataclass
class Config:
    users_path: Path
    themes_path: Path
    answers_path: Path
    topk: int
    w_theme: float  # weights['cat'] -> ThemeMatch
    w_tag: float
    w_region: float
    output_path: Path        # JSON 배열로 저장 (result.json 등)
    summary_csv: Path        # per-user CSV log


def parse_weights(s: str) -> Dict[str, float]:
    """
    Parse weights string like "cat=2.0,tag=1.0,region=1.5".
    Only keys {cat, tag, region} are used. Others are ignored.
    """
    out: Dict[str, float] = {}
    if not s:
        return out
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            print(f"[ERROR] Invalid weight token (expected k=v): {part}", file=sys.stderr)
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
        description="User×Theme mapping evaluator (ThemeMatch + TagOverlap + RegionMatch)."
    )
    # 입력 파일
    p.add_argument("--themes", dest="themes_path", type=Path, help="Path to themes.json")
    p.add_argument("--users", dest="users_path", type=Path, help="Path to user_profiles.json")
    p.add_argument("--answers", dest="answers_path", type=Path, help="Path to answers.json")
    # 파라미터
    p.add_argument("--topk", dest="topk", type=int, help="Top-K (e.g., 5)")
    p.add_argument("--weights", dest="weights", type=str,
                   help='Weights, e.g. "cat=2.0,tag=1.0,region=1.5"  (cat==ThemeMatch)')
    # 출력
    p.add_argument("--output", dest="output_path", type=Path, help="Path to result.json (array)")
    p.add_argument("--log", dest="summary_csv", type=Path, help="Path to summary.csv")
    # 테스트 고정값
    p.add_argument("--test", action="store_true", help="Use fixed test arguments/paths")
    return p.parse_args()


def build_config_from_args(args: argparse.Namespace) -> Config:
    # --test 모드 고정값 주입
    if args.test:
        args.themes_path = Path("./input/themes.json")
        args.users_path = Path("./input/user_profiles.json")
        args.answers_path = Path("./input/answers.json")
        args.topk = 5
        args.weights = "cat=2.0,tag=1.0,region=1.5"
        args.output_path = Path("./output/result.json")
        args.summary_csv = Path("./log/summary.csv")

    # 필수 인자 체크
    missing = []
    if not args.themes_path:  missing.append("--themes")
    if not args.users_path:   missing.append("--users")
    if not args.answers_path: missing.append("--answers")
    if args.topk is None:     missing.append("--topk")
    if not args.weights:      missing.append("--weights")
    if not args.output_path:  missing.append("--output")
    if not args.summary_csv:  missing.append("--log")
    if missing:
        print(f"[ERROR] Missing required arguments: {' '.join(missing)}", file=sys.stderr)
        sys.exit(2)

    # 가중치 파싱 (cat==ThemeMatch)
    w = parse_weights(args.weights)
    w_theme = float(w.get("cat", 0.0))
    w_tag = float(w.get("tag", 0.0))
    w_region = float(w.get("region", 0.0))

    return Config(
        users_path=args.users_path,
        themes_path=args.themes_path,
        answers_path=args.answers_path,
        topk=int(args.topk),
        w_theme=w_theme,
        w_tag=w_tag,
        w_region=w_region,
        output_path=args.output_path,
        summary_csv=args.summary_csv,
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
# Normalization
# =========================

def _norm_str(s: Optional[str]) -> str:
    # 간단 정규화: 트림 + 소문자화 (region 추가 규칙은 요구대로 생략)
    return (s or "").strip().lower()


def _norm_str_list(xs: Optional[Iterable[str]]) -> List[str]:
    # 중복 제거 + 정렬된 리스트(결정론)
    return sorted({_norm_str(x) for x in (xs or []) if isinstance(x, str)})


# =========================
# Domain logic
# =========================

def prepare_themes(themes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    themes.json 항목 정규화:
      - themeId (문자열)
      - tags_norm (정규화된 태그 set)
      - region_norm (정규화된 지역)
    """
    prepped: List[Dict[str, Any]] = []
    for t in themes:
        tid = str(t.get("themeId", "")).strip()
        if not tid:
            continue  # themeId 없는 항목은 스킵
        prepped.append({
            "themeId": tid,
            "tags_norm": set(_norm_str_list(t.get("tags"))),
            "region_norm": _norm_str(t.get("region")),
        })
    # 결정론 보장 위해 themeId 사전순으로 정렬
    prepped.sort(key=lambda x: x["themeId"])
    return prepped


def extract_user_profile(u: Dict[str, Any]) -> Dict[str, Any]:
    """
    user_profiles.json 항목에서 필요한 필드 추출/정규화
      - userId
      - prefer_norm (태그)
      - region_norm
      (avoidTags, category 등은 이번 평가에서 미사용)
    """
    uid_raw = u.get("userId", None)
    uid = str(uid_raw).strip() if isinstance(uid_raw, str) else (uid_raw if uid_raw is not None else None)
    prefer_norm = set(_norm_str_list(u.get("preferTags")))
    region_norm = _norm_str(u.get("region"))
    return {
        "userId": uid,           # 정책 B: 누락 시 None 유지(→ 0점 처리)
        "prefer_norm": prefer_norm,
        "region_norm": region_norm,
    }


def score_one(theme: Dict[str, Any],
              user: Dict[str, Any],
              w_theme: float,
              w_tag: float,
              w_region: float) -> float:
    """
    점수식:
      score = (ThemeMatch * w_theme) + (TagOverlap * w_tag) + (RegionMatch * w_region)
    - ThemeMatch: 1 if |prefer ∩ theme.tags| >= 1 else 0
    - TagOverlap: |prefer ∩ theme.tags|  (상한 없음)
    - RegionMatch: 1 if user.region == theme.region (간단 정규화 비교), else 0
    """
    prefer_overlap = len(user["prefer_norm"] & theme["tags_norm"])
    theme_match = 1 if prefer_overlap >= 1 else 0
    region_match = int(bool(user["region_norm"]) and bool(theme["region_norm"]) and (user["region_norm"] == theme["region_norm"]))
    score = (theme_match * w_theme) + (prefer_overlap * w_tag) + (region_match * w_region)
    return float(score)


def compute_recommendations(users: Sequence[Dict[str, Any]],
                             themes: Sequence[Dict[str, Any]],
                             answers: Sequence[Dict[str, Any]],
                             topk: int,
                             w_theme: float,
                             w_tag: float,
                             w_region: float
                             ) -> Tuple[List[Dict[str, Any]], float, float]:
    """
    전체 추천 계산 + CTR 산출
    반환: (per-user results[], ctr@K, ctr@1)
    """
    if topk <= 0:
        topk = 1

    themes_prepped = prepare_themes(themes)
    if not themes_prepped:
        print("[ERROR] themes.json contains no valid items with 'themeId'.", file=sys.stderr)
        sys.exit(1)

    # answer map
    answer_map: Dict[str, str] = {}
    for a in answers:
        uid = str(a.get("userId", "")).strip()
        tid = str(a.get("answerThemeId", "")).strip()
        if uid and tid:
            answer_map[uid] = tid

    # answers 정합성 체크: 모든 answerThemeId가 themes에 존재해야 함
    theme_ids = {t["themeId"] for t in themes_prepped}
    invalid_answers = sorted({tid for tid in answer_map.values() if tid not in theme_ids})
    if invalid_answers:
        print(f"[ERROR] answerThemeId not found in themes: {', '.join(invalid_answers)}", file=sys.stderr)
        sys.exit(1)

    results: List[Dict[str, Any]] = []
    hit_k_count = 0
    hit1_count = 0

    for idx, u in enumerate(users):
        upd = extract_user_profile(u)

        # 정책 A/B 반영:
        # - 가짜 userId 부여 제거(그대로 None 허용)
        # - answers에 없으면 ans=None → 항상 hit=False
        uid = upd["userId"]

        # 모든 테마 점수 계산
        scored: List[Tuple[float, str]] = []
        for th in themes_prepped:
            s = score_one(th, upd, w_theme, w_tag, w_region)
            scored.append((s, th["themeId"]))

        # 정렬: 점수 내림차순, 동점 시 themeId 오름차순(사전식)
        scored.sort(key=lambda x: (-x[0], x[1]))

        # Top-K themeIds 추출
        top_ids = [tid for _, tid in scored[:topk]]
        top_key = f"top{topk}"

        # 정답/Hit
        ans = answer_map.get(uid) if uid is not None else None
        hit_k = bool(ans in top_ids) if ans else False
        hit1 = bool(ans == (top_ids[0] if top_ids else None)) if ans else False
        if hit_k:
            hit_k_count += 1
        if hit1:
            hit1_count += 1

        results.append({
            "userId": uid,
            top_key: top_ids,
            "answerThemeId": ans,
            "hit": hit_k,
        })

    n = len(users) or 1
    ctr_k = hit_k_count / n
    ctr1 = hit1_count / n

    return results, ctr_k, ctr1


# =========================
# Runner (I/O)
# =========================

def run(cfg: Config) -> None:
    t0 = perf_counter()

    # 입력 로드
    users_data = read_json(cfg.users_path)
    themes_data = read_json(cfg.themes_path)
    answers_data = read_json(cfg.answers_path)

    if not isinstance(users_data, list) or not isinstance(themes_data, list) or not isinstance(answers_data, list):
        print("[ERROR] One or more input files are not JSON arrays.", file=sys.stderr)
        sys.exit(1)

    if not users_data or not themes_data or not answers_data:
        print("[ERROR] One or more input files are empty.", file=sys.stderr)
        sys.exit(1)

    # 계산
    results, ctr_k, ctr1 = compute_recommendations(
        users=users_data,
        themes=themes_data,
        answers=answers_data,
        topk=cfg.topk,
        w_theme=cfg.w_theme,
        w_tag=cfg.w_tag,
        w_region=cfg.w_region,
    )

    elapsed_ms = int((perf_counter() - t0) * 1000)

    # 출력 경로 보장
    ensure_parent(cfg.output_path)
    ensure_parent(cfg.summary_csv)

    # JSON 배열로 저장 (현재 정책: NDJSON 요구 없음)
    with cfg.output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # summary.csv (유저별 결과 로그: 동적 top 컬럼)
    top_key = f"top{cfg.topk}"
    with cfg.summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        header = ["userId", "answerThemeId", "hit"] + [f"top{i}" for i in range(1, cfg.topk + 1)]
        w.writerow(header)
        for r in results:
            tops: List[str] = r.get(top_key, [])
            # 정책 B: None을 공백으로 표시
            uid = r.get("userId") or ""
            ans = r.get("answerThemeId") or ""
            row = [uid, ans, "true" if r.get("hit") else "false"] + tops
            w.writerow(row)

    # 콘솔 출력
    print(f"users={len(users_data)}, ctr@{cfg.topk}={ctr_k:.2f}, ctr@1={ctr1:.2f}, elapsed_ms={elapsed_ms}")
    print("Sample results (top 3):")
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
