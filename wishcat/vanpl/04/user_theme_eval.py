#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
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
    w_cat: float
    w_tag: float
    w_region: float
    avoid_penalty: float
    output_path: Path         # reasion.json
    summary_csv: Path         # per-user log
    run_csv: Path             # run summary


def parse_weights(s: str) -> Dict[str, float]:
    """
    Parse weights string like "cat=2.0,tag=1.0,region=1.5".
    Unknown keys are ignored; missing keys default to 0.0 (handled by caller).
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
        description="User×Theme mapping evaluator (full implementation)."
    )
    # 입력 파일
    p.add_argument("--themes", dest="themes_path", type=Path, help="Path to themes.json")
    p.add_argument("--users", dest="users_path", type=Path, help="Path to user_profiles.json")
    p.add_argument("--answers", dest="answers_path", type=Path, help="Path to answers.json")
    # 파라미터
    p.add_argument("--topk", dest="topk", type=int, help="Top-K (e.g., 5)")
    p.add_argument("--weights", dest="weights", type=str,
                   help='Weights, e.g. "cat=2.0,tag=1.0,region=1.5"')
    p.add_argument("--avoid-penalty", dest="avoid_penalty", type=float,
                   help="Penalty per avoidTag match (e.g., 1.0)")
    # 출력
    p.add_argument("--output", dest="output_path", type=Path, help="Path to reasion.json")
    p.add_argument("--log", dest="summary_csv", type=Path, help="Path to summary.csv")
    p.add_argument("--runlog", dest="run_csv", type=Path, help="Path to run.csv")
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
        args.avoid_penalty = 1.0
        args.output_path = Path("./output/reasion.json")  # 요구사항 고정명
        args.summary_csv = Path("./log/summary.csv")
        args.run_csv = Path("./log/run.csv")

    # 필수 인자 체크
    missing = []
    if not args.themes_path:  missing.append("--themes")
    if not args.users_path:   missing.append("--users")
    if not args.answers_path: missing.append("--answers")
    if args.topk is None:     missing.append("--topk")
    if not args.weights:      missing.append("--weights")
    if args.avoid_penalty is None: missing.append("--avoid-penalty")
    if not args.output_path:  missing.append("--output")
    if not args.summary_csv:  missing.append("--log")
    if not args.run_csv:      missing.append("--runlog")
    if missing:
        print(f"[ERROR] Missing required arguments: {' '.join(missing)}", file=sys.stderr)
        sys.exit(2)

    # 가중치 파싱
    w = parse_weights(args.weights)
    w_cat = float(w.get("cat", 0.0))
    w_tag = float(w.get("tag", 0.0))
    w_region = float(w.get("region", 0.0))

    return Config(
        users_path=args.users_path,
        themes_path=args.themes_path,
        answers_path=args.answers_path,
        topk=int(args.topk),
        w_cat=w_cat,
        w_tag=w_tag,
        w_region=w_region,
        avoid_penalty=float(args.avoid_penalty),
        output_path=args.output_path,
        summary_csv=args.summary_csv,
        run_csv=args.run_csv,
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
# Domain logic
# =========================

def _norm_str(s: Optional[str]) -> str:
    # 단순 공백 트림 + 소문자화 (한글은 영향 없음)
    return (s or "").strip().lower()


def _norm_str_list(xs: Optional[Iterable[str]]) -> List[str]:
    return sorted({_norm_str(x) for x in (xs or []) if isinstance(x, str)})


def prepare_themes(themes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    themes.json 항목 정규화:
      - themeId (문자열)
      - tags_norm (정규화된 태그 set)
      - region_norm (정규화된 지역)
      - categoryId (원본 유지)
    """
    prepped: List[Dict[str, Any]] = []
    for t in themes:
        tid = str(t.get("themeId", "")).strip()
        if not tid:
            # themeId 없는 항목은 스킵
            continue
        prepped.append({
            "themeId": tid,
            "tags_norm": set(_norm_str_list(t.get("tags"))),
            "region_norm": _norm_str(t.get("region")),
            "categoryId": t.get("categoryId"),  # 그대로 둠 (비교만 1/0)
        })
    # 결정론 보장 위해 themeId 사전순으로 한 번 정렬
    prepped.sort(key=lambda x: x["themeId"])
    return prepped


def extract_user_profile(u: Dict[str, Any]) -> Dict[str, Any]:
    """
    user_profiles.json 항목에서 필요한 필드 추출/정규화
      - userId
      - prefer_norm (태그)
      - avoid_norm (태그)
      - region_norm
      - recent_categories (옵션: 'recentCategories' 또는 'categories' 를 지원; 없으면 빈 set)
    """
    uid = str(u.get("userId", "")).strip() or None
    prefer_norm = set(_norm_str_list(u.get("preferTags")))
    avoid_norm = set(_norm_str_list(u.get("avoidTags")))
    region_norm = _norm_str(u.get("region"))

    # 카테고리: 사양에 명시됐지만 실제 파일에 없을 수 있어 optional 처리
    cat_src = u.get("recentCategories", u.get("categories", []))
    if isinstance(cat_src, dict):  # 혹시 dict면 값만 추출
        cat_src = list(cat_src.values())
    recent_categories = set(_norm_str_list(cat_src if isinstance(cat_src, list) else []))

    return {
        "userId": uid,
        "prefer_norm": prefer_norm,
        "avoid_norm": avoid_norm,
        "region_norm": region_norm,
        "recent_categories": recent_categories,
    }


def score_one(theme: Dict[str, Any],
              user: Dict[str, Any],
              w_cat: float,
              w_tag: float,
              w_region: float,
              avoid_penalty: float) -> float:
    """
    점수식:
      score = (category_match * w_cat) + (tag_overlap * w_tag) + (region_match * w_region) - (avoid_overlap * avoid_penalty)
    - category_match: theme.categoryId 가 user.recent_categories 안에 있으면 1 아니면 0
      (user 카테고리 정보가 없으면 0)
    - tag_overlap: |user.prefer_norm ∩ theme.tags_norm|
    - region_match: user.region_norm 과 theme.region_norm 이 비어있지 않고 같으면 1 아니면 0
    - avoid_overlap: |user.avoid_norm ∩ theme.tags_norm|
    """
    # category
    cat = theme.get("categoryId")
    cat_match = 0
    if cat is not None:
        if _norm_str(str(cat)) in user["recent_categories"]:
            cat_match = 1

    # tag overlaps
    prefer_overlap = len(user["prefer_norm"] & theme["tags_norm"])
    avoid_overlap = len(user["avoid_norm"] & theme["tags_norm"])

    # region
    r_user = user["region_norm"]
    r_theme = theme["region_norm"]
    region_match = int(bool(r_user) and bool(r_theme) and (r_user == r_theme))

    score = (cat_match * w_cat) + (prefer_overlap * w_tag) + (region_match * w_region) - (avoid_overlap * avoid_penalty)
    return float(score)


def compute_recommendations(users: Sequence[Dict[str, Any]],
                             themes: Sequence[Dict[str, Any]],
                             answers: Sequence[Dict[str, Any]],
                             topk: int,
                             w_cat: float,
                             w_tag: float,
                             w_region: float,
                             avoid_penalty: float
                             ) -> Tuple[List[Dict[str, Any]], float, float]:
    """
    전체 추천 계산 + CTR 산출
    반환: (per-user results[], ctr@5, ctr@1)
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

    results: List[Dict[str, Any]] = []

    for idx, u in enumerate(users):
        upd = extract_user_profile(u)
        uid = upd["userId"] or f"U{idx+1:03d}"

        # 모든 테마 점수 계산
        scored: List[Tuple[float, str]] = []
        for th in themes_prepped:
            s = score_one(th, upd, w_cat, w_tag, w_region, avoid_penalty)
            scored.append((s, th["themeId"]))

        # 정렬: 점수 내림차순, 동점 시 themeId 오름차순
        # Python sort는 안정적이므로 key로 (-score, themeId) 사용
        scored.sort(key=lambda x: (-x[0], x[1]))

        # Top-K themeIds 추출
        top_ids = [tid for _, tid in scored[:topk]]

        # 정답/Hit
        ans = answer_map.get(uid)
        hit5 = bool(ans in top_ids) if ans else False
        hit1 = bool(ans == (top_ids[0] if top_ids else None)) if ans else False

        results.append({
            "userId": uid,
            "top5": top_ids[:5],              # 명세상 Top-5 기록 (topk가 5가 아닐 수도 있으나 상위 5만 표기)
            "answerThemeId": ans,
            "hit": hit5,
        })

    n = len(users) or 1
    ctr5 = sum(1 for r in results if r["hit"]) / n
    # CTR@1은 실제 상위 1과 비교
    # 위에서 hit1을 따로 보관하지 않았으니 다시 계산
    ctr1_hits = 0
    for idx, u in enumerate(users):
        upd = extract_user_profile(u)
        uid = upd["userId"] or f"U{idx+1:03d}"
        ans = answer_map.get(uid)
        if not ans:
            continue
        # 상위 1 재계산 (효율보다 명확성 우선; n=100·themes~80이라 충분히 빠름)
        scored: List[Tuple[float, str]] = []
        for th in themes_prepped:
            s = score_one(th, upd, w_cat, w_tag, w_region, avoid_penalty)
            scored.append((s, th["themeId"]))
        scored.sort(key=lambda x: (-x[0], x[1]))
        top1 = scored[0][1] if scored else None
        if top1 and ans == top1:
            ctr1_hits += 1
    ctr1 = ctr1_hits / n

    return results, ctr5, ctr1


# =========================
# Runner (I/O orchestration)
# =========================

def run(cfg: Config) -> None:
    t0 = perf_counter()
    run_id = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

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
    results, ctr5, ctr1 = compute_recommendations(
        users=users_data,
        themes=themes_data,
        answers=answers_data,
        topk=cfg.topk,
        w_cat=cfg.w_cat,
        w_tag=cfg.w_tag,
        w_region=cfg.w_region,
        avoid_penalty=cfg.avoid_penalty
    )

    elapsed_ms = int((perf_counter() - t0) * 1000)

    # 출력 경로 보장
    ensure_parent(cfg.output_path)
    ensure_parent(cfg.summary_csv)
    ensure_parent(cfg.run_csv)

    # reasion.json (JSON 배열)
    with cfg.output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # summary.csv (유저별 결과 로그)
    with cfg.summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        header = ["userId", "answerThemeId", "hit", "top1", "top2", "top3", "top4", "top5"]
        w.writerow(header)
        for r in results:
            top = r.get("top5", [])
            row = [
                r.get("userId"),
                r.get("answerThemeId"),
                "true" if r.get("hit") else "false",
                *(top + [""] * (5 - len(top)))  # pad to 5
            ]
            w.writerow(row)

    # run.csv (실행 요약)
    with cfg.run_csv.open("w", encoding="utf-8") as f:
        f.write("run_id,start_time,end_time,users,hit_count,ctr@5,ctr@1,topk,weights,avoid_penalty\n")
        weights_obj = {"cat": cfg.w_cat, "tag": cfg.w_tag, "region": cfg.w_region}
        f.write(f"{run_id},{run_id},{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')},"
                f"{len(users_data)},{int(ctr5*len(users_data))},{ctr5:.2f},{ctr1:.2f},"
                f"{cfg.topk},{json.dumps(weights_obj, ensure_ascii=False)},{cfg.avoid_penalty}\n")

    # 콘솔 출력
    print(f"users={len(users_data)}, ctr@5={ctr5:.2f}, ctr@1={ctr1:.2f}")
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
