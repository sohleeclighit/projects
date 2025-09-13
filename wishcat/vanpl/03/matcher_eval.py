#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import time
from typing import Any, Dict, List, Tuple, Optional
from math import radians, sin, cos, asin, sqrt


# =========================
# 유틸
# =========================
def ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def read_json_array(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"JSON at {path} must be a list of objects.")
    return data


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0088  # km
    rlat1, rlon1, rlat2, rlon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = sin(dlat / 2) ** 2 + cos(rlat1) * cos(rlat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return R * c


def to_id(x: Any) -> str:
    return str(x) if x is not None else ""


def safe_tags(x: Any) -> List[str]:
    if isinstance(x, list):
        return [str(t) for t in x if t is not None]
    return []


def get_scenario_context_latlng(
    s: Dict[str, Any],
) -> Tuple[Optional[float], Optional[float]]:
    ctx = s.get("context") or {}
    lat = ctx.get("lat")
    lng = ctx.get("lng")
    try:
        return (
            (float(lat), float(lng))
            if lat is not None and lng is not None
            else (None, None)
        )
    except Exception:
        return (None, None)


def get_prefer_tags(s: Dict[str, Any]) -> List[str]:
    hints = s.get("userHints") or {}
    return safe_tags(hints.get("prefer"))


# =========================
# 테마 매칭 (태그 기반)
# =========================
def normalize_theme_tag(theme: Any) -> Optional[str]:
    if theme is None:
        return None
    t = str(theme).replace(" ", "")
    return f"테마_{t}"


def theme_matches_by_tags(s_theme: Any, poi_tags: List[str]) -> int:
    ttag = normalize_theme_tag(s_theme)
    if not ttag:
        return 0
    return 1 if ttag in set(safe_tags(poi_tags)) else 0


# =========================
# 점수 계산 & 추천
# =========================
def score_poi_for_scenario(
    s: Dict[str, Any], poi: Dict[str, Any], weights: Dict[str, float]
) -> float:
    th = theme_matches_by_tags(s.get("theme"), poi.get("tags"))
    prefer = set(get_prefer_tags(s))
    poi_tags = set(safe_tags(poi.get("tags")))
    overlap = len(prefer.intersection(poi_tags))

    s_lat, s_lng = get_scenario_context_latlng(s)
    p_lat, p_lng = poi.get("lat"), poi.get("lng")
    dist_km = 0.0
    try:
        if (
            s_lat is not None
            and s_lng is not None
            and p_lat is not None
            and p_lng is not None
        ):
            dist_km = haversine_km(
                float(s_lat), float(s_lng), float(p_lat), float(p_lng)
            )
    except Exception:
        dist_km = 0.0

    return (th * weights["theme"]) + (overlap * weights["tag"]) - (
        dist_km * weights["dist"]
    )


def recommend_top_for_scenario(
    s: Dict[str, Any], catalog: List[Dict[str, Any]], topn: int, weights: Dict[str, float]
) -> Tuple[List[str], float]:
    scored: List[Tuple[str, float]] = []
    for poi in catalog:
        pid = to_id(poi.get("id"))
        sc = score_poi_for_scenario(s, poi, weights)
        scored.append((pid, sc))

    scored.sort(key=lambda x: (-x[1], x[0]))
    top_ids = [pid for pid, _ in scored[:topn]]
    score_top1 = scored[0][1] if scored else float("-inf")
    return top_ids, score_top1


# =========================
# 메인 실행
# =========================
def parse_weights(s: str) -> Dict[str, float]:
    d = {}
    parts = s.split(",")
    for part in parts:
        k, v = part.split("=")
        d[k.strip()] = float(v.strip())
    for key in ("theme", "tag", "dist"):
        if key not in d:
            raise ValueError(f"weight '{key}' missing in {s}")
    return d


def run_once(
    catalog_path: str,
    scenarios_path: str,
    answers_path: str,
    result_path: str,
    log_csv: str,
    topn: int,
    weights: Dict[str, float],
) -> None:
    t0 = time.time()

    catalog = read_json_array(catalog_path)
    scenarios = read_json_array(scenarios_path)
    answers = read_json_array(answers_path)
    ans_map: Dict[str, str] = {
        to_id(a.get("scenarioId")): to_id(a.get("answerId")) for a in answers
    }

    results_json: List[Dict[str, Any]] = []
    rows_csv: List[str] = []
    hit5_count = 0
    hit1_count = 0

    top_key = f"top{topn}"

    for s in scenarios:
        sid = to_id(s.get("scenarioId"))
        answer_id = ans_map.get(sid, "")

        top_ids, score_top1 = recommend_top_for_scenario(s, catalog, topn, weights)
        hit5 = answer_id in top_ids if answer_id else False
        hit1 = (answer_id == top_ids[0]) if (answer_id and top_ids) else False
        if hit5:
            hit5_count += 1
        if hit1:
            hit1_count += 1

        score_answer = None
        if answer_id:
            poi_lookup = next(
                (p for p in catalog if to_id(p.get("id")) == answer_id), None
            )
            if poi_lookup is not None:
                score_answer = score_poi_for_scenario(s, poi_lookup, weights)

        results_json.append(
            {
                "scenarioId": sid,
                top_key: top_ids,
                "answerId": answer_id,
                "hit": hit5,
                "hit1": hit1,
                "scoreTop1": (
                    round(score_top1, 6)
                    if isinstance(score_top1, (int, float))
                    else None
                ),
                "scoreAnswer": (
                    round(score_answer, 6)
                    if isinstance(score_answer, (int, float))
                    else None
                ),
            }
        )

        top_join = ";".join(top_ids)
        rows_csv.append(
            f"{sid},{answer_id},{top_join},{int(hit5)},"
            f"{score_top1 if score_top1 is not None else ''},"
            f"{score_answer if score_answer is not None else ''}"
        )

    n = len(scenarios) if scenarios else 1
    ctr5 = hit5_count / n

    ensure_parent_dir(result_path)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)

    ensure_parent_dir(log_csv)
    with open(log_csv, "w", encoding="utf-8", newline="") as f:
        f.write("scenarioId,answerId,top,hit,score_top1,score_answer\n")
        for line in rows_csv:
            f.write(line + "\n")

    samples = []
    for item in results_json[:3]:
        short = {
            "scenarioId": item["scenarioId"],
            top_key: item[top_key],
            "answerId": item["answerId"],
            "hit": item["hit"],
        }
        samples.append(json.dumps(short, ensure_ascii=False))
    line = f"scenarios={n}, ctr@5={ctr5:.2f}\nSample results (top 3):\n" + "\n".join(
        samples
    )
    print(line)


# =========================
# CLI
# =========================
def parse_args():
    p = argparse.ArgumentParser(
        description="[CTR@5] Location Mapping Engine offline eval"
    )
    p.add_argument("--catalog", help="Path to catalog.json")
    p.add_argument("--scenarios", help="Path to scenarios.json")
    p.add_argument("--answers", help="Path to answers.json")
    p.add_argument("--topk", type=int, default=5, help="Top-N (default=5)")
    p.add_argument(
        "--weights",
        type=str,
        default="theme=3.0,tag=0.8,dist=0.10",
        help='Weights, e.g. "theme=3.0,tag=0.8,dist=0.10"',
    )
    p.add_argument("--output", help="Path to result.json")
    p.add_argument("--log", help="Path to summary.csv")
    p.add_argument("--test", action="store_true", help="Run with default test paths")
    return p.parse_args()


def main():
    args = parse_args()

    if args.test:
        catalog_path = "./input/catalog.json"
        scenarios_path = "./input/scenarios.json"
        answers_path = "./input/answers.json"
        topn = 5
        weights = parse_weights("theme=3.0,tag=0.8,dist=0.10")
        result_path = "./output/result.json"
        log_csv = "./log/summary.csv"
    else:
        if not (
            args.catalog
            and args.scenarios
            and args.answers
            and args.output
            and args.log
        ):
            raise SystemExit(
                "the following arguments are required: --catalog --scenarios --answers --output --log"
            )
        catalog_path = args.catalog
        scenarios_path = args.scenarios
        answers_path = args.answers
        topn = args.topk
        weights = parse_weights(args.weights)
        result_path = args.output
        log_csv = args.log

    run_once(catalog_path, scenarios_path, answers_path, result_path, log_csv, topn, weights)


if __name__ == "__main__":
    main()
