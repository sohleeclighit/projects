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


def theme_matches(s_theme: Any, poi_category: Any) -> int:
    return (
        1
        if (
            s_theme is not None
            and poi_category is not None
            and str(s_theme) == str(poi_category)
        )
        else 0
    )


# =========================
# 점수 계산 & 추천
# =========================
def score_poi_for_scenario(s: Dict[str, Any], poi: Dict[str, Any]) -> float:
    """
    score = (theme match * 3.0) + (tag overlap * 0.8) – (distance_km * 0.10)
    """
    s_theme = s.get("theme")
    poi_cat = poi.get("category")

    # theme match
    th = theme_matches(s_theme, poi_cat)

    # tag overlap
    prefer = set(get_prefer_tags(s))
    poi_tags = set(safe_tags(poi.get("tags")))
    overlap = len(prefer.intersection(poi_tags))

    # distance
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

    score = (th * 3.0) + (overlap * 0.8) - (dist_km * 0.10)
    return score


def recommend_top5_for_scenario(
    s: Dict[str, Any], catalog: List[Dict[str, Any]]
) -> Tuple[List[str], float]:
    """
    catalog 전체를 점수화하고 점수 내림차순, 동점 시 id 오름차순으로 정렬하여 Top-5 ID 반환.
    top1 점수도 함께 반환.
    """
    scored: List[Tuple[str, float]] = []
    for poi in catalog:
        pid = to_id(poi.get("id"))
        sc = score_poi_for_scenario(s, poi)
        scored.append((pid, sc))

    scored.sort(key=lambda x: (-x[1], x[0]))
    top5_ids = [pid for pid, _ in scored[:5]]
    score_top1 = scored[0][1] if scored else float("-inf")
    return top5_ids, score_top1


# =========================
# 메인 실행
# =========================
def run_once(
    catalog_path: str,
    scenarios_path: str,
    answers_path: str,
    result_path: str,
    summary_csv: str,
) -> None:
    t0 = time.time()

    # 입력 로드
    catalog = read_json_array(catalog_path)
    scenarios = read_json_array(scenarios_path)
    answers = read_json_array(answers_path)
    ans_map: Dict[str, str] = {
        to_id(a.get("scenarioId")): to_id(a.get("answerId")) for a in answers
    }

    results_json: List[Dict[str, Any]] = []
    rows_csv: List[str] = []
    hit5_count = 0
    hit1_count = 0  # 내부 집계(콘솔에는 미표시)

    for s in scenarios:
        sid = to_id(s.get("scenarioId"))
        answer_id = ans_map.get(sid, "")

        top5_ids, score_top1 = recommend_top5_for_scenario(s, catalog)
        hit5 = answer_id in top5_ids if answer_id else False
        hit1 = (answer_id == top5_ids[0]) if (answer_id and top5_ids) else False
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
                score_answer = score_poi_for_scenario(s, poi_lookup)

        results_json.append(
            {
                "scenarioId": sid,
                "top5": top5_ids,
                "answerId": answer_id,
                "hit": hit5,
                "hit1": hit1,  # 파일에만 남김
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

        top5_join = ";".join(top5_ids)
        rows_csv.append(
            f"{sid},{answer_id},{top5_join},{int(hit5)},"
            f"{score_top1 if score_top1 is not None else ''},"
            f"{score_answer if score_answer is not None else ''}"
        )

    # CTR 계산
    n = len(scenarios) if scenarios else 1
    ctr5 = hit5_count / n

    # 파일 저장
    ensure_parent_dir(result_path)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)

    ensure_parent_dir(summary_csv)
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        f.write("scenarioId,answerId,top5,hit,score_top1,score_answer\n")
        for line in rows_csv:
            f.write(line + "\n")

    # 콘솔 출력 — 단일 라인 포맷
    samples = []
    for item in results_json[:3]:
        short = {
            "scenarioId": item["scenarioId"],
            "top5": item["top5"],
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
    p.add_argument("--output", help="Path to result.json")
    p.add_argument("--summary", help="Path to summary.csv")
    p.add_argument("--test", action="store_true", help="Use predefined test paths")
    return p.parse_args()


def main():
    args = parse_args()

    if args.test:
        # 요청 사양: senarios.json 철자 유지
        catalog_path = "./input/catalog.json"
        scenarios_path = "./input/scenarios.json"
        answers_path = "./input/answers.json"
        result_path = "./output/result.json"
        summary_csv = "./output/summary.csv"
    else:
        if not (
            args.catalog
            and args.scenarios
            and args.answers
            and args.output
            and args.summary
        ):
            raise SystemExit(
                "the following arguments are required: --catalog --scenarios --answers --output --summary"
            )
        catalog_path = args.catalog
        scenarios_path = args.scenarios
        answers_path = args.answers
        result_path = args.output
        summary_csv = args.summary

    run_once(catalog_path, scenarios_path, answers_path, result_path, summary_csv)


if __name__ == "__main__":
    main()
