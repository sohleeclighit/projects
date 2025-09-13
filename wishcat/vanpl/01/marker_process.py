#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import time
import uuid
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional
from math import radians, sin, cos, asin, sqrt

# =========================
# 설정/상수
# =========================
ALLOWED_CATEGORIES = {"캠핑장", "차박지", "야영지", "RV"}
BLACKLIST_TERMS = ["폐업", "사유지", "군사보호"]
EARTH_RADIUS_M = 6371008.8  # meters
DIST_THRESH_M = 50.0  # 중복 판단 임계값 (m)


# =========================
# 공통 헬퍼
# =========================
def normalize_name(name: str) -> str:
    """이름에서 모든 공백/하이픈 제거 (유니코드 공백 포함)"""
    if not isinstance(name, str):
        return ""
    return re.sub(r"[\s\-]+", "", name)


def parse_iso8601_utc(s: str) -> datetime:
    """ISO8601 문자열을 UTC timezone-aware datetime으로 파싱"""
    if not isinstance(s, str):
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 좌표 사이의 거리(m)"""
    lat1, lon1, lat2, lon2 = map(float, (lat1, lon1, lat2, lon2))
    rlat1, rlon1, rlat2, rlon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = sin(dlat / 2) ** 2 + cos(rlat1) * cos(rlat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return EARTH_RADIUS_M * c


def coords_rounded(m: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """lat/lng를 float 변환 후 소수점 6자리 반올림한 튜플 반환(없으면 None)"""
    lat = m.get("lat")
    lng = m.get("lng")
    try:
        lat_r = round(float(lat), 6) if lat is not None else None
        lng_r = round(float(lng), 6) if lng is not None else None
        return lat_r, lng_r
    except (TypeError, ValueError):
        return None, None


def normalize_coords_one(marker: Dict[str, Any]) -> Dict[str, Any]:
    """위경도를 소수점 6자리로 반올림하여 새 dict 반환"""
    m = marker.copy()
    lat_r, lng_r = coords_rounded(m)
    if lat_r is not None:
        m["lat"] = lat_r
    if lng_r is not None:
        m["lng"] = lng_r
    return m


def contains_blacklisted(name: str, tags: Any) -> bool:
    """name이나 tags 내에 블랙리스트 용어 포함 여부"""
    hay = name or ""
    if isinstance(tags, list):
        hay += " " + " ".join(map(str, tags))
    elif isinstance(tags, str):
        hay += " " + tags
    return any(term in hay for term in BLACKLIST_TERMS)


def strip_internal_keys(m: Dict[str, Any]) -> Dict[str, Any]:
    """내부 키(_로 시작) 제거한 shallow copy 반환"""
    return {k: v for k, v in m.items() if not str(k).startswith("_")}


# =========================
# 단계별 처리 (1→2→3→4)
# =========================
def step1_deduplicate(
    markers: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    1) 중복 제거
       - (a) 이름 정규화 동일  (공백/하이픈 제거)
       - (b) 좌표 50m 이내
       - 같은 장소 내에서는 updated_at 최신만 보존
    반환: (보존리스트, 제거된목록)
    """
    removed = []
    kept: List[Dict[str, Any]] = []

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for m in markers:
        norm = normalize_name(m.get("name", ""))
        m = m.copy()
        m["_norm_name"] = norm
        groups[norm].append(m)

    for _, items in groups.items():
        # 최신순으로 정렬
        items_sorted = sorted(
            items, key=lambda x: parse_iso8601_utc(x.get("updated_at")), reverse=True
        )
        cluster_centers: List[Dict[str, Any]] = []
        for cand in items_sorted:
            lat, lng = cand.get("lat"), cand.get("lng")
            cand_is_dup = False
            for center in cluster_centers:
                clat, clng = center.get("lat"), center.get("lng")
                if None in (lat, lng, clat, clng):
                    continue
                try:
                    d = haversine_m(lat, lng, clat, clng)
                except Exception:
                    continue
                if d <= DIST_THRESH_M:
                    removed.append(cand)
                    cand_is_dup = True
                    break
            if not cand_is_dup:
                cluster_centers.append(cand)
                kept.append(cand)

    # 내부키 정리
    for m in kept + removed:
        m.pop("_norm_name", None)

    return kept, removed


def step2_round_coords(markers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """2) 좌표 정규화(반올림) — 필터 없음"""
    return [normalize_coords_one(m) for m in markers]


def step3_category_filter(
    markers: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """3) 카테고리 필터 (허용 4종 외 제거)"""
    kept, removed = [], []
    for m in markers:
        if m.get("category", "") in ALLOWED_CATEGORIES:
            kept.append(m)
        else:
            removed.append(m)
    return kept, removed


def step4_blacklist_filter(
    markers: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """4) 블랙리스트 필터(name/tags에 금칙어 포함 시 제거)"""
    kept, removed = [], []
    for m in markers:
        if contains_blacklisted(m.get("name", ""), m.get("tags")):
            removed.append(m)
        else:
            kept.append(m)
    return kept, removed


def transform_markers(markers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    요청 순서대로 전체 파이프라인 수행.
    각 단계 후 원본 입력 순서(_idx)로 정렬해 최종 출력 순서를 보장.
    """
    # 1) 중복 제거
    s1_kept, _ = step1_deduplicate(markers)
    s1_kept.sort(key=lambda x: x.get("_idx", 1e18))

    # 2) 좌표 반올림
    s2_kept = step2_round_coords(s1_kept)
    s2_kept.sort(key=lambda x: x.get("_idx", 1e18))

    # 3) 카테고리 필터
    s3_kept, _ = step3_category_filter(s2_kept)
    s3_kept.sort(key=lambda x: x.get("_idx", 1e18))

    # 4) 블랙리스트 필터
    s4_kept, _ = step4_blacklist_filter(s3_kept)
    s4_kept.sort(key=lambda x: x.get("_idx", 1e18))

    return s4_kept


# =========================
# 실행 파이프라인 (결과 파일/기록 + 최소 콘솔 출력)
# =========================
def ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def append_run_csv(
    csv_path: str, run_id: str, input_size: int, output_size: int, elapsed_ms: int
) -> None:
    """run.csv에 헤더가 없으면 추가하고, 1줄 append"""
    ensure_parent_dir(csv_path)
    header_needed = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        if header_needed:
            f.write("run_id,input,output,elapsed_ms\n")
        f.write(f"{run_id},{input_size},{output_size},{elapsed_ms}\n")


def run_once(
    input_file: str, output_file: str, csv_file: str, run_id: str = None
) -> None:
    ensure_parent_dir(output_file)
    ensure_parent_dir(csv_file)

    t0 = time.time()

    # ===== 입력 로드 =====
    with open(input_file, "r", encoding="utf-8") as f:
        raw = json.load(f)
        if not isinstance(raw, list):
            raise ValueError("Input JSON must be a list of marker objects.")
        # 원본 순서 보존용 인덱스 부여
        for i, m in enumerate(raw):
            if isinstance(m, dict):
                m["_idx"] = i
        input_size = len(raw)

    # ===== 변환 =====
    processed = transform_markers(raw)
    output_size = len(processed)

    # ===== 파일 출력 =====
    # (1) result.json (내부 키 제거)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            [strip_internal_keys(x) for x in processed], f, ensure_ascii=False, indent=2
        )

    elapsed_ms = int((time.time() - t0) * 1000)

    # (2) run.csv append
    run_id = run_id or uuid.uuid4().hex[:8]
    append_run_csv(csv_file, run_id, input_size, output_size, elapsed_ms)

    # (3) 콘솔 출력 (요약 + top5만)
    print(f"elapsed={elapsed_ms}ms, input={input_size}, output={output_size}\n")
    print("Sample results (top 5):")
    top5 = processed[:5]
    if top5:
        for item in top5:
            print(json.dumps(strip_internal_keys(item), ensure_ascii=False))
    else:
        print("(no results)")


# =========================
# CLI
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Marker Data Processing Engine")
    parser.add_argument("--input", help="Input JSON file (ignored if --test is used)")
    parser.add_argument("--output", help="Output JSON file (ignored if --test is used)")
    parser.add_argument("--log", help="Run CSV file path (ignored if --test is used)")
    parser.add_argument(
        "--test", action="store_true", help="Run with predefined test paths"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.test:
        input_file = "./input/sample.json"
        output_file = "./output/result.json"
        csv_file = "./output/run.csv"  # 테스트 모드: run.csv 강제
    else:
        if not (args.input and args.output and args.log):
            raise SystemExit(
                "the following arguments are required: --input, --output, --log"
            )
        input_file = args.input
        output_file = args.output
        csv_file = args.log  # 일반 모드: --log 를 run.csv 경로로 사용
    run_once(input_file, output_file, csv_file)


if __name__ == "__main__":
    main()
