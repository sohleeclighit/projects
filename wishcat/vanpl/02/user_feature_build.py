#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import time
import uuid
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from math import radians, sin, cos, asin, sqrt
from collections import Counter

# =========================
# 타임존/상수
# =========================
KST = ZoneInfo("Asia/Seoul")
UTC = ZoneInfo("UTC")

# (Fallback) 한국 시·도(17개) 센트로이드(대략값)
KOREA_SIDO_CENTROIDS = {
    "서울": (37.5665, 126.9780),
    "부산": (35.1796, 129.0756),
    "대구": (35.8714, 128.6014),
    "인천": (37.4563, 126.7052),
    "광주": (35.1595, 126.8526),
    "대전": (36.3504, 127.3845),
    "울산": (35.5384, 129.3114),
    "세종": (36.4800, 127.2890),
    "경기": (37.4138, 127.5183),
    "강원": (37.8228, 128.1555),
    "충북": (36.8000, 127.7000),
    "충남": (36.5184, 126.8000),
    "전북": (35.7175, 127.1530),
    "전남": (34.8679, 126.9910),
    "경북": (36.4919, 128.8889),
    "경남": (35.2383, 128.6920),
    "제주": (33.4996, 126.5312),
}
# 센트로이드 fallback 허용 반경
MAX_REGION_DIST_KM = 250.0


# =========================
# 유틸
# =========================
def ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def write_run_csv(csv_path: str, row: Dict[str, Any]) -> None:
    """
    run.csv를 매 실행 시마다 덮어쓰기 모드로 저장.
    row keys: run_id,start_time,end_time,users,events_used,elapsed_ms
    """
    ensure_parent_dir(csv_path)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        f.write("run_id,start_time,end_time,users,events_used,elapsed_ms\n")
        f.write(
            "{run_id},{start_time},{end_time},{users},{events_used},{elapsed_ms}\n".format(
                **row
            )
        )


def load_input(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of user logs.")
    return data


# -------- 날짜 파싱 (KST 기준) --------
def parse_kst(dt: Any) -> Optional[datetime]:
    """
    입력 문자열/숫자를 한국 시각(Asia/Seoul)으로 해석한 timezone-aware datetime 반환.
    - '...Z'면 UTC로 보고 KST로 변환
    - 타임존 없는 경우 KST로 간주
    - epoch(sec) 숫자면 KST로 변환
    """
    if not dt:
        return None
    if isinstance(dt, (int, float)):
        try:
            return datetime.fromtimestamp(float(dt), tz=KST)
        except Exception:
            return None
    if not isinstance(dt, str):
        return None
    s = dt.strip()
    try:
        if s.endswith("Z"):
            base = datetime.fromisoformat(s[:-1] + "+00:00").astimezone(KST)
            return base
        t = datetime.fromisoformat(s)
        if t.tzinfo is None:
            return t.replace(tzinfo=KST)
        else:
            return t.astimezone(KST)
    except Exception:
        try:
            # 'YYYY-MM-DD HH:MM:SS' 등 흔한 포맷
            t = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
            return t.replace(tzinfo=KST)
        except Exception:
            return None


# -------- 이벤트 공통 처리 --------
def pick_event_time(ev: Dict[str, Any]) -> Optional[datetime]:
    """
    이벤트 객체에서 시각 필드 추정:
    createdAt / created_at / time / timestamp / ts / datetime
    """
    for key in ("createdAt", "created_at", "time", "timestamp", "ts", "datetime"):
        if key in ev and ev[key]:
            return parse_kst(ev[key])
    return None


def get_event_category_id(ev: Dict[str, Any]) -> Optional[int]:
    for key in ("categoryId", "category_id", "catId", "cat_id"):
        if key in ev and ev[key] is not None:
            try:
                return int(ev[key])
            except Exception:
                return None
    return None


def get_event_latlng(ev: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    lat = ev.get("lat") if "lat" in ev else ev.get("latitude")
    lng = (
        ev.get("lng")
        if "lng" in ev
        else ev.get("lon") if "lon" in ev else ev.get("longitude")
    )
    try:
        lat_f = float(lat) if lat is not None else None
        lng_f = float(lng) if lng is not None else None
    except Exception:
        lat_f = None
        lng_f = None
    return lat_f, lng_f


# -------- 거리/지역 매핑 유틸 --------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0088  # km
    rlat1, rlon1, rlat2, rlon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = sin(dlat / 2) ** 2 + cos(rlat1) * cos(rlat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return R * c


# =========================
# 시·도 GeoJSON 로더 & PIP(점-다각형 포함) 판정
# =========================
class SidoIndex:
    """
    간단한 GeoJSON 로더 + bbox 프리체크 + ray casting PIP.
    GeoJSON은 WGS84 경위도, Polygon/MultiPolygon 가정.
    각 feature에는 시·도명이 들어있는 속성이 있어야 함 (기본 'name' 키 탐색).
    """

    def __init__(
        self,
        features: List[Dict[str, Any]],
        name_key_candidates=("name", "ADM_NM", "sidonm", "SIG_KOR_NM"),
    ):
        self.polys: List[
            Tuple[
                str, List[List[Tuple[float, float]]], Tuple[float, float, float, float]
            ]
        ] = []
        # polys 항목: (name, [rings: [(lon,lat),...]], bbox(minx,miny,maxx,maxy))
        for feat in features:
            props = feat.get("properties", {})
            name = None
            for k in name_key_candidates:
                if k in props and props[k]:
                    name = str(props[k])
                    break
            if not name:
                continue
            geom = feat.get("geometry") or {}
            gtype = geom.get("type")
            coords = geom.get("coordinates")
            if not coords:
                continue

            if gtype == "Polygon":
                rings = []
                bbox = [float("inf"), float("inf"), -float("inf"), -float("inf")]
                for ring in coords:
                    ring_ll = []
                    for lon, lat in ring:
                        lon = float(lon)
                        lat = float(lat)
                        ring_ll.append((lon, lat))
                        bbox[0] = min(bbox[0], lon)
                        bbox[1] = min(bbox[1], lat)
                        bbox[2] = max(bbox[2], lon)
                        bbox[3] = max(bbox[3], lat)
                    rings.append(ring_ll)
                self.polys.append((name, rings, tuple(bbox)))
            elif gtype == "MultiPolygon":
                for poly in coords:
                    rings = []
                    bbox = [float("inf"), float("inf"), -float("inf"), -float("inf")]
                    for ring in poly:
                        ring_ll = []
                        for lon, lat in ring:
                            lon = float(lon)
                            lat = float(lat)
                            ring_ll.append((lon, lat))
                            bbox[0] = min(bbox[0], lon)
                            bbox[1] = min(bbox[1], lat)
                            bbox[2] = max(bbox[2], lon)
                            bbox[3] = max(bbox[3], lat)
                        rings.append(ring_ll)
                    self.polys.append((name, rings, tuple(bbox)))

    @staticmethod
    def point_in_ring(lon: float, lat: float, ring: List[Tuple[float, float]]) -> bool:
        inside = False
        n = len(ring)
        for i in range(n):
            x1, y1 = ring[i]
            x2, y2 = ring[(i + 1) % n]
            if (y1 > lat) != (y2 > lat):
                xinters = (x2 - x1) * (lat - y1) / (y2 - y1 + 1e-15) + x1
                if lon <= xinters:  # 경계 포함
                    inside = not inside
        return inside

    @staticmethod
    def point_on_segment(
        lon: float,
        lat: float,
        a: Tuple[float, float],
        b: Tuple[float, float],
        eps=1e-12,
    ) -> bool:
        (x1, y1), (x2, y2) = a, b
        cross = abs((lon - x1) * (y2 - y1) - (lat - y1) * (x2 - x1))
        if cross > eps:
            return False
        dot = (lon - x1) * (x2 - x1) + (lat - y1) * (y2 - y1)
        if dot < -eps:
            return False
        sq_len = (x2 - x1) ** 2 + (y2 - y1) ** 2
        if dot - sq_len > eps:
            return False
        return True

    @classmethod
    def point_in_polygon(
        cls, lon: float, lat: float, rings: List[List[Tuple[float, float]]]
    ) -> bool:
        if not rings:
            return False
        outer = rings[0]
        # 경계(on edge) 체크
        for i in range(len(outer)):
            if cls.point_on_segment(lon, lat, outer[i], outer[(i + 1) % len(outer)]):
                return True
        if not cls.point_in_ring(lon, lat, outer):
            return False
        # holes
        for h in rings[1:]:
            on_edge = any(
                cls.point_on_segment(lon, lat, h[i], h[(i + 1) % len(h)])
                for i in range(len(h))
            )
            if on_edge:
                return True
            if cls.point_in_ring(lon, lat, h):
                return False
        return True

    def find_region(self, lat: float, lon: float) -> Optional[str]:
        if lat is None or lon is None:
            return None
        for name, rings, bbox in self.polys:
            minx, miny, maxx, maxy = bbox
            if not (minx <= lon <= maxx and miny <= lat <= maxy):
                continue
            if self.point_in_polygon(lon, lat, rings):
                return name
        return None


def load_sido_index(geojson_path: Optional[str]) -> Optional["SidoIndex"]:
    if not geojson_path:
        return None
    if not os.path.exists(geojson_path):
        print(
            f"[warn] SIDO GeoJSON not found: {geojson_path} (fallback to centroid-nearest)"
        )
        return None
    try:
        with open(geojson_path, "r", encoding="utf-8") as f:
            gj = json.load(f)
        feats = gj.get("features")
        if not feats or not isinstance(feats, list):
            print(
                f"[warn] Invalid GeoJSON (no features): {geojson_path} (fallback to centroid-nearest)"
            )
            return None
        return SidoIndex(feats)
    except Exception as e:
        print(f"[warn] Failed to load GeoJSON ({e}). Fallback to centroid-nearest.")
        return None


# -------- fallback: 센트로이드 근접 --------
def map_region_by_centroid(lat: Optional[float], lng: Optional[float]) -> Optional[str]:
    if lat is None or lng is None:
        return None
    best_name = None
    best_dist = float("inf")
    for name, (clat, clng) in KOREA_SIDO_CENTROIDS.items():
        d = haversine_km(lat, lng, clat, clng)
        if d < best_dist:
            best_dist = d
            best_name = name
    if best_dist <= MAX_REGION_DIST_KM:
        return best_name
    return None


# =========================
# 처리 로직
# =========================
def build_features(
    users_logs: List[Dict[str, Any]], days: int, sido_index: Optional["SidoIndex"]
) -> Tuple[List[Dict[str, Any]], int]:
    """
    최근 N일 내 이벤트만 포함하여:
      - 활동별 카운트
      - 카테고리 비율 Top3
      - 최근 지역 1~3개 (시·도) [GeoJSON 있으면 PIP, 없으면 센트로이드 fallback]
      - lastActiveAt (UTC Z)
    """
    now_kst = datetime.now(KST)
    cutoff = now_kst - timedelta(days=days)

    results: List[Dict[str, Any]] = []
    total_used = 0

    for u in users_logs:
        user_id = u.get("userId") or u.get("user_id") or u.get("id")

        counts = {"likes": 0, "bookmarks": 0, "participations": 0, "verifications": 0}
        included: List[Tuple[datetime, str, Dict[str, Any]]] = (
            []
        )  # (time_kst, kind, event)
        excluded_by_date: List[Dict[str, Any]] = []  # 콘솔 출력하지 않음

        # 이벤트 수집
        for kind in ("likes", "bookmarks", "participations", "verifications"):
            arr = u.get(kind) or []
            if isinstance(arr, list):
                for ev in arr:
                    if not isinstance(ev, dict):
                        continue
                    t = pick_event_time(ev)
                    if t is None:
                        continue
                    # [cutoff, now_kst] 양끝 포함
                    if cutoff <= t <= now_kst:
                        counts[kind] += 1
                        included.append((t, kind, ev))
                        total_used += 1
                    else:
                        excluded_by_date.append(
                            {"kind": kind, "at": t.isoformat().replace("+09:00", "KST")}
                        )

        # ---- categoryShareTop3 ----
        cat_counter: Counter = Counter()
        for t, kind, ev in included:
            cid = get_event_category_id(ev)
            if cid is not None:
                cat_counter[cid] += 1
        cat_total = sum(cat_counter.values())
        category_share_top3: List[Dict[str, Any]] = []
        if cat_total > 0:
            top = cat_counter.most_common()
            top.sort(key=lambda x: (-x[1], x[0]))  # 동률시 categoryId 오름차순
            for cid, cnt in top[:3]:
                ratio = round(cnt / cat_total, 2)
                category_share_top3.append({"categoryId": cid, "ratio": ratio})

        # ---- recentRegions (최신순 유니크 1~3개) ----
        recent_regions: List[str] = []
        if included:
            included.sort(key=lambda x: x[0], reverse=True)
            seen = set()
            for t, kind, ev in included:
                lat, lng = get_event_latlng(ev)
                region = None
                if lat is not None and lng is not None:
                    if sido_index:
                        region = sido_index.find_region(lat, lng)
                    if not region:
                        region = map_region_by_centroid(lat, lng)
                if region and region not in seen:
                    recent_regions.append(region)
                    seen.add(region)
                if len(recent_regions) >= 3:
                    break

        # ---- lastActiveAt (UTC Z) ----
        last_active_at: Optional[str] = None
        if included:
            latest_kst = max(t for (t, _, _) in included)
            last_active_at = (
                latest_kst.astimezone(UTC).isoformat().replace("+00:00", "Z")
            )

        results.append(
            {
                "userId": user_id,
                "counts": counts,
                "categoryShareTop3": category_share_top3,
                "recentRegions": recent_regions,
                "lastActiveAt": last_active_at,
            }
        )

    return results, total_used


# =========================
# run_id 추출 유틸 (추가)
# =========================
def extract_run_id_from_log_path(log_path: str) -> str:
    """
    파일명이 run{n}.csv 또는 run_{n}.csv 이면 run_id=n, 그 외는 '1'
    - 대소문자 구분 없이 동작
    - 경로 포함 가능 (basename만 사용)
    """
    base = os.path.basename(log_path or "")
    m = re.match(r"(?i)^run_?(\d+)\.csv$", base)  # run1.csv 또는 run_1.csv
    if m:
        return m.group(1)
    return "1"


# =========================
# 실행
# =========================
def run_once(
    input_path: str,
    days: int,
    output_json: str,
    output_csv: str,
    sido_geojson: Optional[str],
) -> None:
    ensure_parent_dir(output_json)
    ensure_parent_dir(output_csv)

    # 시·도 인덱스 로드(선택)
    sido_index = load_sido_index(sido_geojson)

    start_monotonic = time.time()
    start_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    # 입력 로드
    users_logs = load_input(input_path)
    users_count = len(users_logs)

    # 처리
    results, events_used = build_features(users_logs, days, sido_index)

    # 결과 저장
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    elapsed_ms = int((time.time() - start_monotonic) * 1000)
    end_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    # run_id: 규칙 적용 (run{n}.csv / run_{n}.csv → n, 그 외 1)
    run_id_value = extract_run_id_from_log_path(output_csv)

    # 실행 로그(run.csv) — 덮어쓰기
    write_run_csv(
        output_csv,
        {
            "run_id": run_id_value,
            "start_time": start_iso,
            "end_time": end_iso,
            "users": users_count,
            "events_used": events_used,
            "elapsed_ms": elapsed_ms,
        },
    )

    # 콘솔 출력 (기존 포맷 유지)
    print(f"elapsed={elapsed_ms}ms, users={users_count}, events_used={events_used}")
    print("Sample results (top 5):")
    for item in results[:5]:
        print(json.dumps(item, ensure_ascii=False))


# =========================
# CLI
# =========================
def parse_args():
    p = argparse.ArgumentParser(
        description="User Feature Build (counts + top3 + recent regions + lastActiveAt with polygon mapping)"
    )
    p.add_argument("--input", help="Input users_logs.json")
    p.add_argument(
        "--days", type=int, default=30, help="Lookback window in days (default: 30)"
    )
    p.add_argument("--output", help="Output result.json path")
    p.add_argument("--log", help="Output run.csv path")
    p.add_argument("--sido", help="Korea SIDO GeoJSON path (optional)", default=None)
    p.add_argument("--test", action="store_true", help="Use predefined test paths")
    return p.parse_args()


def main():
    args = parse_args()

    if args.test:
        input_path = "./input/users_logs.json"  # 테스트 모드 고정 경로
        output_json = "./output/result.json"
        output_csv = "./output/run.csv"
        sido_geojson = args.sido  # 원하면 --sido로 파일 제공 가능
        days = 30 if args.days is None else args.days
    else:
        if not (args.input and args.output and args.log):
            raise SystemExit(
                "the following arguments are required: --input, --output, --log"
            )
        input_path = args.input
        output_json = args.output
        output_csv = args.log
        sido_geojson = args.sido
        days = args.days

    run_once(input_path, days, output_json, output_csv, sido_geojson)


if __name__ == "__main__":
    main()
