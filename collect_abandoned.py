#!/usr/bin/env python3
"""국가동물보호정보시스템 구조동물 조회 서비스 수집기 (개만 수집)

사용법
  1) 인증키를 환경변수로 지정 (코드나 채팅에 직접 적지 마세요)
       macOS/Linux : export DATA_GO_KR_KEY='발급받은_인증키'
       Windows(PS) : $env:DATA_GO_KR_KEY='발급받은_인증키'
     URL 인코딩된 키(%2F, %3D 등 포함)든 디코딩된 키든 모두 동작합니다.
  2) 먼저 응답 구조 확인
       python collect_abandoned.py --probe
  3) 전체 수집 (중단 후 다시 실행하면 끝난 구간은 건너뜁니다)
       python collect_abandoned.py --start 20210101 --end 20260930 --out abandoned_dogs.csv

필요 패키지: pip install requests
엔드포인트가 바뀌었다면 아래 BASE_URL만 공공데이터포털 명세에 맞게 고치면 됩니다.
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta
from urllib.parse import unquote

import requests

BASE_URL = "https://apis.data.go.kr/1543061/abandonmentPublicService_v2/abandonmentPublic_v2"
UPKIND_DOG = "417000"  # 축종 코드: 개
PAGE_SIZE = 1000
ID_FIELD = "desertionNo"


def extract(payload):
    """JSON 응답에서 (resultCode, resultMsg, totalCount, items) 추출."""
    resp = payload.get("response", payload)
    header = resp.get("header", {}) or {}
    body = resp.get("body", {}) or {}
    items = body.get("items") or []
    if isinstance(items, dict):
        items = items.get("item", [])
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        items = []
    total = int(body.get("totalCount") or 0)
    return str(header.get("resultCode", "")), header.get("resultMsg", ""), total, items


def fetch_page(session, key, bgn, end, page, rows, retries=4):
    params = {
        "serviceKey": key,
        "bgnde": bgn,
        "endde": end,
        "upkind": UPKIND_DOG,
        "pageNo": page,
        "numOfRows": rows,
        "_type": "json",
    }
    last = None
    for attempt in range(retries):
        try:
            r = session.get(BASE_URL, params=params, timeout=30)
        except requests.RequestException as e:
            # 예외 메시지에 serviceKey가 든 URL이 포함되므로 유형만 남긴다
            last = type(e).__name__
            time.sleep(2 ** attempt)
            continue
        if r.status_code in (401, 403):
            # 키 문제는 재시도해도 소용없음. URL(키 포함)은 출력하지 않는다
            raise RuntimeError(f"HTTP {r.status_code}: 인증키를 확인하세요. 응답: {r.text[:300]}")
        if r.status_code >= 400:
            last = f"HTTP {r.status_code}"
            time.sleep(2 ** attempt)
            continue
        try:
            return extract(r.json())
        except ValueError:
            # 인증 오류 등은 XML 텍스트로 오는 경우가 많아 재시도해도 소용없음
            raise RuntimeError(f"JSON이 아닌 응답입니다: {r.text[:300]}")
    raise RuntimeError(f"요청 실패(재시도 소진): {last}")


def windows(start, end, days):
    cur = start
    while cur <= end:
        nxt = min(cur + timedelta(days=days - 1), end)
        yield cur.strftime("%Y%m%d"), nxt.strftime("%Y%m%d")
        cur = nxt + timedelta(days=1)


def get_key(arg_key):
    key = arg_key or os.environ.get("DATA_GO_KR_KEY")
    if not key:
        sys.exit("인증키가 없습니다. 환경변수 DATA_GO_KR_KEY를 설정하거나 --key를 쓰세요.")
    return unquote(key)  # requests가 다시 인코딩하므로 먼저 디코딩


def probe(session, key):
    end = datetime.now()
    bgn = end - timedelta(days=14)
    code, msg, total, items = fetch_page(
        session, key, bgn.strftime("%Y%m%d"), end.strftime("%Y%m%d"), 1, 3
    )
    print(f"resultCode={code} resultMsg={msg} totalCount={total}")
    if not items:
        print("항목이 없습니다. 코드/메시지를 확인하세요.")
        return
    print("필드 목록:", list(items[0].keys()))
    print(json.dumps(items[0], ensure_ascii=False, indent=2))


def collect(session, key, start, end, days, out):
    jsonl_path = out + ".jsonl"
    progress_path = out + ".progress"
    done = set()
    if os.path.exists(progress_path):
        with open(progress_path, encoding="utf-8") as f:
            done = {line.strip() for line in f if line.strip()}

    with open(jsonl_path, "a", encoding="utf-8") as jf, open(progress_path, "a", encoding="utf-8") as pf:
        for bgn, en in windows(start, end, days):
            tag = f"{bgn}-{en}"
            if tag in done:
                continue
            n_seen = 0
            page, pages = 1, 1
            while page <= pages:
                code, msg, total, items = fetch_page(session, key, bgn, en, page, PAGE_SIZE)
                if code not in ("00", "0", "03"):  # 03 = 데이터 없음
                    sys.exit(f"API 오류 resultCode={code} msg={msg} (구간 {tag}, 페이지 {page})")
                pages = max(1, math.ceil(total / PAGE_SIZE))
                for it in items:
                    jf.write(json.dumps(it, ensure_ascii=False) + "\n")
                n_seen += len(items)
                page += 1
                time.sleep(0.3)
            jf.flush()
            pf.write(tag + "\n")
            pf.flush()
            print(f"{tag}: {n_seen}건 (누적 파일: {jsonl_path})", flush=True)

    # JSONL → CSV (중복 제거 + 필드 합집합)
    seen, rows, fields = set(), [], []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rid = row.get(ID_FIELD) or line
            if rid in seen:
                continue
            seen.add(rid)
            rows.append(row)
            for k in row:
                if k not in fields:
                    fields.append(k)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"완료: {len(rows)}건 → {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key")
    ap.add_argument("--start", default="20210101", help="YYYYMMDD")
    ap.add_argument("--end", default=datetime.now().strftime("%Y%m%d"), help="YYYYMMDD")
    ap.add_argument("--days", type=int, default=30, help="한 번에 조회할 기간(일)")
    ap.add_argument("--out", default="abandoned_dogs.csv")
    ap.add_argument("--probe", action="store_true", help="응답 구조만 확인하고 종료")
    args = ap.parse_args()

    key = get_key(args.key)
    session = requests.Session()
    if args.probe:
        probe(session, key)
        return
    start = datetime.strptime(args.start, "%Y%m%d")
    end = datetime.strptime(args.end, "%Y%m%d")
    collect(session, key, start, end, args.days, args.out)


if __name__ == "__main__":
    main()
