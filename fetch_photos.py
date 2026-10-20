#!/usr/bin/env python3
"""공고 사진 표본 내려받기 (1단계: 서버 응답·크기·화질 확인용)

사용법 (프로젝트 폴더에서)
  python fetch_photos.py --dry-run            # 어떤 표본이 뽑히는지만 확인 (다운로드 안 함)
  python fetch_photos.py --n 100              # 100마리의 첫 사진(popfile1)만 받기
  python fetch_photos.py --n 100 --sleep 0.5  # 서버 부하를 더 줄이려면 간격을 늘림

뽑는 규칙
  - happenDt <= --max-date, 처리상태가 종료(입양/안락사/자연사/기증)인 개체만
  - 첫 사진 파일명의 날짜(업로드일)가 접수일로부터 --max-delay 일 이내인 것만 (기본 3일).
    접수 뒤에 올라온 사진(미용 후·입양 후)이 섞이는 누수를 막기 위함. -1이면 제한 없음.
  - 접수 연도별로 고르게 뽑음. --seed로 재현 가능.
저장: photos/<desertionNo>.<확장자>, 기록: photos_meta.csv (상태코드·용량·해상도·해시·업로드 지연일)
이미 받은 파일은 건너뛰므로 중간에 끊겨도 다시 실행하면 이어서 받음.
"""
import argparse
import csv
import hashlib
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

csv.field_size_limit(10**9)
PAT = re.compile(r"/(\d{4})/(\d{2})/(\d{8})\d*(?:\[\d+\])?\.(\w+)$")  # 일부 파일명은 ...123[1].jpg 형태
CLOSED = {"종료(입양)", "종료(안락사)", "종료(자연사)", "종료(기증)"}


def upload_delay(url, happen):
    m = PAT.search(url)
    if not m:
        return None
    try:
        return (datetime.strptime(m.group(3), "%Y%m%d") - datetime.strptime(happen, "%Y%m%d")).days
    except ValueError:
        return None


def load_candidates(path, max_date, max_delay):
    cands, unparsed = [], []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            h = (row.get("happenDt") or "").strip()
            if not h or h > max_date or row.get("processState") not in CLOSED:
                continue
            url = (row.get("popfile1") or "").strip()
            if not url:
                continue
            d = upload_delay(url, h)
            if d is None:
                if len(unparsed) < 5:
                    unparsed.append(url)
                if max_delay >= 0:
                    continue
            elif max_delay >= 0 and not (0 <= d <= max_delay):
                continue
            cands.append({
                "desertionNo": row["desertionNo"], "happenDt": h, "year": h[:4],
                "careNm": row.get("careNm", ""), "processState": row["processState"],
                "delay": "" if d is None else d, "url": url,
            })
    return cands, unparsed


def stratified(cands, n, seed):
    rng = random.Random(seed)
    by_year = defaultdict(list)
    for c in cands:
        by_year[c["year"]].append(c)
    years = sorted(by_year)
    per = math.ceil(n / max(1, len(years)))
    out = []
    for y in years:
        pool = by_year[y]
        out += rng.sample(pool, min(per, len(pool)))
    rng.shuffle(out)
    return out[:n]


def fetch(url, timeout, retries):
    last = ""
    for a in range(retries + 1):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0 (school research; stray-dog-adoption)"})
            with urlopen(req, timeout=timeout) as r:
                return r.status, r.headers.get("Content-Type", ""), r.read()
        except HTTPError as e:
            return e.code, "", b""
        except (URLError, TimeoutError, ConnectionError, OSError) as e:
            last = str(e)
            time.sleep(1 + a)
    return 0, last, b""


def image_size(path):
    try:
        from PIL import Image  # 선택 사항: pip install pillow
        with Image.open(path) as im:
            return im.size
    except Exception:
        return ("", "")


def quant(xs, qs=(0.05, 0.5, 0.95)):
    xs = sorted(xs)
    return [xs[int(len(xs) * q)] for q in qs] if xs else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="abandoned_dogs.csv")
    ap.add_argument("--out", default="photos")
    ap.add_argument("--meta", default="photos_meta.csv")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-date", default="20251231")
    ap.add_argument("--max-delay", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--timeout", type=float, default=20)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cands, unparsed = load_candidates(a.csv, a.max_date, a.max_delay)
    print(f"조건에 맞는 후보 {len(cands):,}마리 (기간 ~{a.max_date}, 업로드 지연 <= {a.max_delay}일)")
    if unparsed:
        print("파일명 날짜를 못 읽은 URL 예시:", *unparsed, sep="\n  ")
    sample = stratified(cands, a.n, a.seed)
    print("표본", len(sample), "마리, 연도별:", dict(sorted(Counter(c["year"] for c in sample).items())))
    print("표본 입양 비율: %.3f" % (sum(c["processState"] == "종료(입양)" for c in sample) / max(1, len(sample))))
    if a.dry_run:
        for c in sample[:3]:
            print(" ", c["desertionNo"], c["happenDt"], c["processState"], c["url"])
        return

    os.makedirs(a.out, exist_ok=True)
    fields = ["desertionNo", "happenDt", "year", "careNm", "processState", "delay", "url",
              "status", "content_type", "bytes", "width", "height", "sha1"]
    done = set()
    if os.path.exists(a.meta):
        with open(a.meta, encoding="utf-8-sig", newline="") as f:
            done = {r["desertionNo"] for r in csv.DictReader(f)}
    new = not os.path.exists(a.meta)
    mf = open(a.meta, "a", encoding="utf-8-sig", newline="")
    w = csv.DictWriter(mf, fieldnames=fields)
    if new:
        w.writeheader()

    status = Counter()
    sizes, dims, hashes = [], [], Counter()
    for i, c in enumerate(sample, 1):
        if c["desertionNo"] in done:
            continue
        ext = os.path.splitext(c["url"])[1].lower() or ".jpg"
        path = os.path.join(a.out, c["desertionNo"] + ext)
        st, ctype, data = fetch(c["url"], a.timeout, 2)
        if st != 200 and ("[" in c["url"]):  # 대괄호가 든 주소는 인코딩해서 한 번 더 시도
            st, ctype, data = fetch(quote(c["url"], safe=":/%"), a.timeout, 2)
        rec = dict(c, status=st, content_type=ctype, bytes=len(data), width="", height="", sha1="")
        if st == 200 and data:
            with open(path, "wb") as f:
                f.write(data)
            rec["sha1"] = hashlib.sha1(data).hexdigest()
            rec["width"], rec["height"] = image_size(path)
            sizes.append(len(data))
            hashes[rec["sha1"]] += 1
            if rec["width"] != "":
                dims.append(min(rec["width"], rec["height"]))
        status[st] += 1
        w.writerow(rec)
        mf.flush()
        if i % 20 == 0:
            print(f"  {i}/{len(sample)} 상태 {dict(status)}", flush=True)
        time.sleep(a.sleep)
    mf.close()

    print("\n== 요약 ==")
    print("HTTP 상태:", dict(status))
    if sizes:
        print("파일 크기(KB) 5/50/95%:", [round(x / 1024) for x in quant(sizes)], "| 총 %.1f MB" % (sum(sizes) / 1e6))
    if dims:
        print("짧은 변 픽셀 5/50/95%:", quant(dims))
    dup = sum(v - 1 for v in hashes.values() if v > 1)
    print("내용이 완전히 같은 중복 사진:", dup)


if __name__ == "__main__":
    main()
