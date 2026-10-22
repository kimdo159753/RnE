#!/usr/bin/env python3
"""공고 사진 표본 내려받기 (+ 개만 상자로 잘라 작게 저장)

사용법 (프로젝트 폴더에서)
  python fetch_photos.py --dry-run                                  # 어떤 표본이 뽑히는지만 확인
  python fetch_photos.py --n 100                                    # 원본 그대로 저장 (1단계 방식)
  python fetch_photos.py --n 1000 --min-date 20220101 --crop --keep-full --out photos1k --meta photos1k_meta.csv
  # 2022~2025 전체(개체당 첫 사진 1장, 상자만 224px·품질 75, 약 2.7GB, 0.5초/장 ≈ 31시간)
  python -u fetch_photos.py --n 300000 --min-date 20220101 --crop --size 224 --quality 75 --out photos_all --meta photos_all_meta.csv

뽑는 규칙
  - --min-date <= happenDt <= --max-date, 처리상태가 종료(입양/안락사/자연사/기증)인 개체만.
    2021년 사진은 서버에서 지워져 404가 난다(2026-10-20 100장 표본: 2021년 20/20 실패, 2022~2025년 79/80 성공).
  - 첫 사진 파일명의 날짜(업로드일)가 접수일로부터 --max-delay 일 이내인 것만 (기본 3일).
    접수 뒤에 올라온 사진(미용 후·입양 후)이 섞이는 누수를 막기 위함. -1이면 제한 없음.
  - 접수 연도별로 고르게 뽑음. --seed로 재현 가능.

--crop: 원본 해상도에서 개를 찾아(torchvision Faster R-CNN MobileNetV3, COCO 네발 동물 라벨) 가장 확실한 개 한 마리의
  상자만 잘라 짧은 변 --size 픽셀로 줄여 저장(photos/crop/). 배경(케이지·바닥·번호표)으로 보호소가 학습되는 것을 줄이기 위함.
  상자는 정사각형으로 늘리지 않고 여백 --margin(비율)만 둔다. 개가 안 잡히면 crop은 저장하지 않고 기록만 남긴다.
  원본은 저장하지 않는다. --keep-full이면 원본을 같은 크기로 줄인 것도 photos/full/에 저장(보호소 정보 비교용).
--crop 없이 쓰면 예전처럼 원본을 photos/<desertionNo>.<확장자>로 저장.
기록: --meta CSV (상태코드·원본 용량·해상도·업로드 지연일·개 인식 결과). 이미 기록된 개체는 건너뛰므로 끊겨도 이어서 받음.
"""
import argparse
import csv
import hashlib
import io
import math
import os
import random
import re
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

csv.field_size_limit(10**9)
PAT = re.compile(r"/(\d{4})/(\d{2})/(\d{8})\d*(?:\[\d+\])?\.(\w+)$")  # 일부 파일명은 ...123[1].jpg 형태
CLOSED = {"종료(입양)", "종료(안락사)", "종료(자연사)", "종료(기증)"}
# torchvision COCO 라벨 번호. 공고 사진은 모두 개인데 모델이 개를 고양이·곰·곰인형·양 등으로 잘못 붙이는 일이 잦아
# (100장 표본에서 못 찾은 8장 전부) 네발 동물 라벨을 모두 개로 본다.
ANIMAL = {17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow", 23: "bear", 88: "teddy bear"}


def upload_delay(url, happen):
    m = PAT.search(url)
    if not m:
        return None
    try:
        return (datetime.strptime(m.group(3), "%Y%m%d") - datetime.strptime(happen, "%Y%m%d")).days
    except ValueError:
        return None


def load_candidates(path, min_date, max_date, max_delay):
    cands, unparsed = [], []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            h = (row.get("happenDt") or "").strip()
            if not h or h < min_date or h > max_date or row.get("processState") not in CLOSED:
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
                "careNm": row.get("careNm", ""), "careRegNo": row.get("careRegNo", ""),
                "processState": row["processState"], "delay": "" if d is None else d, "url": url,
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


class DogCropper:
    """원본에서 개 상자를 찾아 잘라 짧은 변 size로 줄인다."""

    def __init__(self, size, margin, min_score):
        import torch
        from torchvision.models.detection import (FasterRCNN_MobileNet_V3_Large_FPN_Weights,
                                                  fasterrcnn_mobilenet_v3_large_fpn)
        torch.set_grad_enabled(False)
        self.torch = torch
        w = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        self.model = fasterrcnn_mobilenet_v3_large_fpn(weights=w, box_score_thresh=0.2).eval()
        self.to_tensor = w.transforms()
        self.size, self.margin, self.min_score = size, margin, min_score

    def shrink(self, im):
        s = self.size / min(im.size)
        if s < 1:
            im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), resample=3)  # BICUBIC
        return im

    def __call__(self, data):
        from PIL import Image, ImageOps
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        out = self.model([self.to_tensor(im)])[0]
        animal = self.torch.tensor([l in ANIMAL for l in out["labels"].tolist()], dtype=self.torch.bool)
        keep = animal & (out["scores"] >= self.min_score)
        boxes, scores, labels = out["boxes"][keep], out["scores"][keep], out["labels"][keep]
        info = {"width": im.width, "height": im.height, "n_dog": int(keep.sum()), "dog_score": "", "box_frac": "",
                "label": ""}
        crop = None
        if len(scores):
            k = int(scores.argmax())
            x0, y0, x1, y1 = boxes[k].tolist()
            info["label"] = ANIMAL[int(labels[k])]
            mx, my = (x1 - x0) * self.margin, (y1 - y0) * self.margin
            x0, y0 = max(0, x0 - mx), max(0, y0 - my)
            x1, y1 = min(im.width, x1 + mx), min(im.height, y1 + my)
            info["dog_score"] = round(float(scores.max()), 3)
            info["box_frac"] = round((x1 - x0) * (y1 - y0) / (im.width * im.height), 3)
            crop = self.shrink(im.crop((round(x0), round(y0), round(x1), round(y1))))
        return info, crop, self.shrink(im)


def save_jpeg(im, path, quality):
    im.save(path, "JPEG", quality=quality)
    return os.path.getsize(path)


def quant(xs, qs=(0.05, 0.5, 0.95)):
    xs = sorted(xs)
    return [xs[int(len(xs) * q)] for q in qs] if xs else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="abandoned_dogs.csv")
    ap.add_argument("--out", default="photos")
    ap.add_argument("--meta", default="photos_meta.csv")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--min-date", default="00000000")
    ap.add_argument("--max-date", default="20251231")
    ap.add_argument("--max-delay", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--timeout", type=float, default=20)
    ap.add_argument("--workers", type=int, default=2, help="동시에 받는 수(서버 부하 때문에 작게)")
    ap.add_argument("--crop", action="store_true", help="개 상자만 잘라 줄여 저장(원본 저장 안 함)")
    ap.add_argument("--keep-full", action="store_true", help="--crop일 때 원본을 줄인 것도 저장(비교용)")
    ap.add_argument("--size", type=int, default=256, help="저장할 짧은 변 픽셀")
    ap.add_argument("--margin", type=float, default=0.03, help="상자 여백(상자 크기 대비 비율)")
    ap.add_argument("--min-score", type=float, default=0.5, help="개로 인정할 최소 점수")
    ap.add_argument("--quality", type=int, default=85)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cands, unparsed = load_candidates(a.csv, a.min_date, a.max_date, a.max_delay)
    print(f"조건에 맞는 후보 {len(cands):,}마리 (기간 {a.min_date}~{a.max_date}, 업로드 지연 <= {a.max_delay}일)")
    if unparsed:
        print("파일명 날짜를 못 읽은 URL 예시:", *unparsed, sep="\n  ")
    sample = stratified(cands, a.n, a.seed)
    print("표본", len(sample), "마리, 연도별:", dict(sorted(Counter(c["year"] for c in sample).items())))
    print("표본 입양 비율: %.3f" % (sum(c["processState"] == "종료(입양)" for c in sample) / max(1, len(sample))))
    if a.dry_run:
        for c in sample[:3]:
            print(" ", c["desertionNo"], c["happenDt"], c["processState"], c["url"])
        return

    cropper = DogCropper(a.size, a.margin, a.min_score) if a.crop else None
    if cropper:
        os.makedirs(os.path.join(a.out, "crop"), exist_ok=True)
        if a.keep_full:
            os.makedirs(os.path.join(a.out, "full"), exist_ok=True)
    else:
        os.makedirs(a.out, exist_ok=True)
    fields = ["desertionNo", "happenDt", "year", "careNm", "careRegNo", "processState", "delay", "url",
              "status", "content_type", "bytes", "width", "height", "sha1",
              "n_dog", "label", "dog_score", "box_frac", "crop_bytes", "full_bytes", "sec"]
    done = set()
    if os.path.exists(a.meta):
        with open(a.meta, encoding="utf-8-sig", newline="") as f:
            done = {r["desertionNo"] for r in csv.DictReader(f)}
    new = not os.path.exists(a.meta)
    mf = open(a.meta, "a", encoding="utf-8-sig", newline="")
    w = csv.DictWriter(mf, fieldnames=fields)
    if new:
        w.writeheader()

    def get(c):  # 받기 + 요청 사이 대기(작업자별). 받는 동안 앞 사진의 개 찾기가 돌도록 미리 받는다.
        t0 = time.time()
        st, ctype, data = fetch(c["url"], a.timeout, 2)
        if st != 200 and ("[" in c["url"]):  # 대괄호가 든 주소는 인코딩해서 한 번 더 시도
            st, ctype, data = fetch(quote(c["url"], safe=":/%"), a.timeout, 2)
        sec = time.time() - t0
        time.sleep(a.sleep)
        return st, ctype, data, sec

    if os.name == "nt":  # 실행 중에만 Windows 절전 막기(설정은 안 바꿈, 프로세스가 끝나면 풀림)
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    todo = [c for c in sample if c["desertionNo"] not in done]
    print(f"이미 받은 {len(sample) - len(todo):,}마리 건너뜀, 남은 {len(todo):,}마리, 동시 받기 {a.workers}", flush=True)
    pool = ThreadPoolExecutor(a.workers)
    ahead = deque(pool.submit(get, c) for c in todo[:a.workers * 2])
    status, dogs = Counter(), Counter()
    sizes, dims, hashes, saved, secs = [], [], Counter(), [], []
    t_start = time.time()
    for i, c in enumerate(todo, 1):
        st, ctype, data, fsec = ahead.popleft().result()
        nxt = i - 1 + a.workers * 2
        if nxt < len(todo):
            ahead.append(pool.submit(get, todo[nxt]))
        t0 = time.time()
        rec = dict(c, status=st, content_type=ctype, bytes=len(data))
        if st == 200 and data:
            rec["sha1"] = hashlib.sha1(data).hexdigest()
            sizes.append(len(data))
            hashes[rec["sha1"]] += 1
            if cropper:
                try:
                    info, crop, full = cropper(data)
                except Exception as e:  # 깨진 이미지 등
                    info, crop, full = {"n_dog": f"err:{type(e).__name__}"}, None, None
                rec.update(info)
                dogs["개 있음" if crop else "개 없음/오류"] += 1
                if crop:
                    rec["crop_bytes"] = save_jpeg(crop, os.path.join(a.out, "crop", c["desertionNo"] + ".jpg"), a.quality)
                    saved.append(rec["crop_bytes"])
                if full and a.keep_full:
                    rec["full_bytes"] = save_jpeg(full, os.path.join(a.out, "full", c["desertionNo"] + ".jpg"), a.quality)
            else:
                ext = os.path.splitext(c["url"])[1].lower() or ".jpg"
                path = os.path.join(a.out, c["desertionNo"] + ext)
                with open(path, "wb") as f:
                    f.write(data)
                rec["width"], rec["height"] = image_size(path)
            if rec.get("width") not in ("", None):
                dims.append(min(rec["width"], rec["height"]))
        rec["sec"] = round(fsec + time.time() - t0, 2)  # 받기 + 처리 (대기 제외)
        secs.append(rec["sec"])
        status[st] += 1
        w.writerow(rec)
        mf.flush()
        if i % 50 == 0:
            el = time.time() - t_start
            eta = el / i * (len(todo) - i) / 3600
            print(f"  {i:,}/{len(todo):,} 상태 {dict(status)} {dict(dogs)} | {el / i:.2f}초/장, 남은 약 {eta:.1f}시간",
                  flush=True)
    pool.shutdown()
    mf.close()

    print("\n== 요약 ==")
    print("HTTP 상태:", dict(status))
    if sizes:
        print("원본 크기(KB) 5/50/95%:", [round(x / 1024) for x in quant(sizes)], "| 총 %.1f MB" % (sum(sizes) / 1e6))
    if dims:
        print("원본 짧은 변 픽셀 5/50/95%:", quant(dims))
    if cropper:
        print("개 인식:", dict(dogs))
        if saved:
            print("잘라 저장한 크기(KB) 5/50/95%:", [round(x / 1024) for x in quant(saved)],
                  "| 총 %.1f MB" % (sum(saved) / 1e6))
    if secs:
        print("한 장당 처리 시간(초, 대기 제외) 50/95%:", quant(secs, (0.5, 0.95)))
    dup = sum(v - 1 for v in hashes.values() if v > 1)
    print("내용이 완전히 같은 중복 사진:", dup)


if __name__ == "__main__":
    main()
