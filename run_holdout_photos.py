#!/usr/bin/env python3
"""2026년 1~3월 사진 받기 → DINOv2 임베딩 → (재학습이 다 끝난 뒤) 최종 계열로 사진 포함 2026년 평가. 2026-11-02 사용자 승인.

  python run_holdout_photos.py    # 앱과 분리해 실행(WMI + pythonw). 끝난 단계는 v2/run/<단계>.done으로 건너뜀
받기는 지금까지와 같은 miniconda base(같은 검출 환경), 나머지는 이 스크립트를 띄운 venv 파이썬.
"""
import os
import subprocess
import sys
import time
from datetime import datetime

PY = sys.executable
BASE = r"C:\Users\Guest_user\miniconda3\python.exe"
RUN = "v2/run"
FETCH = ["fetch_photos.py", "--n", "300000", "--min-date", "20260101", "--max-date", "20260331", "--crop", "--size", "224",
         "--quality", "75", "--out", "photos_2026", "--meta", "photos_2026_meta.csv"]
NPZ = "emb_cache/photo_dinov2_2026.npz"


def log(m):
    line = f"[{datetime.now():%m-%d %H:%M:%S}] {m}"
    open(os.path.join(RUN, "pipeline_photos.log"), "a", encoding="utf-8").write(line + "\n")


def step(name, cmd):
    if os.path.exists(os.path.join(RUN, name + ".done")):
        return
    for t in (1, 2):
        log(f"{name} 시작 (시도 {t})")
        with open(os.path.join(RUN, name + ".log"), "a", encoding="utf-8") as f:
            if subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT) == 0:
                open(os.path.join(RUN, name + ".done"), "w").write(datetime.now().isoformat())
                log(f"{name} 끝")
                return
        log(f"{name} 실패")
        time.sleep(30)
    log(f"{name} 두 번 실패 → 멈춤")
    sys.exit(1)


def main():
    os.makedirs(RUN, exist_ok=True)
    step("photo26_fetch", [BASE, "-X", "utf8", "-u", *FETCH])
    step("photo26_retry", [BASE, "-X", "utf8", "-u", *FETCH, "--retry-failed"])
    step("photo26_embed", [PY, "-X", "utf8", "-u", "embed_photos.py", "--dir", "photos_2026/crop", "--models", "dinov2", "--tag", "2026"])
    while not os.path.exists(os.path.join(RUN, "holdout_base.done")):  # 모델 선택이 모두 끝난 뒤에만 평가한다
        time.sleep(60)
    win = open("v2/winner.txt").read().strip()
    step("holdout_photo", [PY, "-X", "utf8", "-u", "holdout_2026.py", "--csv", "abandoned_dogs.csv", "--model", f"v2/final_photo_{win}",
                           "--photo", NPZ, "--out", "v2/holdout_photo"])
    step("holdout_photo_base", [PY, "-X", "utf8", "-u", "holdout_2026.py", "--csv", "abandoned_dogs.csv", "--model", f"v2/final_base_{win}",
                                "--photo", NPZ, "--out", "v2/holdout_photo_base"])
    log("모두 끝")


if __name__ == "__main__":
    main()
