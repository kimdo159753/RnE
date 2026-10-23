#!/usr/bin/env python3
"""사진 단계를 처음부터 끝까지 순서대로 돌린다 (Claude 앱과 상관없이 밤새 두기용)

  # 앱과 분리해 실행 (PowerShell, 저장소 폴더에서). 끊겼으면 같은 명령을 다시 실행하면 끝난 단계는 건너뛰고 이어 간다.
  # Start-Process로 띄우면 Claude 앱의 프로세스 묶음에 같이 들어가 앱 쪽이 정리될 때 함께 꺼진다(2026-10-22 21:55 실제로 꺼짐).
  # WMI로 띄우면 부모가 WmiPrvSE가 되어 앱과 완전히 분리된다.
  Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CurrentDirectory=(Get-Location).Path; `
      CommandLine='"C:\\Users\\Guest_user\\venvs\\rne\\Scripts\\python.exe" -u run_photo_pipeline.py'}

단계 (끝나면 photo_pipeline/<단계>.done, 각 단계 로그 photo_pipeline/<단계>.log, 전체 진행 photo_pipeline/pipeline.log)
  1 download    fetch_photos.py 전체 받기를 "남은 0마리"가 될 때까지 다시 실행(이미 받은 개체는 건너뜀).
                --wait-pid로 이미 돌고 있는 받기 프로세스가 있으면 끝날 때까지 기다린다.
  2 retry       --retry-failed: 네트워크 오류(status 0)만 다시 받기.
  3 clean_test  photo_clean_check.py --n 2000 (글씨 지우기·누끼 4종 → 보호소 맞히기 시험).
  4 decide      처리 고르기 규칙(아래) → photo_pipeline/decision.json
  5 clean_all   고른 처리를 전체 사진에 적용(고른 게 없으면 건너뜀).
  6 embed       자른 사진 그대로 3개 모델 + (고른 처리가 있으면) 처리한 사진 3개 모델.
  7 compare     compare_photo.py --group-cv (채택 판정, 보호소 단위 교차검증).
  끝나면 photo_pipeline/SUMMARY.txt에 결정과 판정 줄을 모은다.

처리 고르기 규칙 (CLAUDE.md 계획 2·3을 숫자로 옮긴 것)
  같은 2,000장에서 "가장 비슷한 사진이 같은 보호소" 비율: crop(자른 그대로) / 품종+체중 기준(사진 없이) / 처리 3종.
  줄어든 몫 = (crop - 처리) / (crop - 품종+체중). 가장 많이 줄인 처리의 몫이 0.5 이상이면 그 처리를 전체 적용,
  아니면 자른 사진 그대로. crop이 이미 품종+체중 이하이면 처리하지 않는다.
  어느 쪽이든 자른 사진 임베딩은 계산하고, 처리한 사진 임베딩이 있으면 둘 다 compare 후보에 넣어 검증으로 고른다.
"""
import argparse
import ctypes
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

import pandas as pd

VENV_PY = sys.executable
BASE_PY = r"C:\Users\Guest_user\miniconda3\python.exe"  # 지금까지 사진을 자른 환경(torchvision 0.22)과 같게
FETCH = ["-u", "fetch_photos.py", "--n", "300000", "--min-date", "20220101", "--crop", "--size", "224",
         "--quality", "75", "--out", "photos_all", "--meta", "photos_all_meta.csv"]
MODELS = ["dinov2", "clip", "convnext"]
W = "photo_pipeline"


def log(msg):
    line = f"[{datetime.now():%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(os.path.join(W, "pipeline.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def done(step):
    return os.path.exists(os.path.join(W, step + ".done"))


def mark(step, note=""):
    with open(os.path.join(W, step + ".done"), "w", encoding="utf-8") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {note}\n")
    log(f"단계 끝: {step} {note}")


def run(step, args, py=VENV_PY, tries=2):
    """하위 명령 실행(로그는 photo_pipeline/<step>.log에 덧붙임). 실패하면 한 번 더, 그래도 실패하면 멈춘다."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    for t in range(1, tries + 1):
        log(f"{step} 시작 (시도 {t}): {os.path.basename(py)} {' '.join(args)}")
        with open(os.path.join(W, step + ".log"), "a", encoding="utf-8") as f:
            f.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} 시도 {t} =====\n")
            f.flush()
            rc = subprocess.call([py] + args, stdout=f, stderr=subprocess.STDOUT, env=env)
        if rc == 0:
            return
        log(f"{step} 실패 (종료 코드 {rc})")
        time.sleep(60)
    log(f"{step} 두 번 실패 → 멈춤. 로그 {W}/{step}.log 확인 후 같은 명령으로 다시 실행하면 이 단계부터 이어 간다.")
    sys.exit(1)


def wait_pid(pid):
    """그 PID가 python 프로세스이면 끝날 때까지 기다린다(재부팅 뒤 같은 번호를 다른 프로그램이 쓰는 경우는 무시)."""
    k = ctypes.windll.kernel32
    h = k.OpenProcess(0x00100000 | 0x1000, False, pid)  # SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return
    buf, n = ctypes.create_unicode_buffer(1024), ctypes.c_ulong(1024)
    name = buf.value if k.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(n)) else ""
    if "python" in name.lower():
        log(f"받기 프로세스 {pid} 끝나기를 기다림")
        k.WaitForSingleObject(h, 0xFFFFFFFF)
        log(f"받기 프로세스 {pid} 끝남")
    k.CloseHandle(h)


def last_remaining(step):
    with open(os.path.join(W, step + ".log"), encoding="utf-8", errors="replace") as f:
        found = re.findall(r"남은 ([\d,]+)마리", f.read())
    return int(found[-1].replace(",", "")) if found else None


def decide():
    res = pd.read_csv("photo_clean/clean_check.csv", encoding="utf-8-sig").set_index("사진")["이웃 같은 보호소"]
    crop, bw = res["crop"], res["(사진 없이) 품종+체중"]
    gap = crop - bw
    share = {k: (crop - res[k]) / gap if gap > 0 else 0.0 for k in ("text", "nobg", "text_nobg")}
    best = max(share, key=share.get)
    chosen = best if gap > 0 and share[best] >= 0.5 else None
    out = {"crop": crop, "breed_weight": bw, "values": {k: res[k] for k in share}, "reduced_share": share,
           "chosen": chosen, "rule": "가장 많이 줄인 처리의 (crop-처리)/(crop-품종+체중) >= 0.5이면 전체 적용"}
    json.dump(out, open(os.path.join(W, "decision.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-pid", type=int, default=0, help="이미 돌고 있는 받기 프로세스 번호")
    a = ap.parse_args()
    os.makedirs(W, exist_ok=True)
    # 이 프로세스가 살아 있는 동안만 절전 막기(전원 설정은 안 바꿈)
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    log(f"실행기 시작 (PID {os.getpid()})")

    if not done("download"):
        if a.wait_pid:
            wait_pid(a.wait_pid)
        for _ in range(6):
            run("download", FETCH, py=BASE_PY)
            left = last_remaining("download")
            log(f"받기 남은 개체: {left}")
            if left == 0:
                break
            time.sleep(120)  # 네트워크 문제로 끊겼으면 잠시 뒤 이어 받기
        else:
            log("받기가 여러 번 끝나지 않음 → 받은 데까지로 다음 단계 진행")
        mark("download")

    if not done("retry"):
        run("retry", FETCH + ["--retry-failed"], py=BASE_PY)
        mark("retry")

    if not done("clean_test"):
        run("clean_test", ["-u", "photo_clean_check.py", "--n", "2000", "--out", "photo_clean"])
        mark("clean_test")

    if not done("decide"):
        d = decide()
        mark("decide", f"선택: {d['chosen']} (줄어든 몫 {json.dumps({k: round(v, 2) for k, v in d['reduced_share'].items()})})")
    chosen = json.load(open(os.path.join(W, "decision.json"), encoding="utf-8"))["chosen"]

    if not done("clean_all"):
        if chosen:
            run("clean_all", ["-u", "photo_clean_check.py", "--all", chosen, "--out", "photo_clean_all"])
        mark("clean_all", chosen or "처리 안 함")

    if not done("embed"):
        run("embed", ["-u", "embed_photos.py", "--dir", "photos_all/crop"])
        if chosen:
            run("embed", ["-u", "embed_photos.py", "--dir", f"photo_clean_all/{chosen}", "--tag", chosen])
        mark("embed")

    if not done("compare"):
        photos = [f"{m}=emb_cache/photo_{m}.npz" for m in MODELS]
        if chosen:
            photos += [f"{m}_{chosen}=emb_cache/photo_{m}_{chosen}.npz" for m in MODELS]
        run("compare", ["-u", "compare_photo.py", "--csv", "abandoned_dogs.csv", "--emb", "emb_cache/e5_base_v2.npz",
                        "--search", "model_search", "--photo", *photos, "--out", "photo_compare", "--group-cv"])
        mark("compare")

    with open(os.path.join(W, "compare.log"), encoding="utf-8", errors="replace") as f:
        keep = [l.rstrip() for l in f if any(s in l for s in ("→", "평균 ROC", "ΔPR", "사진 있는 개체"))]
    with open(os.path.join(W, "SUMMARY.txt"), "w", encoding="utf-8") as f:
        f.write(open(os.path.join(W, "decision.json"), encoding="utf-8").read() + "\n\n" + "\n".join(keep) + "\n")
    log("모두 끝. 요약: photo_pipeline/SUMMARY.txt")


if __name__ == "__main__":
    main()
