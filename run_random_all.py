#!/usr/bin/env python3
"""무작위 분할(2026-10-27 팀 결정)로 보고서 숫자를 처음부터 다시 만드는 실행기. 보호소 포함 모델 중심.

  # 앱과 분리해 실행(저장소 폴더에서, WMI + pythonw). 끊기면 같은 명령으로 다시: 끝난 단계는 random_run/<단계>.done으로 건너뜀
  python run_random_all.py

단계 (로그 random_run/<단계>.log, 진행 random_run/pipeline.log)
  1 search      model_search: 로지스틱·SVM·LightGBM·XGBoost·CatBoost·MLP (보호소 포함) → model_search_random/
  2 search_wo   CatBoost만 보호소 제외(기본 모델의 '보호소 뺐을 때' 비교용)
  3 base        train_final(사진 없음, 보호소 제외·포함) → final_model_random_base/   (표 4, 그림 4, 중요도)
  4 whatif      memo_whatif → final_model_random_base/memo_whatif.csv                    (표 6)
  5 ablate      ablate_memo(보호소 포함) → ablation_random/                              (메모 빼 보기)
  6 photo_cmp   compare_photo(3개 이미지 모델) → photo_compare_random/                    (표 7)
  7 photo_eval  final_photo(같은 개체 사진 없음 vs 포함, 보정·정확도) → final_model_photo_random/  (표 8)
  8 final       train_final(사진 포함) → final_model_random/                              (최종 모델 저장)
  9 ablate2     메모 빼 보기 다시(형태소 텍스트 입력까지 제거) → ablation_random2/
"""
import os
import subprocess
import sys
import time
from datetime import datetime

PY = sys.executable
W = "random_run"
COMMON = ["--csv", "abandoned_dogs.csv", "--emb", "emb_cache/e5_base_v2.npz"]
PHOTOS = ["dinov2=emb_cache/photo_dinov2.npz", "clip=emb_cache/photo_clip.npz", "convnext=emb_cache/photo_convnext.npz"]
STEPS = [
    ("search", ["model_search.py", *COMMON, "--split", "random", "--shelters", "with", "--out", "model_search_random",
                "--families", "logistic", "svm", "lightgbm", "xgboost", "catboost", "mlp"]),
    ("search_wo", ["model_search.py", *COMMON, "--split", "random", "--shelters", "without", "--out", "model_search_random",
                   "--families", "catboost"]),
    ("base", ["train_final.py", *COMMON, "--search", "model_search_random", "--split", "random", "--out", "final_model_random_base"]),
    ("whatif", ["memo_whatif.py", "--csv", "abandoned_dogs.csv", "--model", "final_model_random_base"]),
    ("ablate", ["ablate_memo.py", *COMMON, "--search", "model_search_random", "--split", "random", "--shelter", "with",
                "--out", "ablation_random"]),
    ("photo_cmp", ["compare_photo.py", *COMMON, "--search", "model_search_random", "--split", "random", "--photo", *PHOTOS,
                   "--out", "photo_compare_random"]),
    ("photo_eval", ["final_photo.py", *COMMON, "--photo", "emb_cache/photo_dinov2.npz", "--search", "model_search_random",
                    "--splits", "random", "--out", "final_model_photo_random"]),
    ("final", ["train_final.py", *COMMON, "--search", "model_search_random", "--split", "random",
               "--photo", "emb_cache/photo_dinov2.npz", "--out", "final_model_random"]),
    # 5단계는 형태소 텍스트 입력을 남긴 채 '메모 전부 제거'를 했다 → 그 입력까지 빼서 다시(2026-10-28)
    ("ablate2", ["ablate_memo.py", *COMMON, "--search", "model_search_random", "--split", "random", "--shelter", "with",
                 "--variants", "형태소 입력 제거,메모 전부 제거", "--out", "ablation_random2"]),
]


def log(msg):
    line = f"[{datetime.now():%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(os.path.join(W, "pipeline.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    os.makedirs(W, exist_ok=True)
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # 도는 동안만 절전 막기
    log(f"실행기 시작 (PID {os.getpid()})")
    for name, args in STEPS:
        if os.path.exists(os.path.join(W, name + ".done")):
            continue
        for t in (1, 2):
            log(f"{name} 시작 (시도 {t})")
            with open(os.path.join(W, name + ".log"), "a", encoding="utf-8") as f:
                rc = subprocess.call([PY, "-X", "utf8", "-u", *args], stdout=f, stderr=subprocess.STDOUT)
            if rc == 0:
                break
            log(f"{name} 실패 (종료 코드 {rc})")
            time.sleep(30)
        else:
            log(f"{name} 두 번 실패 → 멈춤. 로그 확인 후 같은 명령으로 다시 실행")
            sys.exit(1)
        open(os.path.join(W, name + ".done"), "w").write(datetime.now().isoformat())
        log(f"{name} 끝")
    log("모두 끝")


if __name__ == "__main__":
    main()
