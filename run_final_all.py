#!/usr/bin/env python3
"""허점 수정본 전체 재학습 실행기 (2026-11-02). 빈 메모 제외, CatBoost·LightGBM 모두, 2026년 자료로 마지막 평가.

  python run_final_all.py            # 앱과 분리해 실행(WMI + pythonw). 끊기면 같은 명령으로 이어 함(v2/run/<단계>.done)
  python run_final_all.py --smoke    # 1/40 표본 시험. 결과는 v2_smoke/, 2026년 정답은 보지 않음

최종 계열은 search 단계가 끝나면 **검증 정확도가 가장 높은 단일 계열**로 자동으로 정해 v2/winner.txt에 적는다
(결과를 보기 전에 정한 규칙, 보고서의 "모델 선택은 검증 정확도"와 같다). 시험 구간과 2026년 자료는 고르는 데 쓰지 않는다.
2026년 평가는 최종 계열만, 맨 마지막에 한다.
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

PY = sys.executable
COMMON = ["--csv", "abandoned_dogs.csv", "--emb", "emb_cache/e5_base_v2.npz"]
PHOTOS = ["dinov2=emb_cache/photo_dinov2.npz", "clip=emb_cache/photo_clip.npz", "convnext=emb_cache/photo_convnext.npz"]
DINO = "emb_cache/photo_dinov2.npz"
SINGLES = ["logistic", "svm", "lightgbm", "xgboost", "catboost", "mlp"]


def choose_winner(root):
    d = np.load(os.path.join(root, "search", "preds_with_shelter.npz"), allow_pickle=True)
    acc = {f: float(np.mean((d[f"val_{f}"] >= 0.5) == d["y_val"])) for f in SINGLES}
    w = max(acc, key=acc.get)
    if w not in ("catboost", "lightgbm"):  # 뒷단계 코드는 두 계열만 지원. 다른 계열이면 사람이 봐야 하므로 멈춘다
        raise SystemExit(f"검증 정확도 1등이 {w}: {acc}")
    open(os.path.join(root, "winner.txt"), "w").write(w)
    return w, acc


def steps(root, sm, win):
    s, c = f"{root}/search", COMMON + sm
    cfg = ["--n-cfg", "1"] if sm else []
    out = [
        ("search", ["model_search.py", *c, *cfg, "--split", "random", "--shelters", "with", "--out", s, "--families", *SINGLES]),
        ("search_wo", ["model_search.py", *c, *cfg, "--split", "random", "--shelters", "without", "--out", s, "--families", "catboost", "lightgbm"]),
    ]
    for f in ("lightgbm", "catboost"):
        out += [(f"base_{f}", ["train_final.py", *c, "--search", s, "--split", "random", "--family", f, "--out", f"{root}/final_base_{f}"]),
                (f"final_{f}", ["train_final.py", *c, "--search", s, "--split", "random", "--family", f, "--photo", DINO, "--out", f"{root}/final_photo_{f}"]),
                (f"photo_eval_{f}", ["final_photo.py", *c, "--photo", DINO, "--search", s, "--family", f, "--splits", "random", "time", "--out", f"{root}/photo_eval_{f}"])]
    if win is None:
        return out
    variants = "단어 점수 제거,임베딩 제거,메모 전부 제거" + (",형태소 입력 제거" if win == "catboost" else "")
    out += [
        ("whatif", ["memo_whatif.py", "--csv", "abandoned_dogs.csv", "--model", f"{root}/final_base_{win}"]),
        ("ablate", ["ablate_memo.py", *c, "--search", s, "--split", "random", "--shelter", "with", "--family", win, "--variants", variants, "--out", f"{root}/ablation"]),
        ("photo_cmp", ["compare_photo.py", *c, "--search", s, "--split", "random", "--family", win, "--photo", *PHOTOS, "--out", f"{root}/photo_compare"]),
        ("sensitivity", ["sensitivity_check.py", *c, "--search", s, "--family", win, "--photo", DINO, "--out", f"{root}/sensitivity"]),
    ]
    if not sm:
        out.append(("memo_effects", ["memo_effects.py", "--csv", "abandoned_dogs.csv", "--out", f"{root}/memo_check"]))
    hold = ["holdout_2026.py", "--csv", "abandoned_dogs.csv", "--model", f"{root}/final_base_{win}", "--out", f"{root}/holdout_base"]
    out.append(("holdout_base", hold + (["--test-only"] if sm else [])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    root = "v2_smoke" if a.smoke else "v2"
    sm = ["--stride", "40"] if a.smoke else []
    run = os.path.join(root, "run")
    os.makedirs(run, exist_ok=True)

    def log(m):
        line = f"[{datetime.now():%m-%d %H:%M:%S}] {m}"
        print(line, flush=True)
        open(os.path.join(run, "pipeline.log"), "a", encoding="utf-8").write(line + "\n")

    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # 도는 동안만 절전 막기
    log(f"실행기 시작 (PID {os.getpid()})")
    win = None
    done_steps = set()
    while True:
        wf = os.path.join(root, "winner.txt")
        if win is None and os.path.exists(os.path.join(run, "search.done")):
            win, acc = choose_winner(root)
            log(f"최종 계열 {win} (검증 정확도 { {k: round(v, 4) for k, v in acc.items()} })")
        todo = [(n, x) for n, x in steps(root, sm, win) if n not in done_steps and not os.path.exists(os.path.join(run, n + ".done"))]
        if not todo:
            break
        name, args = todo[0]
        # search 뒤에는 나머지를 계속하기 전에 winner를 정해야 한다
        for t in (1, 2):
            log(f"{name} 시작 (시도 {t})")
            with open(os.path.join(run, name + ".log"), "a", encoding="utf-8") as f:
                rc = subprocess.call([PY, "-X", "utf8", "-u", *args], stdout=f, stderr=subprocess.STDOUT)
            if rc == 0:
                break
            log(f"{name} 실패 (종료 코드 {rc})")
            time.sleep(30)
        else:
            log(f"{name} 두 번 실패 → 멈춤. 로그 확인 후 같은 명령으로 다시 실행")
            sys.exit(1)
        if not (a.smoke and name == "holdout_base"):
            open(os.path.join(run, name + ".done"), "w").write(datetime.now().isoformat())
        else:
            done_steps.add(name)
        log(f"{name} 끝")
    log("모두 끝")


if __name__ == "__main__":
    main()
