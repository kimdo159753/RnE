#!/usr/bin/env python3
"""메모 입력 기여도 확인(하나씩 빼 보기): 최종 CatBoost 설정·같은 분할에서
전체 / 단어 점수(txt_lr) 제거 / 임베딩(PCA32) 제거 / 메모 전부 제거(단어 점수·임베딩·메모 길이)를 비교한다.
빼는 방법: 해당 열을 상수로 채운다(CatBoost는 상수 열로는 나누지 않으므로 제거와 같다).
--variants로 고를 수 있다(전체는 항상 포함). --shelter로 보호소 포함/제외 모델을 고른다.
판정: 전체 대비 ΔPR-AUC의 보호소 클러스터 부트스트랩 95% 구간(검증·시험).

  python ablate_memo.py --csv abandoned_dogs.csv --emb emb_cache/e5_base_v2.npz --search model_search --out ablation
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd

from compare_text_repr import boot_diff
from model_search import Data, fit_catboost, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--search", default="model_search")
    ap.add_argument("--max-date", default="20251231")
    ap.add_argument("--lr-pca", type=int, default=128)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", default="ablation")
    ap.add_argument("--shelter", choices=["both", "with", "without"], default="both")
    ap.add_argument("--variants", default="단어 점수 제거,임베딩 제거,메모 전부 제거")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    D = Data(args)
    txt0, p0, len0 = D.txt.copy(), D.P32.copy(), D.df["mark_len"].copy()
    # (단어 점수, 임베딩, 메모 길이) 사용 여부
    ALL = {"전체": (True, True, True), "단어 점수 제거": (False, True, True),
           "임베딩 제거": (True, False, True), "메모 전부 제거": (False, False, False)}
    variants = {"전체": ALL["전체"], **{v: ALL[v] for v in args.variants.split(",")}}
    g_va, g_te = D.df["care"].values[D.i_va], D.df["care"].values[D.i_te]
    rows = []
    for shelter in {"both": (False, True), "with": (True,), "without": (False,)}[args.shelter]:
        name = "with_shelter" if shelter else "without_shelter"
        cfg = pickle.load(open(os.path.join(args.search, f"catboost_{name}.pkl"), "rb"))["best"]["cfg"]
        preds = {}
        for v, (use_txt, use_emb, use_len) in variants.items():
            D.txt = txt0 if use_txt else np.zeros_like(txt0)
            D.P32 = p0 if use_emb else np.zeros_like(p0)
            D.df["mark_len"] = len0 if use_len else 0
            pv, pt, info = fit_catboost(D, shelter, cfg)
            preds[v] = (pv, pt)
            sv, st = score(D.y[D.i_va], pv), score(D.y[D.i_te], pt)
            print(f"[{name}] {v:<10} 검증 PR {sv['pr_auc']:.4f} ROC {sv['roc_auc']:.4f} | 시험 PR {st['pr_auc']:.4f} "
                  f"ROC {st['roc_auc']:.4f} (트리 {info['iters']})", flush=True)
            rows.append({"모델": name, "변형": v, **{f"검증_{k}": x for k, x in sv.items()}, **{f"시험_{k}": x for k, x in st.items()}})
        for v in list(variants)[1:]:
            (vm, vl, vh), (vrm, _, _) = boot_diff(D.y[D.i_va], preds[v][0], preds["전체"][0], g_va)
            (tm, tl, th), (trm, _, _) = boot_diff(D.y[D.i_te], preds[v][1], preds["전체"][1], g_te)
            verdict = "빼면 확실히 나빠짐" if vh < 0 else ("빼면 확실히 좋아짐" if vl > 0 else "차이 불확실")
            print(f"[{name}] {v} - 전체: 검증 ΔPR {vm:+.4f} [{vl:+.4f}, {vh:+.4f}] ΔROC {vrm:+.4f} | "
                  f"시험 ΔPR {tm:+.4f} [{tl:+.4f}, {th:+.4f}] ΔROC {trm:+.4f} → {verdict}", flush=True)
            rows.append({"모델": name, "변형": f"{v} - 전체", "검증_dPR": vm, "검증_dPR_lo": vl, "검증_dPR_hi": vh,
                         "시험_dPR": tm, "시험_dPR_lo": tl, "시험_dPR_hi": th, "판정": verdict})
    fn = "ablation.csv" if args.shelter == "both" else f"ablation_{args.shelter}_shelter.csv"
    pd.DataFrame(rows).to_csv(os.path.join(args.out, fn), index=False, encoding="utf-8-sig")
    print(f"결과: {args.out}/{fn}")


if __name__ == "__main__":
    main()
