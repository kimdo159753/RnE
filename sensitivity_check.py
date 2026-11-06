#!/usr/bin/env python3
"""허점 점검용 민감도 분석 (2026-11-02). 최종 계열·설정·무작위 분할로 한 가지씩만 바꿔 다시 학습한다.

  python sensitivity_check.py --csv abandoned_dogs.csv --emb emb_cache/e5_base_v2.npz --search v2/search \
      --photo emb_cache/photo_dinov2.npz --family lightgbm --out v2/sensitivity

사진 없는 모델(시험 구간 전체)
  - 중성화 제거: 입양 전에 수술해 값이 바뀌었을 수 있는 입력을 뺐을 때
  - 기증 제외: 기증 뒤 실제로 입양됐을 수 있는 개체를 비입양으로 두지 않았을 때(같은 개체끼리 비교)
사진 포함 모델(사진 있는 개체, 같은 개체끼리 비교)
  - 사진 제거 vs 메모 전부 제거: 중요도 합계가 아니라 실제로 뺐을 때의 차이로 두 입력을 비교
보호소별 입양률 분포도 함께 출력한다(학습 구간 기준, 입양률이 극단인 보호소가 차지하는 비율).
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from accuracy_table import acc, boot_acc_diff
from compare_photo import load_photo, set_photo
from model_search import FIT, Data, score


def report(rows, name, y, base, new, g):
    a = boot_acc_diff(y, new, base, g)
    s0, s1 = score(y, base), score(y, new)
    print(f"{name}: n {len(y):,} 정확도 {acc(y, base) * 100:.2f} → {acc(y, new) * 100:.2f} "
          f"Δ {a[0] * 100:+.2f}%p [{a[1] * 100:+.2f}, {a[2] * 100:+.2f}] | ROC {s0['roc_auc']:.4f} → {s1['roc_auc']:.4f}", flush=True)
    rows.append({"변형": name, "n": len(y), "기준선": 1 - y.mean(), "acc_기준": acc(y, base), "acc_변형": acc(y, new),
                 "dacc": a[0], "dacc_lo": a[1], "dacc_hi": a[2], "roc_기준": s0["roc_auc"], "roc_변형": s1["roc_auc"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--search", required=True)
    ap.add_argument("--photo", required=True)
    ap.add_argument("--family", choices=["catboost", "lightgbm"], required=True)
    ap.add_argument("--max-date", default="20251231")
    ap.add_argument("--lr-pca", type=int, default=128)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--split", default="random")
    ap.add_argument("--out", default="v2/sensitivity")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    D = Data(args)
    D._tree_frame = D.tree_frame
    df = D.df
    cfg = pickle.load(open(os.path.join(args.search, f"{args.family}_with_shelter.pkl"), "rb"))["best"]["cfg"]
    fit = FIT[args.family]
    idx0 = (D.i_tr.copy(), D.i_va.copy(), D.i_te.copy())
    rows = []

    tr = df.iloc[idx0[0]]
    by = tr.groupby("care")["y"].agg(["size", "mean"])
    big = by[by["size"] >= 100]
    print(f"보호소(학습 100마리 이상) {len(big)}곳 입양률 10/50/90% = {big['mean'].quantile([.1, .5, .9]).round(3).tolist()}, "
          f"90% 이상 {int((big['mean'] >= .9).sum())}곳({big.loc[big['mean'] >= .9, 'size'].sum() / len(tr) * 100:.1f}%), "
          f"10% 이하 {int((big['mean'] <= .1).sum())}곳({big.loc[big['mean'] <= .1, 'size'].sum() / len(tr) * 100:.1f}%)", flush=True)

    _, pt0, _ = fit(D, True, cfg)
    y, g = D.y[D.i_te], df["care"].values[D.i_te].astype(str)

    keep = df["neuter"].copy()
    df["neuter"] = "제거"
    _, pt, _ = fit(D, True, cfg)
    df["neuter"] = keep
    report(rows, "중성화 제거", y, pt0, pt, g)

    no_don = (df["outcome"] != "donated").values
    D.i_tr, D.i_va, D.i_te = (i[no_don[i]] for i in idx0)
    _, pt, _ = fit(D, True, cfg)
    report(rows, "기증 제외", D.y[D.i_te], pt0[no_don[idx0[2]]], pt, df["care"].values[D.i_te].astype(str))

    X, has = load_photo(args.photo, df["id"].astype(str).to_numpy())
    D.i_tr, D.i_va, D.i_te = (i[has[i]] for i in idx0)
    pca = PCA(32, random_state=42).fit(X[D.i_tr])
    P = np.full((len(X), 32), np.nan, np.float32)
    P[has] = pca.transform(X[has])
    y, g = D.y[D.i_te], df["care"].values[D.i_te].astype(str)
    set_photo(D, P)
    _, pt_all, _ = fit(D, True, cfg)
    set_photo(D, None)
    _, pt, _ = fit(D, True, cfg)
    report(rows, "사진 제거(사진 포함 모델에서)", y, pt_all, pt, g)
    set_photo(D, P)
    txt0, p0, len0 = D.txt.copy(), D.P32.copy(), df["mark_len"].copy()
    D.txt, D.P32 = np.zeros_like(txt0), np.zeros_like(p0)
    df["mark_len"] = 0
    c = {**cfg, "text_features": False} if args.family == "catboost" else cfg
    _, pt, _ = fit(D, True, c)
    D.txt, D.P32 = txt0, p0
    df["mark_len"] = len0
    report(rows, "메모 전부 제거(사진 포함 모델에서)", y, pt_all, pt, g)

    pd.DataFrame(rows).to_csv(os.path.join(args.out, "sensitivity.csv"), index=False, encoding="utf-8-sig")
    print(f"결과: {args.out}/sensitivity.csv")


if __name__ == "__main__":
    main()
