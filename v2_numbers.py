#!/usr/bin/env python3
"""v2 재학습 결과에서 보고서에 들어갈 숫자를 모아 v2/report_numbers.json으로 저장한다 (2026-11-04). 다시 학습하지 않는다.

  python v2_numbers.py
"""
import json
import pickle

import numpy as np
import pandas as pd

from accuracy_table import boot_acc_diff
from train_adoption import prepare
from memo_categories import classify_series

WIN = open("v2/winner.txt").read().strip()
FAMS = ["lightgbm", "catboost", "xgboost", "mlp", "svm", "logistic"]


def acc(y, p):
    return float(np.mean((p >= 0.5) == y))


def main():
    out = {"winner": WIN}
    raw = pd.read_csv("abandoned_dogs.csv", dtype=str, encoding="utf-8-sig")
    out["raw_n"] = len(raw)
    df = prepare(raw, include_returned=False, max_date=pd.to_datetime("20251231", format="%Y%m%d"))
    out["n"] = len(df)
    out["adopt_rate"] = float(df["y"].mean())
    out["outcome"] = df["outcome"].value_counts().to_dict()
    w = pd.cut(df["weight_kg"], [0, 3, 5, 10, 20, 200], labels=["3kg 이하", "3~5kg", "5~10kg", "10~20kg", "20kg 초과"])
    out["by_weight"] = df.groupby(w, observed=True)["y"].mean().round(4).to_dict()
    b = df.groupby("breed")["y"].agg(["mean", "size"])
    out["by_breed"] = b[b["size"] >= 4000].sort_values("mean").round(4).to_dict("index")
    out["mix_share"] = float(df["is_mix"].mean())
    by = df.groupby("care")["y"].agg(["size", "mean"])
    big = by[by["size"] >= 100]
    out["shelter_big"] = {"n": len(big), "min": float(big["mean"].min()), "max": float(big["mean"].max()),
                          "q10": float(big["mean"].quantile(.1)), "q50": float(big["mean"].quantile(.5)), "q90": float(big["mean"].quantile(.9))}
    out["sido"] = df.groupby("sido")["y"].mean().sort_values().round(4).to_dict()

    # 모델 계열 비교(보호소 포함)
    d = np.load("v2/search/preds_with_shelter.npz", allow_pickle=True)
    yv, yt = d["y_val"], d["y_test"]
    g_te = d["g_test"] if "g_test" in d.files else None
    out["null_val"], out["null_test"] = float(1 - yv.mean()), float(1 - yt.mean())
    out["npz_keys"] = list(d.files)
    t = {}
    for f in FAMS:
        t[f] = {"val": acc(yv, d[f"val_{f}"]), "test": acc(yt, d[f"test_{f}"])}
    out["families"] = t
    out["n_val"], out["n_test"] = int(len(yv)), int(len(yt))
    if g_te is not None:
        g_va = d["g_val"]
        for f in FAMS:
            if f != WIN:
                a = boot_acc_diff(yv, d[f"val_{f}"], d[f"val_{WIN}"], g_va)
                b_ = boot_acc_diff(yt, d[f"test_{f}"], d[f"test_{WIN}"], g_te)
                t[f]["dval"] = [float(x) for x in a]
                t[f]["dtest"] = [float(x) for x in b_]

    # 최종 모델 예측
    for key, m in (("base", f"v2/final_base_{WIN}"), ("photo", f"v2/final_photo_{WIN}")):
        for s in ("with_shelter", "without_shelter"):
            p = pd.read_csv(f"{m}/pred_test_{s}.csv")
            pc = p["p_cal"].values
            y = p["y"].values
            pred = pc >= .5
            tp, fp, fn, tn = [int(x) for x in ((pred & (y == 1)).sum(), (pred & (y == 0)).sum(), (~pred & (y == 1)).sum(), (~pred & (y == 0)).sum())]
            q = pd.qcut(pd.Series(pc).rank(method="first"), 10, labels=False)
            r = pd.Series(y).groupby(q).mean()
            c = pd.DataFrame({"p": pc, "y": y}).groupby(q).mean()
            gap = (c["y"] - c["p"]) * 100
            cal = pickle.load(open(f"{m}/calibrator_{s}.pkl", "rb"))
            out[f"{key}_{s}"] = {"n": len(y), "null": float(1 - y.mean()), "acc": float((pred == y).mean()), "prec": tp / (tp + fp), "rec": tp / (tp + fn),
                                 "spec": tn / (tn + fp), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "bot10": float(r.iloc[0]), "top10": float(r.iloc[-1]),
                                 "gap_min": float(gap.min()), "gap_max": float(gap.max()),
                                 "calibrator": cal.get("method", str(cal.get("calibrator"))[:30]) if isinstance(cal, dict) else str(cal)[:30]}
        out[f"{key}_imp"] = pd.read_csv(f"{m}/importance_with_shelter.csv", index_col=0).iloc[:, 0].round(2).head(8).to_dict()

    # 사진 후보 정확도 차이(같은 개체)
    d = np.load("v2/photo_compare/preds_with_shelter.npz", allow_pickle=True)
    out["photo_npz_keys"] = list(d.files)

    # 메모 항목 등장 비율(표 5)
    cats = classify_series(df["mark"])
    out["memo_cat_rate"] = (cats.mean() * 100).round(1).to_dict()

    # 기본 모델: 보호소를 뺐을 때, 사진을 넣었을 때의 정확도 차이(같은 개체, 보호소 단위 95% 구간)
    a, b_ = [pd.read_csv(f"v2/final_base_{WIN}/pred_test_{s}.csv") for s in ("with_shelter", "without_shelter")]
    gs = df["care"].astype(str).values
    if "care" in a.columns:
        gs = a["care"].astype(str).values
    else:
        gs = None
    if gs is not None:
        out["no_shelter_dacc"] = [float(x) for x in boot_acc_diff(a["y"].values, b_["p_cal"].values, a["p_cal"].values, gs)]
    out["pred_cols"] = list(a.columns)
    d = np.load("v2/photo_compare/preds_with_shelter.npz", allow_pickle=True)
    if {"y_test", "g_test"} <= set(d.files) or "y_te" in d.files:
        pass

    # 마스크 뒤 비어 제외된 메모, 사진 평가 구간별 개체 수
    from train_adoption import mask_leak
    out["split_rates"] = {"val": float(1 - out["null_val"]), "test": float(1 - out["null_test"])}

    json.dump(out, open("v2/report_numbers.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(out, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
