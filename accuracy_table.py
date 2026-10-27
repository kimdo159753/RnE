#!/usr/bin/env python3
"""보고서를 정확도(기준 0.5) 중심으로 바꾸려고 만든 표 (2026-10-26).

model_search가 저장한 계열별 검증·시험 예측(preds_*_shelter.npz)과 final_model의 시험 예측으로
정확도, '모두 비입양' 기준선, CatBoost 대비 정확도 차이의 보호소 클러스터 부트스트랩 95% 구간을 낸다.
모델 선택은 원래대로 검증 PR-AUC로 한 것이고, 이 표는 같은 예측을 정확도로 다시 읽은 것이다.

  python accuracy_table.py --csv abandoned_dogs.csv
"""
import argparse

import numpy as np
import pandas as pd

from train_adoption import prepare

FAMS = ["catboost", "ens_top3", "lightgbm", "xgboost", "mlp", "logistic"]


def acc(y, p, t=0.5):
    return float(np.mean((p >= t) == y))


def boot_acc_diff(y, p_new, p_base, groups, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(groups)
    members = [np.flatnonzero(codes == g) for g in range(len(uniq))]
    hit_new, hit_base = (p_new >= 0.5) == y, (p_base >= 0.5) == y
    d = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        d.append(hit_new[idx].mean() - hit_base[idx].mean())
    return np.mean(d), *np.percentile(d, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--max-date", default="20251231")
    args = ap.parse_args()

    raw = pd.read_csv(args.csv, dtype=str, encoding="utf-8-sig")
    df = prepare(raw, include_returned=False, max_date=pd.to_datetime(args.max_date, format="%Y%m%d"))
    n = len(df)
    n_test = int(n * 0.2)
    n_trall = n - n_test
    n_val = int(n_trall * 0.15)
    g_va = df["care"].values[n_trall - n_val:n_trall]
    g_te = df["care"].values[n_trall:]

    for tag in ("with", "without"):
        d = np.load(f"model_search/preds_{tag}_shelter.npz")
        y_va, y_te = d["y_val"], d["y_test"]
        assert len(y_va) == len(g_va) and len(y_te) == len(g_te)
        print(f"\n== 보호소 {'포함' if tag == 'with' else '제외'} ==")
        print(f"모두 비입양: 검증 {1 - y_va.mean():.4f} / 시험 {1 - y_te.mean():.4f}")
        for f in FAMS:
            pv, pt = d[f"val_{f}"], d[f"test_{f}"]
            line = f"{f:10s} 검증 {acc(y_va, pv):.4f}  시험 {acc(y_te, pt):.4f}"
            if f != "catboost":
                vm, vl, vh = boot_acc_diff(y_va, d["val_catboost"], pv, g_va)
                tm, tl, th = boot_acc_diff(y_te, d["test_catboost"], pt, g_te)
                line += f"  CatBoost-이것 검증 {vm:+.4f} [{vl:+.4f}, {vh:+.4f}]  시험 {tm:+.4f} [{tl:+.4f}, {th:+.4f}]"
            print(line)
        fm = pd.read_csv(f"final_model/pred_test_{tag}_shelter.csv")
        assert np.array_equal(fm["y"].values, y_te)
        print(f"최종 모델 시험: 보정 전 {acc(fm['y'].values, fm['p_raw'].values):.4f} / 보정 후 {acc(fm['y'].values, fm['p_cal'].values):.4f}")


if __name__ == "__main__":
    main()
