#!/usr/bin/env python3
"""메모 항목(memo_categories)이 입양에 주는 영향: 다른 변수를 통제한 로지스틱 회귀

- 통제 변수: 품종(상위 30개 + 기타), 털색(상위 20개 + 기타), 성별, 중성화, 시도, 접수 연월(계절·연도), 나이, 체중
- 두 가지 모형: (1) 보호소 미통제 (2) 보호소 고정효과 통제 — 같은 보호소 안에서 비교한 효과
- 표준오차는 보호소 단위로 묶은(cluster-robust sandwich) 값. 같은 보호소 개체끼리 닮아 있어서다.
  statsmodels GLM은 보호소 더미 300개를 밀집 행렬로 다루다 메모리가 넘쳐서(16GB), 희소 행렬 + 직접 계산으로 바꿨다
- 효과 크기는 평균 한계효과(%p): 모든 개체에 대해 항목을 0→1로 바꿨을 때 예측 입양률 변화의 평균
- 연관 분석이다. 인과 효과가 아니다(메모는 개의 실제 상태를 반영하고, 보호소 글쓰기 습관도 섞인다)

  python memo_effects.py --csv abandoned_dogs.csv --out memo_check
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from memo_categories import CATEGORIES, classify_series
from train_adoption import prepare


def top_or_other(s, n):
    keep = s.value_counts().index[:n]
    return s.where(s.isin(keep), "기타")


def design(df, flags, shelter):
    """희소 행렬로 만든다(보호소 300개 더미를 밀집 행렬로 두면 메모리가 넘친다). 앞쪽 열 = 메모 항목."""
    num = np.column_stack([df["age_years"].fillna(df["age_years"].median()),
                           np.log1p(df["weight_kg"].clip(0, 80).fillna(df["weight_kg"].median()))])
    cats = pd.DataFrame({"품종": top_or_other(df["breed"], 30), "털색": top_or_other(df["color"], 20), "성별": df["sex"],
                         "중성화": df["neuter"], "시도": df["sido"], "연월": df["date"].dt.strftime("%Y-%m")})
    if shelter:
        cats["보호소"] = top_or_other(df["care"], 300)
    oh = OneHotEncoder(drop="first", sparse_output=True).fit_transform(cats)
    X = sparse.hstack([sparse.csr_matrix(flags.values.astype(float)), sparse.csr_matrix(num), oh,
                       sparse.csr_matrix(np.ones((len(df), 1)))]).tocsr()
    return X


def fit(df, flags, shelter):
    """로지스틱 MLE(사실상 무벌점) + 보호소 클러스터 sandwich 표준오차."""
    X, y = design(df, flags, shelter), df["y"].values.astype(float)
    # lbfgs는 보호소 더미 300개가 들어가면 5000회에도 수렴하지 않았다 → 뉴턴 방법(행 많고 열 수백 개에 적합)
    m = LogisticRegression(C=1e6, fit_intercept=False, solver="newton-cholesky", max_iter=200, tol=1e-10).fit(X, y)
    if m.n_iter_[0] >= 200:
        raise RuntimeError("로지스틱 회귀가 수렴하지 않았다")
    beta = m.coef_[0]
    eta = X @ beta
    p = 1 / (1 + np.exp(-eta))
    H = (X.T @ X.multiply((p * (1 - p))[:, None])).toarray()  # 정보 행렬
    groups = pd.factorize(df["care"])[0]
    G = groups.max() + 1
    score = X.multiply((y - p)[:, None]).tocsr()
    S = (sparse.csr_matrix((np.ones(len(y)), (groups, np.arange(len(y)))), shape=(G, len(y))) @ score).toarray()
    Hinv = np.linalg.pinv(H)
    cov = Hinv @ (S.T @ S) @ Hinv * G / (G - 1)
    se = np.sqrt(np.diag(cov))
    out = []
    for j, k in enumerate(flags.columns):
        xk = flags[k].values
        ame = float(np.mean(1 / (1 + np.exp(-(eta + beta[j] * (1 - xk)))) - 1 / (1 + np.exp(-(eta - beta[j] * xk)))))
        z = beta[j] / se[j]
        out.append({"항목": k, "오즈비": np.exp(beta[j]), "오즈비_하한": np.exp(beta[j] - 1.96 * se[j]),
                    "오즈비_상한": np.exp(beta[j] + 1.96 * se[j]), "효과(%p)": ame * 100, "p값": 2 * stats.norm.sf(abs(z))})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--max-date", default="20251231")
    ap.add_argument("--out", default="memo_check")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    raw = pd.read_csv(args.csv, dtype=str, encoding="utf-8-sig")
    df = prepare(raw, include_returned=False, max_date=pd.to_datetime(args.max_date, format="%Y%m%d"))
    flags = classify_series(df["mark"])
    flags = flags[[c for c in CATEGORIES if flags[c].sum() >= 300]]  # 너무 드문 항목은 추정이 불안정해 뺀다
    base = df["y"].mean()

    raw_rate = pd.DataFrame({"항목": flags.columns,
                             "등장": [int(flags[c].sum()) for c in flags.columns],
                             "입양률(있음)": [df.loc[flags[c] == 1, "y"].mean() for c in flags.columns],
                             "입양률(없음)": [df.loc[flags[c] == 0, "y"].mean() for c in flags.columns]})
    print(f"대상 {len(df)}건, 평균 입양률 {base:.3f}", flush=True)
    t1 = fit(df, flags, shelter=False)
    t1.to_csv(os.path.join(args.out, "memo_effects_model1.csv"), index=False, encoding="utf-8-sig")
    print("모형 1 완료", flush=True)
    t2 = fit(df, flags, shelter=True)
    t2.to_csv(os.path.join(args.out, "memo_effects_model2.csv"), index=False, encoding="utf-8-sig")
    print("모형 2 완료", flush=True)
    tab = raw_rate.merge(t1, on="항목").merge(t2[["항목", "오즈비", "효과(%p)", "p값"]], on="항목", suffixes=("", "_보호소통제"))
    tab = tab.sort_values("효과(%p)")
    tab.to_csv(os.path.join(args.out, "memo_effects.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(tab.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
