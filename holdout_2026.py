#!/usr/bin/env python3
"""2026년 접수분으로 최종 평가 (2026-11-02). 무작위 시험 구간은 모델을 고치는 동안 여러 번 봤으므로,
모델 선택에 쓰지 않은 나중 시기 자료로 "앞으로 들어올 개"에 대한 성능을 한 번만 잰다.

  python holdout_2026.py --csv abandoned_dogs.csv --model v2/final_base_lightgbm --out v2/holdout_base
  python holdout_2026.py --csv abandoned_dogs.csv --model v2/final_photo_lightgbm --photo emb_cache/photo_dinov2_2026.npz --out v2/holdout_photo

- 기간은 입양률을 보기 전에 보호중 비율만 보고 정함: 2026-01-01~2026-03-31(보호중 7~11%). 4월부터는 12% 이상이라 뺀다.
- 수집일(2026-10-19)에 끝나지 않은 개는 정답이 없어 빠진다. 그래서 빨리 끝난 개가 조금 더 많이 들어간다.
- 모델·전처리 도구는 저장된 것을 그대로 쓴다(이 자료로 맞추는 것 없음). 결과 파일이 있으면 다시 돌리지 않고 멈춘다.
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd

from model_search import score
from text_analysis import tokenize_all
from train_adoption import calibration_table, prepare
from train_final import apply_cal

START = "20260101"
END = "20260331"


def boot_acc(y, p, g, n=1000, seed=0):
    """정확도·ROC 없이 정확도만 보호소 단위로 다시 뽑은 95% 구간."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(g)
    mem = [np.flatnonzero(codes == k) for k in range(len(uniq))]
    hit = (p >= 0.5) == y
    d = [hit[np.concatenate([mem[k] for k in rng.integers(0, len(mem), len(mem))])].mean() for _ in range(n)]
    return np.percentile(d, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--photo", help="2026년 사진 임베딩 npz. 주면 사진 있는 개체만 평가")
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-only", action="store_true", help="코드 시험용: 정답을 보지 않고 예측까지만")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "holdout.csv")
    if os.path.exists(path):
        raise SystemExit(f"{path}가 이미 있음: 2026년 평가는 한 번만 한다")

    raw = pd.read_csv(args.csv, dtype=str, encoding="utf-8-sig")
    df = prepare(raw, include_returned=False, max_date=pd.to_datetime(END, format="%Y%m%d"))
    df = df[df["date"] >= pd.to_datetime(START, format="%Y%m%d")].reset_index(drop=True)
    prep = pickle.load(open(os.path.join(args.model, "preprocess.pkl"), "rb"))
    family = prep.get("family", "catboost")

    print("메모 임베딩...", flush=True)
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(prep["emb_model"])
    uniq = pd.unique(df["mark"])
    e = st.encode([prep["emb_prefix"] + m for m in uniq], normalize_embeddings=True, batch_size=256, show_progress_bar=False)
    pos = pd.Series(np.arange(len(uniq)), index=uniq)
    pcs = prep["pca32"].transform(e[pos.loc[df["mark"]].to_numpy()])
    toks = tokenize_all(df["mark"])
    txt = prep["txt_model"].predict_proba(prep["tfidf"].transform(toks))[:, 1]

    F = df.copy()
    F["tokens"] = pd.Series(np.asarray(toks), index=df.index).astype(str)
    F["txt_lr"] = txt
    for k in range(32):
        F[f"emb{k}"] = pcs[:, k]
    keep = np.ones(len(F), bool)
    if args.photo:
        from compare_photo import load_photo
        X, has = load_photo(args.photo, F["id"].astype(str).to_numpy())
        if prep.get("photo_pca") is not None:  # 사진 없는 모델이면 같은 개체로 비교하려고 개체만 고른다
            P = np.full((len(X), prep["photo_pca"].n_components_), np.nan, np.float32)
            P[has] = prep["photo_pca"].transform(X[has])
            for k in range(P.shape[1]):
                F[f"photo{k}"] = P[:, k]
        keep = has
    F = F[keep].reset_index(drop=True)
    unseen = ~F["care"].isin(prep["categories"]["care"])
    print(f"평가 대상 {len(F):,}마리, 학습에 없던 보호소의 개 {int(unseen.sum()):,}마리", flush=True)

    rows = []
    for name in ("with_shelter", "without_shelter"):
        if family == "catboost":
            from catboost import CatBoostClassifier
            model = CatBoostClassifier().load_model(os.path.join(args.model, f"catboost_{name}.cbm"))
            names = model.feature_names_
        else:
            model = pickle.load(open(os.path.join(args.model, f"lightgbm_{name}.pkl"), "rb"))
            names = list(model.feature_name_)
        cal = pickle.load(open(os.path.join(args.model, f"calibrator_{name}.pkl"), "rb"))["calibrator"]
        X = F[names].copy()
        for c in ("breed", "color", "sex", "neuter", "sido", "care"):
            if c in names:
                X[c] = X[c].astype(str) if family == "catboost" else pd.Categorical(X[c], categories=prep["categories"][c])
        for c in names:
            if c not in ("breed", "color", "sex", "neuter", "sido", "care", "tokens"):
                X[c] = X[c].astype(np.float32)
        p = apply_cal(cal, model.predict_proba(X)[:, 1])
        if args.test_only:
            print(f"[{name}] 예측 평균 {p.mean():.3f} (코드 시험, 정답은 보지 않음)", flush=True)
            continue
        y = F["y"].values
        s = score(y, p)
        pred = p >= 0.5
        tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
        lo, hi = boot_acc(y, p, F["care"].astype(str).values)
        q = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False)
        r = pd.Series(y).groupby(q).mean()
        row = {"모델": name, "n": len(y), "입양비율": y.mean(), "기준선": 1 - y.mean(), "정확도": (pred == y).mean(), "정확도_lo": lo, "정확도_hi": hi,
               "정밀도": tp / max(1, tp + fp), "재현율": tp / max(1, tp + fn), "ROC": s["roc_auc"], "PR": s["pr_auc"], "Brier": s["brier"],
               "하위10": r.iloc[0], "상위10": r.iloc[-1], "새보호소_n": int(unseen.sum())}
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        print(calibration_table(y, p).to_string(), flush=True)
    if not args.test_only:
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
        print(f"결과: {path}")


if __name__ == "__main__":
    main()
