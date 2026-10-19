#!/usr/bin/env python3
"""유기견 입양 여부 예측: 전처리 → 시간순 분할 → 로지스틱 회귀 vs CatBoost

사용법
  pip install pandas numpy scikit-learn catboost
  python train_adoption.py --csv abandoned_dogs.csv --out results
  python train_adoption.py --csv abandoned_dogs.csv --out results --group-cv
  python train_adoption.py --csv abandoned_dogs.csv --out results_2025 --group-cv --max-date 20251231
  python train_adoption.py --csv abandoned_dogs.csv --out results_emb --group-cv --max-date 20251231 \
      --emb emb_cache/e5_base.npz   # compare_text_repr.py에서 채택된 구성

설계 원칙
  - 입력은 '들어온 시점'에 알 수 있는 정보만 사용 (처리상태/종료일/보호기간은 사용 안 함)
  - 타깃: 보호가 끝난 개체 중 입양=1, 그 외(안락사·자연사·기증 등)=0
    기본은 '반환(주인 찾음)'을 제외합니다. 포함하려면 --include-returned
  - 분할: 접수일 기준 시간순 (앞 80% 학습, 뒤 20% 시험)
  - 보호소 변수를 넣은 모델과 뺀 모델을 함께 비교
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 원본 컬럼명 후보 (API 버전에 따라 이름이 다를 수 있어 앞에서부터 찾음)
CANDIDATES = {
    "id": ["desertionNo"],
    "date": ["happenDt"],
    "state": ["processState"],
    "breed": ["kindNm", "kindFullNm", "kindCd"],
    "color": ["colorCd"],
    "age": ["age"],
    "weight": ["weight"],
    "sex": ["sexCd"],
    "neuter": ["neuterYn"],
    "mark": ["specialMark"],
    "org": ["orgNm"],
    "care": ["careNm"],
}


# 특징 텍스트 누수 제거 (접수 시점 정보만 남긴다)
# 단어만 지우면 "(입양 진행 중)" → "( 중)"처럼 흔적이 남아 그 자체가 결과 신호가 됐다(입양 97%).
# 그래서 누수 단어가 든 구절을 통째로 지운다. 접수 때 이미 아는 표현은 먼저 누수 단어 없는 이름표로 바꿔 보존한다.
KEEP_TAGS = [
    (r"신고자\s*분?\s*(?:가|께서|이)?\s*(?:임시\s*보호\s*후\s*)?입양\s*(?:을\s*)?(?:희망|원함|원해|원하)", " 신고자희망 "),
    (r"공고\s*중\s*입양\s*가능", " 공고중가능 "),  # 보호소 방침
    (r"입양\s*(?:을\s*)?(?:하세요|해\s*주세요|부탁|바랍니다|바래요|가즈아|가자|보내\s*주세요)"
     r"|(?:빠른|좋은)?\s*분양\s*(?:부탁|바랍|희망)", " 새가족호소 "),
]  # 이름표에는 LEAK_WORDS에 걸리는 글자가 없어야 한다
LEAK_WORDS = re.compile(r"입양|임시\s*보호|임보|숙려|예약|기증|분양|반환|보류|안락사|자연사|폐사|전산\s*처리|종료\s*\("
                        r"|귀가\s*(?:예정|조치|완료|함|됨|시킴)")  # '귀가 쫑긋'(귀+조사)은 남긴다
CLAUSE_SPLIT = re.compile(r"[,;()\[\]!~\n/]+|\.(?!\d)")  # 소수점(2.5kg)은 자르지 않는다


def mask_leak(text):
    for pat, tag in KEEP_TAGS:
        text = re.sub(pat, tag, text)
    kept = [c.strip() for c in CLAUSE_SPLIT.split(text) if c.strip() and not LEAK_WORDS.search(c)]
    return ", ".join(kept)


def pick(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None


def classify_outcome(s):
    s = str(s)
    if "입양" in s:
        return "adopted"
    if "반환" in s:
        return "returned"
    if "안락사" in s or "자연사" in s or "폐사" in s:
        return "died"
    if "기증" in s:
        return "donated"
    if "방사" in s:
        return "released"  # 입양 기회와 무관한 종료라 학습에서 제외
    return None  # 보호중/공고중 등 아직 끝나지 않음


def parse_birth_year(s):
    m = re.search(r"(19|20)\d{2}", str(s))
    return int(m.group()) if m else np.nan


def parse_weight(s):
    # 실제 값 예: "25(Kg)", "3.5(Kg)", "3,5(Kg)"(소수점을 쉼표로 쓴 경우)
    m = re.search(r"\d+(\.\d+)?", str(s).replace(",", "."))
    return float(m.group()) if m else np.nan


def prepare(raw, include_returned, max_date=None):
    c = {k: pick(raw, v) for k, v in CANDIDATES.items()}
    missing = [k for k in ("date", "state") if c[k] is None]
    if missing:
        sys.exit(f"필수 컬럼을 찾지 못했습니다: {missing}\n현재 컬럼: {list(raw.columns)}")

    def g(k, default="NA"):
        if c[k] is None:
            return pd.Series(default, index=raw.index)
        return raw[c[k]].fillna(default).astype(str).str.strip()

    df = pd.DataFrame(index=raw.index)
    df["id"] = g("id")
    df["date"] = pd.to_datetime(g("date"), format="%Y%m%d", errors="coerce")
    df["state_raw"] = raw[c["state"]].astype(str)
    df["outcome"] = raw[c["state"]].map(classify_outcome)

    unmatched = df.loc[df["outcome"].isna(), "state_raw"].value_counts().head(10)
    print("\n[처리상태 중 '종료'로 분류되지 않은 값 상위 10개 — 의도와 맞는지 확인하세요]")
    print(unmatched.to_string() if len(unmatched) else "(없음)")

    df["breed"] = g("breed")
    df["color"] = g("color")
    df["sex"] = g("sex")
    df["neuter"] = g("neuter")
    org = g("org")
    df["sido"] = org.str.split().str[0].fillna("NA")
    df["care"] = g("care")
    df["mark_raw"] = g("mark", "")  # 누수 점검용 원문. 모델 입력에는 쓰지 않는다
    uniq = pd.unique(df["mark_raw"])
    df["mark"] = df["mark_raw"].map(dict(zip(uniq, map(mask_leak, uniq))))
    df["mark_len"] = df["mark"].str.len()
    df["is_mix"] = df["breed"].str.contains("믹스|혼합|잡종|MIX|mix", regex=True).astype(int)
    df["weight_kg"] = g("weight", "").map(parse_weight)
    age_raw = g("age", "")
    birth = age_raw.map(parse_birth_year)
    df["age_years"] = (df["date"].dt.year - birth).clip(lower=0)
    df["under_60d"] = age_raw.str.contains("60일미만").astype(int)  # 예: "2025(60일미만)(년생)"
    df["month"] = df["date"].dt.month

    if max_date is not None:
        # 최근 접수분은 아직 보호중이 많아 빨리 끝난 개체만 남으므로 잘라낸다
        n_before = len(df)
        df = df[df["date"] <= max_date]
        print(f"\n접수일 {max_date.date()} 이후 {n_before - len(df)}건 제외")
    df = df.dropna(subset=["date", "outcome"])
    if c["id"]:
        df = df.drop_duplicates("id")
    print("\n[종료된 개체의 결과 분포]")
    print(df["outcome"].value_counts().to_string())
    df = df[df["outcome"] != "released"]
    if not include_returned:
        df = df[df["outcome"] != "returned"]
    df["y"] = (df["outcome"] == "adopted").astype(int)
    df = df.sort_values("date").reset_index(drop=True)
    print(f"\n사용 표본 {len(df)}건, 입양 비율 {df['y'].mean():.3f}, 기간 {df['date'].min().date()} ~ {df['date'].max().date()}")
    return df


def metrics(y, p):
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
    }


def calibration_table(y, p, bins=10):
    t = pd.DataFrame({"y": y, "p": p})
    t["bin"] = pd.qcut(t["p"], bins, duplicates="drop")
    out = t.groupby("bin", observed=True).agg(n=("y", "size"), 예측평균=("p", "mean"), 실제입양률=("y", "mean"))
    return out.round(3)


def make_catboost(cat, text, iters=800):
    return CatBoostClassifier(
        iterations=iters,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=0,
        early_stopping_rounds=50 if iters >= 500 else None,
    )


def load_emb(path, marks):
    """embed_marks.py 캐시에서 mark별 문장 임베딩을 꺼낸다."""
    z = np.load(path, allow_pickle=True)
    idx = {t: i for i, t in enumerate(z["texts"])}
    missing = sum(m not in idx for m in pd.unique(marks))
    if missing:
        sys.exit(f"{path}: 캐시에 없는 문장 {missing}개 — embed_marks.py를 같은 --max-date로 다시 돌리세요")
    return z["emb"][[idx[m] for m in marks]].astype(np.float32)


def emb_pcs(emb, fit_pos, k):
    """학습 구간(fit_pos)에서만 맞춘 PCA로 k차원 주성분 열을 만든다."""
    pca = PCA(k, random_state=42).fit(emb[fit_pos])
    return pd.DataFrame(pca.transform(emb), columns=[f"emb_pc{i}" for i in range(k)]), pca


def run_variant(name, df, cat, num, text, split, outdir, emb=None, emb_k=32):
    n = len(df)
    n_test = int(n * 0.2)
    tr_all, te = df.iloc[: n - n_test], df.iloc[n - n_test:]
    n_val = int(len(tr_all) * 0.15)
    tr, va = tr_all.iloc[: len(tr_all) - n_val], tr_all.iloc[len(tr_all) - n_val:]
    if te["y"].nunique() < 2 or tr["y"].nunique() < 2:
        sys.exit("학습/시험 구간에 입양·비입양이 모두 있어야 합니다. 기간을 넓히거나 타깃을 확인하세요.")

    text_ok = [t for t in text if df[t].replace("", np.nan).nunique() >= 2]
    results = {}

    # 1) 로지스틱 회귀 기준선 (텍스트 제외)
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), cat),
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
    ])
    lr = make_pipeline(pre, LogisticRegression(max_iter=2000))
    lr.fit(tr_all[cat + num], tr_all["y"])
    results["logistic"] = metrics(te["y"], lr.predict_proba(te[cat + num])[:, 1])

    # 2) CatBoost (범주형 + 특징 텍스트 [+ 문장 임베딩 주성분])
    use = cat + num + text_ok
    X = df[use].copy()
    for col in cat + text_ok:
        X[col] = X[col].astype(str).astype(object)  # pandas 3의 str dtype 대신 object로 명시
    if emb is not None:
        P, pca = emb_pcs(emb, np.arange(len(tr)), emb_k)  # df가 날짜순이라 앞 len(tr)행이 학습 구간
        P.index = X.index
        X = pd.concat([X, P], axis=1)
        use = use + list(P.columns)
        np.savez(os.path.join(outdir, f"emb_pca_{name}.npz"), components=pca.components_, mean=pca.mean_)
    cb = make_catboost(cat, text_ok)
    pool_tr = Pool(X.loc[tr.index], tr["y"], cat_features=cat, text_features=text_ok)
    pool_va = Pool(X.loc[va.index], va["y"], cat_features=cat, text_features=text_ok)
    pool_te = Pool(X.loc[te.index], te["y"], cat_features=cat, text_features=text_ok)
    cb.fit(pool_tr, eval_set=pool_va, use_best_model=True)
    p = cb.predict_proba(pool_te)[:, 1]
    results["catboost"] = metrics(te["y"], p)

    print(f"\n===== [{name}] 시험구간 {te['date'].min().date()} ~ {te['date'].max().date()} (n={len(te)}, 입양비율 {te['y'].mean():.3f}) =====")
    print(pd.DataFrame(results).T.round(4).to_string())
    print("\n보정 확인(CatBoost, 예측 확률 10분위):")
    print(calibration_table(te["y"].values, p).to_string())

    imp = pd.Series(cb.get_feature_importance(pool_tr), index=use).sort_values(ascending=False)
    if emb is not None:  # 주성분 32개를 하나로 합쳐 보여 준다 (파일에는 개별 값도 저장)
        pcs = [c for c in use if c.startswith("emb_pc")]
        imp_show = pd.concat([imp.drop(pcs), pd.Series({"emb_pc(합계)": imp[pcs].sum()})]).sort_values(ascending=False)
    else:
        imp_show = imp
    print("\n변수 중요도 상위:")
    print(imp_show.head(12).round(2).to_string())

    imp.to_csv(os.path.join(outdir, f"importance_{name}.csv"), header=["importance"])
    pd.DataFrame({"id": te["id"], "date": te["date"], "y": te["y"], "p_adopt": p}).to_csv(
        os.path.join(outdir, f"pred_test_{name}.csv"), index=False, encoding="utf-8-sig")
    cb.save_model(os.path.join(outdir, f"model_{name}.cbm"))
    return {k: v for k, v in results["catboost"].items()}, results["logistic"]


def group_cv(df, cat, num, text, emb=None, emb_k=32):
    """보호소 단위 교차검증: 처음 보는 보호소에서도 통하는지."""
    text_ok = [t for t in text if df[t].replace("", np.nan).nunique() >= 2]
    use = cat + num + text_ok
    X0 = df[use].copy()
    for col in cat + text_ok:
        X0[col] = X0[col].astype(str).astype(object)
    aucs = []
    for k, (a, b) in enumerate(GroupKFold(5).split(X0, df["y"], groups=df["care"])):
        if df["y"].iloc[b].nunique() < 2:
            continue
        X = X0
        if emb is not None:  # 주성분도 학습 fold에서만 맞춘다
            P = emb_pcs(emb, a, emb_k)[0]
            P.index = X0.index
            X = pd.concat([X0, P], axis=1)
        m = make_catboost(cat, text_ok, iters=300)
        m.fit(Pool(X.iloc[a], df["y"].iloc[a], cat_features=cat, text_features=text_ok))
        pb = m.predict_proba(Pool(X.iloc[b], cat_features=cat, text_features=text_ok))[:, 1]
        aucs.append(roc_auc_score(df["y"].iloc[b], pb))
        print(f"  fold {k + 1}: AUC {aucs[-1]:.4f}")
    print(f"보호소 단위 5-fold AUC 평균 {np.mean(aucs):.4f} (±{np.std(aucs):.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--include-returned", action="store_true")
    ap.add_argument("--group-cv", action="store_true")
    ap.add_argument("--emb", help="embed_marks.py가 만든 임베딩 캐시(.npz). 주면 CatBoost에 주성분을 더한다")
    ap.add_argument("--emb-k", type=int, default=32, help="임베딩 주성분 수")
    ap.add_argument("--max-date", help="YYYYMMDD. 이 날짜까지 접수된 개체만 사용 (보호중 비율이 낮은 시점으로)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    raw = pd.read_csv(args.csv, dtype=str, encoding="utf-8-sig")
    max_date = pd.to_datetime(args.max_date, format="%Y%m%d") if args.max_date else None
    df = prepare(raw, args.include_returned, max_date)

    num = ["age_years", "under_60d", "weight_kg", "month", "is_mix", "mark_len"]
    text = ["mark"]
    cat_no_shelter = ["breed", "color", "sex", "neuter", "sido"]
    cat_with_shelter = cat_no_shelter + ["care"]

    emb = load_emb(args.emb, df["mark"]) if args.emb else None
    run_variant("without_shelter", df, cat_no_shelter, num, text, None, args.out, emb, args.emb_k)
    run_variant("with_shelter", df, cat_with_shelter, num, text, None, args.out, emb, args.emb_k)

    if args.group_cv:
        print("\n===== 보호소 단위 교차검증 (보호소 변수 제외 모델) =====")
        group_cv(df, cat_no_shelter, num, text, emb, args.emb_k)

    print(f"\n결과 파일 저장 위치: {args.out}/")


if __name__ == "__main__":
    main()
