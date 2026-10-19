#!/usr/bin/env python3
"""특징(specialMark) 텍스트 분석: 누수 점검 → 단어별 입양 연관성 → 텍스트 모델의 추가 설명력

사용법
  pip install kiwipiepy
  python text_analysis.py --csv abandoned_dogs.csv --max-date 20251231 --out text_results

모든 통계와 모델 학습은 시간순 앞 80%(학습 구간)에서만 하고, 성능은 뒤 20%(시험 구간)에서 잰다.
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
from kiwipiepy import Kiwi
from scipy.sparse import hstack
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from train_adoption import prepare

KEEP_TAGS = ("NNG", "NNP", "VA", "VV", "MAG", "XR", "SL")
NEGATORS = {"안", "못"}

# 접수 시점 이후에 덧붙었을 수 있는 표현 (결과나 보호 중 경과를 드러냄)
LEAK_PATTERNS = {
    "임시보호/임보": r"임시\s*보호|임보",
    "입양(단어 자체)": r"입양",
    "예약/문의": r"예약|문의",
    "분양": r"분양",
    "주인/보호자 찾음": r"주인\s*(?:찾|확인|연락)|보호자\s*(?:찾|확인|연락)",
    "반환": r"반환",
    "안락사/사망/폐사": r"안락사|사망|폐사|죽",
    "치료중/입원": r"치료\s*중|입원",
    "칩/등록": r"칩|내장|등록",
}


def tokenize_all(texts):
    """중복 문장을 한 번만 분석해 '내용어 + 부정 결합 + 인접 2-gram' 문자열로 돌려준다."""
    kiwi = Kiwi()
    uniq = pd.unique(texts)
    out = {}
    for text, toks in zip(uniq, kiwi.tokenize(list(uniq))):
        words, neg = [], False
        for t in toks:
            if t.tag == "MAG" and t.form in NEGATORS:
                neg = True
                continue
            if t.tag in KEEP_TAGS:
                form = t.form + ("다" if t.tag in ("VA", "VV") else "")
                if t.tag == "SL":
                    form = form.lower()
                words.append(("안" + form) if neg and t.tag in ("VA", "VV") else form)
            neg = False
        bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
        out[text] = " ".join(words + bigrams)
    return texts.map(out)


def log_odds_z(docs, y, min_docs):
    """단어별 입양 연관성: 정보적 디리클레 사전분포를 둔 로그 오즈비 z (Monroe et al. 2008).
    문서 단위(한 개체에 같은 단어가 여러 번 나와도 1회)로 센다."""
    vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, use_idf=False, norm=None, binary=True,
                          min_df=min_docs)
    X = vec.fit_transform(docs)
    vocab = np.array(vec.get_feature_names_out())
    y = np.asarray(y)
    a = np.asarray(X[y == 1].sum(0)).ravel()
    b = np.asarray(X[y == 0].sum(0)).ravel()
    prior = (a + b) / (a + b).sum() * 500  # 사전분포 크기 500
    na, nb, p0 = a.sum(), b.sum(), prior.sum()
    d = (np.log((a + prior) / (na + p0 - a - prior)) - np.log((b + prior) / (nb + p0 - b - prior)))
    var = 1 / (a + prior) + 1 / (b + prior)
    return pd.DataFrame({
        "word": vocab, "n_docs": a + b, "adopt_rate": (a / (a + b)).round(3),
        "log_odds": d.round(3), "z": (d / np.sqrt(var)).round(2),
    }).sort_values("z")


def fit_eval(name, Xtr, ytr, Xte, yte, C=1.0):
    m = LogisticRegression(C=C, max_iter=3000, solver="liblinear")
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    r = {"model": name, "roc_auc": roc_auc_score(yte, p), "pr_auc": average_precision_score(yte, p)}
    print(f"  {name:<34} ROC {r['roc_auc']:.4f}  PR {r['pr_auc']:.4f}")
    return m, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--max-date", default="20251231")
    ap.add_argument("--out", default="text_results")
    ap.add_argument("--min-docs", type=int, default=100, help="단어 통계에 넣을 최소 등장 개체 수")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    raw = pd.read_csv(args.csv, dtype=str, encoding="utf-8-sig")
    df = prepare(raw, include_returned=False, max_date=pd.to_datetime(args.max_date, format="%Y%m%d"))
    n_test = int(len(df) * 0.2)
    tr, te = df.iloc[: len(df) - n_test], df.iloc[len(df) - n_test:]

    # 1) 누수 점검: 결과를 드러내는 표현이 결과별로 얼마나 다르게 나오나 (지우기 전 원문 mark_raw로)
    print("\n===== 1) 누수 의심 표현 (학습 구간 원문, 결과별 등장 비율) =====")
    rows = []
    for label, pat in LEAK_PATTERNS.items():
        hit = tr["mark_raw"].str.contains(pat, regex=True)
        r = {"표현": label, "등장": int(hit.sum()), "입양률(등장)": tr.loc[hit, "y"].mean(),
             "입양률(미등장)": tr.loc[~hit, "y"].mean()}
        for oc in ("adopted", "died", "donated"):
            r[f"비율_{oc}"] = hit[tr["outcome"] == oc].mean()
        rows.append(r)
        ex = tr.loc[hit, "mark_raw"].drop_duplicates().head(3).str[:40].tolist()
        print(f"  {label}: 예시 {ex}")
    leak = pd.DataFrame(rows)
    print(leak.round(4).to_string(index=False))
    leak.to_csv(os.path.join(args.out, "leak_check.csv"), index=False, encoding="utf-8-sig")

    # 2) 단어별 연관성 (prepare에서 mask_leak으로 누수 구절을 지운 mark 사용)
    print("\n형태소 분석 중...")
    df["tokens"] = tokenize_all(df["mark"])
    tr, te = df.iloc[: len(df) - n_test], df.iloc[len(df) - n_test:]
    lo = log_odds_z(tr["tokens"], tr["y"], args.min_docs)
    lo.to_csv(os.path.join(args.out, "word_log_odds.csv"), index=False, encoding="utf-8-sig")
    print(f"\n===== 2) 단어별 입양 연관성 (학습 구간, {args.min_docs}개체 이상 등장한 {len(lo)}개) =====")
    print("[입양과 반대 방향 상위 25]")
    print(lo.head(25).to_string(index=False))
    print("[입양 쪽 상위 25]")
    print(lo.tail(25).iloc[::-1].to_string(index=False))

    # 3) 텍스트의 추가 설명력: 정형 변수만 / 텍스트만 / 둘 다 / 보호소까지 통제
    print("\n===== 3) 로지스틱 회귀 시험 구간 성능 =====")
    cat = ["breed", "color", "sex", "neuter", "sido"]
    num = ["age_years", "under_60d", "weight_kg", "month", "is_mix", "mark_len"]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), cat),
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
    ])
    S_tr, S_te = pre.fit_transform(tr), pre.transform(te)
    sh = OneHotEncoder(handle_unknown="ignore", min_frequency=10)
    H_tr, H_te = sh.fit_transform(tr[["care"]]), sh.transform(te[["care"]])
    tf = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, min_df=20, sublinear_tf=True)
    T_tr, T_te = tf.fit_transform(tr["tokens"]), tf.transform(te["tokens"])
    print(f"  텍스트 어휘 {T_tr.shape[1]}개")

    res = []
    res.append(fit_eval("정형 변수만", S_tr, tr["y"], S_te, te["y"])[1])
    res.append(fit_eval("텍스트만", T_tr, tr["y"], T_te, te["y"])[1])
    res.append(fit_eval("정형 + 텍스트", hstack([S_tr, T_tr]).tocsr(), tr["y"], hstack([S_te, T_te]).tocsr(), te["y"])[1])
    res.append(fit_eval("정형 + 보호소", hstack([S_tr, H_tr]).tocsr(), tr["y"], hstack([S_te, H_te]).tocsr(), te["y"])[1])
    m_full, r = fit_eval("정형 + 보호소 + 텍스트", hstack([S_tr, H_tr, T_tr]).tocsr(), tr["y"],
                         hstack([S_te, H_te, T_te]).tocsr(), te["y"])
    res.append(r)
    pd.DataFrame(res).to_csv(os.path.join(args.out, "model_compare.csv"), index=False, encoding="utf-8-sig")

    # 보호소·정형 변수를 통제한 뒤에도 남는 단어 효과 (보호소별 문구 습관과 분리)
    n_other = S_tr.shape[1] + H_tr.shape[1]
    coef = pd.DataFrame({"word": tf.get_feature_names_out(), "coef_controlled": m_full.coef_[0][n_other:]})
    coef = coef.merge(lo[["word", "n_docs", "adopt_rate", "z"]], on="word", how="left")
    coef = coef[coef["n_docs"].fillna(0) >= args.min_docs].sort_values("coef_controlled")
    coef.to_csv(os.path.join(args.out, "word_coef_controlled.csv"), index=False, encoding="utf-8-sig")
    print(f"\n===== 보호소·정형 변수 통제 후 단어 계수 ({args.min_docs}개체 이상) =====")
    print("[입양과 반대 방향 상위 20]")
    print(coef.head(20).round(3).to_string(index=False))
    print("[입양 쪽 상위 20]")
    print(coef.tail(20).iloc[::-1].round(3).to_string(index=False))
    print(f"\n결과 파일: {args.out}/")


if __name__ == "__main__":
    main()
