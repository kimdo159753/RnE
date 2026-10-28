#!/usr/bin/env python3
"""같은 개에 메모만 바꿔 넣었을 때의 예측 입양 확률 (보고서 표 6, 2026-10-27 다시 만듦)

  python memo_whatif.py --csv abandoned_dogs.csv --model final_model_random_base

기준 개: 경기도 '남양주동물보호협회'(경기도 평균 입양률 수준 보호소)에 들어온 2살·8kg 믹스견.
털색·성별·중성화·접수 월은 같은 보호소의 2~3살·5~10kg 믹스견 중 가장 흔한 값으로 둔다(실행할 때 출력).
메모는 학습 때와 같은 순서로 바꾼다: 누수 구절 제거(mask_leak) → kiwi 형태소 → TF-IDF 로지스틱 점수(txt_lr),
e5 문장 임베딩("query: ", 정규화) → 학습 구간 PCA32. 보호소 포함 모델 + 검증 구간 보정기를 쓴다.
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from text_analysis import tokenize_all
from train_adoption import mask_leak, prepare
from train_final import apply_cal

MEMOS = ["", "순함", "겁이 많지만 애교 많음", "순하지만 피부병 심함", "파보 양성", "공격성 없음", "겁이 많음", "피부병 심함"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", required=True, help="train_final.py 결과 폴더(사진 없는 모델)")
    ap.add_argument("--care-name", default="남양주동물보호협회")
    args = ap.parse_args()

    raw = pd.read_csv(args.csv, dtype=str, encoding="utf-8-sig")
    df = prepare(raw, include_returned=False, max_date=pd.to_datetime("20251231"))
    names = raw.drop_duplicates("careRegNo").set_index("careRegNo")["careNm"]
    care_ids = [c for c in df["care"].unique() if names.get(c, c) == args.care_name or c == args.care_name]
    care = max(care_ids, key=lambda c: (df["care"] == c).sum())
    same = df[(df["care"] == care) & (df["is_mix"] == 1) & df["age_years"].between(2, 3) & df["weight_kg"].between(5, 10)]
    base = {"breed": same["breed"].mode()[0], "color": same["color"].mode()[0], "sex": same["sex"].mode()[0],
            "neuter": same["neuter"].mode()[0], "sido": df.loc[df["care"] == care, "sido"].mode()[0], "care": care,
            "age_years": 2.0, "under_60d": 0, "weight_kg": 8.0, "month": int(same["month"].mode()[0]), "is_mix": 1}
    print(f"보호소 {args.care_name}({care}) 입양률 {df.loc[df['care'] == care, 'y'].mean():.3f}, "
          f"{base['sido']} 평균 {df.loc[df['sido'] == base['sido'], 'y'].mean():.3f}, 비슷한 개 {len(same)}마리")
    print("기준 개:", base)

    prep = pickle.load(open(os.path.join(args.model, "preprocess.pkl"), "rb"))
    model = CatBoostClassifier().load_model(os.path.join(args.model, "catboost_with_shelter.cbm"))
    cal = pickle.load(open(os.path.join(args.model, "calibrator_with_shelter.pkl"), "rb"))["calibrator"]
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(prep["emb_model"])

    marks = [mask_leak(m) for m in MEMOS]
    toks = tokenize_all(pd.Series(marks))
    txt = prep["txt_model"].predict_proba(prep["tfidf"].transform(toks))[:, 1]
    emb = st.encode([prep["emb_prefix"] + (m or "없음") for m in marks], normalize_embeddings=True)
    pcs = prep["pca32"].transform(emb)
    rows = []
    for i, m in enumerate(marks):
        r = {**base, "mark_len": len(m), "txt_lr": txt[i], "tokens": str(toks.iloc[i]),
             **{f"emb{k}": pcs[i, k] for k in range(32)}}
        rows.append(r)
    X = pd.DataFrame(rows)[model.feature_names_]
    for c in ["breed", "color", "sex", "neuter", "sido", "care"]:
        X[c] = X[c].astype(str)
    p = apply_cal(cal, model.predict_proba(X)[:, 1])
    out = pd.DataFrame({"메모": [m or "(메모 없음)" for m in MEMOS], "예측 입양 확률": np.round(p * 100, 1)})
    out = out.sort_values("예측 입양 확률", ascending=False)
    print(out.to_string(index=False))
    out.to_csv(os.path.join(args.model, "memo_whatif.csv"), index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
