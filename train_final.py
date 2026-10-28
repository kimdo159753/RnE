#!/usr/bin/env python3
"""최종 모델 학습 + 확률 보정 + 예측용 도구 저장

model_search.py에서 고른 구성(CatBoost, 계열별 최고 설정)을 같은 입력·같은 분할로 다시 학습한다.
  - 입력: 범주형(품종·색·성별·중성화·시도[·보호소]) + 숫자형 + e5 임베딩 PCA32 + TF-IDF 로지스틱 텍스트 점수
  - 조기 종료는 검증 구간, 확률 보정 방법(없음/Platt/isotonic)도 검증 구간 안 5-fold로 고른다.
  - 시험 구간은 보정 전후 성능과 보정표 보고에만 쓴다.

  python train_final.py --csv abandoned_dogs.csv --emb emb_cache/e5_base_v2.npz --search model_search --out final_model
  # 사진 임베딩 추가(compare_photo.py에서 DINOv2 채택): 사진 있는 개체만 학습·평가한다
  python train_final.py --csv abandoned_dogs.csv --emb emb_cache/e5_base_v2.npz --search model_search \
      --photo emb_cache/photo_dinov2.npz --out final_model_photo_v2
  # 최종(2026-10-27 팀 결정): 무작위 분할 + 사진
  python train_final.py ... --photo emb_cache/photo_dinov2.npz --split random --out final_model_random

--photo: 사진이 있는 개체(2022~2025, 접수 3일 이내 사진)만 남기고 사진 PCA 32개(학습 구간에서만 맞춤)를 입력에 더한다.
  사진 없는 개체를 결측으로 넣으면 "사진이 늦게 올라옴" 자체가 결과 신호가 될 수 있어 넣지 않는다.
  사진 없는 개체는 사진 없는 최종 모델(final_model/)로 예측한다.
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from model_search import NUM, Data, fit_catboost, score
from train_adoption import calibration_table


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))[:, None]


def make_calibrator(kind, p, y):
    if kind == "none":
        return None
    if kind == "platt":
        return LogisticRegression(C=1e6).fit(logit(p), y)
    return IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)


def apply_cal(cal, p):
    if cal is None:
        return p
    if isinstance(cal, LogisticRegression):
        return cal.predict_proba(logit(p))[:, 1]
    return cal.predict(p)


def choose_calibration(pv, yv):
    """검증 구간 안 5-fold로 보정 방법별 Brier를 비교한다(보정기를 맞춘 데이터로 평가하지 않기 위해)."""
    out = {}
    for kind in ("none", "platt", "isotonic"):
        q = np.zeros_like(pv)
        for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(pv, yv):
            q[b] = apply_cal(make_calibrator(kind, pv[a], yv[a]), pv[b])
        out[kind] = float(np.mean((q - yv) ** 2))
    best = min(out, key=out.get)
    # isotonic은 확률을 계단으로 묶어 순위가 뭉개진다(PR-AUC 하락). Platt보다 확실히 나을 때만 쓴다
    if best == "isotonic" and out["platt"] - out["isotonic"] < 5e-4:
        best = "platt"
    return best, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--search", default="model_search", help="model_search.py 결과 폴더 (최고 설정을 읽는다)")
    ap.add_argument("--max-date", default="20251231")
    ap.add_argument("--lr-pca", type=int, default=128)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--photo", help="사진 임베딩 npz(embed_photos.py 결과). 주면 사진 있는 개체만 쓰고 사진 PCA를 더한다")
    ap.add_argument("--photo-pca", type=int, default=32)
    ap.add_argument("--split", choices=["time", "random"], default="time",
                    help="random = 같은 비율(학습 68 / 검증 12 / 시험 20%%)을 무작위로(seed 42). 2026-10-27 팀 결정으로 최종 모델은 random")
    ap.add_argument("--out", default="final_model")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    D = Data(args)
    photo_pca = None
    if args.photo:
        from compare_photo import load_photo, set_photo
        X, has = load_photo(args.photo, D.df["id"].astype(str).to_numpy())
        D.i_tr, D.i_va, D.i_te = (i[has[i]] for i in (D.i_tr, D.i_va, D.i_te))
        photo_pca = PCA(args.photo_pca, random_state=42).fit(X[D.i_tr])
        P = np.full((len(X), args.photo_pca), np.nan, np.float32)
        P[has] = photo_pca.transform(X[has])
        D._tree_frame = D.tree_frame
        set_photo(D, P)
        print(f"사진 있는 개체만: 학습 {len(D.i_tr):,} / 검증 {len(D.i_va):,} / 시험 {len(D.i_te):,}, "
              f"사진 PCA {args.photo_pca} (설명 분산 {photo_pca.explained_variance_ratio_.sum():.2f})", flush=True)
    yv, yt = D.y[D.i_va], D.y[D.i_te]
    for shelter in (False, True):
        name = "with_shelter" if shelter else "without_shelter"
        cfg = pickle.load(open(os.path.join(args.search, f"catboost_{name}.pkl"), "rb"))["best"]["cfg"]
        print(f"\n===== {name}: CatBoost {cfg} =====", flush=True)
        pv, pt, info, model = fit_catboost(D, shelter, cfg, return_model=True)
        kind, briers = choose_calibration(pv, yv)
        cal = make_calibrator(kind, pv, yv)
        pt_cal = apply_cal(cal, pt)
        print(f"트리 {info['iters']}개. 보정 방법 검증 5-fold Brier: { {k: round(v, 5) for k, v in briers.items()} } → {kind}")
        s0, s1 = score(yt, pt), score(yt, pt_cal)
        print(f"시험(보정 전) PR {s0['pr_auc']:.4f} ROC {s0['roc_auc']:.4f} Brier {s0['brier']:.4f}")
        print(f"시험(보정 후) PR {s1['pr_auc']:.4f} ROC {s1['roc_auc']:.4f} Brier {s1['brier']:.4f}")
        pred = pt_cal >= 0.5
        tp, fp = int((pred & (yt == 1)).sum()), int((pred & (yt == 0)).sum())
        fn, tn = int((~pred & (yt == 1)).sum()), int((~pred & (yt == 0)).sum())
        print(f"시험 정확도(보정 확률 0.5 기준) {(tp + tn) / len(yt):.4f} (전부 비입양 {1 - yt.mean():.4f}) "
              f"정밀도 {tp / max(1, tp + fp):.4f} 재현율 {tp / max(1, tp + fn):.4f} | TP {tp} FP {fp} FN {fn} TN {tn}")
        print("보정표(시험, 보정 전):")
        print(calibration_table(yt, pt).to_string())
        print("보정표(시험, 보정 후):")
        print(calibration_table(yt, pt_cal).to_string())

        imp = pd.Series(model.get_feature_importance(), index=model.feature_names_)
        embs = [c for c in imp.index if c.startswith("emb")]
        photos = [c for c in imp.index if c.startswith("photo")]
        show = pd.concat([imp.drop(embs + photos), pd.Series({"emb(합계)": imp[embs].sum()}),
                          pd.Series({"photo(합계)": imp[photos].sum()} if photos else {})]).sort_values(ascending=False)
        print("변수 중요도:")
        print(show.round(2).to_string())

        model.save_model(os.path.join(args.out, f"catboost_{name}.cbm"))
        pickle.dump({"calibrator": cal, "kind": kind}, open(os.path.join(args.out, f"calibrator_{name}.pkl"), "wb"))
        pd.DataFrame({"id": D.df["id"].values[D.i_te], "date": D.df["date"].values[D.i_te], "y": yt,
                      "p_raw": pt, "p_cal": pt_cal}).to_csv(os.path.join(args.out, f"pred_test_{name}.csv"),
                                                            index=False, encoding="utf-8-sig")
        imp.to_csv(os.path.join(args.out, f"importance_{name}.csv"), header=["importance"])

    # 새 개체 예측에 필요한 전처리 도구 (모두 학습 구간에서 맞춘 것)
    cats = {c: list(pd.Index(D.df[c].iloc[D.i_tr].unique())) for c in D.cats(True)}
    pickle.dump({"tfidf": D.tfidf, "txt_model": D.txt_model, "pca32": D.pca32, "categories": cats, "num": NUM,
                 "emb_model": "intfloat/multilingual-e5-base", "emb_prefix": "query: ", "max_date": args.max_date,
                 "photo_pca": photo_pca, "photo_model": "facebook/dinov2-small" if photo_pca is not None else None},
                open(os.path.join(args.out, "preprocess.pkl"), "wb"))
    print(f"\n저장: {args.out}/")


if __name__ == "__main__":
    main()
