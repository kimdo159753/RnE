#!/usr/bin/env python3
"""사진 임베딩을 최종 CatBoost에 넣으면 나아지는가 (CLAUDE.md 사진 계획 5)

  python compare_photo.py --csv abandoned_dogs.csv --emb emb_cache/e5_base_v2.npz --search model_search \
      --photo dinov2=emb_cache/photo_dinov2.npz clip=emb_cache/photo_clip.npz convnext=emb_cache/photo_convnext.npz \
      --out photo_compare --group-cv

비교 방법
  - model_search와 같은 분할·같은 입력·같은 CatBoost 최고 설정. 사진 PCA 32개(학습 구간에서만 맞춤)만 더한다.
  - 사진이 있는 개체만 쓴다(학습·검증·시험 모두, 기준 모델도 같은 개체). 2021년(서버에서 지워짐), 업로드가 접수 3일 뒤인
    사진(누수 위험이라 안 받음), 404는 빠진다. 사진 없음을 결측으로 넣으면 "늦게 올라온 사진"이 결과 신호가 될 수 있어서다.
  - 고르기는 검증으로만: 검증 PR-AUC가 가장 높은 사진 모델 1개.
    채택 = 그 모델의 (사진 - 기준) ΔPR-AUC 보호소 클러스터 부트스트랩 95% 하한이 검증에서 0보다 크고, 시험 ΔPR > 0.
    보호소 제외·포함 모델 각각 판정.
  - --group-cv: 보호소 제외 모델을 보호소 단위 5-fold(학습+검증 구간)로 기준 vs 고른 사진 모델 비교.
    처음 보는 보호소에서도 사진이 도움이 되는지(= 보호소 대신 개 외형을 알려 주는지) 보는 마지막 관문.
    조기 종료용은 학습 fold에서 보호소 15%를 따로 뗀다. 주의: txt_lr은 무작위 fold로 만든 값이라 보호소 정보가 조금 섞인다(할 일 8).
  예측은 out/preds_*.npz로 남겨 다시 돌리면 끝난 변형은 건너뛴다.
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from compare_text_repr import boot_diff
from model_search import Data, fit_catboost, score


def load_photo(path, ids):
    z = np.load(path)
    pos = pd.Series(np.arange(len(z["ids"])), index=z["ids"].astype(str))
    j = pos.reindex(ids).to_numpy()
    X = np.full((len(ids), z["x"].shape[1]), np.nan, np.float32)
    ok = ~np.isnan(j)
    X[ok] = z["x"][j[ok].astype(int)]
    return X, ok


def set_photo(D, P):
    """D.tree_frame에 사진 PCA 열을 붙인다(None이면 원래대로)."""
    D.tree_frame = D._tree_frame if P is None else \
        (lambda shelter: D._tree_frame(shelter).assign(**{f"photo{i}": P[:, i] for i in range(P.shape[1])}))


def run(D, shelter, cfg, P, idx):
    """idx = (학습, 검증, 시험) 행 번호로 CatBoost를 맞춰 (검증 예측, 시험 예측, 정보)."""
    D.i_tr, D.i_va, D.i_te = idx
    set_photo(D, P)
    return fit_catboost(D, shelter, cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--search", default="model_search")
    ap.add_argument("--photo", nargs="+", required=True, help="이름=npz 경로")
    ap.add_argument("--max-date", default="20251231")
    ap.add_argument("--lr-pca", type=int, default=128)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--n-pca", type=int, default=32)
    ap.add_argument("--group-cv", action="store_true")
    ap.add_argument("--out", default="photo_compare")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    D = Data(args)
    D._tree_frame = D.tree_frame
    ids = D.df["id"].astype(str).to_numpy()
    photos, has = {}, np.ones(len(ids), bool)
    for spec in args.photo:
        name, path = spec.split("=", 1)
        photos[name], ok = load_photo(path, ids)
        has &= ok
    tr, va, te = (i[has[i]] for i in (D.i_tr, D.i_va, D.i_te))
    print(f"사진 있는 개체: 학습 {len(tr):,}/{len(D.i_tr):,} 검증 {len(va):,}/{len(D.i_va):,} 시험 {len(te):,}/{len(D.i_te):,}",
          flush=True)
    print(f"  입양 비율(사진 있음): 학습 {D.y[tr].mean():.3f} 검증 {D.y[va].mean():.3f} 시험 {D.y[te].mean():.3f}", flush=True)
    P = {}
    for name, X in photos.items():
        pca = PCA(args.n_pca, random_state=42).fit(X[tr])
        P[name] = np.full((len(ids), args.n_pca), np.nan, np.float32)
        P[name][has] = pca.transform(X[has])
        print(f"  {name}: {X.shape[1]}차원 → PCA {args.n_pca} (설명 분산 {pca.explained_variance_ratio_.sum():.2f})", flush=True)

    y_va, y_te = D.y[va], D.y[te]
    g_va, g_te = D.df["care"].values[va], D.df["care"].values[te]
    rows, chosen = [], {}
    for shelter in (False, True):
        name = "with_shelter" if shelter else "without_shelter"
        cfg = pickle.load(open(os.path.join(args.search, f"catboost_{name}.pkl"), "rb"))["best"]["cfg"]
        cache = os.path.join(args.out, f"preds_{name}.npz")
        preds = dict(np.load(cache)) if os.path.exists(cache) else {}
        for v in ["기준"] + list(P):
            if f"val_{v}" not in preds:
                pv, pt, info = run(D, shelter, cfg, None if v == "기준" else P[v], (tr, va, te))
                preds[f"val_{v}"], preds[f"test_{v}"] = pv, pt
                np.savez(cache, **preds)
                print(f"[{name}] {v} 트리 {info['iters']}", flush=True)
            sv, st = score(y_va, preds[f"val_{v}"]), score(y_te, preds[f"test_{v}"])
            print(f"[{name}] {v:<9} 검증 PR {sv['pr_auc']:.4f} ROC {sv['roc_auc']:.4f} | 시험 PR {st['pr_auc']:.4f} "
                  f"ROC {st['roc_auc']:.4f} Brier {st['brier']:.4f}", flush=True)
            rows.append({"모델": name, "변형": v, **{f"검증_{k}": x for k, x in sv.items()},
                         **{f"시험_{k}": x for k, x in st.items()}})
        best = max(P, key=lambda v: score(y_va, preds[f"val_{v}"])["pr_auc"])
        for v in P:
            (vm, vl, vh), (vrm, _, _) = boot_diff(y_va, preds[f"val_{v}"], preds["val_기준"], g_va)
            (tm, tl, th), (trm, _, _) = boot_diff(y_te, preds[f"test_{v}"], preds["test_기준"], g_te)
            ok = vl > 0 and tm > 0
            verdict = ("채택" if ok else "채택 안 함") if v == best else "(검증 1등 아님)"
            print(f"[{name}] {v} - 기준: 검증 ΔPR {vm:+.4f} [{vl:+.4f}, {vh:+.4f}] ΔROC {vrm:+.4f} | "
                  f"시험 ΔPR {tm:+.4f} [{tl:+.4f}, {th:+.4f}] ΔROC {trm:+.4f} → {verdict}", flush=True)
            rows.append({"모델": name, "변형": f"{v} - 기준", "검증_dPR": vm, "검증_dPR_lo": vl, "검증_dPR_hi": vh,
                         "검증_dROC": vrm, "시험_dPR": tm, "시험_dPR_lo": tl, "시험_dPR_hi": th, "시험_dROC": trm,
                         "판정": verdict})
        chosen[name] = best
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "photo_compare.csv"), index=False, encoding="utf-8-sig")

    if args.group_cv:
        name, v = "without_shelter", chosen["without_shelter"]
        cfg = pickle.load(open(os.path.join(args.search, f"catboost_{name}.pkl"), "rb"))["best"]["cfg"]
        pool = np.concatenate([tr, va])
        groups = D.df["care"].values[pool]
        cv_rows = []
        print(f"\n[보호소 단위 5-fold, 보호소 제외 모델] 기준 vs {v}", flush=True)
        for k, (a, b) in enumerate(GroupKFold(5).split(pool, groups=groups)):
            fit_i, stop_i = next(GroupShuffleSplit(1, test_size=0.15, random_state=k).split(a, groups=groups[a]))
            idx = (pool[a[fit_i]], pool[a[stop_i]], pool[b])
            r = {"fold": k}
            for var in ("기준", v):
                _, pt, _ = run(D, False, cfg, None if var == "기준" else P[var], idx)
                s = score(D.y[pool[b]], pt)
                r.update({f"{var}_pr": s["pr_auc"], f"{var}_roc": s["roc_auc"]})
            print(f"  fold {k}: 기준 PR {r['기준_pr']:.4f} ROC {r['기준_roc']:.4f} | {v} PR {r[f'{v}_pr']:.4f} "
                  f"ROC {r[f'{v}_roc']:.4f}", flush=True)
            cv_rows.append(r)
        cv = pd.DataFrame(cv_rows)
        d_pr, d_roc = cv[f"{v}_pr"] - cv["기준_pr"], cv[f"{v}_roc"] - cv["기준_roc"]
        print(f"  평균 ROC 기준 {cv['기준_roc'].mean():.4f} ± {cv['기준_roc'].std():.4f} / {v} {cv[f'{v}_roc'].mean():.4f} "
              f"± {cv[f'{v}_roc'].std():.4f}")
        print(f"  ΔPR {d_pr.mean():+.4f} (fold별 {np.round(d_pr, 4).tolist()}), ΔROC {d_roc.mean():+.4f}, "
              f"5개 fold 모두 오름: {bool((d_pr > 0).all())}")
        cv.to_csv(os.path.join(args.out, "group_cv.csv"), index=False, encoding="utf-8-sig")
    print(f"결과: {args.out}/")


if __name__ == "__main__":
    main()
