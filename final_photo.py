#!/usr/bin/env python3
"""사진을 넣은 최종 모델: 학습 + 보정 + 평가(정확도 포함), 시간순 분할과 무작위 분할 비교

  python final_photo.py --csv abandoned_dogs.csv --emb emb_cache/e5_base_v2.npz --photo emb_cache/photo_dinov2.npz \
      --search model_search --out final_model_photo

- 구성: train_final.py와 같은 CatBoost 최고 설정 + DINOv2 사진 PCA32(compare_photo.py에서 채택). 사진 있는 개체만
  (2022~2025, 업로드 3일 이내, 사진 받음). 같은 개체로 사진 없는 모델(기준)도 학습해 나란히 본다.
- 분할 2가지(사용자 요청 2026-10-26: "시간순 분할이 문제가 될 수 있음"):
    time   = 설계 기본. 접수일 순 앞 68% 학습 / 12% 검증 / 뒤 20% 시험(model_search.Data와 같은 경계).
    random = 같은 비율을 무작위로(seed 42). 비교용. 미래 정보가 학습에 섞이므로 실제 쓰임(새로 들어온 개 예측)보다 낙관적일 수 있다.
  두 분할 모두 텍스트 점수·임베딩 PCA·사진 PCA·보정은 그 분할의 학습(보정은 검증) 구간에서만 맞춘다.
- 지표: ROC-AUC, PR-AUC, Brier(보정 전후), 정확도(ACC, 사용자 요청) = 보정 확률 0.5 기준 + 검증에서 정확도가 가장
  높은 기준값을 시험에 적용한 값, 전부 비입양으로 찍었을 때 정확도(기준선), 혼동행렬·정밀도·재현율.
  사진 - 기준 차이는 보호소 클러스터 부트스트랩 95% 구간.
- 저장: out/results.csv(모든 조합), out/preds_<분할>_<보호소>.npz, 시간순 사진 모델(최종 후보)은
  out/catboost_<보호소>.cbm, calibrator_<보호소>.pkl, photo_pca.pkl.
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from compare_photo import load_photo, set_photo
from compare_text_repr import boot_diff
from model_search import Data, fit_catboost, score
from train_adoption import calibration_table
from train_final import apply_cal, choose_calibration, make_calibrator


def acc_report(y, p, thr):
    pred = (p >= thr).astype(int)
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    fn, tn = int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    return {"acc": (tp + tn) / len(y), "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
            "specificity": tn / max(1, tn + fp), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def best_acc_threshold(y, p):
    grid = np.round(np.arange(0.2, 0.81, 0.01), 2)
    return float(grid[np.argmax([((p >= t) == y).mean() for t in grid])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--photo", required=True)
    ap.add_argument("--search", default="model_search")
    ap.add_argument("--max-date", default="20251231")
    ap.add_argument("--lr-pca", type=int, default=128)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--splits", nargs="+", default=["time", "random"])
    ap.add_argument("--out", default="final_model_photo")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for split in args.splits:
        args.split = split
        print(f"\n################ 분할: {split} ################", flush=True)
        D = Data(args)
        D._tree_frame = D.tree_frame
        X, has = load_photo(args.photo, D.df["id"].astype(str).to_numpy())
        tr, va, te = (i[has[i]] for i in (D.i_tr, D.i_va, D.i_te))
        y_va, y_te = D.y[va], D.y[te]
        g_te = D.df["care"].values[te]
        d = D.df["date"]
        print(f"사진 있는 개체: 학습 {len(tr):,} 검증 {len(va):,} 시험 {len(te):,} | 입양 비율 학습 {D.y[tr].mean():.3f} "
              f"검증 {y_va.mean():.3f} 시험 {y_te.mean():.3f}", flush=True)
        print(f"  기간: 학습 {d.iloc[tr].min().date()}~{d.iloc[tr].max().date()} 시험 {d.iloc[te].min().date()}~"
              f"{d.iloc[te].max().date()}", flush=True)
        pca = PCA(32, random_state=42).fit(X[tr])
        P = np.full((len(X), 32), np.nan, np.float32)
        P[has] = pca.transform(X[has])
        if split == "time":
            pickle.dump(pca, open(os.path.join(args.out, "photo_pca.pkl"), "wb"))
        D.i_tr, D.i_va, D.i_te = tr, va, te
        for shelter in (False, True):
            name = "with_shelter" if shelter else "without_shelter"
            cfg = pickle.load(open(os.path.join(args.search, f"catboost_{name}.pkl"), "rb"))["best"]["cfg"]
            preds = {}
            for v, PP in (("기준", None), ("사진", P)):
                set_photo(D, PP)
                pv, pt, info, model = fit_catboost(D, shelter, cfg, return_model=True)
                kind, _ = choose_calibration(pv, y_va)
                cal = make_calibrator(kind, pv, y_va)
                pv_c, pt_c = apply_cal(cal, pv), apply_cal(cal, pt)
                thr = best_acc_threshold(y_va, pv_c)
                s_raw, s_cal = score(y_te, pt), score(y_te, pt_c)
                a5, ab = acc_report(y_te, pt_c, 0.5), acc_report(y_te, pt_c, thr)
                preds[v] = pt_c
                row = {"분할": split, "모델": name, "변형": v, "트리": info["iters"], "보정": kind, "n_test": len(te),
                       "시험_입양비율": y_te.mean(), "ROC": s_cal["roc_auc"], "PR": s_cal["pr_auc"],
                       "Brier_보정전": s_raw["brier"], "Brier": s_cal["brier"],
                       "ACC_전부비입양": 1 - y_te.mean(), **{f"{k}@0.5": x for k, x in a5.items()},
                       "기준값_검증ACC최대": thr, **{f"{k}@best": x for k, x in ab.items()}}
                rows.append(row)
                print(f"[{split}/{name}] {v}: 트리 {info['iters']} 보정 {kind} | ROC {s_cal['roc_auc']:.4f} PR {s_cal['pr_auc']:.4f} "
                      f"Brier {s_cal['brier']:.4f} | ACC@0.5 {a5['acc']:.4f} (전부 비입양 {1 - y_te.mean():.4f}) "
                      f"정밀도 {a5['precision']:.3f} 재현율 {a5['recall']:.3f} | 검증 최적 기준 {thr} → ACC {ab['acc']:.4f}",
                      flush=True)
                if split == "time" and v == "사진":
                    model.save_model(os.path.join(args.out, f"catboost_{name}.cbm"))
                    pickle.dump({"calibrator": cal, "kind": kind, "acc_threshold": thr},
                                open(os.path.join(args.out, f"calibrator_{name}.pkl"), "wb"))
                    print("보정표(시험, 보정 후):\n" + calibration_table(y_te, pt_c).to_string(), flush=True)
                    imp = pd.Series(model.get_feature_importance(), index=model.feature_names_)
                    groups = {"사진(합계)": [c for c in imp.index if c.startswith("photo")],
                              "메모 임베딩(합계)": [c for c in imp.index if c.startswith("emb")]}
                    show = imp.drop(sum(groups.values(), []))
                    for k, cols in groups.items():
                        show[k] = imp[cols].sum()
                    show.sort_values(ascending=False).to_csv(os.path.join(args.out, f"importance_{name}.csv"),
                                                             header=["importance"])
                    print("변수 중요도:\n" + show.sort_values(ascending=False).round(2).to_string(), flush=True)
            (m, lo, hi), (rm, rlo, rhi) = boot_diff(y_te, preds["사진"], preds["기준"], g_te)
            rows.append({"분할": split, "모델": name, "변형": "사진 - 기준", "dPR": m, "dPR_lo": lo, "dPR_hi": hi,
                         "dROC": rm, "dROC_lo": rlo, "dROC_hi": rhi,
                         "dACC@0.5": rows[-1]["acc@0.5"] - rows[-2]["acc@0.5"]})
            print(f"[{split}/{name}] 사진 - 기준 (시험): ΔPR {m:+.4f} [{lo:+.4f}, {hi:+.4f}] ΔROC {rm:+.4f} [{rlo:+.4f}, {rhi:+.4f}]",
                  flush=True)
            np.savez(os.path.join(args.out, f"preds_{split}_{name}.npz"), y=y_te, ids=D.df["id"].values[te].astype(str),
                     base=preds["기준"], photo=preds["사진"])
        pd.DataFrame(rows).to_csv(os.path.join(args.out, "results.csv"), index=False, encoding="utf-8-sig")
        del D
    print(f"\n결과: {args.out}/results.csv", flush=True)


if __name__ == "__main__":
    main()
