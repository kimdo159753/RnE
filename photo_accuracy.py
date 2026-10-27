#!/usr/bin/env python3
"""표 7(이미지 모델 비교)에 정확도를 채우는 스크립트 (2026-10-27). 다시 학습하지 않는다.

compare_photo.py가 저장한 예측 캐시(photo_compare/preds_{with,without}_shelter.npz의 val_*/test_*)를
그대로 읽어, 사진 있는 개체의 검증·시험 정확도(보정 전 확률을 0.5로 판정), '모두 비입양' 기준선,
시험 ROC-AUC, 사진 모델과 기준(사진 없음)의 정확도 차이의 보호소 클러스터 부트스트랩 95% 구간을 낸다.
사진 있는 개체를 고르는 방법은 compare_photo.py와 같다(--photo로 준 npz 모두에 id가 있는 개체).

사진 임베딩(emb_cache/photo_*.npz)과 예측 캐시가 있는 PC(팀원 PC)에서 실행:

  python photo_accuracy.py --csv abandoned_dogs.csv --cache photo_compare \
      --photo dinov2=emb_cache/photo_dinov2.npz clip=emb_cache/photo_clip.npz convnext=emb_cache/photo_convnext.npz

결과: photo_compare/photo_accuracy.csv (화면에도 표 7 형식으로 출력)
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from accuracy_table import acc, boot_acc_diff
from train_adoption import prepare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--cache", default="photo_compare")
    ap.add_argument("--photo", nargs="+", required=True, help="이름=npz 경로 (compare_photo.py와 같게)")
    ap.add_argument("--max-date", default="20251231")
    args = ap.parse_args()

    raw = pd.read_csv(args.csv, dtype=str, encoding="utf-8-sig")
    df = prepare(raw, include_returned=False, max_date=pd.to_datetime(args.max_date, format="%Y%m%d"))
    del raw
    # model_search.Data와 같은 시간순 분할
    n = len(df)
    n_test = int(n * 0.2)
    n_trall = n - n_test
    n_val = int(n_trall * 0.15)
    i_va, i_te = np.arange(n_trall - n_val, n_trall), np.arange(n_trall, n)
    ids = df["id"].astype(str).to_numpy()
    has = np.ones(n, bool)
    names = []
    for spec in args.photo:
        name, path = spec.split("=", 1)
        names.append(name)
        has &= np.isin(ids, np.load(path)["ids"].astype(str))
    va, te = i_va[has[i_va]], i_te[has[i_te]]
    y_va, y_te = df["y"].values[va], df["y"].values[te]
    g_va, g_te = df["care"].values[va], df["care"].values[te]
    print(f"사진 있는 개체: 검증 {len(va):,} / 시험 {len(te):,}")

    rows = []
    for tag in ("with_shelter", "without_shelter"):
        path = os.path.join(args.cache, f"preds_{tag}.npz")
        if not os.path.exists(path):
            print(f"[{tag}] 예측 캐시 없음: {path}")
            continue
        d = np.load(path)
        assert len(d["val_기준"]) == len(va) and len(d["test_기준"]) == len(te), "캐시와 사진 있는 개체 수가 다름(--photo 확인)"
        print(f"\n== {'보호소 포함' if tag == 'with_shelter' else '보호소 제외'} ==")
        print(f"{'입력':<12}{'검증 정확도':>10}{'시험 정확도':>10}{'시험 ROC':>9}   시험 Δ정확도 [95% 구간]")
        print(f"{'모두 비입양':<12}{1 - y_va.mean():>10.1%}{1 - y_te.mean():>10.1%}{0.5:>9.3f}")
        rows.append({"모델": tag, "입력": "모두 비입양(기준선)", "검증_acc": 1 - y_va.mean(), "시험_acc": 1 - y_te.mean(), "시험_roc": 0.5})
        for v in ["기준"] + names:
            pv, pt = d[f"val_{v}"], d[f"test_{v}"]
            r = {"모델": tag, "입력": "사진 없음" if v == "기준" else f"+ {v}", "검증_acc": acc(y_va, pv),
                 "시험_acc": acc(y_te, pt), "시험_roc": roc_auc_score(y_te, pt)}
            line = f"{r['입력']:<12}{r['검증_acc']:>10.1%}{r['시험_acc']:>10.1%}{r['시험_roc']:>9.3f}"
            if v != "기준":
                vm, vl, vh = boot_acc_diff(y_va, pv, d["val_기준"], g_va)
                tm, tl, th = boot_acc_diff(y_te, pt, d["test_기준"], g_te)
                r.update({"검증_dacc": vm, "검증_dacc_lo": vl, "검증_dacc_hi": vh, "시험_dacc": tm, "시험_dacc_lo": tl, "시험_dacc_hi": th})
                line += f"   {tm * 100:+.2f}%p [{tl * 100:+.2f}, {th * 100:+.2f}] (검증 {vm * 100:+.2f} [{vl * 100:+.2f}, {vh * 100:+.2f}])"
            print(line)
            rows.append(r)
    out = os.path.join(args.cache, "photo_accuracy.csv")
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n결과: {out}")


if __name__ == "__main__":
    main()
