#!/usr/bin/env python3
"""사진에 보호소 정보가 얼마나 남는지 확인 (원본 축소본 vs 개 상자 자른 것)

  python photo_shelter_check.py --meta photos1k_meta.csv --dir photos1k

방법
  - 이미지 모델(DINOv2 small, 사전학습)로 사진을 벡터로 바꾼다. 결과는 <dir>/emb_<종류>.npz에 저장(다시 돌리면 재사용).
  - 가장 가까운 이웃 시험: 사진마다 코사인 거리로 가장 비슷한 다른 사진을 찾아, 그 사진이 같은 보호소(careRegNo)인지 센다.
    같은 보호소가 표본에 2장 이상 있는 사진만 센다. 우연 기준 = 무작위로 이웃을 골랐을 때 같은 보호소일 확률의 평균.
  - 보호소 분류: 사진 수가 --min-per 이상인 보호소만 남겨 로지스틱 회귀 5-fold 정확도(최빈 보호소 비율과 비교).
  - 참고로 입양 여부 5-fold ROC-AUC(표본이 작아 ±0.03 정도 흔들림).
  두 종류 모두 사진이 있는 같은 개체만 비교한다.
"""
import argparse
import os

import numpy as np
import pandas as pd


def embed(paths, model_name, batch):
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    torch.set_grad_enabled(False)
    proc = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    out = []
    for i in range(0, len(paths), batch):
        ims = [Image.open(p).convert("RGB") for p in paths[i:i + batch]]
        h = model(**proc(images=ims, return_tensors="pt"))
        out.append(h.pooler_output.numpy())  # CLS 토큰
    x = np.concatenate(out)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def nn_same_shelter(x, g):
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    nn = sim.argmax(1)
    cnt = pd.Series(g).map(pd.Series(g).value_counts()).to_numpy()
    m = cnt >= 2
    hit = (g[nn] == g)[m].mean()
    chance = ((cnt - 1) / (len(g) - 1))[m].mean()
    return hit, chance, int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="photos1k_meta.csv")
    ap.add_argument("--dir", default="photos1k")
    ap.add_argument("--model", default="facebook/dinov2-small")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--min-per", type=int, default=8)
    a = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score

    m = pd.read_csv(a.meta, dtype=str, encoding="utf-8-sig")
    m = m[m.status == "200"].copy()
    m["has"] = [all(os.path.exists(os.path.join(a.dir, k, d + ".jpg")) for k in ("full", "crop")) for d in m.desertionNo]
    m = m[m.has].reset_index(drop=True)
    g = m.careRegNo.fillna(m.careNm).to_numpy()
    y = (m.processState == "종료(입양)").astype(int).to_numpy()
    print(f"비교 대상 {len(m)}장, 보호소 {len(set(g))}곳, 입양 비율 {y.mean():.3f}")

    embs = {}
    for kind in ("full", "crop"):
        cache = os.path.join(a.dir, f"emb_{kind}.npz")
        if os.path.exists(cache) and list(np.load(cache)["ids"]) == list(m.desertionNo):
            embs[kind] = np.load(cache)["x"]
        else:
            embs[kind] = embed([os.path.join(a.dir, kind, d + ".jpg") for d in m.desertionNo], a.model, a.batch)
            np.savez(cache, x=embs[kind], ids=m.desertionNo.to_numpy())

    vc = pd.Series(g).value_counts()
    keep = np.isin(g, vc[vc >= a.min_per].index)
    rows = []
    for kind, x in embs.items():
        hit, chance, n = nn_same_shelter(x, g)
        row = {"사진": kind, "이웃 같은 보호소": round(hit, 3), "우연 기준": round(chance, 3), "센 사진 수": n}
        if keep.sum() >= 2 * a.min_per:
            gk = g[keep]
            cv = StratifiedKFold(5, shuffle=True, random_state=0)
            pred = cross_val_predict(LogisticRegression(C=1.0, max_iter=3000), x[keep], gk, cv=cv)
            row.update({"보호소 분류 정확도": round((pred == gk).mean(), 3),
                        "최빈 보호소 비율": round(pd.Series(gk).value_counts(normalize=True).iloc[0], 3),
                        f"분류 대상(보호소 {len(set(gk))}곳)": int(keep.sum())})
        p = cross_val_predict(LogisticRegression(C=0.1, max_iter=3000), x, y,
                              cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
        row["입양 ROC-AUC(참고)"] = round(roc_auc_score(y, p), 3)
        rows.append(row)
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    res.to_csv(os.path.join(a.dir, "shelter_check.csv"), index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
