#!/usr/bin/env python3
"""사진 처리 시험 (2,000장): 글씨 지우기·누끼가 사진 속 보호소 정보를 얼마나 줄이는가

  python photo_clean_check.py --n 2000                     # photos_all/crop에서 무작위 표본 → photo_clean/
  python photo_clean_check.py --n 2000 --skip-make         # 처리 사진은 그대로 두고 시험만 다시
  python photo_clean_check.py --all text_nobg --out photo_clean_all   # 시험으로 고른 처리를 전체 사진에

만드는 사진 4종 (모두 <out>/<종류>/<desertionNo>.jpg, 크기는 자른 사진 그대로)
  crop       자른 사진 그대로(비교 기준)
  text       글씨 지움: EasyOCR(CRAFT) 글씨 상자 → 몇 픽셀 넓힌 마스크 → OpenCV inpaint(TELEA)로 주변 색 메우기.
             검은 상자로 가리면 가린 상자 자체가 보호소 표시가 되므로 메운다. 글씨 없는 사진은 원본 그대로.
  nobg       누끼: rembg(u2net)로 배경을 지우고 모든 사진을 같은 회색(127)으로 채움.
  text_nobg  글씨 지운 뒤 누끼.
  has_text(글씨 있음)는 기록만 하고 모델 입력에는 쓰지 않는다(보호소 신호). 해석용.

시험 (photo_shelter_check.py와 같은 방식, DINOv2-small CLS 임베딩)
  - 가장 비슷한 사진이 같은 보호소일 비율. 같은 표본에서 우연 기준과 "품종이 같고 체중이 가장 비슷한 개" 기준을 함께 낸다.
    판단: 처리 후 값이 품종+체중 기준에 가까워지면 전체 적용, 거의 안 줄면 자른 사진 그대로(CLAUDE.md 계획 2).
  - 보호소 분류 정확도(사진 --min-per장 이상인 보호소만), 입양 ROC(참고).
  - 눈으로 확인할 예시: <out>/examples_text.jpg(글씨 찾은 사진, 원본|마스크|지운 것), <out>/examples_nobg.jpg.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from photo_shelter_check import nn_same_shelter
from train_adoption import parse_weight

KINDS = ("crop", "text", "nobg", "text_nobg")


def load_sample(meta, csv_path, photo_dir, n, seed, ids=None):
    """ids를 주면 그 개체들(지난번 표본)을 같은 순서로. meta가 늘어나면 같은 seed라도 표본이 달라지기 때문."""
    m = pd.read_csv(meta, dtype=str, encoding="utf-8-sig")
    m = m.drop_duplicates("desertionNo", keep="last")  # 다시 받은 개체는 마지막 줄이 현재 기록
    if ids is not None:
        m = m.set_index("desertionNo").loc[list(ids)].reset_index()
    else:
        m = m[m.status == "200"]
        m = m[[os.path.exists(os.path.join(photo_dir, "crop", d + ".jpg")) for d in m.desertionNo]]
        m = m.sample(min(n, len(m)), random_state=seed).reset_index(drop=True)
    raw = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig", usecols=["desertionNo", "kindNm", "weight"])
    raw = raw.drop_duplicates("desertionNo", keep="last")
    m = m.merge(raw, on="desertionNo", how="left")
    m["weight_kg"] = m.weight.map(parse_weight)
    return m


def contact_sheet(rows, path, cell=160, cols=None):
    """rows: [[PIL 이미지, ...], ...] → 한 줄에 한 예시를 이어 붙여 저장"""
    from PIL import Image
    if not rows:
        return
    cols = cols or max(len(r) for r in rows)
    sheet = Image.new("RGB", (cell * cols, cell * len(rows)), (255, 255, 255))
    for i, r in enumerate(rows):
        for j, im in enumerate(r):
            im = im.convert("RGB").copy()
            im.thumbnail((cell, cell))
            sheet.paste(im, (j * cell, i * cell))
    sheet.save(path, "JPEG", quality=85)


def make_text(m, src, out, gpu, dilate, quality, n_examples, skip=False):
    import cv2
    import easyocr
    from PIL import Image
    reader = easyocr.Reader(["ko", "en"], gpu=gpu, verbose=False)
    os.makedirs(os.path.join(out, "text"), exist_ok=True)
    recs, examples = [], []
    t0 = time.time()
    for i, d in enumerate(m.desertionNo, 1):
        if skip and os.path.exists(os.path.join(out, "text", d + ".jpg")):  # 전체 적용 이어 하기
            continue
        bgr = cv2.imread(os.path.join(src, d + ".jpg"))
        h, w = bgr.shape[:2]
        # 224px라 3배로 키워서 찾는다. 기본값(2배, 0.7/0.4)은 사진 아래 설명 글의 오른쪽 절반을 놓쳤다(60장 시험).
        horiz, free = reader.detect(bgr, mag_ratio=3.0, text_threshold=0.5, low_text=0.3, min_size=8)
        mask = np.zeros((h, w), np.uint8)
        for x0, x1, y0, y1 in horiz[0]:
            cv2.rectangle(mask, (max(0, int(x0)), max(0, int(y0))), (min(w - 1, int(x1)), min(h - 1, int(y1))), 255, -1)
        for pts in free[0]:
            cv2.fillPoly(mask, [np.array(pts, np.int32)], 255)
        n_box = len(horiz[0]) + len(free[0])
        if n_box:
            mask = cv2.dilate(mask, np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8))
            fixed = cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)
        else:
            fixed = bgr
        cv2.imwrite(os.path.join(out, "text", d + ".jpg"), fixed, [cv2.IMWRITE_JPEG_QUALITY, quality])
        recs.append({"desertionNo": d, "n_text_box": n_box, "text_frac": round(float((mask > 0).mean()), 4)})
        if n_box and len(examples) < n_examples:
            to_pil = lambda a: Image.fromarray(cv2.cvtColor(a, cv2.COLOR_BGR2RGB))
            examples.append([to_pil(bgr), Image.fromarray(mask), to_pil(fixed)])
        if i % 500 == 0:
            print(f"  글씨 {i}/{len(m)} ({(time.time() - t0) / i:.2f}초/장)", flush=True)
    contact_sheet(examples, os.path.join(out, "examples_text.jpg"))
    return pd.DataFrame(recs)


def make_nobg(m, out, pairs, model, quality, n_examples, skip=False):
    """pairs: [(원본 폴더, 저장할 종류)]"""
    from PIL import Image
    from rembg import new_session, remove
    sess = new_session(model)
    recs, examples = {}, []
    for src_dir, dst_kind in pairs:
        os.makedirs(os.path.join(out, dst_kind), exist_ok=True)
        t0 = time.time()
        for i, d in enumerate(m.desertionNo, 1):
            if skip and os.path.exists(os.path.join(out, dst_kind, d + ".jpg")):
                continue
            im = Image.open(os.path.join(src_dir, d + ".jpg")).convert("RGB")
            rgba = remove(im, session=sess)
            bg = Image.new("RGB", im.size, (127, 127, 127))
            bg.paste(rgba, mask=rgba.getchannel("A"))
            bg.save(os.path.join(out, dst_kind, d + ".jpg"), "JPEG", quality=quality)
            if dst_kind == "nobg":
                recs[d] = round(float((np.asarray(rgba.getchannel("A")) > 127).mean()), 3)
                if len(examples) < n_examples:
                    examples.append([im, bg])
            if i % 500 == 0:
                print(f"  누끼({dst_kind}) {i}/{len(m)} ({(time.time() - t0) / i:.2f}초/장)", flush=True)
    # 한 줄에 4쌍(원본|누끼)
    contact_sheet([sum(examples[k:k + 4], []) for k in range(0, len(examples), 4)], os.path.join(out, "examples_nobg.jpg"))
    return pd.Series(recs, name="fg_frac")


def embed(paths, model_name, batch, device):
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    torch.set_grad_enabled(False)
    proc = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    out = []
    for i in range(0, len(paths), batch):
        ims = [Image.open(p).convert("RGB") for p in paths[i:i + batch]]
        inp = {k: v.to(device) for k, v in proc(images=ims, return_tensors="pt").items()}
        out.append(model(**inp).pooler_output.float().cpu().numpy())  # CLS 토큰
    x = np.concatenate(out)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def nn_breed_weight(breed, wkg, g, seed):
    """품종이 같은 다른 개 중 체중(로그)이 가장 가까운 개가 같은 보호소인 비율. 사진 없이 낼 수 있는 기준."""
    rng = np.random.default_rng(seed)
    cnt = pd.Series(g).map(pd.Series(g).value_counts()).to_numpy()
    lw = np.log1p(np.nan_to_num(wkg, nan=-1.0).clip(0))
    hits, chance = [], []
    for _, idx in pd.Series(np.arange(len(g))).groupby(pd.Series(breed).fillna("NA").to_numpy()):
        idx = idx.to_numpy()
        if len(idx) < 2:
            continue
        for i in idx:
            if cnt[i] < 2:
                continue
            o = idx[idx != i]
            diff = np.abs(lw[o] - lw[i]) + rng.random(len(o)) * 1e-9  # 같은 체중이면 무작위
            hits.append(g[o[diff.argmin()]] == g[i])
            chance.append((cnt[i] - 1) / (len(g) - 1))
    return float(np.mean(hits)), float(np.mean(chance)), len(hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="photos_all_meta.csv")
    ap.add_argument("--csv", default="abandoned_dogs.csv")
    ap.add_argument("--dir", default="photos_all")
    ap.add_argument("--out", default="photo_clean")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dilate", type=int, default=3, help="글씨 마스크를 넓힐 픽셀")
    ap.add_argument("--rembg-model", default="u2net")
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--model", default="facebook/dinov2-small")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--min-per", type=int, default=8)
    ap.add_argument("--skip-make", action="store_true")
    ap.add_argument("--all", choices=["text", "nobg", "text_nobg"],
                    help="시험 대신 <dir>/crop 전체에 이 처리를 적용해 <out>/<종류>/에 저장(끊겨도 이어 함)")
    a = ap.parse_args()

    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    gpu = torch.cuda.is_available()
    os.makedirs(a.out, exist_ok=True)
    if a.all:
        crop = os.path.join(a.dir, "crop")
        allm = pd.DataFrame({"desertionNo": sorted(f[:-4] for f in os.listdir(crop) if f.endswith(".jpg"))})
        print(f"전체 적용 {a.all}: {len(allm):,}장 → {a.out}/{a.all}/", flush=True)
        if a.all in ("text", "text_nobg"):
            make_text(allm, crop, a.out, gpu, a.dilate, a.quality, 0, skip=True)
        if a.all != "text":
            src = os.path.join(a.out, "text") if a.all == "text_nobg" else crop
            make_nobg(allm, a.out, [(src, a.all)], a.rembg_model, a.quality, 0, skip=True)
        print("전체 적용 끝", flush=True)
        return
    info_path = os.path.join(a.out, "photo_info.csv")
    reuse = a.skip_make and os.path.exists(info_path)
    info = pd.read_csv(info_path, dtype={"desertionNo": str}) if reuse else None
    m = load_sample(a.meta, a.csv, a.dir, a.n, a.seed, ids=info.desertionNo if reuse else None)
    g = m.careRegNo.fillna(m.careNm).to_numpy()
    y = (m.processState == "종료(입양)").astype(int).to_numpy()
    print(f"표본 {len(m)}장, 보호소 {len(set(g))}곳, 입양 비율 {y.mean():.3f}, GPU {gpu}", flush=True)

    if not reuse:
        os.makedirs(os.path.join(a.out, "crop"), exist_ok=True)
        import shutil
        for d in m.desertionNo:
            shutil.copyfile(os.path.join(a.dir, "crop", d + ".jpg"), os.path.join(a.out, "crop", d + ".jpg"))
        info = make_text(m, os.path.join(a.dir, "crop"), a.out, gpu, a.dilate, a.quality, 24)
        pairs = [(os.path.join(a.out, "crop"), "nobg"), (os.path.join(a.out, "text"), "text_nobg")]
        info["fg_frac"] = info.desertionNo.map(make_nobg(m, a.out, pairs, a.rembg_model, a.quality, 24))
        info.to_csv(info_path, index=False, encoding="utf-8-sig")
    m = m.merge(info, on="desertionNo", how="left")

    # 글씨: 얼마나 있고, 보호소에 몰려 있는가(해석용)
    has = m.n_text_box.fillna(0) > 0
    by_sh = pd.DataFrame({"g": g, "has": has}).groupby("g").has.agg(["mean", "size"])
    by_sh = by_sh[by_sh["size"] >= a.min_per]
    print(f"\n글씨 있음 {has.mean():.3f} ({has.sum()}장), 글씨 면적 중앙값(있는 사진) {m.text_frac[has].median():.3f}")
    print(f"  입양 비율: 글씨 있음 {y[has].mean():.3f} / 없음 {y[~has].mean():.3f}")
    if len(by_sh):
        print(f"  보호소별 글씨 비율({len(by_sh)}곳, 사진 {a.min_per}장 이상) 10/50/90%:",
              np.round(by_sh["mean"].quantile([.1, .5, .9]).to_numpy(), 3).tolist())
    print(f"누끼 전경 면적 5/50/95%: {np.round(m.fg_frac.quantile([.05, .5, .95]).to_numpy(), 3).tolist()}")

    hit, chance, n_bw = nn_breed_weight(m.kindNm.to_numpy(), m.weight_kg.to_numpy(), g, a.seed)
    rows = [{"사진": "(사진 없이) 품종+체중", "이웃 같은 보호소": round(hit, 3), "우연 기준": round(chance, 3), "센 수": n_bw}]
    vc = pd.Series(g).value_counts()
    keep = np.isin(g, vc[vc >= max(a.min_per, 5)].index)  # 5-fold라 보호소당 5장 이상
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    for kind in KINDS:
        cache = os.path.join(a.out, f"emb_{kind}.npz")
        if os.path.exists(cache) and list(np.load(cache)["ids"]) == list(m.desertionNo):
            x = np.load(cache)["x"]
        else:
            x = embed([os.path.join(a.out, kind, d + ".jpg") for d in m.desertionNo], a.model, a.batch,
                      "cuda" if gpu else "cpu")
            np.savez(cache, x=x, ids=m.desertionNo.to_numpy().astype(str))  # object 배열이면 다시 못 읽음
        hit, chance, n = nn_same_shelter(x, g)
        row = {"사진": kind, "이웃 같은 보호소": round(hit, 3), "우연 기준": round(chance, 3), "센 수": n}
        gk = g[keep]
        if len(set(gk)) >= 2:
            pred = cross_val_predict(LogisticRegression(C=1.0, max_iter=3000), x[keep], gk, cv=cv)
            row.update({"보호소 분류 정확도": round((pred == gk).mean(), 3),
                        "최빈 보호소 비율": round(pd.Series(gk).value_counts(normalize=True).iloc[0], 3),
                        "분류 보호소 수": len(set(gk)), "분류 사진 수": int(keep.sum())})
        p = cross_val_predict(LogisticRegression(C=0.1, max_iter=3000), x, y, cv=cv, method="predict_proba")[:, 1]
        row["입양 ROC-AUC(참고)"] = round(roc_auc_score(y, p), 3)
        rows.append(row)
    res = pd.DataFrame(rows)
    print("\n" + res.to_string(index=False))
    res.to_csv(os.path.join(a.out, "clean_check.csv"), index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
