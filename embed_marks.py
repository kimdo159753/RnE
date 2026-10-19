#!/usr/bin/env python3
"""특징(specialMark) 문장 임베딩 계산 → 캐시 저장

prepare()에서 누수 표현을 지운 mark를 쓰고, 같은 문장은 한 번만 계산한다.
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install sentence-transformers
  python embed_marks.py --csv abandoned_dogs.csv --model jhgan/ko-sroberta-multitask --out emb_cache/ko_sroberta.npz
  python embed_marks.py --csv abandoned_dogs.csv --model intfloat/multilingual-e5-base --prefix "query: " --out emb_cache/e5_base.npz
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from train_adoption import prepare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--max-date", default="20251231")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prefix", default="", help='e5 계열은 "query: "')
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--limit", type=int, help="속도 측정용: 앞의 N개 문장만")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    raw = pd.read_csv(args.csv, dtype=str, encoding="utf-8-sig")
    df = prepare(raw, include_returned=False, max_date=pd.to_datetime(args.max_date, format="%Y%m%d"))
    texts = pd.unique(df["mark"])
    if args.limit:
        texts = texts[: args.limit]
    print(f"\n고유 문장 {len(texts)}개 임베딩 ({args.model})", flush=True)

    model = SentenceTransformer(args.model, device="cpu")
    # 긴 문장 순으로 묶어 패딩 낭비를 줄인다
    order = np.argsort([len(t) for t in texts])
    t0 = time.time()
    emb = model.encode([args.prefix + (texts[i] or "없음") for i in order], batch_size=args.batch,
                       normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    out = np.empty_like(emb)
    out[order] = emb
    dt = time.time() - t0
    print(f"완료: {dt:.0f}초 ({len(texts) / dt:.0f}문장/초), 차원 {out.shape[1]}")
    np.savez_compressed(args.out, texts=np.array(texts, dtype=object), emb=out.astype(np.float16),
                        model=args.model, prefix=args.prefix)
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
