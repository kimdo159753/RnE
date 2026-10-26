#!/usr/bin/env python3
"""메모 규칙 분류기(memo_categories) 채점: memo_check/eval_sample.csv 200개를 gold_eval_sample.txt 정답과 비교.

정답은 14개 항목만 판정했다(코드 → 항목은 gold 파일 머리말). 지표: 전체(마이크로) 정밀도·재현율, 항목별 정밀도·재현율,
메모 단위 완전 일치 수.

  python memo_eval.py                                     # 1차 표본(seed 11, 규칙 수정 전에 판정)
  python memo_eval.py --sample memo_check/eval_sample2.csv --gold memo_check/gold_eval_sample2.txt   # 2차(seed 23, 수정 후 새 표본)
"""
import argparse
import re

import pandas as pd

from memo_categories import classify

CODES = {"G": "성격_온순", "V": "성격_활발", "F": "성격_겁경계", "A": "성격_공격", "I": "건강_감염병", "S": "건강_피부",
         "J": "건강_외상거동", "W": "건강_쇠약", "M": "관리_미용옷", "C": "관리_목줄", "R": "출신_떠돌이", "P": "외모_칭찬",
         "Y": "나이_새끼", "B": "동반_무리"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="memo_check/eval_sample.csv")
    ap.add_argument("--gold", default="memo_check/gold_eval_sample.txt")
    ap.add_argument("--show-errors", action="store_true", help="틀린 메모를 정답·예측과 함께 출력")
    args = ap.parse_args()
    s = pd.read_csv(args.sample, encoding="utf-8-sig", dtype=str)
    gold = {}
    for line in open(args.gold, encoding="utf-8"):
        m = re.match(r"^(\d+):(.*)$", line.strip())
        if m:
            gold[int(m.group(1))] = {CODES[c] for c in m.group(2).split()}
    assert len(gold) == len(s), (len(gold), len(s))
    rows, exact = [], 0
    for i, text in enumerate(s["mark"].fillna("")):
        pred = {k for k, v in classify(text).items() if v and k in CODES.values()}
        exact += pred == gold[i]
        if args.show_errors and pred != gold[i]:
            print(f"{i}: 놓침 {sorted(gold[i] - pred)} 잘못 {sorted(pred - gold[i])} | {text}")
        for k in CODES.values():
            rows.append((k, k in pred, k in gold[i]))
    df = pd.DataFrame(rows, columns=["항목", "예측", "정답"])
    tp, fp, fn = (df.예측 & df.정답).sum(), (df.예측 & ~df.정답).sum(), (~df.예측 & df.정답).sum()
    print(f"정밀도 {tp / (tp + fp):.3f}  재현율 {tp / (tp + fn):.3f}  (TP {tp}, FP {fp}, FN {fn})  완전 일치 {exact}/{len(s)}")
    g = df.groupby("항목").apply(lambda d: pd.Series({
        "정답수": int(d.정답.sum()), "정밀도": (d.예측 & d.정답).sum() / max(1, d.예측.sum()),
        "재현율": (d.예측 & d.정답).sum() / max(1, d.정답.sum())}), include_groups=False)
    print(g.round(3).to_string())


if __name__ == "__main__":
    main()
