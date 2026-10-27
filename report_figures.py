#!/usr/bin/env python3
"""R&E 보고서용 그림 3개 (흑백 인쇄에서도 읽히도록 회색 + 파랑 한 가지)

  python report_figures.py --out report_fig
  fig1_flow.png    연구 절차 흐름도
  fig_eda.png      체중 구간별·품종별 입양률
  fig2_decile.png  예측 확률 10분위별 실제 입양률 (최종 모델, 보호소 제외, 보정 후)
  fig_percentile_shelter.png  예측 확률 백분위별 실제 입양률 (최종 모델, 보호소 포함, 보정 후) — 보고서 그림 3
  fig3_effects.png 메모 항목별 입양 오즈비와 95% 구간 (memo_check/memo_effects*.csv)
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

FONT = r"C:\Windows\Fonts\Hancom Gothic Regular.ttf"
font_manager.fontManager.addfont(FONT)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False
BLUE, GREY, DARK = "#2a78d6", "#9a9a9a", "#333333"


def fig_flow(path):
    steps = [("데이터 수집", "공공데이터 API\n433,422마리"),
             ("정제·정답 정의", "2021~2025 접수\n320,426마리"),
             ("메모 누수 제거", "결과 문구 구절 삭제\n10,693건"),
             ("변수 만들기", "범주·숫자 12개\n+ 메모 33개"),
             ("모델 비교", "5개 모델 계열\n검증 구간 선택"),
             ("최종 평가·해석", "시험 구간 1회\n메모 항목 효과")]
    fig, ax = plt.subplots(figsize=(9.0, 2.0), dpi=200)
    ax.set_xlim(0, 6); ax.set_ylim(0, 1); ax.axis("off")
    for i, (h, b) in enumerate(steps):
        x = i + 0.06
        ax.add_patch(FancyBboxPatch((x, 0.12), 0.82, 0.76, boxstyle="round,pad=0.01,rounding_size=0.04",
                                    fc="#eef4fb" if i in (2, 4) else "#f4f4f4", ec=BLUE if i in (2, 4) else GREY, lw=1))
        ax.text(x + 0.41, 0.66, h, ha="center", va="center", fontsize=10, color=DARK, weight="bold")
        ax.text(x + 0.41, 0.36, b, ha="center", va="center", fontsize=8, color=DARK, linespacing=1.3)
        if i < 5:
            ax.annotate("", xy=(i + 1.05, 0.5), xytext=(x + 0.86, 0.5), arrowprops=dict(arrowstyle="->", color=GREY, lw=1))
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_eda(path):
    """체중 구간별·주요 품종별 입양률 (2021~2025 접수, 보호 종료 개체)"""
    from train_adoption import prepare
    raw = pd.read_csv("abandoned_dogs.csv", dtype=str, encoding="utf-8-sig")
    df = prepare(raw, include_returned=False, max_date=pd.to_datetime("20251231"))
    base = df["y"].mean() * 100
    w = pd.cut(df["weight_kg"], [0, 3, 5, 10, 20, 200], labels=["3kg 이하", "3~5kg", "5~10kg", "10~20kg", "20kg 초과"])
    gw = df.groupby(w, observed=True)["y"].agg(["mean", "size"])
    b = df.groupby("breed")["y"].agg(["mean", "size"])
    b = b[b["size"] >= 4000].sort_values("mean")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.9), dpi=200, gridspec_kw={"width_ratios": [1.15, 1]})
    for ax, g, title in [(axes[0], gw, "체중 구간별"), (axes[1], b, "품종별 (4,000마리 이상)")]:
        x = np.arange(len(g))
        ax.bar(x, g["mean"] * 100, color="#c9d9ee", width=0.62)
        for i, (m, n) in enumerate(zip(g["mean"], g["size"])):
            ax.text(i, m * 100 + 1.5, f"{m * 100:.1f}%", ha="center", fontsize=7.5, color=DARK)
        ax.axhline(base, color=GREY, ls="--", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(g.index, fontsize=7.5)
        ax.set_ylim(0, 80); ax.tick_params(labelsize=7.5)
        ax.set_title(title, fontsize=9, color=DARK)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    axes[0].set_ylabel("입양률 (%)", fontsize=8)
    axes[1].text(-0.45, base + 2.5, f"전체 {base:.1f}%", fontsize=7.5, color="#555")  # 믹스견 막대(30%) 위 빈 곳
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_decile(path):
    t = pd.read_csv("final_model/pred_test_without_shelter.csv")
    t["bin"] = pd.qcut(t["p_cal"], 10, labels=False)
    g = t.groupby("bin").agg(pred=("p_cal", "mean"), real=("y", "mean"))
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
    x = np.arange(1, 11)
    ax.bar(x, g["real"] * 100, color=[BLUE if i in (0, 9) else "#c9d9ee" for i in range(10)], width=0.65, label="실제 입양률")
    ax.plot(x, g["pred"] * 100, "o-", color=DARK, ms=3.5, lw=1, label="예측 확률 평균")
    ax.axhline(t["y"].mean() * 100, color=GREY, ls="--", lw=0.8)
    ax.text(0.45, t["y"].mean() * 100 + 1.5, f"전체 {t['y'].mean() * 100:.1f}%", ha="left", fontsize=8, color="#555")
    for i in (0, 9):
        ax.text(x[i], g["real"].iloc[i] * 100 + 2, f"{g['real'].iloc[i] * 100:.1f}%", ha="center", fontsize=8, color=DARK)
    ax.set_xticks(x); ax.set_xticklabels([f"{i}" for i in x], fontsize=8)
    ax.set_xlabel("예측 확률 10분위 (1 = 가장 낮음)", fontsize=9); ax.set_ylabel("%", fontsize=9)
    ax.set_ylim(0, 85); ax.tick_params(labelsize=8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.98))
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_percentile(path):
    """예측 확률 백분위(1~100)별 실제 입양률 (최종 모델, 보호소 포함, 보정 후)"""
    t = pd.read_csv("final_model/pred_test_with_shelter.csv")
    t["bin"] = pd.qcut(t["p_cal"].rank(method="first"), 100, labels=False)
    g = t.groupby("bin").agg(pred=("p_cal", "mean"), real=("y", "mean"))
    fig, ax = plt.subplots(figsize=(6.4, 3.3), dpi=200)
    x = np.arange(1, 101)
    ax.bar(x, g["real"] * 100, color="#a8c4ea", width=0.85, label="실제 입양률")
    ax.plot(x, g["pred"] * 100, "-", color=DARK, lw=1.2, label="예측 확률 평균")
    ax.axhline(t["y"].mean() * 100, color=GREY, ls="--", lw=0.8)
    ax.text(1, t["y"].mean() * 100 + 1.5, f"전체 {t['y'].mean() * 100:.1f}%", ha="left", fontsize=8, color="#555")
    ax.set_xlim(0, 101); ax.set_ylim(0, 100)
    ax.set_xticks([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.set_xlabel("예측 확률 백분위 (1 = 가장 낮음, 100 = 가장 높음)", fontsize=9); ax.set_ylabel("실제 입양률 (%)", fontsize=9)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


NAMES = {"성격_공격": "공격성", "건강_감염병": "감염병 양성", "건강_쇠약": "쇠약", "건강_외상거동": "외상·거동",
         "성격_겁경계": "겁·경계", "건강_피부": "피부 질환", "출신_떠돌이": "떠돌이·포획", "출신_농장": "농장 출신",
         "서식_양식": "보호소 기록 양식", "나이_새끼": "새끼 표현", "동반_무리": "동반 입소", "출신_주인언급": "주인 언급",
         "관리_목줄": "목줄", "관리_칩있음": "칩 있음", "성격_활발": "활발", "외모_소형": "소형",
         "외모_칭찬": "외모 칭찬", "성격_온순": "온순", "관리_미용옷": "미용·옷"}


def fig_effects(path):
    a = pd.read_csv("memo_check/memo_effects.csv", encoding="utf-8-sig")
    b = pd.read_csv("memo_check/memo_effects_model2.csv", encoding="utf-8-sig").set_index("항목")
    # 보호소 기록 양식은 회귀에는 통제 변수로 남기되 그림에서는 뺀다(개의 특징이 아니라 보호소 표시, 본문에서 따로 설명, 2026-10-23 팀 의견)
    # 등장 2,000건 미만(농장 출신·칩·소형)도 신뢰구간이 너무 넓어 그림에서 뺀다(회귀에는 그대로, 2026-10-23)
    a = a[(a["항목"] != "서식_양식") & (a["등장"] >= 2000)]
    a["lo2"], a["hi2"] = a["항목"].map(b["오즈비_하한"]), a["항목"].map(b["오즈비_상한"])
    a = a.sort_values("오즈비_보호소통제", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.6, 5.2), dpi=200)
    y = np.arange(len(a))
    sig = (a["lo2"] > 1) | (a["hi2"] < 1)
    for i, r in a.iterrows():
        c = BLUE if sig[i] else GREY
        ax.plot([r["lo2"], r["hi2"]], [i, i], color=c, lw=1.6)
        ax.plot(r["오즈비_보호소통제"], i, "o", color=c, ms=5)
        ax.plot(r["오즈비"], i, "o", mfc="white", mec="#555", ms=4, mew=0.8)
        ax.text(2.05, i, f"{r['효과(%p)_보호소통제']:+.1f}".replace("-0.0", "0.0").replace("+0.0", "0.0"),
                va="center", ha="right", fontsize=7.5, color=DARK if sig[i] else "#777")
    ax.axvline(1, color="#666", lw=0.8)
    ax.set_xscale("log"); ax.set_xlim(0.34, 2.1)
    ax.set_xticks([0.4, 0.5, 0.7, 1, 1.4]); ax.set_xticklabels(["0.4", "0.5", "0.7", "1", "1.4"], fontsize=8)
    ax.minorticks_off()
    ax.set_yticks(y); ax.set_yticklabels([NAMES[k] for k in a["항목"]], fontsize=8)
    ax.text(2.05, -1.2, "%p", ha="right", fontsize=7.5, color="#555")
    ax.set_ylim(len(a) - 0.4, -1.6)
    ax.set_xlabel("입양 오즈비 (로그 축, 1보다 작으면 입양에 불리)", fontsize=8.5)
    ax.plot([], [], "o", color=BLUE, label="보호소 통제 (선: 95% 신뢰구간)")
    ax.plot([], [], "o", mfc="white", mec="#555", label="보호소 미통제")
    ax.plot([], [], "o", color=GREY, label="신뢰구간이 1을 포함")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="report_fig")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig_flow(os.path.join(args.out, "fig1_flow.png"))
    fig_eda(os.path.join(args.out, "fig_eda.png"))
    fig_decile(os.path.join(args.out, "fig2_decile.png"))
    fig_percentile(os.path.join(args.out, "fig_percentile_shelter.png"))
    fig_effects(os.path.join(args.out, "fig3_effects.png"))
    print("saved", os.listdir(args.out))


if __name__ == "__main__":
    main()
