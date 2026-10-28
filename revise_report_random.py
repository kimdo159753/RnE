#!/usr/bin/env python3
"""보고서(hwpx)를 무작위 분할 재학습 결과 + SVM 비교로 고친다 (2026-10-28 사용자 요청).

  python revise_report_random.py --source "<원본>.hwpx" --numbers random_numbers.json --out "<새 이름>.hwpx"

원본에서 내용을 뽑고(rebuild_report_hwpx.extract) 문단·표·그림·참고문헌을 바꾼 뒤 학교 양식으로 다시 조립한다.
숫자는 --numbers(JSON)에서 읽어 문장에 넣는다. 그림 3(연구 절차)·4(백분위)는 report_fig_random/의 새 그림으로 바꾼다.
문단을 찾을 때는 원본 문장의 앞부분으로 찾고, 못 찾으면 멈춘다(원본이 바뀌었으면 고쳐야 함).
"""
import argparse
import json
import os
import shutil
import tempfile

import rebuild_report_hwpx as rb

TEMPLATE = "2026 R&E 프로그램 보고서_(팀코드).hwpx"


def find(body, kind, start):
    hits = [i for i, b in enumerate(body) if b[0] == kind and len(b) > 1 and str(b[1]).startswith(start)]
    assert len(hits) == 1, f"{kind} '{start}' 문단 {len(hits)}개"
    return hits[0]


def set_p(body, start, text, kind="P"):
    body[find(body, kind, start)] = [kind, text]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--numbers", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    N = json.load(open(args.numbers, encoding="utf-8"))
    img_dir = tempfile.mkdtemp(prefix="hwpx_imgs_")
    c = rb.extract(args.source, img_dir)
    body = [list(b) for b in c["body"]]

    # ── 요약·키워드
    c["summary"] = (
        "본 연구는 공간과 인력이 한정된 보호소가 어떤 유기견을 먼저 지원할지 결정하도록 돕는 것을 목표로 하며, 이를 위한 구체적인 목적은 세 가지이다. "
        "첫째, 유기견이 보호소에 들어온 시점에 알 수 있는 정보만으로 입양 가능성을 예측하는 모델을 만들고 여러 기계학습 모델을 공정한 조건에서 비교한다. "
        "둘째, 보호소 직원이 자유롭게 작성하는 비정형 텍스트인 특징 메모를 예측에 활용하는 방법을 찾고 그 효과를 검증한다. "
        "셋째, 메모에 담긴 성격, 건강, 관리 흔적 등의 특징이 다른 조건을 통제한 뒤에도 입양과 어떤 관련을 갖는지 분석한다. "
        "국가동물보호정보시스템 공공데이터에서 2021~2026년 개 433,422마리의 기록을 수집하고, 2025년까지 접수되어 보호가 끝난 320,426마리의 입양 여부를 예측하였다. "
        "결과가 덧붙은 메모 문구를 구절 단위로 지워 정보 누수를 막고, 자료를 무작위로 섞어 학습·검증·시험 구간을 나누었다. "
        "로지스틱 회귀, 서포트 벡터 머신(SVM), LightGBM, XGBoost, CatBoost, 다층 퍼셉트론 여섯 계열을 같은 조건에서 비교한 결과 "
        "CatBoost와 LightGBM이 가장 우수하였고 둘의 차이는 불확실하여, 범주형 변수 처리에 강점이 있는 CatBoost를 최종 모델로 선택하였다. "
        f"메모와 정형 정보를 쓰는 기본 모델의 시험 구간 정확도는 {N['base_acc']}로, 모든 개를 비입양으로 판정할 때({N['base_null']})보다 {N['base_gain']} 높았고, "
        f"{N['memo_sum']} "
        f"공고 사진의 이미지 임베딩을 더한 최종 모델은 사진이 있는 개체에서 정확도 {N['ph_acc']}(기준선 {N['ph_null']})로 같은 개체의 사진 없는 모델({N['ph0_acc']})보다 높았으며, "
        f"예측 상위 10%의 실제 입양률은 {N['ph_top']}, 하위 10%는 {N['ph_bot']}였다. "
        "메모 항목별로 로지스틱 회귀를 적용한 결과, 다른 조건과 보호소를 통제하였을 때 감염병 양성(-11.3%p), 공격성(-11.0%p), 쇠약(-9.9%p), 외상·거동 문제(-9.4%p), "
        "겁·경계(-8.2%p) 메모는 낮은 입양 가능성과, 미용·옷 같은 관리 흔적(+6.0%p)과 온순(+5.8%p) 메모는 높은 입양 가능성과 관련되었다. "
        "다만 무작위 분할에서는 같은 날·같은 보호소의 개체가 학습과 시험에 함께 들어가므로, 성능은 앞으로 들어올 새 개체에 대해 다소 낙관적인 값이다. "
        "이 결과는 보호소가 정한 지원 정책에 따라 접수 단계에서 지원 대상을 고르는 데 활용될 수 있다.")
    c["keywords"] = c["keywords"].replace("기계학습, CatBoost", "기계학습, CatBoost, 서포트 벡터 머신")

    # ── Ⅱ-2 비교 모델에 SVM 추가
    set_p(body, "2. 비교 모델", "2. 비교 모델: 로지스틱 회귀, 서포트 벡터 머신, 다층 퍼셉트론", kind="S")
    i = find(body, "P", "다층 퍼셉트론은 여러 층의")
    body[i] = ["P", body[i][1].replace("다섯 계열", "여섯 계열")]
    body.insert(i, ["P",
        "서포트 벡터 머신(SVM)[17]은 입양 개체와 비입양 개체를 가르는 경계 가운데, 경계에서 양쪽의 가장 가까운 개체까지의 거리(마진)가 가장 큰 경계를 찾는 모델이다. "
        "경계를 정하는 데에는 경계 가까이의 개체(서포트 벡터)만 영향을 주므로 멀리 떨어진 이상값에 덜 흔들린다. "
        "커널 함수를 쓰면 곡선 경계도 만들 수 있지만 계산량이 개체 수의 제곱 이상으로 늘어 20만 마리가 넘는 본 자료에는 쓰기 어려워, "
        "본 연구는 직선(초평면) 경계를 쓰는 선형 SVM을 사용하였다. SVM의 출력은 확률이 아니라 경계로부터의 거리이므로, "
        "Platt 보정[6]으로 확률로 바꾸어 다른 모델과 같은 방식으로 평가하였다."])

    i = find(body, "P", "본 연구는 결과를 정확도(ACC) 중심으로")
    body[i] = ["P", body[i][1].replace("정확도가 약 66%가 나오므로", "정확도가 약 64%가 나오므로")]
    assert "약 64%" in body[i][1]

    # ── Ⅱ-5 누수와 분할
    set_p(body, "5. 데이터 누수와 시간순 평가", "5. 데이터 누수와 평가 자료의 분할", kind="S")
    set_p(body, "따라서 입력 변수가 언제 기록되는지",
        "따라서 입력 변수가 언제 기록되는지 확인하고, 학습·시험 구간을 나눈 뒤 평균이나 주성분 같은 통계는 학습 구간으로만 구해야 한다. "
        "자료를 무작위로 섞어 나누면 모든 시기가 각 구간에 고르게 들어가 시기에 따른 치우침이 없지만, 같은 날 같은 보호소에 들어온 개들이 학습과 시험 구간에 함께 들어가 "
        "그 개들의 결과가 서로의 예측에 간접적으로 도움을 줄 수 있다. 이 경우 평가 점수는 앞으로 들어올 새 개체를 예측할 때보다 다소 높게 나올 수 있다. "
        "또한 모델과 설정을 고르는 일은 검증 구간에서만 하고, 시험 구간은 최종 성능을 보고할 때 한 번만 사용해야 시험 구간이 선택 과정에 간접적으로 쓰이는 일을 막을 수 있다.")

    # ── Ⅲ-2 정답 정의와 무작위 분할
    set_p(body, "2. 정답 정의와 시간순 분할", "2. 정답 정의와 무작위 분할", kind="S")
    set_p(body, "정답은 보호가 끝난 개 중 입양을 1",
        "정답은 보호가 끝난 개 중 입양을 1, 안락사·자연사·기증을 0으로 두었다. 기증은 개인이나 단체로 소유권이 넘어간 경우로, 국가동물보호정보시스템에서는 "
        "그 뒤 실제로 입양되었는지를 확인할 수 없어 보호소를 통한 입양과 구분하였다. 전체 자료를 무작위로 섞어 학습 68%, 검증 12%, 시험 20%로 나누었다(표 2). "
        "접수일 순서로 나누면 시기마다 입양률이 달라 구간 사이에 치우침이 생기므로(예: 2024년 3~10월 접수분의 입양률 31.2%, 그 앞 시기 36.6%), "
        f"모든 시기가 고르게 섞이도록 무작위 분할을 택하였고, 그 결과 세 구간의 입양 비율은 {N['split_rates']}로 고르게 되었다. "
        "모델의 종류와 설정을 고르는 모든 결정은 검증 구간에서만 하였고, 시험 구간은 최종 성능을 보고할 때 한 번만 사용하였다. "
        "다만 이 분할에서는 시험 개체의 77.8%가 같은 날·같은 보호소의 개체를 학습 구간에 두고 있어, 그 개체들의 입양 비율만으로도 ROC-AUC 0.721의 예측이 가능하였다. "
        f"접수일 순서로 나누어 평가한 사전 실험(보호소 포함 기본 모델 시험 정확도 73.2%, ROC-AUC 0.766)보다 본 연구의 성능({N['base_acc']}, {N['base_roc']})이 높게 나온 데에는 "
        "이 같은 시기 정보가 일부 작용하였을 수 있으므로, 이후 제시하는 성능은 앞으로 들어올 새 개체에 대해서는 다소 낙관적인 값으로 보아야 한다.")
    cap, rel, left, rows = c["tables"]["t2"]
    c["tables"]["t2"] = (cap.replace("학습·검증·시험 구간", "학습·검증·시험 구간 (무작위 분할)"), rel, left,
                         [["구간", "접수 기간", "개체 수", "입양 비율"]] + N["t2_rows"])

    # ── Ⅲ-4 입력 변수와 모델 선택
    i = find(body, "P", "입력 변수는 범주형 6개")
    body[i] = ["P", body[i][1].replace(
        "다른 하나는 768차원 문장 임베딩 벡터를 학습 구간에서 구한 주성분으로 32차원 벡터로 줄인 것이다.",
        "다른 하나는 768차원 문장 임베딩 벡터를 학습 구간에서 구한 주성분으로 32차원 벡터로 줄인 것이다. "
        "CatBoost에는 여기에 더해 형태소로 나눈 메모를 그대로 읽는 자체 텍스트 입력 기능을 쓸지도 설정 후보로 두었다.")]
    set_p(body, "모델은 로지스틱 회귀, LightGBM",
        "모델은 로지스틱 회귀, 선형 SVM, LightGBM, XGBoost, CatBoost, 다층 퍼셉트론(MLP) 여섯 계열을 비교하였다. "
        "계열마다 설정 8개(로지스틱 회귀와 SVM은 규제 강도 4개)를 무작위로 시도해 검증 구간에서 가장 좋은 설정을 고른 뒤, 계열 간 비교와 최종 모델 선택은 검증 정확도로 하였다. "
        "1등이 우연이 아닌지는 보호소 단위로 표본을 다시 뽑는 클러스터 부트스트랩 1,000회로 성능 차이의 95% 신뢰구간을 구해 판단하였다. "
        "사전 점검으로 아무 정보가 없는 무작위 임베딩을 넣어 보았을 때, 검증 점수만으로는 무작위 임베딩이 1등으로 뽑혔지만 이 신뢰구간 기준으로는 걸러졌다. "
        "트리 모델에는 검증 구간 성능이 100번 연속 나아지지 않으면 학습을 멈추는 조기 종료를 적용하였고, 앞선 실험에서 트리 수 상한(800개)에 걸렸던 문제를 피하기 위해 "
        "상한을 5,000개로 늘렸다. 전체 과정은 그림 3과 같다.")

    # ── Ⅳ-1 모델 비교 + 표 3
    set_p(body, "표 3은 다섯 계열의", N["t3_text"])
    cap, rel, left, rows = c["tables"]["t3"]
    c["tables"]["t3"] = (cap, rel, left, [rows[0]] + N["t3_rows"])

    # ── Ⅳ-2 기본 모델
    set_p(body, "기본 모델은 범주형 6개", N["base_text"])
    c["tables"]["t4"] = (c["tables"]["t4"][0], c["tables"]["t4"][1], c["tables"]["t4"][2], [["지표", "값"]] + N["t4_rows"])
    set_p(body, "그림 4는 시험 구간 64,085마리", N["fig4_text"])
    i = find(body, "P", "정확도는 기준값 0.5에서")
    body[i] = ["P", body[i][1].replace("재현율이 39.8%에 그친 것은", f"재현율이 {N['base_rec']}에 머문 것은")]
    set_p(body, "메모의 기여는 메모에서 만든 입력을", N["memo_text"])

    # ── Ⅳ-3 메모만 바꾼 예측(표 6)
    set_p(body, "둘째, 같은 개(경기도에서", N["t6_text"])
    c["tables"]["t6"] = (c["tables"]["t6"][0], c["tables"]["t6"][1], c["tables"]["t6"][2], [["메모", "예측 입양 확률"]] + N["t6_rows"])

    # ── Ⅳ-5 사진
    set_p(body, "사진은 사전 학습된 이미지 모델 세 가지", N["photo_text1"])
    c["tables"]["t7"] = (c["tables"]["t7"][0], c["tables"]["t7"][1], c["tables"]["t7"][2], [["입력", "검증 ROC-AUC", "시험 ROC-AUC"]] + N["t7_rows"])
    set_p(body, "세 모델 모두 성능을 높였고", N["photo_text2"])
    c["tables"]["t8"] = (c["tables"]["t8"][0].replace("57,210", "44,249"), c["tables"]["t8"][1], c["tables"]["t8"][2],
                         [["지표", "사진 없음", "사진 포함(최종 모델)"]] + N["t8_rows"])
    set_p(body, "최종 모델의 정확도는 74.7%", N["photo_text3"])

    # ── Ⅴ 결론
    set_p(body, "본 연구는 공간과 인력이 한정된 보호소가 어떤 유기견을 먼저 지원할지 결정하도록 돕기 위해", N["concl1"])
    set_p(body, "이 결과는 보호소가 접수 단계에서", N["concl2"])
    set_p(body, "후속 연구로는", N["concl4"])

    # ── 그림 3·4 교체, 참고문헌 [17]
    for key, fn in (("f3", "fig1_flow.png"), ("f4", "fig_percentile_shelter.png")):
        name, w, cap = c["figs"][key]
        new = key + "_random.png"
        shutil.copy(os.path.join("report_fig_random", fn), os.path.join(img_dir, new))
        c["figs"][key] = (new, w, cap)
    c["refs"] = [r for r in c["refs"] if not r.startswith("[17]")] + [
        "[17] Cortes, C., & Vapnik, V.(1995). Support-vector networks. Machine Learning, 20(3), 273-297."]

    c["body"] = [tuple(b) for b in body]
    rb.build(c, img_dir, TEMPLATE, args.out)


if __name__ == "__main__":
    main()
