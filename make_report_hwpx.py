#!/usr/bin/env python3
"""R&E 보고서 양식(.hwpx)에 연구 내용을 채워 새 .hwpx를 만든다.

양식의 문단·표·그림 XML을 그대로 복제하고 글자만 바꾼다(글꼴·크기·문단 모양·표 테두리는 양식 그대로).
  - 첫 쪽(작성 요령)은 지운다. 머리말·쪽 번호·구역 설정은 남긴다.
  - 팀코드·저자·지도교사는 양식의 자리표시를 그대로 둔다(사용자가 채움).

  python make_report_hwpx.py --template "2026 R&E 프로그램 보고서_(팀코드).hwpx" --figs report_fig \
      --out "2026 R&E 프로그램 보고서_(팀코드)_작성본.hwpx"
"""
import argparse
import random
import re
import zipfile
from xml.sax.saxutils import escape

from PIL import Image

MM = 283.465  # HWPUNIT per mm
TEXT_W = 49604

# ---------------------------------------------------------------- 내용
TITLE = "보호소 접수 정보와 특징 메모를 이용한 유기견 입양 가능성 예측"
SUMMARY = (
    "본 연구는 유기견이 보호소에 들어온 시점에 알 수 있는 정보만으로 입양 가능성을 예측하는 모델을 만들고, "
    "보호소 직원이 작성하는 특징 메모가 입양과 어떤 관련이 있는지 분석하였다. 국가동물보호정보시스템 공공데이터에서 "
    "2021~2026년 개 433,422마리의 기록을 수집하고, 2025년까지 접수되어 보호가 끝난 320,426마리의 입양 여부를 예측하였다. "
    "결과가 덧붙은 메모 문구를 구절 단위로 지워 정보 누수를 막고, 접수일 순서로 학습·검증·시험 구간을 나누었다. "
    "로지스틱 회귀, LightGBM, XGBoost, CatBoost, 다층 퍼셉트론을 같은 조건에서 비교한 결과 CatBoost가 가장 우수하였으며, "
    "시험 구간에서 ROC-AUC 0.740, PR-AUC 0.596을 보였다. 예측 상위 10%의 실제 입양률은 72.3%, 하위 10%는 8.0%였다. "
    "특징 메모는 모델 중요도의 약 58%를 차지하였고, 다른 조건과 보호소를 통제하였을 때 공격성(-11.0%p), 쇠약(-9.9%p), "
    "외상·거동 문제(-9.4%p), 겁·경계(-8.2%p) 메모는 낮은 입양 가능성과, 미용·옷 같은 관리 흔적(+6.0%p)과 온순(+5.8%p) "
    "메모는 높은 입양 가능성과 관련되었다. 이 결과는 입양이 어려운 개를 접수 단계에서 찾아 지원을 집중하는 데 활용될 수 있다."
)
KEYWORDS = "키워드 : 유기견 입양, 기계학습, CatBoost, 텍스트 임베딩, 데이터 누수"

# 본문 블록: ("H", 장 제목) ("S", 절 제목) ("P", 문단) ("T", 표 이름) ("F", 그림 이름) ("GAP",)
BODY = [
    ("H", "Ⅰ. 연구의 필요성 및 목적"),
    ("P", "보호소에 들어온 유기견의 상당수는 입양되지 못한다. 본 연구가 수집한 2021~2026년 기록에서 보호가 끝난 개 가운데 "
          "입양은 125,455마리였지만 안락사 119,967마리, 자연사 83,175마리로 입양되지 못하고 생을 마친 개가 더 많았다. "
          "보호소의 공간과 인력은 한정되어 있으므로, 입양이 어려울 개를 접수 단계에서 미리 알 수 있다면 홍보, 치료, "
          "임시보호 연계 같은 지원을 그 개에게 먼저 집중할 수 있다."),
    ("P", "국가동물보호정보시스템은 구조된 동물의 품종, 나이, 체중 같은 정형 정보와 함께 보호소 직원이 자유롭게 적은 특징 "
          "메모를 공개한다. 메모에는 성격, 건강, 발견 경위처럼 입양 희망자가 실제로 눈여겨보는 내용이 담기지만 형식이 정해져 "
          "있지 않아 분석이 어렵다. 또한 일부 메모에는 '입양완료'처럼 결과가 나중에 덧붙어 있어, 그대로 쓰면 성능이 실제보다 "
          "높게 평가되는 문제가 생긴다."),
    ("P", "이에 본 연구의 목적은 다음과 같다. 첫째, 접수 시점에 알 수 있는 정보만으로 입양 가능성을 예측하는 모델을 만들고 "
          "여러 기계학습 모델을 공정한 조건에서 비교한다. 둘째, 비정형 텍스트인 특징 메모를 예측에 활용하는 방법을 찾고 그 효과를 "
          "검증한다. 셋째, 메모에 담긴 성격, 건강, 관리 흔적 등의 특징이 다른 조건을 통제한 뒤에도 입양과 어떤 관련을 갖는지 분석한다."),
    ("GAP",),
    ("H", "Ⅱ. 이론적 배경"),
    ("S", "1. 그래디언트 부스팅과 CatBoost"),
    ("P", "그래디언트 부스팅은 작은 결정 트리를 차례로 더해 가며, 앞선 트리가 틀린 부분을 다음 트리가 보완하도록 학습하는 앙상블 "
          "방법이다. 표 형태의 데이터에서 변수 사이의 비선형 관계와 상호작용을 잘 잡아 널리 쓰이며, 대표적인 구현으로 XGBoost, "
          "LightGBM, CatBoost가 있다. CatBoost[2]는 품종이나 보호소처럼 값의 종류가 많은 범주형 변수를 다루기 위해, 각 범주를 그 "
          "범주의 평균 결과로 바꾸되 해당 개체보다 앞선 데이터만 사용하는 순서형 목표 통계량을 쓴다. 이 방식은 정답 정보가 입력으로 "
          "새어 들어가 과적합되는 문제를 줄인다."),
    ("P", "비교 대상으로 쓴 로지스틱 회귀는 각 변수의 효과를 더해 확률을 계산하는 선형 모델로, 해석이 쉽지만 변수 사이의 비선형 관계를 "
          "잡지 못한다. 다층 퍼셉트론은 여러 층의 인공 신경망으로, 범주형 변수를 학습 가능한 벡터로 바꾸고 고차원의 문장 임베딩을 그대로 "
          "입력할 수 있다는 장점이 있다."),
    ("S", "2. 형태소 분석과 문장 임베딩"),
    ("P", "한국어는 조사와 어미가 붙어 같은 단어도 여러 형태로 나타나므로, 텍스트를 분석하려면 형태소 단위로 나누는 과정이 필요하다. "
          "본 연구는 한국어 형태소 분석기 Kiwi[3]로 메모를 명사, 형용사, 동사 등 내용어로 나누고, 단어의 등장 빈도에 희귀도를 곱한 "
          "TF-IDF로 바꾸었다. 그러나 TF-IDF는 '순함'과 '온순하다'처럼 뜻이 같은 다른 표현을 서로 다른 단어로 취급한다. 문장 "
          "임베딩은 사전 학습된 언어 모델이 문장 전체의 의미를 수백 차원의 숫자 벡터로 바꾸는 방법으로, 뜻이 비슷한 문장은 가까운 "
          "벡터를 갖는다. 본 연구는 다국어 문장 임베딩 모델 multilingual-e5-base[4]를 사용하였다."),
    ("S", "3. 평가 지표와 데이터 누수"),
    ("P", "ROC-AUC는 무작위로 고른 입양 개체가 비입양 개체보다 높은 점수를 받을 확률이다. PR-AUC는 입양으로 예측한 개체 중 실제 "
          "입양 비율(정밀도)과 실제 입양 개체 중 찾아낸 비율(재현율)의 관계를 요약한 값으로, 한쪽 결과가 적을 때 모델 간 차이를 더 잘 "
          "드러낸다. Brier 점수는 예측 확률과 실제 결과(0 또는 1)의 차이를 제곱해 평균한 값으로, 작을수록 확률이 정확하다. 확률 "
          "보정은 모델이 낸 점수가 실제 비율과 맞도록 조정하는 과정으로, Platt 보정은 점수를 로지스틱 함수로 다시 맞추어 순위는 "
          "그대로 두고 확률값만 바로잡는다[6]."),
    ("P", "데이터 누수는 예측 시점에는 알 수 없는 정보가 입력에 섞여 들어가는 현상이다. 누수가 있으면 평가 점수는 높아지지만 실제로 "
          "쓰면 성능이 떨어진다. 따라서 입력 변수가 언제 기록되는지 확인하고, 미래를 예측하는 상황과 같도록 데이터를 시간 순서대로 "
          "나누어 평가해야 한다."),
    ("GAP",),
    ("H", "Ⅲ. 연구방법 및 절차"),
    ("S", "1. 데이터 수집과 정제"),
    ("P", "공공데이터포털의 구조동물 조회 서비스[1]에서 축종을 개로 한정하여, 발견일 기준 2021년 1월 1일부터 2026년 10월 19일까지의 "
          "기록 433,422건(51개 항목)을 수집하였다. 처리 결과는 입양 125,455, 안락사 119,967, 자연사 83,175, 반환 69,796, "
          "기증 19,466, 보호중 15,273건 등이었다. 2026년 접수분은 아직 보호 중인 개가 많아(2026년 9월 접수분의 52%) 결과가 빨리 "
          "난 개만 남는 치우침이 생기므로 제외하였다. 그 결과 2025년 12월 31일까지 접수되어 입양, 안락사, 자연사, 기증으로 보호가 "
          "끝난 320,426마리를 분석하였으며, 이 중 입양 비율은 35.5%였다. 주인에게 돌아간 반환은 입양과 성격이 달라 제외하였다. "
          "사용한 항목과 제외한 항목은 표 1과 같다."),
    ("T", "items"),
    ("P", "그림 1은 분석 대상의 체중 구간별, 주요 품종별 입양률이다. 3kg 이하 소형견의 입양률은 41.3%였으나 10~20kg은 23.8%로 "
          "낮았고, 전체의 84%를 차지하는 믹스견은 30.3%로 푸들(64.0%), 말티즈(65.7%), 포메라니안(71.8%)보다 크게 낮았다. 이처럼 "
          "정형 정보만으로도 입양률 차이가 뚜렷하지만, 같은 품종과 체중 안에서도 성격과 건강에 따라 결과가 달라지므로 메모 정보가 함께 "
          "필요하다."),
    ("F", "eda"),
    ("S", "2. 정답 정의와 시간순 분할"),
    ("P", "정답은 보호가 끝난 개 중 입양을 1, 안락사·자연사·기증을 0으로 두었다. 미래의 개를 예측하는 상황과 같도록 데이터를 접수일 "
          "순서로 나누어, 앞 80% 가운데 앞부분 85%를 학습 구간, 뒷부분 15%를 검증 구간, 나머지 20%를 시험 구간으로 정하였다(표 2). "
          "모델의 종류와 설정을 고르는 모든 결정은 검증 구간에서만 하였고, 시험 구간은 최종 성능을 보고할 때 한 번만 사용하였다."),
    ("T", "split"),
    ("S", "3. 특징 메모의 누수 제거"),
    ("P", "특징 메모 가운데 일부에는 입양이 끝난 뒤 '입양완료', '입양숙려기간 진행중', '(입양예약)', '임시보호 및 기증예정'처럼 "
          "결과가 덧붙어 있었다. 예를 들어 '입양숙려기간'이 들어 있는 메모의 입양률은 98%에 달하였다. 처음에는 이런 단어만 "
          "지웠으나, '(입양 진행 중)'이 '( 중)'으로 남는 것처럼 지우고 남은 흔적 자체가 결과를 알려 주는 문제가 확인되었다(입양률 97%)."),
    ("P", "그래서 쉼표, 마침표, 괄호 등을 기준으로 메모를 구절 단위로 나눈 뒤, 결과를 드러내는 단어(입양, 임시보호, 예약, 기증, 분양, "
          "반환, 안락사, 자연사 등)가 든 구절을 통째로 지웠다. 단, '신고자 입양희망'처럼 접수 때 이미 알 수 있는 표현은 먼저 다른 "
          "이름표로 바꾸어 보존하였다. 그 결과 2025년까지의 메모 388,021건 중 10,693건에서 구절이 지워졌고, 남은 메모에서 결과를 "
          "드러내는 단어는 발견되지 않았다."),
    ("S", "4. 입력 변수와 모델 선택"),
    ("P", "입력 변수는 범주형 5개(품종, 털색, 성별, 중성화, 시도), 숫자형 6개(나이, 생후 60일 미만 여부, 체중, 접수 월, 믹스견 여부, "
          "메모 길이), 메모에서 만든 숫자 33개로 구성하였다. 메모는 두 가지로 바꾸었다. 하나는 형태소 TF-IDF에 로지스틱 회귀를 적용해 "
          "얻은 입양 확률 점수 1개이고, 다른 하나는 문장 임베딩(768차원)을 학습 구간에서 구한 주성분 32개로 줄인 것이다. 보호소는 "
          "영향이 매우 커서(보호소별 입양률 6~98%), 보호소를 입력에 넣은 모델과 뺀 모델을 함께 만들었다. 보호소를 뺀 모델은 처음 "
          "보는 보호소에도 쓸 수 있고 개 자체의 특징에 집중한다."),
    ("P", "모델은 로지스틱 회귀, LightGBM, XGBoost, CatBoost, 다층 퍼셉트론(MLP) 다섯 계열과 상위 3개 모델의 평균 앙상블을 "
          "비교하였다. 계열마다 설정 8개를 무작위로 시도하고 검증 구간의 PR-AUC로 가장 좋은 설정을 골랐다. 1등이 우연이 아닌지는 "
          "보호소 단위로 표본을 다시 뽑는 클러스터 부트스트랩 1,000회로 성능 차이의 95% 신뢰구간을 구해 판단하였다. 사전 점검으로 "
          "아무 정보가 없는 무작위 임베딩을 넣어 보았을 때, 검증 점수만으로는 무작위 임베딩이 1등으로 뽑혔지만 이 신뢰구간 기준으로는 "
          "걸러졌다. 트리 모델에는 검증 구간 성능이 100번 연속 나아지지 않으면 학습을 멈추는 조기 종료를 적용하였고, 앞선 실험에서 트리 수 "
          "상한(800개)에 걸렸던 문제를 피하기 위해 상한을 5,000개로 늘렸다. 전체 과정은 그림 2와 같다."),
    ("F", "fig1"),
    ("GAP",),
    ("H", "Ⅳ. 연구결과 및 분석"),
    ("S", "1. 모델 비교"),
    ("P", "표 3은 다섯 계열과 앙상블의 시험 구간 성능이다. CatBoost는 단일 모델 중 검증과 시험 모두에서 가장 높았고, 나머지 네 계열과의 "
          "PR-AUC 차이는 신뢰구간이 모두 0보다 커서 확실한 차이로 판정되었다(+0.007~+0.039). 상위 3개 평균 앙상블은 검증 점수가 조금 "
          "더 높았으나 CatBoost와의 차이가 불확실하였고(보호소 제외 모델, 95% 신뢰구간 -0.003~+0.005), 시험 구간에서는 오히려 "
          "낮았다. 따라서 더 단순한 CatBoost 단일 모델을 최종 모델로 선택하였다. 최종 설정은 트리 깊이 8, L2 정규화 30이었다."),
    ("T", "models"),
    ("S", "2. 최종 모델 성능"),
    ("P", "최종 모델의 시험 구간 성능은 표 4와 같다. 보호소를 뺀 모델은 ROC-AUC 0.740, PR-AUC 0.596이었고, 보호소를 넣으면 각각 "
          "0.766, 0.638로 높아졌다. 예측 확률은 검증 구간으로 Platt 보정[6]을 하였다. 그림 3처럼 예측 확률이 높은 구간일수록 실제 "
          "입양률이 꾸준히 높아져, 예측 상위 10%의 실제 입양률은 72.3%, 하위 10%는 8.0%로 약 9배 차이가 났다. 다만 중간 구간에서는 "
          "예측이 실제보다 4~5%p 낮았는데, 보정에 쓴 검증 구간(2024년 3~10월)의 입양률이 31.2%로 다른 구간보다 낮았기 때문으로 보인다."),
    ("T", "final"),
    ("F", "fig2"),
    ("P", "확률 0.5를 기준으로 입양 여부를 판정하면 보호소를 뺀 모델의 정확도는 71.3%, 정밀도는 61.5%, 재현율은 42.8%였다. 모든 "
          "개를 비입양으로 판정해도 정확도가 65.8%이므로, 이 모델은 하나의 기준으로 입양 여부를 가르기보다 입양이 어려운 개의 우선순위를 "
          "정하는 데 더 적합하다. 변수 중요도는 메모 임베딩(45.0)과 메모 단어 점수(12.7)가 합쳐 약 58%를 차지하였고, 시도(9.2), "
          "체중(8.9), 품종(8.4)이 뒤를 이었다."),
    ("P", "보호소를 넣은 모델의 성능이 더 높은 것은 보호소마다 안락사 시점, 공간, 홍보 수준이 달라 입양률이 6%에서 98%까지 차이 나기 "
          "때문이다. 이미 운영 중인 보호소 안에서 우선순위를 정할 때는 보호소 포함 모델이, 새 보호소에 적용하거나 개 자체의 특징을 해석할 "
          "때는 보호소 제외 모델이 적절하다."),
    ("S", "3. 메모 해석의 검증"),
    ("P", "모델이 메모를 의미에 맞게 해석하는지 확인하기 위해 두 가지를 검증하였다. 첫째, 메모를 성격, 건강, 관리 흔적, 출신, 외모 등 "
          "14개 항목으로 나누는 규칙 분류기를 만들고(표 5), '공격성 없음', '파보 음성' 같은 부정 표현은 해당 항목으로 세지 않도록 하였다. "
          "무작위 메모 200개로 규칙을 다듬은 뒤, 겹치지 않는 새 메모 200개를 따로 판정하여 비교한 결과 정밀도 0.962, 재현율 0.900이었다. "
          "외모 칭찬(재현율 0.71)과 외상·거동(0.67)은 표현이 다양해 놓치는 경우가 상대적으로 많았다."),
    ("T", "cats"),
    ("P", "둘째, 같은 개(경기도, 2살, 8kg 믹스견)에 메모만 바꾸어 넣고 최종 모델의 예측 변화를 보았다(표 6). 성격 문제나 건강 문제가 "
          "적힌 메모는 예측을 크게 낮추었고, 장점과 단점이 섞인 메모는 그 사이 값을 보였다. 반면 '공격성 없음'은 11.3%로 공격성이 있는 "
          "메모와 비슷하게 낮게 예측되었다. 실제 데이터에서 '공격성·입질 없음'이 적힌 343건의 입양률은 27.7%였으므로, 모델이 부정 "
          "표현을 충분히 읽지 못하는 한계가 확인되었다. 다만 이런 표현은 전체 메모의 0.1% 수준이다."),
    ("T", "memo"),
    ("S", "4. 입양과 관련된 메모 특징"),
    ("P", "각 메모 항목이 입양과 어떤 관련이 있는지 알아보기 위해 품종, 털색, 성별, 중성화, 시도, 접수 연월, 나이, 체중을 통제한 로지스틱 "
          "회귀를 적용하였다. 같은 보호소의 개끼리는 비슷한 경향을 보이므로 표준오차는 보호소 단위로 묶어 계산하였고[5], 보호소를 추가로 "
          "통제한 모형도 함께 추정하였다. 그림 4는 그 결과이며, 오른쪽 숫자는 항목이 있을 때와 없을 때의 평균 입양 확률 차이(%p)이다."),
    ("F", "fig3"),
    ("P", "보호소를 통제하였을 때 감염병 양성(-11.3%p), 공격성(-11.0%p), 쇠약(-9.9%p), 외상·거동 문제(-9.4%p), 겁·경계(-8.2%p), "
          "피부 질환(-7.1%p), 떠돌이·포획(-6.0%p)은 낮은 입양 가능성과 관련되었다. 반대로 미용·옷 등 관리 흔적(+6.0%p), 온순(+5.8%p), "
          "외모 칭찬(+4.6%p), 활발(+3.0%p)은 높은 입양 가능성과 관련되었다. 동반 입소, 목줄, 칩, 주인 언급은 다른 조건을 통제하면 "
          "뚜렷한 관련이 없었다."),
    ("P", "두 모형의 차이에서도 의미 있는 점이 드러났다. '개체관리번호', '꼬리단미 안됨' 같은 보호소 기록 양식은 보호소를 통제하지 않으면 "
          "-7.0%p였으나 통제하면 -1.1%p로 사라졌다. 이 표현이 개의 특징이 아니라 특정 보호소를 가리키는 표지였다는 뜻이다. 반대로 감염병 "
          "양성은 보호소를 통제할 때만 크게 불리하게 나타났는데, 감염 검사를 기록하는 보호소의 입양률이 원래 높아 보호소 간 비교에서 "
          "효과가 가려졌던 것으로 해석된다."),
    ("GAP",),
    ("H5", "Ⅴ. 결론 및 고찰"),
    ("P", "본 연구는 유기견이 보호소에 들어온 시점의 정보만으로 입양 가능성을 예측하는 모델을 만들고, 비정형 텍스트인 특징 메모의 역할을 "
          "분석하였다. 첫째, 접수 정보만으로도 입양 가능성을 의미 있게 구분할 수 있었다. 최종 CatBoost 모델은 시험 구간에서 ROC-AUC "
          "0.740을 보였고, 예측 상위 10%와 하위 10%의 실제 입양률은 72.3%와 8.0%로 9배 차이가 났다. 둘째, 특징 메모는 가장 중요한 "
          "정보였다. 메모를 문장 임베딩과 단어 점수로 바꾸어 넣으면 정형 정보만 쓸 때보다 성능이 확실히 높아졌고, 모델 중요도의 약 58%를 "
          "차지하였다. 셋째, 메모에 적힌 성격, 건강, 관리 흔적은 다른 조건을 통제한 뒤에도 입양과 뚜렷하게 관련되었다. 넷째, 메모에는 "
          "결과가 덧붙은 문구가 섞여 있어, 구절 단위로 지우지 않으면 성능이 부풀려진다는 점을 확인하였다."),
    ("P", "이 결과는 보호소가 입양이 어려울 개를 접수 단계에서 찾아 홍보, 치료, 임시보호 연계를 먼저 지원하는 데 활용될 수 있다. 다만 "
          "본 연구의 결과는 연관성이며 인과 관계가 아니다. 메모는 개의 실제 상태를 반영하므로, 메모를 좋게 바꾸어 쓴다고 입양이 늘어난다고 "
          "말할 수는 없다. 또한 입양은 입양 희망자의 선택과 우연에 크게 좌우되어 완벽한 예측은 어렵고, 모델이 부정 표현을 충분히 읽지 "
          "못하며, 메모 분류기의 정답 판정을 사람이 아닌 AI가 수행하였다는 한계가 있다."),
    ("P", "실제 활용 방안으로는, 공고를 올리는 시점에 모델이 예측한 입양 확률이 하위 20%에 드는 개를 따로 표시하여 SNS 홍보나 임시보호 "
          "연계 대상으로 먼저 검토하는 방법을 생각할 수 있다. 이때 확률의 절대값은 시기에 따라 달라질 수 있으므로, 같은 시기에 들어온 개들 "
          "사이의 순위를 보는 것이 안전하다."),
    ("P", "후속 연구로는 공고 사진과 접수 당시 보호소의 보호 두수를 변수로 추가하고, 입양까지 걸린 시간을 다루는 생존분석으로 보호소가 준 "
          "시간이 부족해 입양되지 못한 경우를 구분하는 연구가 필요하다."),
]

TABLES = {
    "items": ("표 1. 사용한 항목과 제외한 항목", [0.17, 0.53, 0.30], [2],
              [["구분", "항목", "이유"],
               ["사용", "품종, 털색, 나이, 체중, 성별, 중성화, 특징 메모, 발견일, 관할 시도, 보호소", "접수 시점에 알 수 있음"],
               ["제외(누수)", "처리 상태, 종료 사유, 수정 시각, 마이크로칩 번호, 건강검진·접종 기록", "결과가 드러나거나 접수 뒤 기록"],
               ["제외(중복·무관)", "품종 코드, 공고번호, 보호소 연락처·주소, 안내문", "다른 항목과 같거나 개와 무관"],
               ["미사용(후보)", "공고 사진, 발견 장소, 공고 기간", "후속 연구에서 추가"]]),
    "split": ("표 2. 학습·검증·시험 구간", [0.2, 0.4, 0.2, 0.2], [],
              [["구간", "접수 기간", "개체 수", "입양 비율"],
               ["학습", "2021.01.01~2024.03.18", "217,890", "36.6%"],
               ["검증", "2024.03.18~2024.10.30", "38,451", "31.2%"],
               ["시험", "2024.10.30~2025.12.31", "64,085", "34.2%"]]),
    "models": ("표 3. 모델 계열별 시험 구간 성능", [0.28, 0.18, 0.18, 0.18, 0.18], [],
               [["모델", "PR-AUC(보호소 제외)", "ROC-AUC(보호소 제외)", "PR-AUC(보호소 포함)", "ROC-AUC(보호소 포함)"],
                ["CatBoost", "0.596", "0.740", "0.638", "0.766"],
                ["상위 3개 평균 앙상블", "0.595", "0.739", "0.640", "0.767"],
                ["LightGBM", "0.587", "0.734", "0.635", "0.763"],
                ["XGBoost", "0.584", "0.734", "0.626", "0.759"],
                ["다층 퍼셉트론", "0.580", "0.721", "0.623", "0.753"],
                ["로지스틱 회귀", "0.573", "0.725", "0.611", "0.747"]]),
    "final": ("표 4. 최종 모델(CatBoost)의 시험 구간 성능", [0.5, 0.25, 0.25], [],
              [["지표", "보호소 제외", "보호소 포함"],
               ["ROC-AUC", "0.740", "0.766"],
               ["PR-AUC", "0.596", "0.638"],
               ["Brier 점수(보정 후)", "0.190", "0.181"],
               ["예측 상위 10%의 실제 입양률", "72.3%", "77.2%"],
               ["예측 하위 10%의 실제 입양률", "8.0%", "6.3%"]]),
    "cats": ("표 5. 메모 항목 분류 기준과 등장 비율 (분석 대상 320,426마리)", [0.22, 0.58, 0.20], [1],
             [["항목", "판정 표현의 예", "메모 비율"],
              ["성격 - 온순", "순함, 온순, 얌전, 사람을 잘 따름, 애교", "32.4%"],
              ["성격 - 겁·경계", "겁 많음, 소심, 경계심, 예민", "17.3%"],
              ["성격 - 공격", "공격성, 입질, 사나움, 으르렁거림", "3.6%"],
              ["건강 - 피부", "피부병, 진드기, 탈모, 구더기", "4.8%"],
              ["건강 - 외상·거동", "골절, 교통사고, 기립 불가, 종양", "2.9%"],
              ["건강 - 쇠약", "야윔, 탈수, 기력 없음, 설사", "2.2%"],
              ["건강 - 감염병", "파보·홍역 양성, 심장사상충 감염", "1.5%"],
              ["관리 흔적", "미용, 옷 착용, 하네스, 이름표", "2.6%"],
              ["출신 - 떠돌이", "떠돌이, 배회, 포획, 들개", "5.6%"],
              ["외모 칭찬", "귀여움, 예쁨, 잘생김, 미모", "8.3%"]]),
    "memo": ("표 6. 메모만 바꾸었을 때의 예측 입양 확률 (같은 개, 보호소 제외 모델)", [0.6, 0.4], [],
             [["메모", "예측 입양 확률"],
              ["(메모 없음)", "27.9%"],
              ["순함", "25.7%"],
              ["겁이 많지만 애교 많음", "25.4%"],
              ["겁이 많음", "11.5%"],
              ["공격성 없음", "11.3%"],
              ["순하지만 피부병 심함", "11.0%"],
              ["파보 양성", "10.6%"],
              ["피부병 심함", "8.6%"]]),
}

FIGS = {  # 이름: (파일, 폭 mm, 캡션) — 번호는 본문에 나오는 순서
    "eda": ("fig_eda.png", 165, "[그림 1] 체중 구간별·주요 품종별 입양률"),
    "fig1": ("fig1_flow.png", 165, "[그림 2] 연구 절차"),
    "fig2": ("fig2_decile.png", 135, "[그림 3] 예측 확률 10분위별 실제 입양률 (보호소 제외 모델)"),
    "fig3": ("fig3_effects.png", 140, "[그림 4] 메모 항목별 입양 오즈비와 95% 신뢰구간 (오른쪽: 평균 효과 %p, 보호소 통제)"),
}

REFS = [
    "[1]공공데이터포털(2026). 국가동물보호정보시스템 구조동물 조회 서비스. https://www.data.go.kr/data/15098931/openapi.do (2026년 10월 19일 접속).",
    "[2]Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A.(2018). CatBoost: unbiased boosting with categorical features. Advances in Neural Information Processing Systems, 31.",
    "[3]Lee, M. Kiwi: Korean Intelligent Word Identifier(kiwipiepy). https://github.com/bab2min/kiwipiepy",
    "[4]Wang, L., Yang, N., Huang, X., Yang, L., Majumder, R., & Wei, F.(2024). Multilingual E5 Text Embeddings: A Technical Report. arXiv:2402.05672.",
    "[5]Cameron, A. C., & Miller, D. L.(2015). A Practitioner's Guide to Cluster-Robust Inference. Journal of Human Resources, 50(2), 317-372.",
    "[6]Platt, J.(1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. Advances in Large Margin Classifiers, 10(3), 61-74.",
]


# ---------------------------------------------------------------- XML 조립
def lineseg(w=TEXT_W, h=900):
    return (f'<hp:linesegarray><hp:lineseg textpos="0" vertpos="0" vertsize="{h}" textheight="{h}" baseline="{int(h * 0.85)}" '
            f'spacing="{int(h * 0.6)}" horzpos="0" horzsize="{w}" flags="393216"/></hp:linesegarray>')


def para(pp, cp, text=""):
    t = f"<hp:t>{escape(text)}</hp:t>" if text else "<hp:t/>"
    return (f'<hp:p id="0" paraPrIDRef="{pp}" styleIDRef="0" pageBreak="0" columnBreak="0" merged="0">'
            f'<hp:run charPrIDRef="{cp}">{t}</hp:run>{lineseg()}</hp:p>')


def uid():
    return str(random.randint(1_000_000_000, 2_000_000_000))


def table_xml(tpl, name):
    caption, rel, left_cols, rows = TABLES[name]
    pre = tpl[: tpl.index("<hp:tr>")]
    post = tpl[tpl.rindex("</hp:tr>") + len("</hp:tr>"):]
    cell_tpl = re.search(r"<hp:tc .*?</hp:tc>", tpl, re.S).group(0)
    total = 48746
    widths = [int(total * r) for r in rel]
    widths[-1] = total - sum(widths[:-1])
    nr, nc = len(rows), len(rel)
    trs = []
    for r, row in enumerate(rows):
        kind = 0 if r == 0 else (1 if r == 1 else (3 if r == nr - 1 else 2))
        ids = [(14, 15, 16), (11, 12, 13), (7, 3, 8), (9, 6, 10)][kind]
        tcs = []
        for c, text in enumerate(row):
            bf = ids[0] if c == 0 else (ids[2] if c == nc - 1 else ids[1])
            pp = 40 if (c in left_cols and r > 0) else 27
            tc = cell_tpl
            tc = re.sub(r'borderFillIDRef="\d+"', f'borderFillIDRef="{bf}"', tc, count=1)
            tc = re.sub(r'paraPrIDRef="\d+"', f'paraPrIDRef="{pp}"', tc, count=1)
            tc = re.sub(r"<hp:t>[^<]*</hp:t>", f"<hp:t>{escape(text)}</hp:t>", tc, count=1)
            tc = re.sub(r'horzsize="\d+"', f'horzsize="{widths[c] - 1020}"', tc, count=1)
            tc = re.sub(r'<hp:cellAddr colAddr="\d+" rowAddr="\d+"/>', f'<hp:cellAddr colAddr="{c}" rowAddr="{r}"/>', tc)
            tc = re.sub(r'<hp:cellSz width="\d+" height="\d+"/>', f'<hp:cellSz width="{widths[c]}" height="1465"/>', tc)
            tcs.append(tc)
        trs.append("<hp:tr>" + "".join(tcs) + "</hp:tr>")
    pre = re.sub(r'rowCnt="\d+" colCnt="\d+"', f'rowCnt="{nr}" colCnt="{nc}"', pre)
    pre = re.sub(r'<hp:sz width="48746" widthRelTo="ABSOLUTE" height="\d+"', f'<hp:sz width="48746" widthRelTo="ABSOLUTE" height="{1465 * nr}"', pre)
    pre = re.sub(r'(<hp:tbl id=")\d+', lambda m: m.group(1) + uid(), pre)
    pre = re.sub(r"(<hp:caption .*?<hp:t>)[^<]*(</hp:t>)", lambda m: m.group(1) + escape(caption) + m.group(2), pre, flags=re.S)
    return pre + "".join(trs) + post


def figure_xml(tpl, name, bin_id):
    fn, width_mm, caption = FIGS[name]
    w_px, h_px = Image.open(FIG_DIR + "/" + fn).size
    W = int(width_mm * MM)
    H = int(W * h_px / w_px)
    x = tpl
    x = re.sub(r'(<hp:pic id=")\d+', lambda m: m.group(1) + uid(), x)
    x = re.sub(r'instid="\d+"', f'instid="{uid()}"', x)
    x = re.sub(r'<hp:offset x="\d+" y="\d+"/>', '<hp:offset x="0" y="0"/>', x, count=1)
    x = re.sub(r'<hp:orgSz width="\d+" height="\d+"/>', f'<hp:orgSz width="{W}" height="{H}"/>', x)
    x = re.sub(r'<hp:curSz width="\d+" height="\d+"/>', f'<hp:curSz width="{W}" height="{H}"/>', x)
    x = re.sub(r'<hp:rotationInfo angle="0" centerX="\d+" centerY="\d+"', f'<hp:rotationInfo angle="0" centerX="{W // 2}" centerY="{H // 2}"', x)
    x = re.sub(r"<hp:renderingInfo>.*?</hp:renderingInfo>",
               '<hp:renderingInfo><hc:transMatrix e1="1" e2="0" e3="0" e4="0" e5="1" e6="0"/>'
               '<hc:scaMatrix e1="1" e2="0" e3="0" e4="0" e5="1" e6="0"/>'
               '<hc:rotMatrix e1="1" e2="0" e3="0" e4="0" e5="1" e6="0"/></hp:renderingInfo>', x, flags=re.S)
    x = re.sub(r'binaryItemIDRef="\w+"', f'binaryItemIDRef="{bin_id}"', x)
    x = re.sub(r"<hp:imgRect>.*?</hp:imgRect>",
               f'<hp:imgRect><hc:pt0 x="0" y="0"/><hc:pt1 x="{W}" y="0"/><hc:pt2 x="{W}" y="{H}"/><hc:pt3 x="0" y="{H}"/></hp:imgRect>', x, flags=re.S)
    x = re.sub(r'<hp:imgClip left="0" right="\d+" top="0" bottom="\d+"/>', f'<hp:imgClip left="0" right="{W}" top="0" bottom="{H}"/>', x)
    x = re.sub(r'<hp:imgDim dimwidth="\d+" dimheight="\d+"/>', f'<hp:imgDim dimwidth="{W}" dimheight="{H}"/>', x)
    x = re.sub(r'<hp:sz width="\d+" widthRelTo="ABSOLUTE" height="\d+"', f'<hp:sz width="{W}" widthRelTo="ABSOLUTE" height="{H}"', x)
    x = re.sub(r"<hp:shapeComment>.*?</hp:shapeComment>", f"<hp:shapeComment>그림입니다.\n원본 그림의 이름: {fn}</hp:shapeComment>", x, flags=re.S)
    x = re.sub(r'lastWidth="\d+"', f'lastWidth="{W}"', x)
    x = re.sub(r"(<hp:caption .*?<hp:t>)[^<]*(</hp:t>)", lambda m: m.group(1) + escape(caption) + m.group(2), x, flags=re.S)
    x = re.sub(r"<hp:ctrl><hp:newNum[^>]*/></hp:ctrl>", "", x)
    return x


def strip_memo(p):
    """양식 작성자의 메모(MEMO 필드)를 지운다. 누름틀(CLICK_HERE)은 남긴다."""
    ids = re.findall(r'<hp:fieldBegin id="(\d+)" type="MEMO"', p)
    p = re.sub(r'<hp:ctrl><hp:fieldBegin id="\d+" type="MEMO".*?</hp:fieldBegin></hp:ctrl>', "", p, flags=re.S)
    for i in ids:
        p = re.sub(rf'<hp:ctrl><hp:fieldEnd beginIDRef="{i}"[^>]*/></hp:ctrl>', "", p)
    return p


def set_text(p, new):
    """문단의 첫 번째 보이는 글자 묶음을 바꾸고 나머지 글자는 비운다(필드 구조는 유지)."""
    done = False

    def rep(m):
        nonlocal done
        if not done and m.group(1).strip():
            done = True
            return f"<hp:t>{escape(new)}</hp:t>"
        return m.group(0)
    return re.sub(r"<hp:t>([^<]*)</hp:t>", rep, p)


def build_section(parts):
    p = parts
    # 0: 구역 설정·머리말·쪽 번호는 남기고 '작성 요령' 제목과 안내 메모를 지운다
    p0 = p[0]
    cut = p0.index('<hp:run charPrIDRef="33">')
    p0 = p0[:cut] + '<hp:run charPrIDRef="10"><hp:t/></hp:run>' + p0[p0.rindex("<hp:linesegarray>"):]
    out = [p0]
    # 1(작성 요령 표), 2(빈 줄)는 건너뛴다
    out.append(strip_memo(p[3]).replace("<hp:t>수학1A</hp:t>", "<hp:t>(팀코드)</hp:t>"))  # 누름틀 안 팀코드 자리
    out.append(p[4])
    out.append(set_text(p[5], TITLE))                     # 과제명
    out += [p[6], strip_memo(p[7]), p[8], p[9], p[10]]    # 저자·지도교사(자리표시 그대로), 요약 머리
    out.append(set_text(p[11], SUMMARY))
    out.append(p[12])
    out.append(set_text(p[13], KEYWORDS))
    out.append(p[14])                                     # 단 설정 문단
    tbl_tpl, fig_tpl = p[27], p[21]
    bins = {}
    for blk in BODY:
        k = blk[0]
        if k == "H":
            out.append(para(20, 26, blk[1]))
        elif k == "H5":
            out.append(para(27, 26, blk[1]))
        elif k == "S":
            out.append(para(17, 27, blk[1]))
        elif k == "P":
            out.append(para(29, 20, blk[1]))
        elif k == "GAP":
            out.append(para(5, 26))
        elif k == "T":
            out.append(table_xml(tbl_tpl, blk[1]))
            out.append(para(25, 15))                      # 표 아래 빈 줄(양식과 같음)
        elif k == "F":
            bins[blk[1]] = f"image{len(bins) + 2}"
            out.append(figure_xml(fig_tpl, blk[1], bins[blk[1]]))
    out.append(para(15, 18))
    out.append(para(18, 26, "참고문헌"))
    out += [para(38, 20, r) for r in REFS]
    return out, bins


def estimate_pages():
    """대략적인 쪽수 추정(한글 프로그램 없이). 본문 높이 237mm(A4 - 여백 15mm×2 - 머리말·꼬리말 15mm×2),
    바탕 9pt 한 줄 약 58자·5.08mm(줄 간격 160%). 표와 그림은 쪽을 넘기지 못하므로 남은 공간이 모자라면 다음 쪽으로 보낸다.
    실제 쪽수와 ±0.5쪽 정도 차이 날 수 있다."""
    import math
    H, line = 237.0, 5.08
    blocks = [64 + math.ceil(len(SUMMARY) / 60) * line]  # 제목·저자·요약 묶음
    kinds = ["fixed"]
    for b in BODY:
        k = b[0]
        if k == "P":
            blocks.append(math.ceil((len(b[1]) + 2) / 58) * line); kinds.append("text")
        elif k in ("H", "H5"):
            blocks.append(8.5); kinds.append("fixed")
        elif k == "S":
            blocks.append(7.8); kinds.append("fixed")
        elif k == "GAP":
            blocks.append(6.8); kinds.append("fixed")
        elif k == "T":
            cap, rel, _, rows = TABLES[b[1]]
            h = 8.1
            for row in rows:
                n = max(math.ceil(len(t) / max(1, (172 * r - 3.6) / 2.9)) for t, r in zip(row, rel))
                h += 5.17 + (n - 1) * line
            blocks.append(h + line); kinds.append("obj")
        elif k == "F":
            fn, wmm, cap = FIGS[b[1]]
            wp, hp = Image.open(FIG_DIR + "/" + fn).size
            blocks.append(wmm * hp / wp + math.ceil(len(cap) / 50) * line + 3); kinds.append("obj")
    blocks.append(5 + 8.5 + sum(math.ceil(len(r) / 62) * 4.6 + 1.4 for r in REFS)); kinds.append("text")
    page, used = 1, 0.0
    for h, k in zip(blocks, kinds):
        if k == "obj" and used + h > H:
            page, used = page + 1, h
            continue
        used += h
        while used > H:
            page, used = page + 1, used - H
    return page - 1 + used / H


def main():
    global FIG_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--figs", default="report_fig")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    FIG_DIR = args.figs
    random.seed(7)

    zin = zipfile.ZipFile(args.template)
    sec = zin.read("Contents/section0.xml").decode("utf-8")
    start = sec.index(">", sec.index("<hs:sec")) + 1
    end = sec.rindex("</hs:sec>")
    body = sec[start:end]
    tok = re.compile(r"<hp:p\b[^>]*?(/?)>|</hp:p>")
    parts, depth, cur = [], 0, None
    for m in tok.finditer(body):
        if m.group(0).startswith("</"):
            depth -= 1
            if depth == 0:
                parts.append(body[cur:m.end()])
        elif m.group(1) != "/":
            if depth == 0:
                cur = m.start()
            depth += 1
    assert len(parts) == 55, f"양식 문단 수가 예상과 다름: {len(parts)}"
    new_parts, bins = build_section(parts)
    new_sec = sec[:start] + "".join(new_parts) + sec[end:]

    hpf = zin.read("Contents/content.hpf").decode("utf-8")
    items = "".join(f'<opf:item id="{bid}" href="BinData/{bid}.png" media-type="image/png" isEmbeded="1"/>' for bid in bins.values())
    hpf = hpf.replace('<opf:item id="section0"', items + '<opf:item id="section0"')
    hpf = re.sub(r'<opf:title xml:space="preserve">[^<]*</opf:title>', f'<opf:title xml:space="preserve">{escape(TITLE)}</opf:title>', hpf)

    preview = "\r\n".join([TITLE, "", "요약", SUMMARY])[:1000]
    with zipfile.ZipFile(args.out, "w") as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename == "Contents/section0.xml":
                data = new_sec.encode("utf-8")
            elif info.filename == "Contents/content.hpf":
                data = hpf.encode("utf-8")
            elif info.filename == "Preview/PrvText.txt":
                data = preview.encode("utf-8")  # 양식의 미리보기 글자도 UTF-8
            comp = zipfile.ZIP_STORED if info.filename == "mimetype" else zipfile.ZIP_DEFLATED
            zout.writestr(info.filename, data, compress_type=comp)
        for name, bid in bins.items():
            zout.write(f"{FIG_DIR}/{FIGS[name][0]}", f"BinData/{bid}.png", compress_type=zipfile.ZIP_DEFLATED)
    print("saved", args.out, "| 본문 글자 수", sum(len(b[1]) for b in BODY if b[0] == "P"), "| 그림", list(bins.values()),
          f"| 추정 쪽수 {estimate_pages():.2f}")


if __name__ == "__main__":
    main()
