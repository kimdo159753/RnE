#!/usr/bin/env python3
"""특징 메모 → 주제별 항목(여러 개 동시 가능) 규칙 분류기

모델 입력이 아니라 해석·검증용이다. 누수 제거가 끝난 mark(train_adoption.prepare)를 입력으로 쓴다.
부정 표현("공격성 없음", "파보 음성", "경계심 X")은 해당 항목으로 세지 않는다.

  python memo_categories.py --csv abandoned_dogs.csv --sample 200 --out memo_check
"""
import argparse
import os
import re

import pandas as pd

# 단어 뒤에 붙으면 '없다'는 뜻이 되는 표현 (같은 구절 안, 짧은 거리)
NEG_AFTER = (r"(?:성|심|증|적|기|질|병)?\s*(?:은|는|이|가|도|을|를)?\s*(?:전혀\s*|별로\s*|크게\s*)?"
             r"(?:없|x|X|×|무\b|안\s*함|안함|아님|아니|적음|거의\s*없|보이지\s*않|하지\s*않|않)")

HANGUL_NUM = r"(?:\d+|두|세|네|다섯|여섯|일곱|여덟|아홉|열)"
CATEGORIES = {
    # 성격
    "성격_온순": r"순하|순함|순한|온순|얌전|착하|착함|착한|유순|온화|친근|애교|친화|사교|사랑둥이"
                r"|사람\s*(?:을|를|에게|이랑)?\s*(?:잘|매우|굉장히|몹시|너무|무척|아주)?\s*(?:따르|따름|따라|좋아)"
                r"|사람\s*손\s*(?:을|이)?\s*(?:잘\s*)?(?:탄|타)",
    "성격_활발": r"활발|활달|명랑|발랄|에너지|활동적",
    "성격_겁경계": r"겁|소심|경계|예민|두려|무서워|피하|낯가림|낯을\s*가|친해질\s*시간",
    "성격_공격": r"공격|입질|사나|으르렁|물려|물음|무는|물기|물어|위협|이빨을\s*보",
    # 건강
    "건강_감염병": r"(?:파보|홍역|코로나|디스템퍼|심장\s*사상충|사상충|장염|원충|켄넬|개선충)"
                 r"[^,.+]{0,8}?(?:양성|감염|확진|의심|있음|걸림)"  # '+'는 '파보+사상충'처럼 나열에 쓰여 양성 표시로 보지 않는다
                 r"|^\s*(?:심장\s*)?사상충\s*$",  # 구절에 '심장사상충'만 적은 경우는 보유로 본다(코로나 등은 접종 기록과 헷갈려 제외)
    "건강_피부": r"피부병|피부\s*질환|피부염|피부\s*(?:상태)?\s*(?:불량|나쁨|안\s*좋)|(?<![가-힣])옴|옴\s*진드기|진드기|곰팡이"
               r"|탈모|구더기|습진|털\s*(?:이|부분이)?\s*(?:많이\s*)?빠짐",
    "건강_외상거동": r"골절|교통사고|절뚝|파행|기립\s*불|못\s*일어|일어나지\s*못|일어서지\s*못|못\s*걷|걷지\s*못|마비|외상|출혈|상처"
                  r"|다리\s*(?:를|가)?\s*(?:불편|다침|부상|저)|발육\s*불량|실명|안구\s*(?:돌출|손상)|탈장|종양",
    "건강_쇠약": r"야윔|야위|마름|말랐|말랏|말라|탈수|기력|영양\s*실조|빈혈|쇠약|노령|노견|설사|구토",
    # 관리 흔적 (보호소가 해 준 미용, '미용을 하면'처럼 가정은 아래 EXCLUDE로 뺀다)
    "관리_미용옷": r"미용|옷\s*(?:을|입|착용)|옷착용|옷입|하네스|이름표|리본|염색|발톱\s*(?:정리|관리)|빗질|관리\s*(?:받은|된|됨)",
    "관리_목줄": r"목줄|목걸이|목띠|목태|목테|줄\s*(?:이|을)?\s*달린|체인\s*줄",
    "관리_칩있음": r"(?:내장)?칩\s*(?:이|은)?\s*(?:있|O|o|확인\s*됨|삽입)|동물\s*등록\s*(?:되어|됨|돼)",
    # 출신·경위
    "출신_떠돌이": r"떠돌|들개|야생|배회|포획|구조\s*틀|산\s*(?:에서|속)|유기\s*생활|길\s*생활|동네\s*개|마을\s*개",
    "출신_농장": r"농장|번식장|사육장|개\s*공장|도사견|육견",
    "출신_주인언급": r"견주|주인|보호자\s*(?:분|가|님)?|파양|인수|소유권",
    "출신_신고자희망": r"신고자희망|신고자\s*분?\s*(?:이|가)?\s*(?:우선|키우|데려)",
    # 외모·나이 표현
    "외모_칭찬": r"이쁘|이쁨|이쁜|이뻐|예쁘|예쁨|예쁜|귀엽|귀여|귀요미|잘생|미모|매력|공주|왕자|인형|미남|미녀",
    "외모_소형": r"소형|작은\s*(?:체구|체형|몸집|편|사이즈|아이)|작음|(?:체구|몸집|체형)\s*(?:가|이)?\s*작",
    "나이_새끼": r"새끼|애기|아기|자견|강아지|\d+\s*개월|\d+\s*일\s*(?:령|추정|정도)|생후|젖먹이|포유|태어난\s*지|어린",
    # 동반 입소
    "동반_무리": rf"어미|(?<!단)(?<!장)모견|동배|형제|남매|자매|{HANGUL_NUM}\s*마리|둘이|"
                r"(?:함께|같이)\s*(?:입소|구조|포획|들어|발견|옴)",
    # 보호소 기록 양식 (개의 특징이라기보다 서식)
    "서식_양식": r"개체\s*관리\s*번호|꼬리\s*단미|귀\s*쫑긋|귀\s*접힘|코\s*검정|털\s*(?:상태)?\s*(?:때탐|양호)",
}
# 부정 표현을 따로 확인할 항목(나쁜 특징·질병은 '없음'이 흔하다). 쇠약은 '기력 없음'이 곧 쇠약이라 넣지 않는다
NEGATABLE = {"성격_온순", "성격_겁경계", "성격_공격", "건강_피부", "건강_외상거동", "관리_목줄", "관리_미용옷", "출신_주인언급"}
# 맞더라도 그 구절이 이 표현이면 세지 않는다 (가정, 보호소가 한 일, 없다는 뜻)
EXCLUDE = {
    "관리_미용옷": re.compile(r"미용\s*(?:을|만)?\s*(?:하면|시키면|필요|요함|해야)|(?:뭉침|엉킴)[^,]*미용\s*(?:했|함|실시)|보호소에서\s*미용"),
    "동반_무리": re.compile(r"(?:어미|모견)\s*(?:없이|미\s*발견|못\s*찾|모름|불명)"),
    # 그 개가 어미인 경우: '새끼 포유중', '새끼와 같이 입소', '새끼 낳은'
    "나이_새끼": re.compile(r"새끼\s*(?:와|랑|하고)?\s*(?:같이|함께)|새끼\s*(?:포유|수유|낳|출산)|성견"),
}
COMPILED = {k: re.compile(v) for k, v in CATEGORIES.items()}
CLAUSES = re.compile(r"[,;/·\n]+|\.(?!\d)")


def classify(text):
    """메모 → {항목: 0/1}. 구절 단위로 보고, 부정 표현이 바로 뒤에 붙은 경우는 세지 않는다."""
    out = dict.fromkeys(CATEGORIES, 0)
    for clause in CLAUSES.split(text or ""):
        clause = clause.strip()
        if not clause:
            continue
        for k, rx in COMPILED.items():
            if k in EXCLUDE and EXCLUDE[k].search(clause):
                continue
            for m in rx.finditer(clause):
                if k in NEGATABLE and re.match(NEG_AFTER, clause[m.end(): m.end() + 12]):
                    continue
                out[k] = 1
                break
    return out


def classify_series(s):
    uniq = pd.unique(s)
    table = pd.DataFrame([classify(t) for t in uniq], index=uniq)
    return table.loc[s.values].set_index(s.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--max-date", default="20251231")
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--out", default="memo_check")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from train_adoption import prepare
    raw = pd.read_csv(args.csv, dtype=str, encoding="utf-8-sig")
    df = prepare(raw, include_returned=False, max_date=pd.to_datetime(args.max_date, format="%Y%m%d"))
    C = classify_series(df["mark"])
    print("\n항목별 등장 비율과 입양률 (전체 학습 대상)")
    rows = [{"항목": k, "등장": int(C[k].sum()), "비율": C[k].mean(), "입양률": df.loc[C[k] == 1, "y"].mean()} for k in C]
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(f"항목 하나도 없음: {(C.sum(1) == 0).mean():.3f}")

    # 검증용 무작위 표본 (비어 있지 않은 메모)
    pool = df[df["mark"].str.len() >= 2]
    s = pool.sample(args.sample, random_state=7)
    sheet = pd.concat([s[["id", "mark"]].reset_index(drop=True), C.loc[s.index].reset_index(drop=True)], axis=1)
    sheet.to_csv(os.path.join(args.out, "check_sheet.csv"), index=False, encoding="utf-8-sig")
    print(f"\n검증 시트: {args.out}/check_sheet.csv ({len(sheet)}건)")


if __name__ == "__main__":
    main()
