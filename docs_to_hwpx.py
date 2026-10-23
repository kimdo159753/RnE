#!/usr/bin/env python3
"""Claude Docs 편집본(마크다운으로 내보낸 것) → 학교 양식 .hwpx (2026-10-22).

보고서 내용은 Claude Docs 문서 "유기견 입양 예측 R&E 보고서 (편집본)"에서 고치고,
마지막에 그 문서를 마크다운으로 내보내 이 스크립트로 hwpx를 만든다. 서식은 rebuild_report_hwpx.build(학교 양식 복제)를 쓴다.

  python docs_to_hwpx.py report_docs/export.md --out 유기견_예측_최종.hwpx

문서 쓰는 규칙(이 규칙대로 써야 hwpx로 옮겨진다):
  ## 표지 아래 "- 팀코드: / - 제목: 줄1 / 줄2 / - 저자: / - 지도:" 목록, "### 요약" 다음 문단 = 요약, "키워드 :"로 시작하는 줄
  ## Ⅰ.~Ⅴ. = 장, ### 1. = 절, 나머지 문단 = 본문
  "표 N. 제목" 문단 바로 다음 표 = 표, 그림 자리 다음 "[그림 N] 제목" 문단 = 그림 캡션(그림 파일은 report_docs/img/fN.*)
  ## 참고문헌 아래 문단 하나 = 참고문헌 하나
"""
import argparse
import glob
import html
import json
import os
import re

import rebuild_report_hwpx as rb

FIG_W = {"f1": 150, "f2": 150, "f3": 130, "f4": 125}  # mm, 지금 hwpx와 같은 폭
# 표 번호별 열 너비(rel)와 왼쪽 정렬 열(left) — 유기견_예측_보호소포함.hwpx에서 뽑아 둔 것
LAYOUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_docs", "table_layout.json")
LAYOUT = json.load(open(LAYOUT_PATH, encoding="utf-8")) if os.path.exists(LAYOUT_PATH) else {}


def unescape_md(s):
    # Docs 내보내기는 [ ] ~ 등을 \[ \] \~ 와 &#91; 로 감싼다
    s = html.unescape(s)
    return re.sub(r"\\([\\`*_{}\[\]()#+\-.!~|>])", r"\1", s)


def blocks(md):
    """빈 줄로 나눈 덩어리. 표는 한 덩어리."""
    return [b.strip("\n") for b in re.split(r"\n\s*\n", md) if b.strip()]


def table_rows(b):
    lines = [l.strip() for l in b.splitlines() if l.strip().startswith("|")]
    rows = [[c.strip() for c in l.strip("|").split("|")] for l in lines]
    return [r for r in rows if not all(re.fullmatch(r":?-{3,}:?", c) for c in r)]


def parse(md, img_dir):
    c = {"title": [], "body": [], "refs": [], "tables": {}, "figs": {}, "team": "(팀코드)"}
    section, pending_cap, pending_fig = None, None, None
    for b in blocks(unescape_md(md)):
        if b.startswith("# ") or re.match(r"^\w{3} \d+, \d{4} ·", b):  # 문서 제목, 날짜·작성자 줄
            continue
        if b.startswith("## "):
            h = b[3:].strip()
            section = "cover" if h == "표지" else "refs" if h == "참고문헌" else "body"
            if section == "body":
                if c["body"]:
                    c["body"].append(("GAP",))
                c["body"].append(("H5" if h.startswith("Ⅴ") else "H", h))
            continue
        if section == "cover":
            if b.startswith("### 요약"):
                continue
            if b.startswith("- "):
                for line in b.splitlines():
                    k, _, v = line[2:].partition(":")
                    v = v.strip()
                    if k.strip() == "팀코드":
                        c["team"] = v
                    elif k.strip() == "제목":
                        c["title"] = [t.strip() for t in v.split(" / ")]
                    elif k.strip() == "저자":
                        c["authors"] = v
                    elif k.strip() == "지도":
                        c["advisors"] = v
            elif b.startswith("키워드"):
                c["keywords"] = b
            else:
                c["summary"] = (c.get("summary", "") + " " + b).strip()
        elif section == "refs":
            c["refs"].append(" ".join(b.split()))
        elif section == "body":
            if b.startswith("### "):
                c["body"].append(("S", b[4:].strip()))
            elif b.startswith("|"):
                rows = table_rows(b)
                name = f"t{len(c['tables']) + 1}"
                nc = len(rows[0])
                rel = [1 / nc] * nc
                left = [j for j in range(nc) if sum(len(r[j]) for r in rows[1:]) / max(1, len(rows) - 1) > 14]
                lay = LAYOUT.get((pending_cap or "").split(".")[0])
                if lay and len(lay["rel"]) == nc:  # 열 수가 그대로면 지금 hwpx의 열 너비를 쓴다
                    rel, left = lay["rel"], lay["left"]
                c["tables"][name] = (pending_cap or "", rel, left, rows)
                c["body"].append(("T", name))
                pending_cap = None
            elif re.match(r"^표 \d+\.", b):
                pending_cap = b
            elif re.match(r"^!\[|^\[image", b):  # 그림 자리 (내보내기에서는 [image: 그림 N] 로 바뀜)
                m = re.search(r"그림\s*(\d+)", b)
                pending_fig = f"f{m.group(1)}" if m else f"f{len(c['figs']) + 1}"
            elif re.match(r"^\[그림 \d+\]", b):
                k = pending_fig or f"f{re.match(r'^.그림 (\d+)', b).group(1)}"
                src = sorted(glob.glob(os.path.join(img_dir, k + ".*")))
                assert src, f"그림 파일 없음: {img_dir}/{k}.*"
                c["figs"][k] = (os.path.basename(src[0]), FIG_W.get(k, 150), b)
                c["body"].append(("F", k))
                pending_fig = None
            else:
                c["body"].append(("P", " ".join(b.split())))
    assert pending_cap is None and pending_fig is None, "표/그림 캡션 짝이 안 맞음"
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("md")
    ap.add_argument("--out", required=True)
    ap.add_argument("--template", default="2026 R&E 프로그램 보고서_(팀코드).hwpx")
    ap.add_argument("--img-dir", default="report_docs/img")
    args = ap.parse_args()
    c = parse(open(args.md, encoding="utf-8").read(), args.img_dir)
    rb.build(c, args.img_dir, args.template, args.out)


if __name__ == "__main__":
    main()
