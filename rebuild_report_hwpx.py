#!/usr/bin/env python3
"""다른 사람이 고쳐 양식이 틀어진 보고서(.hwpx)에서 내용(글·표·그림·참고문헌)만 뽑아,
학교 양식 .hwpx의 서식(글꼴·크기·문단 모양·표 테두리·캡션·머리말·여백)으로 다시 조립한다.
조립은 make_report_hwpx.py의 기능을 그대로 쓴다(양식 문단·표·그림 XML 복제).

  python rebuild_report_hwpx.py --template "2026 R&E 프로그램 보고서_(팀코드).hwpx" \
      --source "유기견_예측_A수정.hwpx" --out "유기견_예측_A수정_양식복원.hwpx"

내용 판별 규칙(원본 파일의 글 기준):
  첫 줄 "2026학년도 R&E 프로그램 (팀코드)" → 팀코드 / 그다음 제목 줄들 / "·"가 든 저자 줄 / "지도교사" 줄 /
  "요약" 다음 문단 = 요약 / "키워드"로 시작 = 키워드 / "Ⅰ."~"Ⅴ." = 장 제목 / "1. " 형태 = 절 제목 /
  "표 N." 바로 다음 표 = 표와 캡션 / 그림 바로 다음 "[그림 N]" = 그림 캡션 / "참고문헌" 뒤 "[n]" = 참고문헌 / 나머지 = 본문
"""
import argparse
import os
import re
import tempfile
import zipfile
from html import unescape

from PIL import Image

import make_report_hwpx as mr


def split_paras(sec):
    start = sec.index(">", sec.index("<hs:sec")) + 1
    body = sec[start:sec.rindex("</hs:sec>")]
    tok = re.compile(r"<hp:p\b[^>]*?(/?)>|</hp:p>")
    paras, depth, cur = [], 0, None
    for m in tok.finditer(body):
        if m.group(0).startswith("</"):
            depth -= 1
            if depth == 0:
                paras.append(body[cur:m.end()])
        elif m.group(1) != "/":
            if depth == 0:
                cur = m.start()
            depth += 1
    return paras


def text_of(xml):
    return unescape("".join(re.findall(r"<hp:t>([^<]*)</hp:t>", xml))).strip()


def table_grid(p):
    rows = []
    for tr in re.findall(r"<hp:tr>(.*?)</hp:tr>", p, re.S):
        cells = re.findall(r"<hp:tc\b.*?</hp:tc>", tr, re.S)
        rows.append([" ".join(text_of(par) for par in re.findall(r"<hp:p\b.*?</hp:p>", c, re.S)).strip() for c in cells])
    widths = [int(w) for w in re.findall(r'<hp:cellSz width="(\d+)"', re.search(r"<hp:tr>(.*?)</hp:tr>", p, re.S).group(1))]
    rel = [w / sum(widths) for w in widths]
    nc = len(rel)
    left = [c for c in range(nc) if sum(len(r[c]) for r in rows[1:]) / max(1, len(rows) - 1) > 14]
    return rows, rel, left


def extract(src_path, img_dir):
    z = zipfile.ZipFile(src_path)
    sec = z.read("Contents/section0.xml").decode("utf-8")
    paras = split_paras(sec)
    hpf = z.read("Contents/content.hpf").decode("utf-8")
    bins = dict(re.findall(r'<opf:item id="(\w+)" href="([^"]+)"', hpf))
    c = {"title": [], "body": [], "refs": [], "tables": {}, "figs": {}}
    texts = [text_of(re.sub(r"<hp:(tbl|pic)\b.*?</hp:\1>", "", p, flags=re.S)) for p in paras]
    i, state = 0, "front"
    pending_cap = None
    while i < len(paras):
        p, t = paras[i], texts[i]
        if "<hp:tbl" in p:
            rows, rel, left = table_grid(p)
            name = f"t{len(c['tables']) + 1}"
            c["tables"][name] = (pending_cap or "", rel, left, rows)
            c["body"].append(("T", name)); pending_cap = None
        elif "<hp:pic" in p:
            bid = re.search(r'binaryItemIDRef="(\w+)"', p).group(1)
            w = int(re.search(r'<hp:sz width="(\d+)"', p).group(1))
            cap = texts[i + 1] if i + 1 < len(texts) and texts[i + 1].startswith("[그림") else ""
            fn = os.path.basename(bins[bid])
            open(os.path.join(img_dir, fn), "wb").write(z.read(bins[bid]))
            name = f"f{len(c['figs']) + 1}"
            c["figs"][name] = (fn, round(w / mr.MM), cap)
            c["body"].append(("F", name))
            if cap:
                i += 1
        elif state == "front":
            m = re.match(r"2026학년도 R&E 프로그램\s*\((.+)\)", t)
            if m:
                c["team"] = m.group(1)
            elif t and "·" in t and "지도" not in t and "authors" not in c:
                c["authors"] = t
            elif t.startswith("지도교사"):
                c["advisors"] = t
            elif t == "요약":
                c["summary"] = texts[i + 1]; i += 1
            elif t.startswith("키워드"):
                c["keywords"] = t; state = "body"
            elif t:
                c["title"].append(t)
        elif state == "body":
            if re.match(r"^[ⅠⅡⅢⅣⅤⅥⅦ]\.", t):
                if c["body"]:
                    c["body"].append(("GAP",))
                c["body"].append(("H5" if t.startswith("Ⅴ") else "H", t))
            elif t == "참고문헌":
                state = "refs"
            elif re.match(r"^표 \d+\.", t):
                pending_cap = t
            elif re.match(r"^\d+\.\s", t):
                c["body"].append(("S", t))
            elif t:
                c["body"].append(("P", t))
        else:
            if t:
                c["refs"].append(t)
        i += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fig-scale", type=float, default=1.0, help="그림 폭 배율(쪽수 맞추기용, 내용은 그대로)")
    args = ap.parse_args()
    img_dir = tempfile.mkdtemp(prefix="hwpx_imgs_")
    c = extract(args.source, img_dir)
    c["figs"] = {k: (fn, round(w * args.fig_scale), cap) for k, (fn, w, cap) in c["figs"].items()}

    # make_report_hwpx의 내용 변수를 원본 파일 내용으로 바꿔 끼운 뒤 같은 조립을 쓴다
    mr.TITLE = " ".join(c["title"])
    mr.SUMMARY, mr.KEYWORDS = c["summary"], c["keywords"]
    mr.BODY, mr.TABLES, mr.FIGS, mr.REFS = c["body"], c["tables"], c["figs"], c["refs"]
    mr.FIG_DIR = img_dir

    zin = zipfile.ZipFile(args.template)
    sec = zin.read("Contents/section0.xml").decode("utf-8")
    start = sec.index(">", sec.index("<hs:sec")) + 1
    end = sec.rindex("</hs:sec>")
    parts = split_paras(sec)
    assert len(parts) == 55, f"양식 문단 수가 예상과 다름: {len(parts)}"
    out, bins = mr.build_section(parts)
    xml = "".join(out)
    # 팀코드·저자·지도교사
    xml = xml.replace("<hp:t>(팀코드)</hp:t>", f"<hp:t>{mr.escape(c.get('team', '(팀코드)'))}</hp:t>")
    if "authors" in c:
        xml = xml.replace("<hp:t>저자명 · 저자명 · 저자명 · 저자명</hp:t>", f"<hp:t>{mr.escape(c['authors'])}</hp:t>")
    if "advisors" in c:
        xml = xml.replace("<hp:t>지도교사 OOO · 지도교수 OOO</hp:t>", f"<hp:t>{mr.escape(c['advisors'])}</hp:t>")
    # 원본처럼 제목을 두 줄로: 양식의 제목 문단을 줄 수만큼 복제
    if len(c["title"]) > 1:
        t_esc = mr.escape(mr.TITLE)
        m = re.search(r"<hp:p\b(?:(?!<hp:p\b).)*?<hp:t>" + re.escape(t_esc) + r"</hp:t>.*?</hp:p>", xml, re.S)
        one = m.group(0)
        xml = xml.replace(one, "".join(one.replace(f"<hp:t>{t_esc}</hp:t>", f"<hp:t>{mr.escape(line)}</hp:t>") for line in c["title"]))
    new_sec = sec[:start] + xml + sec[end:]

    hpf = zin.read("Contents/content.hpf").decode("utf-8")
    items = ""
    for name, bid in bins.items():
        fn = c["figs"][name][0]
        mt = "image/jpg" if fn.lower().endswith((".jpg", ".jpeg")) else "image/png"
        items += f'<opf:item id="{bid}" href="BinData/{bid}{os.path.splitext(fn)[1]}" media-type="{mt}" isEmbeded="1"/>'
    hpf = hpf.replace('<opf:item id="section0"', items + '<opf:item id="section0"')
    hpf = re.sub(r'<opf:title xml:space="preserve">[^<]*</opf:title>', f'<opf:title xml:space="preserve">{mr.escape(mr.TITLE)}</opf:title>', hpf)
    with zipfile.ZipFile(args.out, "w") as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename == "Contents/section0.xml":
                data = new_sec.encode("utf-8")
            elif info.filename == "Contents/content.hpf":
                data = hpf.encode("utf-8")
            elif info.filename == "Preview/PrvText.txt":
                data = "\r\n".join([mr.TITLE, "", "요약", mr.SUMMARY])[:1000].encode("utf-8")
            comp = zipfile.ZIP_STORED if info.filename == "mimetype" else zipfile.ZIP_DEFLATED
            zout.writestr(info.filename, data, compress_type=comp)
        for name, bid in bins.items():
            fn = c["figs"][name][0]
            zout.write(os.path.join(img_dir, fn), f"BinData/{bid}{os.path.splitext(fn)[1]}", compress_type=zipfile.ZIP_DEFLATED)
    print("saved", args.out)
    print("팀코드", c.get("team"), "| 제목", c["title"], "| 본문 문단", sum(b[0] == "P" for b in c["body"]),
          "| 표", len(c["tables"]), "| 그림", len(c["figs"]), "| 참고문헌", len(c["refs"]),
          f"| 추정 쪽수 {mr.estimate_pages():.2f}")


if __name__ == "__main__":
    main()
