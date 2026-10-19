#!/usr/bin/env python3
"""특징 텍스트 표현 비교: 없음 / TF-IDF(형태소) / 문장 임베딩 / 둘 다

판정 규칙 (시험 구간을 고르는 데 쓰지 않기 위해)
  - 시간순: 학습 = 앞 80% 중 앞 85%, 검증 = 앞 80% 중 뒤 15%, 시험 = 뒤 20%
  - 후보 선택은 검증 구간 PR-AUC로만 한다. 시험 구간 성능은 선택이 끝난 뒤 한 번만 보고한다.
  - 로지스틱(빠름, 전 조합)과 CatBoost(최종 모델 형태, 주요 조합)를 모두 본다.

  python compare_text_repr.py --csv abandoned_dogs.csv --emb emb_cache/ko_sroberta.npz emb_cache/e5_base.npz
"""
import argparse
import os

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.sparse import csr_matrix, hstack
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from text_analysis import tokenize_all
from train_adoption import make_catboost, prepare

CAT = ["breed", "color", "sex", "neuter", "sido"]
NUM = ["age_years", "under_60d", "weight_kg", "month", "is_mix", "mark_len"]


def score(y, p):
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p), "brier": brier_score_loss(y, p)}


def boot_diff(y, p_new, p_base, groups, n_boot=1000, seed=0):
    """보호소 단위 클러스터 부트스트랩으로 (새 모델 - 기준) PR-AUC·ROC-AUC 차이의 평균과 95% 구간.
    같은 보호소 개체끼리 닮아 있어서 개체 단위로 뽑으면 구간이 지나치게 좁아진다."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(groups)
    members = [np.flatnonzero(codes == g) for g in range(len(uniq))]
    d_pr, d_roc = [], []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        d_pr.append(average_precision_score(yy, p_new[idx]) - average_precision_score(yy, p_base[idx]))
        d_roc.append(roc_auc_score(yy, p_new[idx]) - roc_auc_score(yy, p_base[idx]))
    q = lambda a: (np.mean(a), *np.percentile(a, [2.5, 97.5]))
    return q(d_pr), q(d_roc)


def report_diffs(title, preds, base, y, groups, split):
    """preds: {이름: 예측}. base 대비 차이와 구간을 출력하고 표로 돌려준다."""
    print(f"\n[{title}] {split} 구간, 기준 = {base} (보호소 클러스터 부트스트랩 1000회)")
    rows = []
    for name, p in preds.items():
        if name == base:
            continue
        (pm, pl, ph), (rm, rl, rh) = boot_diff(y, p, preds[base], groups)
        sure = "확실히 나음" if pl > 0 else ("확실히 못함" if ph < 0 else "차이 불확실")
        print(f"  {name:<26} ΔPR {pm:+.4f} [{pl:+.4f}, {ph:+.4f}]  ΔROC {rm:+.4f} [{rl:+.4f}, {rh:+.4f}]  → {sure}")
        rows.append({"title": title, "split": split, "base": base, "model": name, "dPR": pm, "dPR_lo": pl, "dPR_hi": ph,
                     "dROC": rm, "dROC_lo": rl, "dROC_hi": rh, "verdict": sure})
    return rows


def load_emb(path, marks):
    z = np.load(path, allow_pickle=True)
    idx = {t: i for i, t in enumerate(z["texts"])}
    missing = sum(m not in idx for m in pd.unique(marks))
    if missing:
        raise SystemExit(f"{path}: 캐시에 없는 문장 {missing}개 — embed_marks.py를 같은 --max-date로 다시 돌리세요")
    return z["emb"][[idx[m] for m in marks]].astype(np.float32), str(z["model"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--max-date", default="20251231")
    ap.add_argument("--emb", nargs="+", required=True)
    ap.add_argument("--pca", type=int, default=32, help="CatBoost에 넣을 임베딩 주성분 수")
    ap.add_argument("--lr-pca", type=int, default=128, help="로지스틱에 넣을 임베딩 주성분 수")
    ap.add_argument("--stride", type=int, default=1, help="시험 실행용: N개마다 1개만 사용")
    ap.add_argument("--out", default="text_repr_results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    raw = pd.read_csv(args.csv, dtype=str, encoding="utf-8-sig")
    df = prepare(raw, include_returned=False, max_date=pd.to_datetime(args.max_date, format="%Y%m%d"))
    df = df.iloc[:: args.stride].reset_index(drop=True)
    n = len(df)
    n_test = int(n * 0.2)
    n_trall = n - n_test
    n_val = int(n_trall * 0.15)
    i_tr, i_va, i_te = np.arange(n_trall - n_val), np.arange(n_trall - n_val, n_trall), np.arange(n_trall, n)
    i_trall = np.arange(n_trall)
    y = df["y"].values
    print(f"학습 {len(i_tr)} / 검증 {len(i_va)} / 시험 {len(i_te)}")

    print("형태소 분석 중...", flush=True)
    df["tokens"] = tokenize_all(df["mark"])
    embs = {}
    for p in args.emb:
        e, name = load_emb(p, df["mark"])
        embs[name.split("/")[-1]] = e
        print(f"임베딩 {name}: {e.shape}")

    # ---------- 로지스틱: 모든 조합 ----------
    def blocks(fit_idx):
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), CAT),
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUM),
        ])
        sh = OneHotEncoder(handle_unknown="ignore", min_frequency=10)
        tf = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, min_df=20, sublinear_tf=True)
        fit = df.iloc[fit_idx]
        pre.fit(fit), sh.fit(fit[["care"]]), tf.fit(fit["tokens"])
        out = {"S": pre.transform(df), "H": sh.transform(df[["care"]]), "T": tf.transform(df["tokens"])}
        for k, e in embs.items():
            # 768차원 그대로는 메모리가 부족해 학습 구간에서 맞춘 주성분으로 줄인다
            pc = make_pipeline(PCA(args.lr_pca, random_state=42), StandardScaler()).fit(e[fit_idx])
            out["E:" + k] = csr_matrix(pc.transform(e).astype(np.float32))
        return out

    combos = []
    for shelter in ("", "H"):
        for text in ["", "T"] + [f"E:{k}" for k in embs] + [f"T+E:{k}" for k in embs]:
            combos.append("+".join(x for x in ["S", shelter, text] if x))

    def build(B, combo):
        return hstack([B[c] for c in combo.split("+")]).tocsr()

    print("\n===== 로지스틱 회귀: 학습→검증 (선택용) =====", flush=True)
    B = blocks(i_tr)
    rows = []
    val_pred = {}
    for combo in combos:
        X = build(B, combo)
        best = None
        for C in (0.1, 0.3, 1.0):
            m = LogisticRegression(C=C, max_iter=3000, solver="liblinear").fit(X[i_tr], y[i_tr])
            p = m.predict_proba(X[i_va])[:, 1]
            s = score(y[i_va], p)
            rows.append({"combo": combo, "C": C, **{f"val_{k}": v for k, v in s.items()}})
            if best is None or s["pr_auc"] > best[0]:
                best = (s["pr_auc"], C, s["roc_auc"], p)
        val_pred[combo] = best[3]
        print(f"  {combo:<26} C={best[1]:<4} 검증 ROC {best[2]:.4f} PR {best[0]:.4f}", flush=True)
    val = pd.DataFrame(rows)
    best_c = val.loc[val.groupby("combo")["val_pr_auc"].idxmax()].set_index("combo")

    print("\n===== 로지스틱 회귀: 학습+검증으로 다시 맞춘 뒤 시험 (보고용, 조합별 C는 검증에서 고정) =====", flush=True)
    B = blocks(i_trall)
    test_rows, test_pred = [], {}
    for combo in combos:
        X = build(B, combo)
        C = best_c.loc[combo, "C"]
        m = LogisticRegression(C=C, max_iter=3000, solver="liblinear").fit(X[i_trall], y[i_trall])
        test_pred[combo] = m.predict_proba(X[i_te])[:, 1]
        s = score(y[i_te], test_pred[combo])
        test_rows.append({"combo": combo, "C": C, "val_pr_auc": best_c.loc[combo, "val_pr_auc"],
                          "val_roc_auc": best_c.loc[combo, "val_roc_auc"], **{f"test_{k}": v for k, v in s.items()}})
    del B
    lr = pd.DataFrame(test_rows)
    lr.to_csv(os.path.join(args.out, "logistic_compare.csv"), index=False, encoding="utf-8-sig")
    print(lr.round(4).to_string(index=False))

    g_va, g_te = df["care"].values[i_va], df["care"].values[i_te]
    diffs = []
    for base in ("S+T", "S+H+T"):
        keep = lambda d: {k: v for k, v in d.items() if ("H" in k.split("+")) == ("H" in base.split("+"))}
        diffs += report_diffs(f"로지스틱 {base} 기준", keep(val_pred), base, y[i_va], g_va, "검증")
        diffs += report_diffs(f"로지스틱 {base} 기준", keep(test_pred), base, y[i_te], g_te, "시험")

    # ---------- CatBoost: 텍스트 없음 / text_features / 임베딩 PCA / 둘 다 (임베딩 모델마다) ----------
    emb_cols = {}
    P_all = []
    for k, e in embs.items():
        pca = PCA(args.pca, random_state=42).fit(e[i_tr])
        cols = [f"{k}_pc{i}" for i in range(args.pca)]
        P_all.append(pd.DataFrame(pca.transform(e), columns=cols))
        emb_cols[k] = cols
        print(f"\nCatBoost용 {k} PCA {args.pca}차원 설명 분산 {pca.explained_variance_ratio_.sum():.3f}")
    P = pd.concat(P_all, axis=1)

    variants = {"none": ([], []), "text": (["mark"], [])}
    for k, cols in emb_cols.items():
        variants[f"emb:{k}"] = ([], cols)
        variants[f"text+emb:{k}"] = (["mark"], cols)
    cb_rows, cb_val, cb_test = [], {False: {}, True: {}}, {False: {}, True: {}}
    for shelter in (False, True):
        cat = CAT + (["care"] if shelter else [])
        for vname, (text, embcols) in variants.items():
            X = pd.concat([df[cat + NUM + text].reset_index(drop=True), P[embcols]], axis=1)
            for c in cat + text:
                X[c] = X[c].astype(str).astype(object)

            def pool(idx):
                return Pool(X.iloc[idx], y[idx], cat_features=cat, text_features=text or None)

            m = make_catboost(cat, text)
            m.fit(pool(i_tr), eval_set=pool(i_va), use_best_model=True)
            pv, pt = m.predict_proba(pool(i_va))[:, 1], m.predict_proba(pool(i_te))[:, 1]
            cb_val[shelter][vname], cb_test[shelter][vname] = pv, pt
            sv, st = score(y[i_va], pv), score(y[i_te], pt)
            cb_rows.append({"shelter": shelter, "text": vname, "best_iter": m.get_best_iteration(),
                            **{f"val_{k}": v for k, v in sv.items()}, **{f"test_{k}": v for k, v in st.items()}})
            print(f"  CatBoost shelter={shelter!s:<5} {vname:<28} 검증 PR {sv['pr_auc']:.4f} ROC {sv['roc_auc']:.4f} | "
                  f"시험 PR {st['pr_auc']:.4f} ROC {st['roc_auc']:.4f} Brier {st['brier']:.4f}", flush=True)
    cb = pd.DataFrame(cb_rows)
    cb.to_csv(os.path.join(args.out, "catboost_compare.csv"), index=False, encoding="utf-8-sig")
    print("\n===== CatBoost 요약 =====")
    print(cb.round(4).to_string(index=False))
    for shelter in (False, True):
        tag = f"CatBoost 보호소 {'포함' if shelter else '제외'}"
        diffs += report_diffs(tag, cb_val[shelter], "text", y[i_va], g_va, "검증")
        diffs += report_diffs(tag, cb_test[shelter], "text", y[i_te], g_te, "시험")
    pd.DataFrame(diffs).to_csv(os.path.join(args.out, "bootstrap_diffs.csv"), index=False, encoding="utf-8-sig")

    print("\n===== 채택 판정: 검증에서 '확실히 나음'이고 시험에서도 ΔPR > 0인 것만 =====")
    d = pd.DataFrame(diffs)
    ok = []
    w = d.pivot_table(index=["title", "model"], columns="split", values=["dPR", "dPR_lo", "dPR_hi"])
    for (title, model), r in w.iterrows():
        if r[("dPR_lo", "검증")] > 0 and r[("dPR", "시험")] > 0:
            ok.append(f"{title} / {model}: 검증 ΔPR {r[('dPR', '검증')]:+.4f}, 시험 ΔPR {r[('dPR', '시험')]:+.4f} "
                      f"[{r[('dPR_lo', '시험')]:+.4f}, {r[('dPR_hi', '시험')]:+.4f}]")
    print("\n".join(ok) if ok else "채택할 것 없음 — 현재 방식(TF-IDF/CatBoost text_features) 유지")
    print(f"\n결과 파일: {args.out}/")


if __name__ == "__main__":
    main()
