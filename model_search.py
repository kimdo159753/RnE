#!/usr/bin/env python3
"""모델 계열 비교: 로지스틱 / LightGBM / XGBoost / CatBoost / MLP (+ 단순 평균 앙상블)

공정성 규칙
  - 같은 시간순 분할: 학습 = 앞 80% 중 앞 85%, 검증 = 앞 80% 중 뒤 15%, 시험 = 뒤 20%
  - 같은 정보: 범주형(품종·색·성별·중성화·시도[·보호소]), 숫자형, 특징 텍스트, 문장 임베딩(e5)
    표현만 모델에 맞게 바꾼다. 학습 구간에서만 맞춘다(인코더, PCA, 텍스트 점수).
  - 계열마다 설정 조합을 정해진 수만큼 무작위로 시도하고, 조기 종료와 선택은 검증 구간으로만 한다.
  - 시험 구간은 계열별 최종 1개씩만 보고한다. 1등이 확실한지는 보호소 클러스터 부트스트랩으로 본다.
  - 계열별 결과를 out/에 저장하고, 다시 실행하면 끝난 계열은 건너뛴다.

  python model_search.py --csv abandoned_dogs.csv --emb emb_cache/e5_base_v2.npz --out model_search
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from compare_text_repr import boot_diff
from text_analysis import tokenize_all
from train_adoption import load_emb, prepare

CAT = ["breed", "color", "sex", "neuter", "sido"]
NUM = ["age_years", "under_60d", "weight_kg", "month", "is_mix", "mark_len"]
FAMILIES = ["logistic", "lightgbm", "xgboost", "catboost", "mlp"]


def score(y, p):
    return {"pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p), "brier": brier_score_loss(y, p)}


def sample_configs(space, n, seed):
    rng = np.random.default_rng(seed)
    return [{k: v[rng.integers(len(v))] for k, v in space.items()} for _ in range(n)]


class Data:
    """모든 계열이 같이 쓰는 입력. 학습 구간(i_tr)에서만 맞춘다."""

    def __init__(self, args):
        raw = pd.read_csv(args.csv, dtype=str, encoding="utf-8-sig")
        df = prepare(raw, include_returned=False, max_date=pd.to_datetime(args.max_date, format="%Y%m%d"))
        df = df.iloc[:: args.stride].reset_index(drop=True)
        n = len(df)
        n_test = int(n * 0.2)
        n_trall = n - n_test
        n_val = int(n_trall * 0.15)
        self.i_tr = np.arange(n_trall - n_val)
        self.i_va = np.arange(n_trall - n_val, n_trall)
        self.i_te = np.arange(n_trall, n)
        self.df, self.y = df, df["y"].values
        print(f"학습 {len(self.i_tr)} / 검증 {len(self.i_va)} / 시험 {len(self.i_te)}", flush=True)

        print("형태소 분석...", flush=True)
        df["tokens"] = tokenize_all(df["mark"])
        self.tfidf = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, min_df=20, sublinear_tf=True)
        self.tfidf.fit(df["tokens"].iloc[self.i_tr])
        self.T = self.tfidf.transform(df["tokens"])

        self.E = load_emb(args.emb, df["mark"])  # 768차원 원본 (MLP용)
        self.pca32 = PCA(32, random_state=42).fit(self.E[self.i_tr])
        self.P32 = self.pca32.transform(self.E).astype(np.float32)
        pca128 = PCA(args.lr_pca, random_state=42).fit(self.E[self.i_tr])
        self.P128 = StandardScaler().fit(pca128.transform(self.E[self.i_tr])).transform(pca128.transform(self.E))

        # 텍스트 점수(트리 모델용 스태킹 입력): TF-IDF 로지스틱. 학습 구간은 5-fold 바깥 예측, 나머지는 학습 구간 전체로 맞춘 예측
        print("텍스트 점수(OOF)...", flush=True)
        txt = np.zeros(n)
        for a, b in KFold(5, shuffle=True, random_state=0).split(self.i_tr):
            m = LogisticRegression(C=0.3, max_iter=3000, solver="liblinear").fit(self.T[self.i_tr[a]], self.y[self.i_tr[a]])
            txt[self.i_tr[b]] = m.predict_proba(self.T[self.i_tr[b]])[:, 1]
        m = LogisticRegression(C=0.3, max_iter=3000, solver="liblinear").fit(self.T[self.i_tr], self.y[self.i_tr])
        self.txt_model = m  # 새 개체 예측 때 같은 텍스트 점수를 만들려고 남겨 둔다
        rest = np.concatenate([self.i_va, self.i_te])
        txt[rest] = m.predict_proba(self.T[rest])[:, 1]
        self.txt = txt.astype(np.float32)

    def cats(self, shelter):
        return CAT + (["care"] if shelter else [])

    def tree_frame(self, shelter):
        """범주형은 학습 구간에 나온 값만 범주로 두고(나머지는 결측), 숫자 + 임베딩 PCA32 + 텍스트 점수."""
        X = pd.DataFrame(index=self.df.index)
        for c in self.cats(shelter):
            cats = pd.Index(self.df[c].iloc[self.i_tr].unique())
            X[c] = pd.Categorical(self.df[c], categories=cats)
        for c in NUM:
            X[c] = self.df[c].astype(np.float32)
        for i in range(32):
            X[f"emb{i}"] = self.P32[:, i]
        X["txt_lr"] = self.txt
        return X


# ---------------- 계열별 학습 함수: (검증 예측, 시험 예측, 정보) 반환 ----------------

def fit_logistic(D, shelter, cfg):
    oh = OneHotEncoder(handle_unknown="ignore", min_frequency=10).fit(D.df[D.cats(shelter)].iloc[D.i_tr])
    num = D.df[NUM].astype(float)
    imp = SimpleImputer(strategy="median").fit(num.iloc[D.i_tr])
    sc = StandardScaler().fit(imp.transform(num.iloc[D.i_tr]))
    X = hstack([oh.transform(D.df[D.cats(shelter)]), csr_matrix(sc.transform(imp.transform(num))), D.T,
                csr_matrix(D.P128.astype(np.float32))]).tocsr()
    m = LogisticRegression(C=cfg["C"], max_iter=3000, solver="liblinear").fit(X[D.i_tr], D.y[D.i_tr])
    return m.predict_proba(X[D.i_va])[:, 1], m.predict_proba(X[D.i_te])[:, 1], {}


def fit_lightgbm(D, shelter, cfg):
    import lightgbm as lgb
    X = D.tree_frame(shelter)
    m = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.03, verbose=-1, n_jobs=-1, random_state=42, **cfg)
    m.fit(X.iloc[D.i_tr], D.y[D.i_tr], eval_set=[(X.iloc[D.i_va], D.y[D.i_va])],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return m.predict_proba(X.iloc[D.i_va])[:, 1], m.predict_proba(X.iloc[D.i_te])[:, 1], {"iters": m.best_iteration_}


def fit_xgboost(D, shelter, cfg):
    import xgboost as xgb
    X = D.tree_frame(shelter)
    m = xgb.XGBClassifier(n_estimators=5000, learning_rate=0.03, tree_method="hist", enable_categorical=True,
                          early_stopping_rounds=100, eval_metric="logloss", n_jobs=-1, random_state=42, **cfg)
    m.fit(X.iloc[D.i_tr], D.y[D.i_tr], eval_set=[(X.iloc[D.i_va], D.y[D.i_va])], verbose=False)
    return m.predict_proba(X.iloc[D.i_va])[:, 1], m.predict_proba(X.iloc[D.i_te])[:, 1], {"iters": m.best_iteration}


def fit_catboost(D, shelter, cfg, return_model=False):
    from catboost import CatBoostClassifier, Pool
    cat = D.cats(shelter)
    X = D.tree_frame(shelter)
    for c in cat:
        X[c] = X[c].astype(str).astype(object)  # CatBoost는 문자열 범주를 직접 받는다
    text = []
    if cfg["text_features"]:  # CatBoost 자체 텍스트 처리 (kiwi 형태소 토큰 입력)
        X["tokens"] = D.df["tokens"].astype(str).astype(object)
        text = ["tokens"]
    cfg = {k: v for k, v in cfg.items() if k != "text_features"}
    pool = lambda idx: Pool(X.iloc[idx], D.y[idx], cat_features=cat, text_features=text or None)
    m = CatBoostClassifier(iterations=5000, learning_rate=0.06, loss_function="Logloss", eval_metric="Logloss",
                           od_type="Iter", od_wait=100, random_seed=42, verbose=0, thread_count=-1, **cfg)
    m.fit(pool(D.i_tr), eval_set=pool(D.i_va), use_best_model=True)
    out = (m.predict_proba(pool(D.i_va))[:, 1], m.predict_proba(pool(D.i_te))[:, 1], {"iters": m.get_best_iteration()})
    return out + (m,) if return_model else out


def fit_mlp(D, shelter, cfg):
    """범주형 임베딩 + 숫자형 + e5 768차원 원본 + 텍스트 점수 → MLP. 검증 PR-AUC로 조기 종료."""
    import torch
    import torch.nn as nn
    torch.manual_seed(42)
    cat = D.cats(shelter)
    codes, cards = [], []
    for c in cat:
        vals = pd.Index(D.df[c].iloc[D.i_tr].value_counts().loc[lambda s: s >= 5].index)
        codes.append(vals.get_indexer(D.df[c]) + 1)  # 0 = 드물거나 처음 보는 값
        cards.append(len(vals) + 1)
    C = torch.tensor(np.stack(codes, 1), dtype=torch.long)
    num = D.df[NUM].astype(float)
    miss = num.isna().astype(np.float32).values
    imp = SimpleImputer(strategy="median").fit(num.iloc[D.i_tr])
    sc = StandardScaler().fit(imp.transform(num.iloc[D.i_tr]))
    Xn = np.hstack([sc.transform(imp.transform(num)), miss, D.E, D.txt[:, None]]).astype(np.float32)
    Xn = torch.tensor(Xn)
    y = torch.tensor(D.y, dtype=torch.float32)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.embs = nn.ModuleList([nn.Embedding(k, min(32, (k + 1) // 2)) for k in cards])
            d_in = sum(e.embedding_dim for e in self.embs) + Xn.shape[1]
            h = cfg["hidden"]
            self.net = nn.Sequential(nn.Linear(d_in, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(cfg["dropout"]),
                                     nn.Linear(h, h // 2), nn.BatchNorm1d(h // 2), nn.ReLU(), nn.Dropout(cfg["dropout"]),
                                     nn.Linear(h // 2, 1))

        def forward(self, c, x):
            return self.net(torch.cat([e(c[:, i]) for i, e in enumerate(self.embs)] + [x], 1)).squeeze(1)

    net = Net()
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    lossf = nn.BCEWithLogitsLoss()

    def predict(idx):
        net.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(idx), 8192):
                b = idx[s: s + 8192]
                out.append(torch.sigmoid(net(C[b], Xn[b])).numpy())
        return np.concatenate(out)

    best, best_state, bad, ep = -1, None, 0, 0
    rng = np.random.default_rng(0)
    for ep in range(40):
        net.train()
        perm = rng.permutation(D.i_tr)
        for s in range(0, len(perm), 1024):
            b = perm[s: s + 1024]
            opt.zero_grad()
            lossf(net(C[b], Xn[b]), y[b]).backward()
            opt.step()
        pr = average_precision_score(D.y[D.i_va], predict(D.i_va))
        if pr > best:
            best, bad = pr, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 4:
                break
    net.load_state_dict(best_state)
    return predict(D.i_va), predict(D.i_te), {"epochs": ep + 1}


SPACES = {
    "logistic": {"C": [0.03, 0.1, 0.3, 1.0]},
    "lightgbm": {"num_leaves": [31, 63, 127, 255], "min_child_samples": [20, 100, 300], "colsample_bytree": [0.6, 0.8, 1.0],
                 "subsample": [0.7, 0.9, 1.0], "subsample_freq": [1], "reg_lambda": [0.0, 1.0, 10.0],
                 "cat_smooth": [10, 50], "min_data_per_group": [50, 200], "max_cat_to_onehot": [4]},
    "xgboost": {"max_depth": [4, 6, 8, 10], "min_child_weight": [1, 5, 20], "subsample": [0.7, 0.9, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0], "reg_lambda": [1.0, 10.0], "max_cat_to_onehot": [4]},
    "catboost": {"depth": [6, 8], "l2_leaf_reg": [3, 10, 30], "text_features": [True, False]},
    "mlp": {"hidden": [256, 512], "dropout": [0.2, 0.4], "lr": [1e-3, 3e-3], "wd": [1e-5, 1e-4]},
}
FIT = {"logistic": fit_logistic, "lightgbm": fit_lightgbm, "xgboost": fit_xgboost, "catboost": fit_catboost, "mlp": fit_mlp}


def run_family(D, fam, shelter, n_cfg, out):
    path = os.path.join(out, f"{fam}_{'with' if shelter else 'without'}_shelter.pkl")
    if os.path.exists(path):
        print(f"  [{fam} shelter={shelter}] 이미 있음 → 건너뜀", flush=True)
        return pickle.load(open(path, "rb"))
    space = SPACES[fam]
    if fam == "logistic":
        cfgs = [{"C": c} for c in space["C"]]
    elif fam == "catboost":
        cfgs = sample_configs(space, n_cfg, seed=1)
    else:
        cfgs = sample_configs(space, n_cfg, seed=1)
    cfgs = [dict(t) for t in {tuple(sorted(c.items())) for c in cfgs}]  # 중복 제거
    trials, best = [], None
    for cfg in cfgs:
        t0 = time.time()
        pv, pt, info = FIT[fam](D, shelter, cfg)
        sv = score(D.y[D.i_va], pv)
        trials.append({"cfg": cfg, "val": sv, "sec": time.time() - t0, **info})
        print(f"  [{fam} shelter={shelter}] {cfg} → 검증 PR {sv['pr_auc']:.4f} ROC {sv['roc_auc']:.4f} "
              f"({time.time() - t0:.0f}초, {info})", flush=True)
        if best is None or sv["pr_auc"] > best["val"]["pr_auc"]:
            best = {"cfg": cfg, "val": sv, "pv": pv, "pt": pt, "sec": time.time() - t0, **info}
    res = {"family": fam, "shelter": shelter, "best": best, "trials": trials}
    pickle.dump(res, open(path, "wb"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--max-date", default="20251231")
    ap.add_argument("--out", default="model_search")
    ap.add_argument("--n-cfg", type=int, default=8, help="계열별 무작위 설정 조합 수 (로지스틱은 C 4개 고정)")
    ap.add_argument("--lr-pca", type=int, default=128)
    ap.add_argument("--stride", type=int, default=1, help="시험 실행용")
    ap.add_argument("--families", nargs="+", default=FAMILIES)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    D = Data(args)
    g_va, g_te = D.df["care"].values[D.i_va], D.df["care"].values[D.i_te]
    summary = []
    for shelter in (False, True):
        tag = "보호소 포함" if shelter else "보호소 제외"
        print(f"\n========== {tag} ==========", flush=True)
        res = {fam: run_family(D, fam, shelter, args.n_cfg, args.out) for fam in args.families}
        pv = {f: r["best"]["pv"] for f, r in res.items()}
        pt = {f: r["best"]["pt"] for f, r in res.items()}
        # 단순 평균 앙상블: 전체 / 검증 상위 3개 계열
        order = sorted(pv, key=lambda f: -average_precision_score(D.y[D.i_va], pv[f]))
        for name, members in [("ens_all", order), ("ens_top3", order[:3])]:
            pv[name] = np.mean([pv[f] for f in members], 0)
            pt[name] = np.mean([pt[f] for f in members], 0)
        rows = []
        for f in pv:
            sv, st = score(D.y[D.i_va], pv[f]), score(D.y[D.i_te], pt[f])
            r = res.get(f, {}).get("best", {})
            rows.append({"setting": tag, "model": f, "cfg": json.dumps(r.get("cfg", {"members": order if f == "ens_all" else order[:3]}),
                         ensure_ascii=False), **{f"val_{k}": v for k, v in sv.items()},
                         **{f"test_{k}": v for k, v in st.items()}, "fit_sec": r.get("sec")})
        tab = pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False)
        print(f"\n[{tag}] 계열별 최종 (검증 PR-AUC 순)")
        print(tab.drop(columns=["cfg", "setting"]).round(4).to_string(index=False))
        winner = tab.iloc[0]["model"]
        print(f"\n[{tag}] 검증 1등 = {winner}. 나머지와의 차이 (1등 - 상대, 보호소 클러스터 부트스트랩 1000회)")
        for f in tab["model"].iloc[1:]:
            (vm, vl, vh), _ = boot_diff(D.y[D.i_va], pv[winner], pv[f], g_va)
            (tm, tl, th), _ = boot_diff(D.y[D.i_te], pt[winner], pt[f], g_te)
            sure = "확실히 나음" if vl > 0 else "차이 불확실"
            print(f"  vs {f:<10} 검증 ΔPR {vm:+.4f} [{vl:+.4f}, {vh:+.4f}] {sure:<7} | 시험 ΔPR {tm:+.4f} [{tl:+.4f}, {th:+.4f}]")
            tab.loc[tab.model == f, ["val_dPR_vs_winner", "val_dPR_lo", "test_dPR_vs_winner"]] = [-vm, -vh, -tm]
        summary.append(tab)
        np.savez(os.path.join(args.out, f"preds_{'with' if shelter else 'without'}_shelter.npz"),
                 **{f"val_{k}": v for k, v in pv.items()}, **{f"test_{k}": v for k, v in pt.items()},
                 y_val=D.y[D.i_va], y_test=D.y[D.i_te])
    pd.concat(summary).to_csv(os.path.join(args.out, "summary.csv"), index=False, encoding="utf-8-sig")
    print(f"\n결과 파일: {args.out}/")


if __name__ == "__main__":
    main()
