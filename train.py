"""End-to-end training on PaySim.

    python scripts/download_data.py
    python train.py --csv data/paysim.csv                 # real 6.36M rows (~20-30 min on 4 cores)
    python train.py --synthetic 200000 --n-iter 3         # tiny smoke test (CI)

Pipeline: load -> leakage-free tabular features -> causal transaction-graph features
-> tuned XGBoost (time-aware CV, PR-AUC) -> F1 threshold on validation PR curve
-> untouched test evaluation + ablation -> reports/, models/, data/sample_transactions.csv
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import shap  # noqa: E402
from scipy.stats import loguniform, randint, uniform  # noqa: E402
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,  # noqa: E402
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from fraudpipe import graph  # noqa: E402
from fraudpipe.features import LEAKY_COLS, TABULAR_FEATURES, build_tabular  # noqa: E402
from fraudpipe.synthetic import generate  # noqa: E402

MODELS, REPORTS, DATA = Path("models"), Path("reports"), Path("data")
DTYPES = dict(step="int16", amount="float32", oldbalanceOrg="float32", newbalanceOrig="float32",
              oldbalanceDest="float32", newbalanceDest="float32", isFraud="int8", isFlaggedFraud="int8")


def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    i = int(np.nanargmax(f1))
    return float(thr[i]), float(f1[i])


def metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    return dict(pr_auc=float(average_precision_score(y, p)), roc_auc=float(roc_auc_score(y, p)),
                precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred)), f1=float(f1_score(y, pred)),
                threshold=float(thr), alert_rate=float(pred.mean()),
                tp=int(((pred == 1) & (y == 1)).sum()), fp=int(((pred == 1) & (y == 0)).sum()),
                fn=int(((pred == 0) & (y == 1)).sum()), tn=int(((pred == 0) & (y == 0)).sum()))


def fit_xgb(X, y, n_iter, seed):
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    kw = dict(tree_method="hist", eval_metric="aucpr", n_jobs=-1, scale_pos_weight=spw ** 0.5,
              random_state=seed)
    space = dict(n_estimators=randint(150, 450), max_depth=randint(3, 9),
                 learning_rate=loguniform(0.03, 0.3), subsample=uniform(0.6, 0.4),
                 colsample_bytree=uniform(0.6, 0.4), min_child_weight=randint(1, 10),
                 gamma=uniform(0, 3), reg_lambda=loguniform(0.1, 10))
    if len(X) > 600_000:  # tune on a time-ordered subsample; refit on everything
        idx = np.sort(np.random.default_rng(seed).choice(len(X), 600_000, replace=False))
        Xs, ys = X.iloc[idx], y[idx]
    else:
        Xs, ys = X, y
    search = RandomizedSearchCV(XGBClassifier(**kw), space, n_iter=n_iter, scoring="average_precision",
                                cv=TimeSeriesSplit(3), random_state=seed, n_jobs=1, refit=False)
    search.fit(Xs, ys)
    final = XGBClassifier(**kw, **search.best_params_).fit(X, y)
    return final, search.best_params_, float(search.best_score_)


def make_reports(y_te, probs, thr, model, X_te, results):
    REPORTS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, p in probs.items():
        pr, rc, _ = precision_recall_curve(y_te, p)
        ax[0].plot(rc, pr, label=f"{name} (AP={results[name]['pr_auc']:.3f})")
    ax[0].set(xlabel="Recall", ylabel="Precision", title="Precision-recall (held-out test)")
    ax[0].legend(loc="lower left")
    names = list(results)
    x = np.arange(3)
    for i, n in enumerate(names):
        ax[1].bar(x + i * 0.38, [results[n]["precision"], results[n]["recall"], results[n]["f1"]], 0.38, label=n)
    ax[1].set_xticks(x + 0.19, ["precision", "recall", "F1"])
    ax[1].set(ylim=(0, 1.05), title="At validation-chosen threshold")
    ax[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(REPORTS / "pr_curve_and_ablation.png", dpi=130)
    plt.close(fig)

    rng = np.random.default_rng(0)
    pos, neg = np.where(y_te == 1)[0], np.where(y_te == 0)[0]
    idx = np.concatenate([rng.choice(pos, min(len(pos), 700), replace=False),
                          rng.choice(neg, 1500, replace=False)])
    xs = X_te.iloc[idx]
    sv = shap.TreeExplainer(model).shap_values(xs)
    plt.figure()
    shap.summary_plot(sv, xs, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(REPORTS / "shap_summary.png", dpi=130, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--csv")
    g.add_argument("--synthetic", type=int, metavar="N_ROWS")
    ap.add_argument("--n-iter", type=int, default=8)
    ap.add_argument("--graph-blocks", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--show-leakage", action="store_true")
    a = ap.parse_args()
    t0 = time.time()

    df = (pd.read_csv(a.csv, dtype=DTYPES) if a.csv else generate(a.synthetic, seed=a.seed))
    df = df.sort_values("step", kind="stable").reset_index(drop=True)
    y = df["isFraud"].to_numpy()
    n = len(df)
    print(f"rows={n:,} frauds={y.sum():,} ({y.mean():.3%})", flush=True)

    # temporal split 70/15/15 -- blocks (5% each) align with the split edges
    i_tr, i_va = int(n * 0.70), int(n * 0.85)
    tr, va, te = slice(0, i_tr), slice(i_tr, i_va), slice(i_va, n)

    tab = build_tabular(df)
    print(f"tabular features done ({time.time()-t0:.0f}s)", flush=True)
    gf, _ = graph.causal_features(df, a.graph_blocks)
    print(f"causal graph features done ({time.time()-t0:.0f}s)", flush=True)
    full = pd.concat([tab, gf], axis=1)
    all_feats = list(full.columns)

    results, probs, models = {}, {}, {}
    for vname, feats in {"tabular_only": TABULAR_FEATURES, "tabular_plus_graph": all_feats}.items():
        print(f"\n== {vname} ({len(feats)} features) ==", flush=True)
        model, params, cv = fit_xgb(full.iloc[tr][feats], y[tr], a.n_iter, a.seed)
        thr, f1v = best_f1_threshold(y[va], model.predict_proba(full.iloc[va][feats])[:, 1])
        pt = model.predict_proba(full.iloc[te][feats])[:, 1]
        m = metrics(y[te], pt, thr)
        m.update(cv_pr_auc=cv, val_f1=f1v, best_params=params)
        results[vname], probs[vname], models[vname] = m, pt, (model, thr, feats)
        print({k: round(v, 4) for k, v in m.items() if isinstance(v, float)}, flush=True)

    if a.show_leakage:  # illustration only: what happens if the leaky columns are kept
        leaky = full.copy()
        for c in LEAKY_COLS:
            leaky[c] = df[c].to_numpy()
        feats = all_feats + LEAKY_COLS
        mdl, _, _ = fit_xgb(leaky.iloc[tr][feats], y[tr], 3, a.seed)
        pt = mdl.predict_proba(leaky.iloc[te][feats])[:, 1]
        results["WITH_LEAKY_COLUMNS_do_not_use"] = dict(pr_auc=float(average_precision_score(y[te], pt)))
        print("leaky PR-AUC (inflated):", results["WITH_LEAKY_COLUMNS_do_not_use"], flush=True)

    # ---- artifacts: production model = tabular + graph ----
    model, thr, feats = models["tabular_plus_graph"]
    MODELS.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    nodes = graph.build_node_table(df)  # history snapshot used at serving time
    deg = nodes["in_deg"] + nodes["out_deg"]
    compact = nodes[deg >= 2].astype("float32")  # unseen / 1-edge accounts -> cold-start zeros
    compact.to_parquet(MODELS / "node_features.parquet", compression="zstd")
    joblib.dump(dict(model=model, features=feats, threshold=thr), MODELS / "model.joblib")

    make_reports(y[te], probs, thr, model, full.iloc[te][feats],
                 {k: results[k] for k in probs})

    # real held-out sample for the dashboard demo (all test frauds up to 400 + 600 legit)
    rng = np.random.default_rng(1)
    te_idx = np.arange(i_va, n)
    fr = te_idx[y[te] == 1]
    lg = te_idx[y[te] == 0]
    pick = np.sort(np.concatenate([rng.choice(fr, min(len(fr), 400), replace=False),
                                   rng.choice(lg[df["type"].to_numpy()[lg] != "PAYMENT"], 600, replace=False)]))
    sample = pd.concat([df.iloc[pick][["step", "type", "amount", "nameOrig", "oldbalanceOrg", "nameDest",
                                       "oldbalanceDest", "isFraud"]],
                        tab.iloc[pick][["orig_tx_count_24h", "orig_time_since_last",
                                        "dest_tx_count_24h", "dest_time_since_last"]]], axis=1)
    sample.to_csv(DATA / "sample_transactions.csv", index=False)

    results["meta"] = dict(rows=int(n), frauds=int(y.sum()), fraud_rate=float(y.mean()),
                           n_train=i_tr, n_val=i_va - i_tr, n_test=n - i_va, test_frauds=int(y[te].sum()),
                           graph_blocks=a.graph_blocks, train_seconds=round(time.time() - t0, 1),
                           source=a.csv or f"synthetic:{a.synthetic}")
    (REPORTS / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nartifacts saved ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
