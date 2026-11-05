"""Train one model on one feature set, tune on val, write row-level scores.

Protocol (agreed at kickoff): fit on train, pick hyperparameters by val PR-AUC,
score test ONCE at the freeze. Test labels are never read here. Test features are
only built when --score-test is passed, so nothing can be tuned on test by accident.

    python -m src.models.train --model {lr,mlp,lgbm,rf} --features {txn,txn_graph}
        [--groups g1,g3]     only these graph groups (R2's ablation)
        [--delay 1]          use the 1-day label-delay graph features
        [--score-test]       also write test scores (freeze only)

Outputs:
    reports/scores/{model}_{feature_set}_{split}.parquet   txn_id, score
    reports/tables/hyperparams.csv                         one row per (model, feature_set), upserted
    reports/tables/tuning/{model}_{feature_set}.csv         val PR-AUC for every grid point
    models/{model}_{feature_set}.joblib                     fitted model(s) + transformers (gitignored)

Imbalance: LR uses class_weight="balanced", LightGBM scale_pos_weight, Random Forest
class_weight="balanced_subsample" (a robustness model: raw graph columns, 300 trees), and the MLP
trains on all train fraud plus 10 legit rows per fraud (same sample for txn and
txn_graph). The MLP is fitted with 5 seeds; its score is the mean of the 5
probabilities, and the grid point is chosen by mean val PR-AUC over the seeds.
"""
import argparse
import json
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.config import LABEL_DELAY, MODEL_DIR, SCORE_DIR, TABLE_DIR
from src.models.data import (FeatureBuilder, feature_set_name, graph_code_hash, load_graph,
                             load_splits, parse_groups)
from src.models.grids import (LGBM_EARLY_STOP, MLP_SEEDS, MODELS, grid_points, make_model,
                              undersample)

HYPERPARAM_FILE = TABLE_DIR / "hyperparams.csv"


# --- fitting ------------------------------------------------------------------

def _fit_lr(params, X_tr, y_tr, X_va, y_va):
    m = make_model("lr", params).fit(X_tr, y_tr)
    return [m], average_precision_score(y_va, m.predict_proba(X_va)[:, 1]), {}


def _fit_mlp(params, X_tr, y_tr, X_va, y_va, train):
    models, aps = [], []
    for s in MLP_SEEDS:
        ids = undersample(train, s)
        m = make_model("mlp", params, seed=s).fit(X_tr.loc[ids], y_tr.loc[ids])
        models.append(m)
        aps.append(average_precision_score(y_va, m.predict_proba(X_va)[:, 1]))
    # score = mean probability over seeds; tune on the mean of the per-seed PR-AUCs
    return models, float(np.mean(aps)), {"val_pr_auc_sd": float(np.std(aps, ddof=1)),
                                          "val_pr_auc_seeds": [round(a, 6) for a in aps]}


def _fit_lgbm(params, X_tr, y_tr, X_va, y_va):
    from lightgbm import early_stopping
    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    m = make_model("lgbm", params, scale_pos_weight=spw)
    m.fit(X_tr, y_tr, eval_X=(X_va,), eval_y=(y_va,),
          callbacks=[early_stopping(LGBM_EARLY_STOP, verbose=False)])
    ap = average_precision_score(y_va, m.predict_proba(X_va)[:, 1])
    return [m], ap, {"best_iteration": int(m.best_iteration_ or m.n_estimators), "scale_pos_weight": spw}


def _fit_rf(params, X_tr, y_tr, X_va, y_va):
    m = make_model("rf", params).fit(X_tr, y_tr)
    return [m], average_precision_score(y_va, m.predict_proba(X_va)[:, 1]), {}


def predict(models, X):
    """Mean positive-class probability over the fitted model(s) (one for LR/LGBM/RF, 5 for MLP)."""
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def tune(model, X_tr, y_tr, X_va, y_va, train, log=print):
    rows, best = [], None
    for params in grid_points(model):
        t0 = time.time()
        if model == "lr":
            models, ap, extra = _fit_lr(params, X_tr, y_tr, X_va, y_va)
        elif model == "mlp":
            models, ap, extra = _fit_mlp(params, X_tr, y_tr, X_va, y_va, train)
        elif model == "rf":
            models, ap, extra = _fit_rf(params, X_tr, y_tr, X_va, y_va)
        else:
            models, ap, extra = _fit_lgbm(params, X_tr, y_tr, X_va, y_va)
        rows.append({"params": json.dumps(params), "val_pr_auc": ap,
                     **{k: v for k, v in extra.items() if not isinstance(v, list)},
                     "seconds": round(time.time() - t0, 1)})
        log(f"  {json.dumps(params):<70} val PR-AUC {ap:.4f}  ({rows[-1]['seconds']}s)")
        if best is None or ap > best["val_pr_auc"]:
            best = {"params": params, "val_pr_auc": ap, "models": models, **extra}
    return best, pd.DataFrame(rows)


# --- outputs ------------------------------------------------------------------

def write_scores(txn_ids, scores, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"txn_id": np.asarray(txn_ids, dtype="int64"),
                        "score": np.asarray(scores, dtype="float32")})
    assert out.txn_id.is_unique
    out.to_parquet(path, index=False)
    return out


def upsert_hyperparams(row, path=HYPERPARAM_FILE):
    """Replace the row for (model, feature_set) if it exists, otherwise append."""
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        old = old[~((old.model == row["model"]) & (old.feature_set == row["feature_set"]))]
        new = pd.concat([old, new], ignore_index=True)
    order = {m: i for i, m in enumerate(MODELS)}
    new = new.sort_values(["feature_set", "model"], key=lambda s: s.map(order) if s.name == "model" else s)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)
    return new


# --- main ---------------------------------------------------------------------

def run(model, features, groups=None, delay=LABEL_DELAY, score_test=False,
        splits=None, graph=None, score_dir=SCORE_DIR, table_dir=TABLE_DIR,
        model_dir=MODEL_DIR, save_model=True, log=print):
    """Train, tune and write outputs. `splits` / `graph` can be passed in (tests use toy data)."""
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {MODELS}")
    t0 = time.time()
    groups = parse_groups(groups)
    fs = feature_set_name(features, groups, delay)
    wanted = ["train", "val"] + (["test"] if score_test else [])
    splits = load_splits(wanted) if splits is None else {k: splits[k] for k in wanted}
    if graph is None:
        graph = load_graph(features, delay)

    train, val = splits["train"], splits["val"]
    builder = FeatureBuilder(features, groups, scale_graph=(model not in ("lgbm", "rf"))).fit(train, graph)
    X_tr, X_va = builder.transform(train, graph), builder.transform(val, graph)
    y_tr = pd.Series(train.fraud.to_numpy(), index=X_tr.index)
    y_va = val.fraud.to_numpy()
    log(f"{model} / {fs}: {X_tr.shape[1]} features, {len(X_tr):,} train rows, {len(X_va):,} val rows")

    best, grid_log = tune(model, X_tr, y_tr, X_va, y_va, train, log=log)

    write_scores(X_va.index, predict(best["models"], X_va), score_dir / f"{model}_{fs}_val.parquet")
    if score_test:
        X_te = builder.transform(splits["test"], graph)
        write_scores(X_te.index, predict(best["models"], X_te), score_dir / f"{model}_{fs}_test.parquet")
        log("  test scores written")
    else:
        log("  test scoring skipped (pass --score-test at the freeze)")

    (table_dir / "tuning").mkdir(parents=True, exist_ok=True)
    grid_log.to_csv(table_dir / "tuning" / f"{model}_{fs}.csv", index=False)
    row = {
        "model": model, "feature_set": fs, "params": json.dumps(best["params"]),
        "val_pr_auc": round(best["val_pr_auc"], 6),
        "val_pr_auc_sd": round(best["val_pr_auc_sd"], 6) if "val_pr_auc_sd" in best else None,
        "best_iteration": best.get("best_iteration"),
        "n_features": X_tr.shape[1],
        "label_delay": delay if features == "txn_graph" else None,
        "graph_code_hash": graph_code_hash() if features == "txn_graph" else None,
        "test_scored": score_test,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    upsert_hyperparams(row, table_dir / "hyperparams.csv")

    if save_model:
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "feature_set": fs, "params": best["params"],
                     "models": best["models"], "builder": builder, "columns": list(X_tr.columns),
                     "label_delay": delay}, model_dir / f"{model}_{fs}.joblib")

    extra = f" (sd {best['val_pr_auc_sd']:.4f} over {len(MLP_SEEDS)} seeds)" if "val_pr_auc_sd" in best else ""
    log(f"  best {json.dumps(best['params'])}: val PR-AUC {best['val_pr_auc']:.4f}{extra}, "
        f"{X_tr.shape[1]} features, {time.time() - t0:.0f}s")
    return row


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--features", required=True, choices=["txn", "txn_graph"])
    p.add_argument("--groups", default=None, help="comma-separated graph groups, e.g. g1,g3")
    p.add_argument("--delay", type=int, default=LABEL_DELAY, help="G5 label delay in days (7 or 1)")
    p.add_argument("--score-test", action="store_true", help="also write test scores (freeze only)")
    a = p.parse_args(argv)
    run(a.model, a.features, a.groups, a.delay, a.score_test)


if __name__ == "__main__":
    main()
