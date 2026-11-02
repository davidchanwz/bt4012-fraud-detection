"""Evaluation harness (R4): reads score files, writes the results tables. Never retrains.

Usage:
    uv run python -m src.eval.metrics                  # the six main runs on val
    uv run python -m src.eval.metrics --split test     # ONCE, at the freeze (needs --score-test files)
    # R2's ablation, e.g. txn_graph vs txn_graph_g1g3:
    uv run python -m src.eval.metrics --models lr,lgbm --feature-sets txn_graph,txn_graph_g1g3 \
        --pairs txn_graph:txn_graph_g1g3 --name ablation

Reads reports/scores/{model}_{feature_set}_{split}.parquet (columns txn_id, score)
and the split's labels. Writes to reports/tables/:
    results_{name}_{split}.csv          model, feature_set, split, metric, value, ci_low, ci_high, subset, n_rows, n_fraud
    results_{name}_paired_{split}.csv   model, split, metric, diff, ci_low, ci_high, feature_set_base, feature_set_new, subset
    results_{name}_budget_{split}.csv   one row per model x feature set x alerting policy: alerts, TP, FP,
                                        recall, fraud amount caught, and the normal-looking breakdown
    results_{name}_budget_curve_{split}.csv   recall / precision / FP at every budget from 0.1% to 5%
                                        (point estimates), per subgroup: for the figure and the UI's slider
    results_{name}_fp_profile_{split}.csv     the legit rows in each budget queue, by category: what the
                                        analysts' false positives look like (FP burden, §4.4)
Figures from these tables: uv run python -m src.eval.figures

Method (see src/eval/core.py for the conventions):
- PR-AUC primary, ROC-AUC secondary.
- Precision / recall / F1 / FP at the F1-maximising threshold picked on VAL for that
  run. On test, the val threshold is applied unchanged.
- Recall, precision, FP, fraud amount caught at each review budget (default 1%, 2%).
- Every metric on all rows and on the subgroups: normal_looking (primary),
  normal_looking_strict, and normal_looking_q90 (90th-percentile sensitivity check).
- 95% percentile CIs from a stratified bootstrap (1,000 resamples, fraud and legit
  resampled separately). Resample b is the same rows for every model, so the
  txn_graph - txn difference is a paired bootstrap.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import SCORE_DIR, SEED, SPLIT_DIR, TABLE_DIR
from src.eval.core import (
    SortedRun, budget_tag, choose_threshold, full_name, metric_names, n_alerts, stratified_weights,
)
from src.features.subgroup import fit_thresholds, flag_normal_looking

MODELS = ("lr", "mlp", "lgbm")
FEATURE_SETS = ("txn", "txn_graph")
BUDGETS = (0.01, 0.02)
SUBSETS = ("all", "normal_looking", "normal_looking_strict", "normal_looking_q90")
N_BOOT = 1000
SENSITIVITY_QUANTILE = 0.90
CURVE_BUDGETS = tuple(round(0.001 * i, 3) for i in range(1, 51))      # 0.1%, 0.2%, ..., 5.0%

MAIN_COLS = ["model", "feature_set", "split", "metric", "value", "ci_low", "ci_high",
             "subset", "n_rows", "n_fraud"]
PAIRED_COLS = ["model", "split", "metric", "diff", "ci_low", "ci_high",
               "feature_set_base", "feature_set_new", "subset"]


# --- inputs -------------------------------------------------------------------

def add_subgroup_flags(df, train):
    """normal_looking / _strict use the fixed 95th-percentile definition; _q90 is the sensitivity check.
    Thresholds come from train only."""
    by_cat, global_cut = fit_thresholds(train)
    flags = flag_normal_looking(df, by_cat, global_cut)
    by_cat90, _ = fit_thresholds(train, quantile=SENSITIVITY_QUANTILE)
    flags["normal_looking_q90"] = flag_normal_looking(df, by_cat90, global_cut)["normal_looking"]
    return pd.concat([df, flags], axis=1)


def load_labels(split, split_dir=SPLIT_DIR):
    """The split's rows, sorted by txn_id, with labels, amounts and subgroup flags."""
    train = pd.read_parquet(split_dir / "train.parquet")
    df = train if split == "train" else pd.read_parquet(split_dir / f"{split}.parquet")
    df = add_subgroup_flags(df, train)
    return df.sort_values("txn_id").reset_index(drop=True)


def score_path(scores_dir, model, feature_set, split):
    return Path(scores_dir) / f"{model}_{feature_set}_{split}.parquet"


def parse_score_name(stem):
    """'lr_txn_graph_g1g3_val' -> ('lr', 'txn_graph_g1g3', 'val'). Model is the first token, split the last."""
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"not a score file name: {stem}")
    return parts[0], "_".join(parts[1:-1]), parts[-1]


def load_scores(path, txn_ids):
    """Score column aligned to txn_ids. Fails loudly on a wrong schema, duplicates or missing rows."""
    df = pd.read_parquet(path)
    missing = {"txn_id", "score"} - set(df.columns)
    if missing:
        raise ValueError(f"{Path(path).name}: missing columns {sorted(missing)}")
    if df.txn_id.duplicated().any():
        raise ValueError(f"{Path(path).name}: duplicate txn_id")
    s = df.set_index("txn_id")["score"].reindex(txn_ids)
    if s.isna().any():
        raise ValueError(f"{Path(path).name}: {int(s.isna().sum())} split rows have no score")
    if len(df) != len(txn_ids):
        raise ValueError(f"{Path(path).name}: {len(df)} rows but the split has {len(txn_ids)}")
    return s.to_numpy(dtype=float)


# --- public helper for R2 / R3 ------------------------------------------------

def score_predictions(y, score, txn_id=None, amount=None, threshold=None, budgets=BUDGETS):
    """Point metrics for one set of predictions, no files and no bootstrap.

    If threshold is None it's picked on these same rows (fine on val, never on test).
    Returns a flat dict, e.g. {'pr_auc': ..., 'recall_at_1pct': ..., ...}.
    """
    y = np.asarray(y)
    n = len(y)
    txn_id = np.arange(n) if txn_id is None else txn_id
    amount = np.ones(n) if amount is None else amount
    thr = choose_threshold(y, score) if threshold is None else threshold
    run = SortedRun(score, y, txn_id, amount, {"all": np.ones(n, bool)})
    out = {full_name(s, m): v for (s, m), v in run.evaluate(thr, budgets).items()}
    out["threshold"] = thr
    return out


# --- evaluation ---------------------------------------------------------------

def evaluate_runs(labels, runs, thresholds, budgets=BUDGETS, subsets=SUBSETS, n_boot=N_BOOT,
                  seed=SEED, log=print):
    """Point estimates and bootstrap replicates for each run.

    runs: {(model, feature_set): score array aligned to labels}
    thresholds: {(model, feature_set): val threshold}
    Returns (points, boots): points[run] = {(subset, metric): value};
    boots[run] = array (n_boot, n_metrics) in metric_names() order.
    """
    y = labels.fraud.to_numpy()
    masks = {sub: np.ones(len(labels), bool) if sub == "all" else labels[sub].to_numpy() == 1
             for sub in subsets}
    sorted_runs = {k: SortedRun(s, y, labels.txn_id.to_numpy(), labels.amount.to_numpy(), masks)
                   for k, s in runs.items()}
    names = metric_names(budgets, subsets)
    points = {k: r.evaluate(thresholds[k], budgets) for k, r in sorted_runs.items()}
    boots = {k: np.empty((n_boot, len(names))) for k in runs}
    t0 = time.time()
    for b in range(n_boot):
        w = stratified_weights(y, b, seed)          # same rows for every run -> paired
        for k, r in sorted_runs.items():
            res = r.evaluate(thresholds[k], budgets, weights=w)
            boots[k][b] = [res[nm] for nm in names]
        if (b + 1) % 200 == 0:
            log(f"  bootstrap {b + 1}/{n_boot} ({time.time() - t0:.0f}s)")
    return points, boots


def ci(dist):
    """95% percentile interval, ignoring NaN replicates."""
    dist = dist[~np.isnan(dist)]
    if len(dist) == 0:
        return np.nan, np.nan
    lo, hi = np.percentile(dist, [2.5, 97.5])
    return float(lo), float(hi)


def main_table(labels, points, boots, split, budgets=BUDGETS, subsets=SUBSETS):
    names = metric_names(budgets, subsets)
    sizes = {sub: (len(labels), int(labels.fraud.sum())) if sub == "all"
             else (int(labels[sub].sum()), int(labels.fraud[labels[sub] == 1].sum())) for sub in subsets}
    rows = []
    for (model, fs), pt in points.items():
        for j, (sub, m) in enumerate(names):
            lo, hi = ci(boots[(model, fs)][:, j])
            rows.append([model, fs, split, full_name(sub, m), pt[(sub, m)], lo, hi, sub, *sizes[sub]])
    return pd.DataFrame(rows, columns=MAIN_COLS)


def paired_table(points, boots, pairs, split, budgets=BUDGETS, subsets=SUBSETS):
    """new - base for every model that has both runs, CI from the paired bootstrap."""
    names = metric_names(budgets, subsets)
    rows = []
    models = sorted({m for m, _ in points}, key=lambda m: MODELS.index(m) if m in MODELS else 99)
    for base, new in pairs:
        for model in models:
            if (model, base) not in points or (model, new) not in points:
                continue
            d = boots[(model, new)] - boots[(model, base)]
            for j, (sub, m) in enumerate(names):
                diff = points[(model, new)][(sub, m)] - points[(model, base)][(sub, m)]
                lo, hi = ci(d[:, j])
                rows.append([model, split, full_name(sub, m), diff, lo, hi, base, new, sub])
    return pd.DataFrame(rows, columns=PAIRED_COLS)


def budget_table(labels, runs, thresholds, split, budgets=BUDGETS, subsets=SUBSETS):
    """Operational view for R1's business numbers and the FP-burden analysis (point estimates).
    One row per run x policy: each review budget, plus the val-F1 threshold for comparison."""
    y = labels.fraud.to_numpy()
    masks = {sub: np.ones(len(labels), bool) if sub == "all" else labels[sub].to_numpy() == 1
             for sub in subsets}
    n_days = labels.step.nunique()
    rows = []
    for (model, fs), s in runs.items():
        run = SortedRun(s, y, labels.txn_id.to_numpy(), labels.amount.to_numpy(), masks)
        policies = [(f"budget_{budget_tag(b)}", dict(policy="budget", budget=b)) for b in budgets]
        policies.append(("val_f1_threshold", dict(policy="threshold", threshold=thresholds[(model, fs)])))
        for name, kw in policies:
            c = run.policy_counts(**kw)
            a = c["all"]
            row = {
                "model": model, "feature_set": fs, "split": split, "policy": name,
                "n_rows": len(labels), "n_days": n_days, "n_alerts": int(c["n_alerts"]),
                "alerts_per_day": c["n_alerts"] / n_days,
                "n_fraud": int(a["n_fraud"]),
                "max_recall": min(c["n_alerts"], a["n_fraud"]) / a["n_fraud"],
                "tp": int(a["tp"]), "fp": int(a["fp"]), "fp_per_day": a["fp"] / n_days,
                "recall": a["tp"] / a["n_fraud"],
                "precision": a["tp"] / c["n_alerts"] if c["n_alerts"] else np.nan,
                "fraud_amount_caught": a["amount_caught"], "fraud_amount_total": a["amount_total"],
                "amount_share": a["amount_caught"] / a["amount_total"],
            }
            for sub in subsets[1:]:
                d = c[sub]
                row[f"{sub}_fraud"] = int(d["n_fraud"])
                row[f"{sub}_tp"] = int(d["tp"])
                row[f"{sub}_recall"] = d["tp"] / d["n_fraud"] if d["n_fraud"] else np.nan
                row[f"{sub}_fp"] = int(d["fp"])              # legit rows in the queue that look normal
            if name.startswith("budget"):
                b = kw["budget"]
                assert row["n_alerts"] == n_alerts(len(labels), b)
            rows.append(row)
    return pd.DataFrame(rows)


def budget_curve(labels, runs, split, budgets=CURVE_BUDGETS, subsets=SUBSETS):
    """Recall, precision and FP at every budget in `budgets` (point estimates; CIs are in the main
    table at 1% and 2%). Same whole-split queue as everywhere else."""
    y = labels.fraud.to_numpy()
    masks = {sub: np.ones(len(labels), bool) if sub == "all" else labels[sub].to_numpy() == 1
             for sub in subsets}
    n_days = labels.step.nunique()
    rows = []
    for (model, fs), s in runs.items():
        run = SortedRun(s, y, labels.txn_id.to_numpy(), labels.amount.to_numpy(), masks)
        for b in budgets:
            c = run.policy_counts("budget", budget=b)
            k = c["n_alerts"]
            for sub in subsets:
                d = c[sub]
                rows.append({
                    "model": model, "feature_set": fs, "split": split, "budget": b,
                    "n_alerts": int(k), "alerts_per_day": k / n_days, "subset": sub,
                    "n_fraud": int(d["n_fraud"]), "tp": int(d["tp"]), "fp": int(d["fp"]),
                    "recall": d["tp"] / d["n_fraud"] if d["n_fraud"] else np.nan,
                    "precision": d["tp"] / (d["tp"] + d["fp"]) if d["tp"] + d["fp"] else np.nan,
                    "max_recall": min(k, d["n_fraud"]) / d["n_fraud"] if sub == "all" else np.nan,
                    "amount_share": d["amount_caught"] / d["amount_total"] if d["amount_total"] else np.nan,
                })
    return pd.DataFrame(rows)


def fp_profile(labels, runs, split, budgets=BUDGETS):
    """The legit transactions inside each budget queue, by category: who the analysts are
    reviewing for nothing. fp_rate = share of that category's legit rows that get queued."""
    legit_by_cat = labels[labels.fraud == 0].groupby("category", observed=True).size()
    rows = []
    for (model, fs), s in runs.items():
        ranked = labels.assign(score=s).sort_values(["score", "txn_id"], ascending=[False, True])
        for b in budgets:
            queue = ranked.head(n_alerts(len(labels), b))
            fp = queue[queue.fraud == 0]
            g = fp.groupby("category", observed=True)
            t = pd.DataFrame({"fp": g.size(), "fp_normal_looking": g.normal_looking.sum(),
                              "fp_median_amount": g.amount.median()})
            t = t[t.fp > 0].sort_values("fp", ascending=False)
            t["fp_share"] = t.fp / len(fp)
            t["legit_in_split"] = legit_by_cat.reindex(t.index).to_numpy()
            t["fp_rate"] = t.fp / t.legit_in_split
            t = t.rename_axis("category").reset_index()
            t.insert(0, "policy", f"budget_{budget_tag(b)}")
            t.insert(0, "split", split)
            t.insert(0, "feature_set", fs)
            t.insert(0, "model", model)
            rows.append(t)
    return pd.concat(rows, ignore_index=True)


# --- running it ---------------------------------------------------------------

def thresholds_from_val(run_keys, scores_dir, val_labels):
    out = {}
    for model, fs in run_keys:
        p = score_path(scores_dir, model, fs, "val")
        if not p.exists():
            raise FileNotFoundError(f"{p.name} is needed to pick the threshold on val")
        out[(model, fs)] = choose_threshold(val_labels.fraud.to_numpy(),
                                            load_scores(p, val_labels.txn_id.to_numpy()))
    return out


def run(split="val", models=MODELS, feature_sets=FEATURE_SETS, pairs=(("txn", "txn_graph"),),
        name="main", budgets=BUDGETS, n_boot=N_BOOT, scores_dir=SCORE_DIR, out_dir=TABLE_DIR,
        split_dir=SPLIT_DIR, seed=SEED, log=print):
    t0 = time.time()
    if split == "test":
        log("*** Scoring TEST. This should happen once, at the freeze (Sun 15 Nov). ***")
    labels = load_labels(split, split_dir)
    val_labels = labels if split == "val" else load_labels("val", split_dir)

    keys = [(m, fs) for m in models for fs in feature_sets
            if score_path(scores_dir, m, fs, split).exists()]
    absent = [f"{m}_{fs}" for m in models for fs in feature_sets if (m, fs) not in keys]
    if absent:
        log(f"no {split} scores for: {', '.join(absent)} (skipped)")
    if not keys:
        raise FileNotFoundError(f"no score files for split '{split}' in {scores_dir}")

    runs = {k: load_scores(score_path(scores_dir, *k, split), labels.txn_id.to_numpy()) for k in keys}
    thresholds = thresholds_from_val(keys, scores_dir, val_labels)
    log(f"{split}: {len(labels):,} rows, {int(labels.fraud.sum()):,} fraud; {len(keys)} runs; "
        f"{n_boot} bootstrap resamples")

    points, boots = evaluate_runs(labels, runs, thresholds, budgets, SUBSETS, n_boot, seed, log)
    main = main_table(labels, points, boots, split, budgets)
    for k, thr in thresholds.items():                     # record the threshold used (no CI)
        main.loc[len(main)] = [*k, split, "threshold", thr, np.nan, np.nan, "all",
                               len(labels), int(labels.fraud.sum())]
    paired = paired_table(points, boots, pairs, split, budgets)
    budget = budget_table(labels, runs, thresholds, split, budgets)
    tables = {
        "": main, "paired_": paired, "budget_": budget,
        "budget_curve_": budget_curve(labels, runs, split),
        "fp_profile_": fp_profile(labels, runs, split, budgets),
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for kind, df in tables.items():
        paths[kind] = out_dir / f"results_{name}_{kind}{split}.csv"
        df.to_csv(paths[kind], index=False, float_format="%.6g")
    log(summary(main, paired, budgets))
    log("wrote " + ", ".join(p.name for p in paths.values()) + f" in {time.time() - t0:.0f}s")
    return main, paired, budget


def summary(main, paired, budgets=BUDGETS):
    """Short console view. The CSVs are the source of every reported number."""
    tag = budget_tag(budgets[0])
    show = ["pr_auc", "f1", f"recall_at_{tag}", f"recall_at_{tag}_normal_looking",
            f"recall_at_{tag}_normal_looking_strict"]
    v = main[main.metric.isin(show)].copy()
    v["cell"] = [f"{a:.3f} [{lo:.3f}, {hi:.3f}]" for a, lo, hi in zip(v.value, v.ci_low, v.ci_high)]
    wide = v.pivot(index=["model", "feature_set"], columns="metric", values="cell")[show]
    ceiling = main.loc[main.metric == f"max_recall_at_{tag}", "value"].iloc[0]
    lines = ["", wide.to_string(), f"(best possible recall@{tag}: {ceiling:.2%})"]
    if len(paired):
        p = paired[paired.metric.isin(show)].copy()
        p["cell"] = [f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]" for d, lo, hi in zip(p["diff"], p.ci_low, p.ci_high)]
        pw = p.pivot(index=["model", "feature_set_base", "feature_set_new"], columns="metric",
                     values="cell")[show]
        lines += ["", "paired difference (new - base):", pw.to_string()]
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--feature-sets", default=",".join(FEATURE_SETS))
    ap.add_argument("--pairs", default="txn:txn_graph",
                    help="comma-separated base:new pairs for the paired bootstrap")
    ap.add_argument("--name", default="main", help="output name: results_{name}_{split}.csv")
    ap.add_argument("--budgets", default=",".join(str(b) for b in BUDGETS))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--scores-dir", type=Path, default=SCORE_DIR)
    ap.add_argument("--out-dir", type=Path, default=TABLE_DIR)
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    pairs = [tuple(p.split(":")) for p in a.pairs.split(",") if p]
    run(split=a.split, models=a.models.split(","), feature_sets=a.feature_sets.split(","),
        pairs=pairs, name=a.name, budgets=tuple(float(b) for b in a.budgets.split(",")),
        n_boot=a.n_boot, scores_dir=a.scores_dir, out_dir=a.out_dir)


if __name__ == "__main__":
    main()
