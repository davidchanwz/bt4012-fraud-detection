"""Read-only access to the frozen outputs in reports/tables/ (and data/processed/ for the case viewer).

The dashboard never trains, scores or recomputes a metric: every number it shows is a row
that already exists in a CSV written by the pipeline. Missing files return None, so a tab
can show "not available yet" instead of crashing (ablation, item-A EDA and the test split
arrive later).

Result families: R4's metrics.py writes results_{name}_{kind}{split}.csv. The main six runs
use name="main" (LR, MLP, LightGBM); the Random Forest robustness check uses name="rf".
results() stacks both so every tab can treat RF as a fourth model.
"""
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import CLEAN_FILE, TABLE_DIR

# Overridable so tests can point the app at a copy with files removed (empty states).
TABLE_DIR = Path(os.environ.get("DASHBOARD_TABLE_DIR", TABLE_DIR))
CLEAN_FILE = Path(os.environ.get("DASHBOARD_CLEAN_FILE", CLEAN_FILE))

SPLITS = ("val", "test")
MODELS = ("lr", "mlp", "lgbm", "rf")
FEATURE_SETS = ("txn", "txn_graph")
SUBSETS = ("all", "normal_looking", "normal_looking_strict", "normal_looking_q90")
FAMILIES = {"main": ("lr", "mlp", "lgbm"), "rf": ("rf",)}
KINDS = ("", "paired", "budget", "budget_curve", "fp_profile")

MODEL_LABELS = {"lr": "Logistic regression", "mlp": "MLP", "lgbm": "LightGBM", "rf": "Random forest"}
MODEL_SHORT = {"lr": "LR", "mlp": "MLP", "lgbm": "LightGBM", "rf": "RF"}
FEATURE_LABELS = {"txn": "txn", "txn_graph": "txn + graph"}
SUBSET_LABELS = {
    "all": "All fraud",
    "normal_looking": "Normal-looking",
    "normal_looking_strict": "Strict normal-looking",
    "normal_looking_q90": "Normal-looking (q90 check)",
}
SUBSET_HELP = {
    "all": "Every fraud case in the split.",
    "normal_looking": "Fraud with an amount at or below the 95th percentile of legit amounts in the same "
                      "category (train thresholds). The primary subgroup.",
    "normal_looking_strict": "Fraud with an amount at or below 72.81, the strict subgroup.",
    "normal_looking_q90": "Same as normal-looking but at the 90th percentile: a sensitivity check, "
                          "not a way to pick the best number.",
}
GROUP_LABELS = {"txn": "transaction", "g1": "customer history", "g2": "merchant history",
                "g3": "customer-merchant edge", "g4": "network structure", "g5": "past labels"}

# Base metric names as written by src/eval/metrics.py. On a subset the name gets "_{subset}".
METRIC_LABELS = {
    "pr_auc": "PR-AUC",
    "roc_auc": "ROC-AUC",
    "f1": "F1 at val threshold",
    "precision": "Precision at val threshold",
    "recall": "Recall at val threshold",
    "recall_at_1pct": "Recall @ 1% budget",
    "precision_at_1pct": "Precision @ 1% budget",
    "fp_at_1pct": "False positives @ 1% budget",
    "amount_share_at_1pct": "Fraud value caught @ 1%",
    "recall_at_2pct": "Recall @ 2% budget",
    "precision_at_2pct": "Precision @ 2% budget",
    "fp_at_2pct": "False positives @ 2% budget",
    "amount_share_at_2pct": "Fraud value caught @ 2%",
}
COUNT_METRICS = {"fp", "fp_at_1pct", "fp_at_2pct", "amount_caught_at_1pct", "amount_caught_at_2pct"}

# Sources the group still owes; the tabs and the sidebar status panel read this list.
PENDING = {
    "eda": {"title": "Normal-looking fraud vs legit EDA", "owner": "R4 Rayson (item A)",
            "files": ["normal_looking_comparison.csv", "normal_looking_category_mix.csv"]},
    "ablation": {"title": "Ablation (leave one group out, label-free vs labels-only)", "owner": "R2 Jayden",
                 "files": ["results_ablation_{split}.csv", "results_ablation_paired_{split}.csv"]},
    "delay": {"title": "Label delay, 7-day vs 1-day", "owner": "R2 Jayden",
              "files": ["results_delay_{split}.csv", "results_delay_paired_{split}.csv"]},
    "test": {"title": "Test split (after the freeze, Sun 15 Nov)", "owner": "everyone",
             "files": ["results_main_test.csv", "results_main_paired_test.csv"]},
}


@dataclass(frozen=True)
class Selection:
    """The sidebar state every tab receives."""
    split: str
    model: str
    feature_sets: tuple          # ("txn", "txn_graph"), or one of them
    subset: str


# --- reading ------------------------------------------------------------------

def table_path(name):
    return TABLE_DIR / name


@st.cache_data(show_spinner=False)
def _read_csv(path, mtime):            # mtime is part of the cache key, so edits show up
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _read_parquet(path, mtime, columns=None):
    return pd.read_parquet(path, columns=columns)


def load(name):
    """A CSV from reports/tables/, or None if it doesn't exist yet."""
    p = table_path(name)
    return _read_csv(str(p), p.stat().st_mtime) if p.exists() else None


def exists(name):
    return table_path(name).exists()


def results(kind="", split="val", families=tuple(FAMILIES)):
    """results_{family}_{kind}{split}.csv for every family that exists, stacked. None if none exist."""
    stem = f"{kind}_" if kind else ""
    parts = []
    for fam in families:
        df = load(f"results_{fam}_{stem}{split}.csv")
        if df is not None:
            parts.append(df.assign(family=fam))
    return pd.concat(parts, ignore_index=True) if parts else None


def families(split="val"):
    """Every results_{name}_{split}.csv family on disk (main, rf, and whatever R2 names the ablation)."""
    suffixes = tuple(f"_{k}" for k in KINDS if k)
    names = []
    for p in sorted(TABLE_DIR.glob(f"results_*_{split}.csv")):
        name = p.stem[len("results_"):-len(f"_{split}")]
        if name and not name.endswith(suffixes) and not name.endswith("_mock"):
            names.append(name)
    return names


def variant_results(kind="", split="val"):
    """Rows for feature sets other than plain txn / txn_graph (ablation: txn_graph_g1g3, ..._delay1),
    from any results family. None until R2's runs are evaluated."""
    df = results(kind, split, families=tuple(families(split)))
    if df is None:
        return None
    col = "feature_set_new" if kind == "paired" else "feature_set"
    out = df[~df[col].isin(FEATURE_SETS)]
    return None if out.empty else out.reset_index(drop=True)


def available_splits():
    """Splits with main results. test only appears after the freeze."""
    return [s for s in SPLITS if exists(f"results_main_{s}.csv")]


def available_models(split="val"):
    df = results("", split)
    return [m for m in MODELS if df is not None and m in set(df.model)]


def metric_name(base, subset="all"):
    return base if subset == "all" else f"{base}_{subset}"


def metric(split, model, feature_set, base, subset="all"):
    """One row (value, ci_low, ci_high, n_rows, n_fraud) as a dict, or None."""
    df = results("", split)
    if df is None:
        return None
    r = df[(df.model == model) & (df.feature_set == feature_set) & (df.metric == metric_name(base, subset))]
    return None if r.empty else r.iloc[0].to_dict()


def paired(split, model, base, subset="all"):
    """Paired txn -> txn_graph difference for one metric: dict with diff, ci_low, ci_high, or None."""
    df = results("paired", split)
    if df is None:
        return None
    r = df[(df.model == model) & (df.metric == metric_name(base, subset))
           & (df.feature_set_base == "txn") & (df.feature_set_new == "txn_graph")]
    return None if r.empty else r.iloc[0].to_dict()


def ceiling(split, budget="1pct"):
    """Best possible recall at the budget (queue smaller than the fraud count), from R4's table."""
    df = results("", split)
    if df is None:
        return None
    r = df[df.metric == f"max_recall_at_{budget}"]
    return None if r.empty else float(r.value.iloc[0])


def clean_data(columns=None):
    """data/processed/banksim_clean.parquet (rebuilt per machine), or None after a fresh clone."""
    if not CLEAN_FILE.exists():
        return None
    return _read_parquet(str(CLEAN_FILE), CLEAN_FILE.stat().st_mtime, tuple(columns) if columns else None)


def status(split="val"):
    """What is present and what is still owed, for the sidebar panel and the empty states."""
    rows = [{"key": "main", "title": "Main results (LR, MLP, LightGBM)", "owner": "R3 / R4",
             "files": [f"results_main_{split}.csv"], "present": exists(f"results_main_{split}.csv")},
            {"key": "rf", "title": "Random forest robustness check", "owner": "R3",
             "files": [f"results_rf_{split}.csv"], "present": exists(f"results_rf_{split}.csv")},
            {"key": "cases", "title": "Case-study shortlist", "owner": "R3",
             "files": ["case_study_counts.csv"], "present": exists("case_study_counts.csv")},
            {"key": "processed", "title": "data/processed (case viewer history)", "owner": "local rebuild",
             "files": ["data/processed/banksim_clean.parquet"], "present": CLEAN_FILE.exists()}]
    variants = variant_results("", split)
    variant_fs = set() if variants is None else set(variants.feature_set)
    found_in_any_family = {     # R2 may evaluate these under --name ablation rather than their own name
        "ablation": any("_delay" not in f for f in variant_fs),
        "delay": any("_delay" in f for f in variant_fs),
    }
    for key, p in PENDING.items():
        files = [f.format(split="test" if key == "test" else split) for f in p["files"]]
        present = all(exists(f) for f in files) or found_in_any_family.get(key, False)
        rows.append({"key": key, **p, "files": files, "present": present})
    return rows


def pending_files(key, split="val"):
    return [f.format(split=split) for f in PENDING[key]["files"]]
