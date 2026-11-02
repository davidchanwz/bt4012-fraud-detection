"""Main results table for report §4.1: 3 models x {txn, txn_graph} with CIs and paired differences.

Reads R4's tables (src/eval/metrics.py writes them; this script never computes a metric):
    reports/tables/results_main.csv     model, feature_set, split, metric, value, ci_low, ci_high
    reports/tables/results_paired.csv   model, split, metric, diff, ci_low, ci_high   (txn_graph - txn)
and writes reports/tables/results_main.md. The schema is a PROPOSAL until Rayson (R4)
confirms it; if his files differ, only read_main / read_paired should need changing.

    python -m src.models.results_table [--split test] [--main PATH --paired PATH --out PATH]

Until the real files exist, run it on the fixture:
    python -m src.models.results_table --main tests/fixtures/results_main_mock.csv \
        --paired tests/fixtures/results_paired_mock.csv --out reports/tables/results_main_mock.md
"""
import argparse

import pandas as pd

from src.config import ROOT, TABLE_DIR
from src.models.grids import MODELS

# recall@1% can't exceed this on test: 1,113 alerts for 1,200 fraud (see docs/plan.md)
CEILINGS = {("test", "recall_at_1pct"): 1113 / 1200}

METRICS = [            # (column, label); only those present in the file are shown
    ("pr_auc", "PR-AUC"),
    ("roc_auc", "ROC-AUC"),
    ("f1", "F1 @ val thr."),
    ("recall_at_1pct", "Recall @1%"),
    ("recall_at_2pct", "Recall @2%"),
    ("recall_at_1pct_normal_looking", "Recall @1% (normal-looking)"),
]
MODEL_LABELS = {"lr": "Logistic regression", "mlp": "MLP", "lgbm": "LightGBM"}
FS_LABELS = {"txn": "txn", "txn_graph": "txn + graph"}

MAIN_COLS = ["model", "feature_set", "split", "metric", "value", "ci_low", "ci_high"]
PAIRED_COLS = ["model", "split", "metric", "diff", "ci_low", "ci_high"]


def _read(path, cols):
    df = pd.read_csv(path)
    missing = sorted(set(cols) - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing columns {missing}; expected {cols}")
    return df


def read_main(path):
    return _read(path, MAIN_COLS)


def read_paired(path):
    return _read(path, PAIRED_COLS)


def fmt_ci(v, lo, hi, signed=False):
    if pd.isna(v):
        return "–"
    s = f"{v:+.3f}" if signed else f"{v:.3f}"
    return s if pd.isna(lo) or pd.isna(hi) else f"{s} [{lo:.3f}, {hi:.3f}]"


def build_table(main, paired, split="test"):
    """Long tables -> one wide frame: rows = model x feature set (+ paired diff row), cols = metrics."""
    main, paired = main[main.split == split], paired[paired.split == split]
    metrics = [(m, lab) for m, lab in METRICS if m in set(main.metric)]
    rows = []
    for model in [m for m in MODELS if m in set(main.model)]:
        for fs in ["txn", "txn_graph"]:
            sub = main[(main.model == model) & (main.feature_set == fs)].set_index("metric")
            if sub.empty:
                continue
            row = {"Model": MODEL_LABELS[model], "Features": FS_LABELS[fs]}
            for m, lab in metrics:
                row[lab] = fmt_ci(*sub.loc[m, ["value", "ci_low", "ci_high"]]) if m in sub.index else "–"
            rows.append(row)
        sub = paired[paired.model == model].set_index("metric")
        if not sub.empty:
            row = {"Model": "", "Features": "Δ (paired)"}
            for m, lab in metrics:
                row[lab] = fmt_ci(*sub.loc[m, ["diff", "ci_low", "ci_high"]], signed=True) if m in sub.index else "–"
            rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(table, split):
    cols = list(table.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in table.iterrows()]
    notes = [f"Split: {split}. Point estimate [95% bootstrap CI]. Δ = txn+graph minus txn, paired bootstrap."]
    if (split, "recall_at_1pct") in CEILINGS and "Recall @1%" in cols:
        c = CEILINGS[(split, "recall_at_1pct")]
        notes.append(f"Best possible recall@1% on {split} is {c:.2%} (1,113 alerts for 1,200 fraud).")
    return "\n".join(lines + [""] + notes) + "\n"


def run(main_path, paired_path, out, split="test", log=print):
    table = build_table(read_main(main_path), read_paired(paired_path), split)
    md = to_markdown(table, split)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    log(md)
    log(f"saved {out}")
    return table


def main(argv=None):
    path = lambda s: ROOT / s if not s.startswith("/") else s
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--main", default=TABLE_DIR / "results_main.csv", type=path)
    p.add_argument("--paired", default=TABLE_DIR / "results_paired.csv", type=path)
    p.add_argument("--out", default=TABLE_DIR / "results_main.md", type=path)
    p.add_argument("--split", default="test", choices=["val", "test"])
    a = p.parse_args(argv)
    run(a.main, a.paired, a.out, a.split)


if __name__ == "__main__":
    main()
