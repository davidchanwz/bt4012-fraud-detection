"""Fake score files, so downstream scripts can be built before the real runs exist.

Writes {model}_{txn,txn_graph}_{split}.parquet with the REAL txn_ids and random
scores into reports/scores_mock/ (gitignored). The txn_graph scores lean a bit
more on the label than txn, so case_studies.py finds some graph-only catches.
Never point anything that produces report numbers at this directory.

    python -m src.models.mock_scores [--out reports/scores_mock] [--splits val test]
"""
import argparse

import numpy as np

from src.config import ROOT, SEED
from src.models.data import load_splits
from src.models.grids import MAIN_MODELS
from src.models.train import write_scores

MOCK_DIR = ROOT / "reports" / "scores_mock"


def mock_scores(y, signal, rng):
    """Score = signal * label + noise, squashed to (0, 1)."""
    z = signal * np.asarray(y) + rng.normal(size=len(y))
    return 1 / (1 + np.exp(-z))


def write_mock(out=MOCK_DIR, splits=("val", "test"), seed=SEED):
    rng = np.random.default_rng(seed)
    parts = load_splits(list(splits))
    written = []
    for name, df in parts.items():
        for m in MAIN_MODELS:
            for fs, signal in [("txn", 2.0), ("txn_graph", 2.6)]:
                path = out / f"{m}_{fs}_{name}.parquet"
                write_scores(df.txn_id, mock_scores(df.fraud, signal, rng), path)
                written.append(path)
    return written


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=MOCK_DIR, type=lambda s: ROOT / s if not s.startswith("/") else s)
    p.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    a = p.parse_args(argv)
    for path in write_mock(a.out, a.splits):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
