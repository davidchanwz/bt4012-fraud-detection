"""Run the six main experiments: 3 models x {txn, txn_graph}. One command for a full rerun.

    python -m src.models.run_main                  # val scores only (weeks 2-3)
    python -m src.models.run_main --score-test     # at the freeze, once
    python -m src.models.run_main --models lr lgbm # subset

Same grids as src/models/grids.py for every run (see src/models/train.py).
"""
import argparse

from src.models.grids import MODELS
from src.models.train import run


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    p.add_argument("--features", nargs="+", default=["txn", "txn_graph"], choices=["txn", "txn_graph"])
    p.add_argument("--score-test", action="store_true")
    a = p.parse_args(argv)
    rows = [run(m, f, score_test=a.score_test) for f in a.features for m in a.models]
    print()
    for r in rows:
        sd = f" ± {r['val_pr_auc_sd']:.4f}" if r["val_pr_auc_sd"] is not None else ""
        print(f"{r['model']:<5} {r['feature_set']:<10} val PR-AUC {r['val_pr_auc']:.4f}{sd}")


if __name__ == "__main__":
    main()
