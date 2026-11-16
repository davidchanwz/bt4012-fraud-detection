"""R2's ablation and label-delay runs: train the variants, then evaluate them. One command.

    python -m src.models.run_ablation                  # train + evaluate on val
    python -m src.models.run_ablation --score-test     # at the freeze: also write test scores
    python -m src.models.run_ablation --eval-only --split test   # evaluate test after --score-test

Variants (LR and LightGBM, same grids as the main runs, see src/models/train.py):
    leave one group out   txn_graph minus G1 / G2 / G3 / G5
    label-free            G1-G3 only (no past labels; G4 isn't built)   = "minus G5"
    labels only           G5 only
    label delay           all groups with the 1-day label-delay features

The base runs (txn, txn_graph) come from run_main and are only re-evaluated here.

Outputs (via src/eval/metrics.py, which never retrains):
    results_ablation_{,paired_,budget_,budget_curve_,fp_profile_}{split}.csv
        paired: each variant vs txn_graph (what dropping groups costs), and
                label-free / labels-only vs txn (what each half adds on its own)
    results_delay_{...}{split}.csv
        paired: txn_graph_delay1 vs txn_graph (what a 1-day delay would gain)
"""
import argparse

from src.eval import metrics
from src.models.data import feature_set_name
from src.models.train import run

MODELS = ["lr", "lgbm"]
ALL_GROUPS = ["g1", "g2", "g3", "g5"]

LEAVE_ONE_OUT = [[g for g in ALL_GROUPS if g != drop] for drop in ALL_GROUPS]
LABELS_ONLY = ["g5"]
LABEL_FREE = ["g1", "g2", "g3"]          # same set as "minus G5"
ABLATION_GROUPS = LEAVE_ONE_OUT + [LABELS_ONLY]
DELAY = 1


def ablation_sets():
    return [feature_set_name("txn_graph", g) for g in ABLATION_GROUPS]


def train_all(models=MODELS, score_test=False):
    rows = []
    for m in models:
        for groups in ABLATION_GROUPS:
            rows.append(run(m, "txn_graph", groups=",".join(groups), score_test=score_test))
        rows.append(run(m, "txn_graph", delay=DELAY, score_test=score_test))
    return rows


def evaluate(split="val", models=MODELS, n_boot=metrics.N_BOOT):
    variants = ablation_sets()
    label_free = feature_set_name("txn_graph", LABEL_FREE)
    labels_only = feature_set_name("txn_graph", LABELS_ONLY)
    pairs = [("txn_graph", v) for v in variants] + [("txn", label_free), ("txn", labels_only)]
    metrics.run(split=split, models=models, feature_sets=["txn", "txn_graph"] + variants,
                pairs=pairs, name="ablation", n_boot=n_boot)

    delay_fs = feature_set_name("txn_graph", delay=DELAY)
    metrics.run(split=split, models=models, feature_sets=["txn_graph", delay_fs],
                pairs=[("txn_graph", delay_fs)], name="delay", n_boot=n_boot)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=MODELS, choices=["lr", "mlp", "lgbm", "rf"])
    p.add_argument("--score-test", action="store_true", help="also write test scores (freeze only)")
    p.add_argument("--eval-only", action="store_true", help="skip training, only evaluate")
    p.add_argument("--split", choices=["val", "test"], default="val")
    a = p.parse_args(argv)

    if not a.eval_only:
        rows = train_all(a.models, a.score_test)
        print()
        for r in rows:
            print(f"{r['model']:<5} {r['feature_set']:<22} val PR-AUC {r['val_pr_auc']:.4f}")
    evaluate(a.split, a.models)


if __name__ == "__main__":
    main()
