"""Hyperparameter grids and model factories.

The SAME grids are used for txn and txn_graph runs, so the only difference
between a pair of runs is the feature set. Don't tune one side separately.
"""
from itertools import product

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from src.config import SEED

MAIN_MODELS = ["lr", "mlp", "lgbm"]      # the main experiment and the main results table
MODELS = MAIN_MODELS + ["rf"]            # rf is a robustness check (appendix), added after proposal feedback

GRIDS = {
    "lr": {"C": [1e-3, 1e-2, 1e-1, 1, 10, 100]},
    "mlp": {"hidden_layer_sizes": [(32,), (64, 32)], "alpha": [1e-4, 1e-2]},
    "lgbm": {"num_leaves": [15, 31, 63], "learning_rate": [0.05, 0.1], "min_child_samples": [20, 100]},
    # min_samples_leaf matters a lot here (val PR-AUC ~0.90 at 20, ~0.94 at 1), so the grid includes 1
    "rf": {"max_depth": [20, None], "min_samples_leaf": [1, 5, 20]},
}

MLP_SEEDS = [0, 1, 2, 3, 4]
MLP_LEGIT_PER_FRAUD = 10     # undersampling ratio for the MLP train set
LGBM_MAX_TREES = 1000
LGBM_EARLY_STOP = 50
RF_TREES = 300


def grid_points(model):
    grid = GRIDS[model]
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in product(*(grid[k] for k in keys))]


def make_model(model, params, seed=0, scale_pos_weight=None):
    if model == "lr":
        return LogisticRegression(class_weight="balanced", max_iter=2000, **params)
    if model == "mlp":
        return MLPClassifier(early_stopping=True, validation_fraction=0.1, n_iter_no_change=10,
                             max_iter=200, random_state=SEED + seed, **params)
    if model == "lgbm":
        from lightgbm import LGBMClassifier    # imported lazily: needs libomp on macOS
        return LGBMClassifier(n_estimators=LGBM_MAX_TREES, scale_pos_weight=scale_pos_weight,
                              metric="average_precision", random_state=SEED, deterministic=True,
                              force_row_wise=True, verbose=-1, **params)
    if model == "rf":
        return RandomForestClassifier(n_estimators=RF_TREES, max_features="sqrt", n_jobs=-1,
                                      class_weight="balanced_subsample", random_state=SEED, **params)
    raise ValueError(f"unknown model {model!r}; choose from {MODELS}")


def undersample(train, seed, ratio=MLP_LEGIT_PER_FRAUD):
    """txn_ids of all train fraud plus `ratio` legit rows per fraud, drawn with seed SEED + seed.

    Depends only on the train labels and the seed, so txn and txn_graph runs
    with the same seed train on exactly the same rows.
    """
    ids = train.sort_values("txn_id")
    fraud = ids.loc[ids.fraud == 1, "txn_id"].to_numpy()
    legit = ids.loc[ids.fraud == 0, "txn_id"].to_numpy()
    n = min(len(legit), ratio * len(fraud))
    picked = np.random.default_rng(SEED + seed).choice(legit, size=n, replace=False)
    return np.sort(np.concatenate([fraud, picked]))
