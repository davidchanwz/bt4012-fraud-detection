"""Checks for src/models (training contract, no test leakage). Run with: pytest -q

All tests use toy data, so they run without the real pipeline.
"""
import numpy as np
import pandas as pd
import pytest

from src.config import LABEL_DELAY
from src.features.graph import build_graph_features
from src.models import train as T
from src.models.data import FeatureBuilder, feature_set_name
from src.models.grids import MLP_LEGIT_PER_FRAUD, undersample


def toy_clean(n_rows=1500, n_days=30, seed=0):
    """Cleaned-data lookalike: amount drives fraud so the models have signal."""
    rng = np.random.default_rng(seed)
    cats = {"M0": "food", "M1": "food", "M2": "travel", "M3": "health", "M4": "leisure"}
    merch = rng.choice(list(cats), n_rows)
    amount = rng.gamma(2, 30, n_rows).round(2)
    fraud = (rng.random(n_rows) < 0.02 + 0.3 * (amount > 150)).astype("int8")
    return pd.DataFrame({
        "txn_id": np.arange(n_rows),
        "step": np.sort(rng.integers(0, n_days, n_rows)),
        "customer": rng.choice([f"C{i}" for i in range(40)], n_rows),
        "age": rng.choice(list("0123456U"), n_rows),
        "gender": rng.choice(list("FMEU"), n_rows),
        "merchant": merch,
        "category": [cats[m] for m in merch],
        "amount": amount,
        "log_amount": np.log1p(amount),
        "zero_amount": (amount == 0).astype("int8"),
        "fraud": fraud,
    })


@pytest.fixture(scope="module")
def toy():
    df = toy_clean()
    splits = {"train": df[df.step < 20], "val": df[(df.step >= 20) & (df.step < 25)],
              "test": df[df.step >= 25]}
    return splits, build_graph_features(df, LABEL_DELAY)


@pytest.fixture
def run_toy(toy, tmp_path, monkeypatch):
    splits, graph = toy
    # small grids so the tests stay fast
    monkeypatch.setitem(T.grid_points.__globals__["GRIDS"], "lr", {"C": [0.1, 1]})
    monkeypatch.setitem(T.grid_points.__globals__["GRIDS"], "mlp",
                        {"hidden_layer_sizes": [(8,)], "alpha": [1e-3]})

    def go(model="lr", features="txn", score_test=False, **kw):
        return T.run(model, features, splits=splits, graph=graph if features == "txn_graph" else None,
                     score_test=score_test, score_dir=tmp_path / "scores", table_dir=tmp_path / "tables",
                     model_dir=tmp_path / "models", log=lambda *a: None, **kw)
    return go, tmp_path, splits


# --- naming -------------------------------------------------------------------

@pytest.mark.parametrize("args, name", [
    (("txn",), "txn"),
    (("txn_graph",), "txn_graph"),
    (("txn_graph", "g3,g1"), "txn_graph_g1g3"),
    (("txn_graph", ["g1", "g2", "g3", "g5"]), "txn_graph"),
    (("txn_graph", None, 1), "txn_graph_delay1"),
    (("txn_graph", "g5", 1), "txn_graph_g5_delay1"),
])
def test_feature_set_name(args, name):
    assert feature_set_name(*args) == name


@pytest.mark.parametrize("args", [("txn", "g1"), ("txn", None, 1), ("txn_graph", "g9"),
                                  ("txn_graph", None, 3), ("graph",)])
def test_feature_set_name_rejects_bad_args(args):
    with pytest.raises(ValueError):
        feature_set_name(*args)


# --- features -----------------------------------------------------------------

def test_feature_builder_fits_on_train_only(toy):
    splits, graph = toy
    b = FeatureBuilder("txn_graph", "g1,g3").fit(splits["train"], graph)
    X = b.transform(splits["train"], graph)
    # standardised on train -> train means ~0 for the graph columns
    assert np.allclose(X[b.graph_columns].mean(), 0, atol=1e-8)
    assert all(c.split("_")[0] in {"g1", "g3"} for c in b.graph_columns)
    Xv = b.transform(splits["val"], graph)
    assert list(Xv.columns) == list(X.columns) and Xv.index.name == "txn_id"
    assert (Xv.index == splits["val"].txn_id.to_numpy()).all()


# --- undersampling ------------------------------------------------------------

def test_undersample_keeps_all_fraud_and_ratio(toy):
    train = toy[0]["train"]
    ids = undersample(train, seed=0)
    picked = train.set_index("txn_id").loc[ids]
    assert set(train.loc[train.fraud == 1, "txn_id"]) <= set(ids)
    n_fraud = int(train.fraud.sum())
    assert (picked.fraud == 0).sum() == min((train.fraud == 0).sum(), MLP_LEGIT_PER_FRAUD * n_fraud)


def test_undersample_depends_only_on_seed(toy):
    train = toy[0]["train"]
    shuffled = train.sample(frac=1, random_state=3)          # row order must not matter
    assert (undersample(train, 2) == undersample(shuffled, 2)).all()
    assert not np.array_equal(undersample(train, 0), undersample(train, 1))


# --- training contract --------------------------------------------------------

def test_scores_written_for_val_only_by_default(run_toy):
    go, tmp, splits = run_toy
    go("lr", "txn")
    s = pd.read_parquet(tmp / "scores" / "lr_txn_val.parquet")
    assert list(s.columns) == ["txn_id", "score"]
    assert s.txn_id.is_unique and len(s) == len(splits["val"])
    assert set(s.txn_id) == set(splits["val"].txn_id)
    assert s.score.between(0, 1).all()
    assert not (tmp / "scores" / "lr_txn_test.parquet").exists()


def test_score_test_flag_writes_test(run_toy):
    go, tmp, splits = run_toy
    go("lr", "txn_graph", score_test=True)
    s = pd.read_parquet(tmp / "scores" / "lr_txn_graph_test.parquet")
    assert len(s) == len(splits["test"])


def test_test_split_is_never_touched_without_flag(toy, tmp_path, monkeypatch):
    """Without --score-test, run() must not even look at the test split."""
    splits, _ = toy
    poisoned = dict(splits, test=None)                    # any access would crash
    monkeypatch.setitem(T.grid_points.__globals__["GRIDS"], "lr", {"C": [1]})
    T.run("lr", "txn", splits=poisoned, score_dir=tmp_path, table_dir=tmp_path,
          save_model=False, log=lambda *a: None)


def test_models_only_fit_on_train_rows(run_toy, monkeypatch):
    go, _, splits = run_toy
    train_ids = set(splits["train"].txn_id)
    seen = []
    real = T.make_model

    def spy(*a, **kw):
        m = real(*a, **kw)
        fit = m.fit

        def checked_fit(X, y, **fkw):
            seen.append(len(X))
            assert set(X.index) <= train_ids
            return fit(X, y, **fkw)
        m.fit = checked_fit
        return m
    monkeypatch.setattr(T, "make_model", spy)
    go("lr", "txn", save_model=False)        # the spy wrapper can't be pickled
    go("mlp", "txn", save_model=False)
    assert seen and max(seen) <= len(train_ids)


def test_hyperparams_upsert_replaces_row(run_toy):
    go, tmp, _ = run_toy
    go("lr", "txn")
    go("lr", "txn")
    go("lr", "txn_graph")
    hp = pd.read_csv(tmp / "tables" / "hyperparams.csv")
    assert len(hp) == 2
    assert set(hp.feature_set) == {"txn", "txn_graph"}


def test_mlp_records_seed_sd(run_toy):
    go, tmp, _ = run_toy
    row = go("mlp", "txn")
    assert row["val_pr_auc_sd"] is not None and row["val_pr_auc_sd"] >= 0


def test_reruns_are_deterministic(run_toy):
    go, tmp, _ = run_toy
    go("mlp", "txn")
    a = pd.read_parquet(tmp / "scores" / "mlp_txn_val.parquet")
    go("mlp", "txn")
    b = pd.read_parquet(tmp / "scores" / "mlp_txn_val.parquet")
    pd.testing.assert_frame_equal(a, b)


def test_lgbm_runs_if_available(run_toy, monkeypatch):
    pytest.importorskip("lightgbm")
    go, tmp, _ = run_toy
    monkeypatch.setitem(T.grid_points.__globals__["GRIDS"], "lgbm",
                        {"num_leaves": [7], "learning_rate": [0.1], "min_child_samples": [20]})
    row = go("lgbm", "txn_graph")
    assert row["best_iteration"] >= 1
    assert (tmp / "scores" / "lgbm_txn_graph_val.parquet").exists()
