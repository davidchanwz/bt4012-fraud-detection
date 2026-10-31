"""Leakage checks for the graph features. Run with: pytest -q

The rule: a feature for a row on day t may only use rows from days < t,
and g5_ (label) features may only use labels from days <= t - label_delay.
Each test breaks something the features must NOT depend on and checks
that nothing changes.
"""
import numpy as np
import pandas as pd
import pytest

from src.config import CLEAN_FILE
from src.features.graph import GraphScaler, build_graph_features, group_columns

DELAY = 3


def toy(n_rows=400, n_days=15, seed=0):
    rng = np.random.default_rng(seed)
    cats = {"M0": "food", "M1": "food", "M2": "travel", "M3": "health", "M4": "leisure"}
    merch = rng.choice(list(cats), n_rows)
    return pd.DataFrame({
        "txn_id": np.arange(n_rows),
        "step": np.sort(rng.integers(0, n_days, n_rows)),
        "customer": rng.choice([f"C{i}" for i in range(20)], n_rows),
        "merchant": merch,
        "category": [cats[m] for m in merch],
        "amount": rng.gamma(2, 30, n_rows).round(2),
        "fraud": (rng.random(n_rows) < 0.1).astype("int8"),
    })


def label_free(cols):
    return group_columns(cols, ["g1", "g2", "g3"])


# --- hand-checked example -----------------------------------------------------

def test_values_on_tiny_example():
    df = pd.DataFrame({
        "txn_id": [0, 1, 2, 3, 4],
        "step": [0, 0, 1, 2, 2],
        "customer": ["A", "A", "A", "A", "B"],
        "merchant": ["X", "Y", "X", "X", "X"],
        "category": ["food"] * 5,
        "amount": [10.0, 30.0, 20.0, 40.0, 5.0],
        "fraud": [0, 1, 0, 0, 0],
    })
    f = build_graph_features(df, label_delay=1)
    # row 1 is on day 0: same-day row 0 must not count
    assert f.loc[1, "g1_cust_n_prev"] == 0
    # row 3 (A, day 2) sees rows 0, 1, 2
    assert f.loc[3, "g1_cust_n_prev"] == 3
    assert f.loc[3, "g1_cust_amt_mean"] == pytest.approx(20.0)
    assert f.loc[3, "g1_cust_n_merchants"] == 2
    assert f.loc[3, "g1_cust_days_since_last"] == 1
    assert f.loc[3, "g3_pair_n_prev"] == 2          # A-X on days 0 and 1
    assert f.loc[3, "g3_pair_share"] == pytest.approx(2 / 3)
    assert f.loc[3, "g2_merch_n_prev"] == 2         # X on days 0 and 1
    assert f.loc[3, "g2_merch_n_customers"] == 1    # only A so far; B is same-day
    assert f.loc[3, "g5_cust_fraud_n"] == 1         # row 1's label, known from day 1
    # row 4: B is new, and has never used X
    assert f.loc[4, "g1_cust_is_new"] == 1 and f.loc[4, "g3_pair_is_new"] == 1
    assert f.loc[4, "g2_merch_n_customers"] == 1


# --- leakage tests ------------------------------------------------------------

def test_day_zero_has_no_history():
    df = toy()
    f = build_graph_features(df, DELAY)
    day0 = df.loc[df.step == 0, "txn_id"]
    counts = [c for c in f if "_n_" in c or c.endswith("_n") or c.endswith("_n_prev")]
    assert (f.loc[day0, counts] == 0).all().all()
    assert (f.loc[day0, "g1_cust_is_new"] == 1).all()


@pytest.mark.parametrize("t", [1, 5, 9, 14])
def test_truncation_future_rows_change_nothing(t):
    """Features for day-t rows are identical whether or not later days exist."""
    df = toy()
    full = build_graph_features(df, DELAY)
    cut = build_graph_features(df[df.step <= t], DELAY)
    rows = df.loc[df.step == t, "txn_id"]
    pd.testing.assert_frame_equal(cut.loc[rows], full.loc[rows], check_exact=False, rtol=1e-12)


def test_same_day_rows_change_nothing():
    """Dropping every other row from the same day leaves a row's features unchanged."""
    df = toy()
    full = build_graph_features(df, DELAY)
    for t in [3, 8, 12]:
        target = df[df.step == t].iloc[0]
        kept = df[(df.step != t) | (df.txn_id == target.txn_id)]
        alone = build_graph_features(kept, DELAY)
        pd.testing.assert_series_equal(alone.loc[target.txn_id], full.loc[target.txn_id],
                                       check_exact=False, rtol=1e-12)


@pytest.mark.parametrize("t", [4, 9, 14])
def test_unknown_labels_change_nothing(t):
    """Shuffling labels the bank wouldn't know yet (days > t - delay) leaves day-t rows unchanged."""
    df = toy()
    full = build_graph_features(df, DELAY)
    shuffled = df.copy()
    late = shuffled.step > t - DELAY
    shuffled.loc[late, "fraud"] = np.random.default_rng(1).permutation(shuffled.loc[late, "fraud"].to_numpy())
    assert (shuffled.fraud != df.fraud).any()
    after = build_graph_features(shuffled, DELAY)
    rows = df.loc[df.step == t, "txn_id"]
    pd.testing.assert_frame_equal(after.loc[rows], full.loc[rows], check_exact=False, rtol=1e-12)


def test_label_free_groups_ignore_labels():
    df = toy()
    full = build_graph_features(df, DELAY)
    flipped = df.assign(fraud=1 - df.fraud)
    after = build_graph_features(flipped, DELAY)
    cols = label_free(full.columns)
    assert len(cols) > 0
    pd.testing.assert_frame_equal(after[cols], full[cols])


def test_label_delay_must_be_positive():
    with pytest.raises(AssertionError):
        build_graph_features(toy(), label_delay=0)


def test_row_order_does_not_matter():
    df = toy()
    a = build_graph_features(df, DELAY)
    b = build_graph_features(df.sample(frac=1, random_state=0), DELAY)
    pd.testing.assert_frame_equal(a, b)


def test_scaler_fits_on_train_only():
    df = toy()
    f = build_graph_features(df, DELAY)
    train = df.loc[df.step <= 9, "txn_id"]
    test = df.loc[df.step > 9, "txn_id"]
    sc = GraphScaler().fit(f.loc[train])
    Xtr, Xte = sc.transform(f.loc[train]), sc.transform(f.loc[test])
    assert np.allclose(Xtr.mean(), 0, atol=1e-9)
    assert not Xte.isna().any().any() and list(Xte.columns) == list(f.columns)


# --- same checks on the real data (skipped if the pipeline hasn't run) ----------

needs_data = pytest.mark.skipif(not CLEAN_FILE.exists(), reason="run the pipeline first")


@pytest.fixture(scope="module")
def real():
    df = pd.read_parquet(CLEAN_FILE)
    return df, build_graph_features(df)


@needs_data
def test_real_one_row_per_transaction(real):
    df, f = real
    assert len(f) == len(df) and f.index.is_unique and not f.isna().any().any()


@needs_data
@pytest.mark.parametrize("t", [0, 119, 120, 150, 179])
def test_real_truncation(real, t):
    """Includes the split boundaries: val/test rows only see what came before them."""
    df, full = real
    cut = build_graph_features(df[df.step <= t])
    rows = df.loc[df.step == t, "txn_id"]
    pd.testing.assert_frame_equal(cut.loc[rows], full.loc[rows], check_exact=False, rtol=1e-9)


@needs_data
def test_real_unknown_labels_change_nothing(real):
    df, full = real
    t = 150
    from src.config import LABEL_DELAY
    shuffled = df.copy()
    late = shuffled.step > t - LABEL_DELAY
    shuffled.loc[late, "fraud"] = np.random.default_rng(0).permutation(shuffled.loc[late, "fraud"].to_numpy())
    after = build_graph_features(shuffled)
    rows = df.loc[df.step == t, "txn_id"]
    pd.testing.assert_frame_equal(after.loc[rows], full.loc[rows], check_exact=False, rtol=1e-9)
