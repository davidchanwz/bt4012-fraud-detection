"""Sanity checks for the week-1 data pipeline. Run with: pytest -q"""
import pandas as pd
import pytest

from src.config import CLEAN_FILE, SPLIT_DIR, TRAIN_END, VAL_END
from src.data.clean import clean
from src.features.encode import TransactionEncoder
from src.features.subgroup import fit_thresholds, flag_normal_looking


def tiny_raw():
    return pd.DataFrame({
        "step": ["0", "0", "1"],
        "customer": ["'C1'", "'C2'", "'C1'"],
        "age": ["'4'", "'U'", "'4'"],
        "gender": ["'M'", "'E'", "'M'"],
        "zipcodeOri": ["'28007'"] * 3,
        "merchant": ["'M1'", "'M2'", "'M1'"],
        "zipMerchant": ["'28007'"] * 3,
        "category": ["'es_travel'", "'es_food'", "'es_travel'"],
        "amount": ["10.5", "0", "999.0"],
        "fraud": ["0", "0", "1"],
    })


def test_clean_strips_quotes_and_drops_zip():
    df = clean(tiny_raw())
    assert "zipcodeOri" not in df and "zipMerchant" not in df
    assert df.customer.tolist() == ["C1", "C2", "C1"]
    assert set(df.category.astype(str)) == {"travel", "food"}
    assert df.zero_amount.tolist() == [0, 1, 0]
    assert df.txn_id.tolist() == [0, 1, 2]


def test_clean_rejects_non_constant_zip():
    raw = tiny_raw()
    raw.loc[0, "zipcodeOri"] = "'99999'"
    with pytest.raises(AssertionError):
        clean(raw)


def test_encoder_ignores_unseen_category():
    df = clean(tiny_raw())
    enc = TransactionEncoder().fit(df.iloc[:2])
    X = enc.transform(df)
    assert X.shape[0] == 3
    assert not X.isna().any().any()


# --- checks on the real processed files (skipped if pipeline not run yet) ---
needs_data = pytest.mark.skipif(not (SPLIT_DIR / "test.parquet").exists(),
                                reason="run the pipeline first")


@needs_data
def test_splits_are_chronological_and_disjoint():
    tr = pd.read_parquet(SPLIT_DIR / "train.parquet")
    va = pd.read_parquet(SPLIT_DIR / "val.parquet")
    te = pd.read_parquet(SPLIT_DIR / "test.parquet")
    assert tr.step.max() == TRAIN_END and va.step.min() == TRAIN_END + 1
    assert va.step.max() == VAL_END and te.step.min() == VAL_END + 1
    assert tr.step.max() < va.step.min() <= va.step.max() < te.step.min()
    ids = [set(x.txn_id) for x in (tr, va, te)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    full = pd.read_parquet(CLEAN_FILE)
    assert len(tr) + len(va) + len(te) == len(full)


@needs_data
def test_every_split_has_enough_fraud():
    for name in ["train", "val", "test"]:
        p = pd.read_parquet(SPLIT_DIR / f"{name}.parquet")
        assert p.fraud.sum() >= 1000, name


@needs_data
def test_subgroup_thresholds_only_use_train():
    tr = pd.read_parquet(SPLIT_DIR / "train.parquet")
    te = pd.read_parquet(SPLIT_DIR / "test.parquet")
    by_cat, g = fit_thresholds(tr)
    # thresholds should match a manual calc on train legit rows only
    legit = tr[tr.fraud == 0]
    assert g == pytest.approx(legit.amount.quantile(0.95))
    manual = legit.groupby("category", observed=True).amount.quantile(0.95)
    assert (by_cat - manual).abs().max() < 1e-9
    flags = flag_normal_looking(te, by_cat, g)
    assert flags.normal_looking.isin([0, 1]).all()
