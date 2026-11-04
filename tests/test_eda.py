"""Checks for the normal-looking fraud vs legit comparison in src/eda.py. Run with: pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src.eda import NL_FEATURES, category_weights, normal_looking_comparison, smd


def test_category_weights_give_legit_the_fraud_mix():
    fraud = pd.Series(["a"] * 30 + ["b"] * 70)
    legit = pd.Series(["a"] * 900 + ["b"] * 50 + ["c"] * 50)      # c has no fraud
    w = category_weights(legit, fraud)
    mix = pd.Series(w).groupby(legit).sum() / w.sum()
    assert mix["a"] == pytest.approx(0.3) and mix["b"] == pytest.approx(0.7)
    assert mix["c"] == 0


def test_smd_signs_and_scale():
    rng = np.random.default_rng(0)
    a, b = rng.normal(1, 1, 5000), rng.normal(0, 1, 5000)
    assert smd(a, b) == pytest.approx(1.0, abs=0.05)                # one sd apart
    assert smd(b, a) == pytest.approx(-smd(a, b))
    assert smd(b, b) == pytest.approx(0)
    w = np.r_[np.zeros(2500), np.ones(2500)]                         # weights pick a subset
    assert smd(a, b, w) == pytest.approx(smd(a, b[2500:]))


def test_comparison_on_toy_rows():
    rng = np.random.default_rng(1)
    n = 400
    train = pd.DataFrame({
        "txn_id": np.arange(n), "category": rng.choice(["food", "health"], n),
        "amount": rng.gamma(2, 20, n), "fraud": (rng.random(n) < 0.2).astype(int),
        "normal_looking": (rng.random(n) < 0.8).astype(int),
    })
    cols = [c for c, _, _ in NL_FEATURES if c != "amount"]
    graph = pd.DataFrame(rng.random((n, len(cols))), columns=cols, index=pd.Index(np.arange(n), name="txn_id"))
    feat, mix, (f, l, w) = normal_looking_comparison(train, graph)
    assert list(feat.feature) == [c for c, _, _ in NL_FEATURES]
    nl = train[train.normal_looking == 1]
    assert len(f) == nl.fraud.sum() and len(l) == (nl.fraud == 0).sum()
    assert mix.share_of_nl_fraud.sum() == pytest.approx(1) and mix.share_of_nl_legit.sum() == pytest.approx(1)
    assert feat.smd_vs_same_category_legit.notna().all()
