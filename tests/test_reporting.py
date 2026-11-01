"""Checks for the case-study selection and the §4.1 table formatter. Run with: pytest -q"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models import results_table as R
from src.models.case_studies import select, top_k_flags

FIXTURES = Path(__file__).parent / "fixtures"


# --- case studies -------------------------------------------------------------

def toy_split():
    """10 rows, 4 fraud. With a 20% budget each model flags 2 rows."""
    df = pd.DataFrame({
        "step": 0, "customer": "C", "merchant": "M", "category": "food", "amount": 10.0,
        "fraud":                 [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        "normal_looking":        [0, 1, 1, 0, 1, 1, 1, 1, 1, 1],
        "normal_looking_strict": [0, 0, 1, 0, 1, 1, 1, 1, 1, 1],
    }, index=pd.Index(range(100, 110), name="txn_id"))
    # txn flags rows 100, 103; graph flags rows 100, 102
    s_txn = pd.Series([.9, .1, .2, .8, .3, .3, .3, .3, .3, .3], index=df.index)
    s_graph = pd.Series([.9, .1, .8, .2, .3, .3, .3, .3, .3, .3], index=df.index)
    return df, s_txn, s_graph


def test_top_k_flags_size_and_ties():
    s = pd.Series([.5, .5, .5, .1], index=[3, 1, 2, 0])
    f = top_k_flags(s, 0.5)
    assert f.sum() == 2
    assert f.loc[1] == 1 and f.loc[2] == 1 and f.loc[3] == 0     # ties broken by txn_id


def test_select_counts_and_candidates():
    df, s_txn, s_graph = toy_split()
    counts, cand = select(df, s_txn, s_graph, budget=0.2, top=5)
    assert counts["caught_both"] == 1            # row 100
    assert counts["caught_graph_only"] == 1      # row 102
    assert counts["caught_txn_only"] == 1        # row 103
    assert counts["missed_both"] == 1            # row 101
    assert counts["graph_only_normal_looking"] == 1
    assert cand.txn_id.tolist() == [102]
    assert cand.loc[0, "rank_gain"] > 0


def test_select_handles_no_graph_only():
    df, s_txn, _ = toy_split()
    counts, cand = select(df, s_txn, s_txn, budget=0.2, top=5)
    assert counts["caught_graph_only"] == 0 and cand.empty


# --- results table ------------------------------------------------------------

def test_results_table_on_fixture(tmp_path):
    out = tmp_path / "t.md"
    table = R.run(FIXTURES / "results_main_mock.csv", FIXTURES / "results_paired_mock.csv", out,
                  log=lambda *a: None)
    assert len(table) == 9                                     # 3 models x (txn, txn+graph, Δ)
    md = out.read_text()
    assert "92.75%" in md                                      # ceiling printed with recall@1%
    assert "+0.030 [0.010, 0.050]" in md


def test_results_table_rejects_wrong_schema(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"model": ["lr"], "value": [0.5]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        R.read_main(bad)


def test_fmt_ci():
    assert R.fmt_ci(0.5, 0.4, 0.6) == "0.500 [0.400, 0.600]"
    assert R.fmt_ci(0.5, np.nan, np.nan) == "0.500"
    assert R.fmt_ci(np.nan, 0, 1) == "–"
    assert R.fmt_ci(0.02, -0.01, 0.05, signed=True) == "+0.020 [-0.010, 0.050]"
