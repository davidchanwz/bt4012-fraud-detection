"""Checks for the §4.5 business numbers: arithmetic, agreement with R4's tables and the dashboard."""
import pandas as pd
import pytest

from src.config import TABLE_DIR
from src.eval import business as B

BUDGET_COLS = dict(split="val", n_days=10, n_alerts=100, alerts_per_day=10.0, max_recall=1.0,
                   fraud_amount_total=1000.0)


def toy_budget():
    return pd.DataFrame([
        dict(BUDGET_COLS, model="lr", feature_set="txn", policy="budget_1pct", tp=50, fp=50, fp_per_day=5.0,
             recall=0.5, precision=0.5, fraud_amount_caught=800.0, amount_share=0.8),
        dict(BUDGET_COLS, model="lr", feature_set="txn_graph", policy="budget_1pct", tp=80, fp=20, fp_per_day=2.0,
             recall=0.8, precision=0.8, fraud_amount_caught=860.0, amount_share=0.86),
    ])


def toy_paired():
    return pd.DataFrame({"model": ["lr", "lr"], "metric": ["amount_caught_at_1pct", "recall_at_1pct"],
                         "diff": [60.0, 0.3], "ci_low": [10.0, 0.2], "ci_high": [90.0, 0.4],
                         "subset": ["all", "all"]})


def test_workload_arithmetic():
    r = B.business_table(toy_budget(), minutes=6, shift_hours=4).iloc[0]
    assert r.analyst_hours_per_day == pytest.approx(1.0)        # 10 alerts x 6 min
    assert r.analysts_needed == pytest.approx(0.25)
    assert r.frauds_caught_per_day == pytest.approx(5.0)
    assert r.hours_on_false_alerts_per_day == pytest.approx(0.5)
    assert r.reviews_per_fraud_caught == pytest.approx(2.0)
    assert r.amount_caught_per_day == pytest.approx(80.0)
    assert r.amount_missed_per_day == pytest.approx(20.0)


def test_no_catches_gives_nan_not_a_crash():
    b = toy_budget().assign(tp=0)
    assert B.business_table(b).reviews_per_fraud_caught.isna().all()


def test_uplift_and_paired_ci():
    r = B.uplift_table(toy_budget(), toy_paired(), minutes=6).iloc[0]
    assert r.extra_frauds_caught == 30
    assert r.false_alerts_avoided_per_day == pytest.approx(3.0)
    assert r.analyst_hours_freed_per_day == pytest.approx(0.3)
    assert r.extra_amount_caught == pytest.approx(60.0)
    assert (r.extra_amount_ci_low, r.extra_amount_ci_high) == (10.0, 90.0)
    assert (r.recall_diff_ci_low, r.recall_diff_ci_high) == (0.2, 0.4)
    assert r.amount_per_net_extra_fraud == pytest.approx(2.0)


def test_missing_paired_metric_leaves_ci_empty():
    r = B.uplift_table(toy_budget(), toy_paired().iloc[:0]).iloc[0]
    assert pd.isna(r.extra_amount_ci_low)


def test_no_net_gain_leaves_amount_per_fraud_empty():
    b = toy_budget()
    b.loc[b.feature_set == "txn_graph", "tp"] = 50
    assert pd.isna(B.uplift_table(b, toy_paired()).iloc[0].amount_per_net_extra_fraud)


def test_paired_ci_ignores_subgroup_rows():
    p = pd.concat([toy_paired().assign(subset="normal_looking", ci_low=-1.0), toy_paired()])
    assert B.paired_ci(p, "lr", "budget_1pct", "recall")[1] == 0.2


def test_default_minutes_match_the_dashboard():
    pytest.importorskip("streamlit")
    from app.tabs.budget import DEFAULT_MINUTES
    assert B.MINUTES_PER_REVIEW == DEFAULT_MINUTES


def test_real_val_tables_agree_with_r4(tmp_path):
    """Run on a copy of R4's real tables, then tie the output back to those tables."""
    budget = pd.read_csv(TABLE_DIR / "results_main_budget_val.csv")
    paired = pd.read_csv(TABLE_DIR / "results_main_paired_val.csv")
    budget.to_csv(tmp_path / "results_main_budget_val.csv", index=False)
    paired.to_csv(tmp_path / "results_main_paired_val.csv", index=False)
    biz, up = B.run("val", tables=tmp_path, log=lambda *_: None)

    assert len(biz) == len(budget) and len(up) == len(budget) // 2
    assert (tmp_path / "business_val.csv").exists() and (tmp_path / "business_uplift_val.csv").exists()
    # the extra amount we derive from the budget table is the point estimate of R4's paired diff
    for _, r in up[up.policy != "val_f1_threshold"].iterrows():
        metric = B.PAIRED_METRIC[r.policy].format("amount_caught")
        row = paired[(paired.model == r.model) & (paired.metric == metric) & (paired.subset == "all")].iloc[0]
        assert r.extra_amount_caught == pytest.approx(row["diff"], abs=1.0)
        assert r.extra_amount_ci_low == row.ci_low
