"""Checks for src/eval (R4's evaluation harness). Run with: pytest -q

Toy data only, so these run without the real pipeline. The main idea being tested:
metrics with weight 1 per row must equal sklearn, and integer weights must equal
the same metrics on rows repeated that many times (that's what a bootstrap is).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score,
    roc_auc_score,
)

from src.eval import metrics as M
from src.eval.core import SortedRun, choose_threshold, n_alerts, stratified_weights
from src.features.subgroup import fit_thresholds

FIXTURES = Path(__file__).parent / "fixtures"
BUDGETS = (0.01, 0.05)


def toy_scores(n=2000, seed=0, rate=0.05):
    """Labels, scores with ties (rounded), ids, amounts and two subset masks."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < rate).astype(int)
    s = np.round(np.clip(0.3 * y + rng.normal(0.3, 0.2, n), 0, 1), 2)    # rounding -> many ties
    ids = rng.permutation(n) + 1000
    amt = rng.gamma(2, 40, n).round(2)
    masks = {"all": np.ones(n, bool), "low": amt < 60, "odd": ids % 2 == 1}
    return y, s, ids, amt, masks


# --- basic conventions ----------------------------------------------------------

def test_n_alerts_matches_plan():
    assert n_alerts(111_306, 0.01) == 1113           # CLAUDE.md: 1,113 alerts at 1%
    assert n_alerts(111_306, 0.02) == 2226
    assert n_alerts(100, 0.29) == 29                 # not 28 from float error
    assert n_alerts(10, 0.01) == 1                   # never an empty queue


def test_ceiling_is_92_75_on_test_sized_data():
    y = np.zeros(111_306, int)
    y[:1200] = 1
    out = M.score_predictions(y, np.random.default_rng(0).random(len(y)))
    assert out["max_recall_at_1pct"] == pytest.approx(0.9275)


# --- point estimates against sklearn ---------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_point_metrics_match_sklearn(seed):
    y, s, ids, amt, masks = toy_scores(seed=seed)
    thr = choose_threshold(y, s)
    res = SortedRun(s, y, ids, amt, masks).evaluate(thr, BUDGETS)
    pred = (s >= thr).astype(int)
    for sub, m in masks.items():
        assert res[(sub, "pr_auc")] == pytest.approx(average_precision_score(y[m], s[m]))
        assert res[(sub, "roc_auc")] == pytest.approx(roc_auc_score(y[m], s[m]))
        assert res[(sub, "precision")] == pytest.approx(precision_score(y[m], pred[m]))
        assert res[(sub, "recall")] == pytest.approx(recall_score(y[m], pred[m]))
        assert res[(sub, "f1")] == pytest.approx(f1_score(y[m], pred[m]))
        assert res[(sub, "fp")] == ((pred == 1) & (y == 0) & m).sum()


def test_choose_threshold_maximises_f1():
    y, s, *_ = toy_scores(seed=3)
    best = max(f1_score(y, (s >= t).astype(int)) for t in np.unique(s))
    assert f1_score(y, (s >= choose_threshold(y, s)).astype(int)) == pytest.approx(best)
    p, r, t = precision_recall_curve(y, s)                  # same answer as sklearn's curve
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    assert choose_threshold(y, s) == pytest.approx(t[np.argmax(f1[:-1])])


def test_budget_by_brute_force():
    y, s, ids, amt, masks = toy_scores(seed=4)
    res = SortedRun(s, y, ids, amt, masks).evaluate(0.5, BUDGETS)
    df = pd.DataFrame({"s": s, "y": y, "id": ids, "amt": amt, "low": masks["low"]})
    q = df.sort_values(["s", "id"], ascending=[False, True]).head(n_alerts(len(df), 0.05))
    assert res[("all", "recall_at_5pct")] == pytest.approx(q.y.sum() / y.sum())
    assert res[("all", "fp_at_5pct")] == (q.y == 0).sum()
    assert res[("all", "amount_caught_at_5pct")] == pytest.approx(q.amt[q.y == 1].sum())
    assert res[("low", "recall_at_5pct")] == pytest.approx((q.y * q.low).sum() / (df.y * df.low).sum())
    assert res[("low", "fp_at_5pct")] == ((q.y == 0) & q.low).sum()


def test_ties_broken_by_txn_id():
    # same example as R3's top_k_flags test: ids 1 and 2 win the tie, 3 doesn't
    s = np.array([.5, .5, .5, .1])
    ids = np.array([3, 1, 2, 0])
    y = np.array([1, 0, 0, 1])                       # the tied fraud row (id 3) is left out
    res = SortedRun(s, y, ids, np.ones(4), {"all": np.ones(4, bool)}).evaluate(0.9, (0.5,))
    assert res[("all", "recall_at_50pct")] == 0.0
    assert res[("all", "fp_at_50pct")] == 2


def test_subgroup_budget_uses_whole_queue():
    # 10 rows, 20% budget -> 2 alerts. The subgroup's fraud ranks 3rd, so it's missed,
    # even though it's the top-ranked row inside the subgroup.
    s = np.linspace(1, 0.1, 10)
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    sub = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1], bool)
    res = SortedRun(s, y, np.arange(10), np.ones(10), {"all": np.ones(10, bool), "sub": sub}).evaluate(
        0.95, (0.2,))
    assert res[("sub", "recall_at_20pct")] == 0.0
    assert np.isnan(res[("sub", "precision_at_20pct")])       # no subgroup rows in the queue
    assert res[("sub", "pr_auc")] == 1.0                       # but it ranks first within the subgroup


# --- weights = bootstrap ---------------------------------------------------------

def test_integer_weights_equal_repeated_rows():
    y, s, ids, amt, masks = toy_scores(n=600, seed=5)
    w = np.random.default_rng(1).integers(0, 4, len(y))
    thr = choose_threshold(y, s)
    weighted = SortedRun(s, y, ids, amt, masks).evaluate(thr, BUDGETS, weights=w)
    rep = lambda a: np.repeat(a, w)
    repeated = SortedRun(rep(s), rep(y), rep(ids), rep(amt), {k: rep(m) for k, m in masks.items()}
                         ).evaluate(thr, BUDGETS)
    assert weighted.keys() == repeated.keys()
    for k in weighted:
        assert weighted[k] == pytest.approx(repeated[k], nan_ok=True), k


def test_stratified_weights():
    y = np.r_[np.ones(30, int), np.zeros(970, int)]
    w0 = stratified_weights(y, 0, seed=4012)
    assert w0[y == 1].sum() == 30 and w0[y == 0].sum() == 970      # class counts fixed
    assert np.array_equal(w0, stratified_weights(y, 0, seed=4012))  # same b -> same rows (paired)
    assert not np.array_equal(w0, stratified_weights(y, 1, seed=4012))


# --- files, schema and the full run --------------------------------------------

def test_parse_score_name():
    assert M.parse_score_name("lr_txn_val") == ("lr", "txn", "val")
    assert M.parse_score_name("lgbm_txn_graph_g1g3_delay1_test") == ("lgbm", "txn_graph_g1g3_delay1", "test")


def test_load_scores_rejects_bad_files(tmp_path):
    ids = np.arange(5)
    bad = {
        "nocol": pd.DataFrame({"txn_id": ids, "p": 0.1}),
        "dup": pd.DataFrame({"txn_id": [0, 0, 1, 2, 3], "score": 0.1}),
        "short": pd.DataFrame({"txn_id": ids[:4], "score": 0.1}),
    }
    for name, df in bad.items():
        df.to_parquet(tmp_path / f"{name}.parquet")
        with pytest.raises(ValueError):
            M.load_scores(tmp_path / f"{name}.parquet", ids)


def test_subgroup_default_unchanged():
    train = pd.DataFrame({"category": ["a"] * 50 + ["b"] * 50, "amount": np.arange(100.0),
                          "fraud": [0] * 100})
    by_cat, cut = fit_thresholds(train)
    by_cat95, cut95 = fit_thresholds(train, quantile=0.95)
    assert by_cat.equals(by_cat95) and cut == cut95


def make_toy_project(root, seed=0, shuffle_test_labels=False):
    """Split files plus score files for lr on txn and txn_graph (graph = better)."""
    rng = np.random.default_rng(seed)
    split_dir, score_dir = root / "splits", root / "scores"
    split_dir.mkdir()
    score_dir.mkdir()
    start = 0
    for split, n in [("train", 3000), ("val", 2000), ("test", 2000)]:
        y = (rng.random(n) < 0.05).astype("int8")
        df = pd.DataFrame({
            "txn_id": np.arange(start, start + n), "step": np.repeat(np.arange(10), n // 10),
            "category": rng.choice(["food", "travel"], n), "amount": rng.gamma(2, 30, n).round(2),
            "fraud": y,
        })
        start += n
        if split != "train":
            noise = rng.normal(0, 0.3, n)
            for fs, signal in [("txn", 0.4), ("txn_graph", 0.8)]:
                pd.DataFrame({"txn_id": df.txn_id, "score": (signal * y + noise).astype("float32")}
                             ).to_parquet(score_dir / f"lr_{fs}_{split}.parquet")
        if split == "test" and shuffle_test_labels:
            df["fraud"] = rng.permutation(df.fraud.to_numpy())
        df.to_parquet(split_dir / f"{split}.parquet")
    return split_dir, score_dir


def run_toy(root, **kw):
    split_dir, score_dir = make_toy_project(root, **kw)
    return M.run(split="test", models=["lr"], scores_dir=score_dir, out_dir=root / "tables",
                 split_dir=split_dir, n_boot=50, log=lambda *a: None)


def test_end_to_end_schema_and_files(tmp_path):
    main, paired, budget = run_toy(tmp_path)
    for f in ["results_main_test.csv", "results_main_paired_test.csv", "results_main_budget_test.csv"]:
        assert (tmp_path / "tables" / f).exists()
    # first columns are exactly R3's proposed schema (tests/fixtures), so results_table.py can read them
    assert list(main.columns[:7]) == list(pd.read_csv(FIXTURES / "results_main_mock.csv").columns)
    assert list(paired.columns[:6]) == list(pd.read_csv(FIXTURES / "results_paired_mock.csv").columns)
    assert {"pr_auc", "recall_at_1pct", "recall_at_1pct_normal_looking", "max_recall_at_1pct",
            "f1_normal_looking_strict", "threshold"} <= set(main.metric)
    ok = main.ci_low.notna()
    assert (main.ci_low[ok] <= main.ci_high[ok]).all()
    # the better graph scores should show up as a positive paired difference
    d = paired.set_index("metric").loc["pr_auc"]
    assert d["diff"] > 0 and d.ci_low > 0
    assert set(budget.policy) == {"budget_1pct", "budget_2pct", "val_f1_threshold"}
    b1 = budget[budget.policy == "budget_1pct"].iloc[0]
    assert b1.n_alerts == n_alerts(2000, 0.01) and b1.tp + b1.fp == b1.n_alerts


def test_threshold_comes_from_val_not_test(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    thr = lambda main: main[main.metric == "threshold"].set_index("feature_set").value
    a, _, _ = run_toy(tmp_path / "a")
    b, _, _ = run_toy(tmp_path / "b", shuffle_test_labels=True)
    pd.testing.assert_series_equal(thr(a), thr(b))         # test labels can't move the threshold


def test_paired_difference_is_zero_for_identical_runs():
    y, s, ids, amt, _ = toy_scores(n=1000, seed=6)
    labels = pd.DataFrame({"txn_id": ids, "fraud": y, "amount": amt, "step": 0,
                           **{sub: 1 for sub in M.SUBSETS[1:]}})
    runs = {("lr", "txn"): s, ("lr", "txn_graph"): s.copy()}
    thr = {k: 0.5 for k in runs}
    points, boots = M.evaluate_runs(labels, runs, thr, n_boot=30, log=lambda *a: None)
    paired = M.paired_table(points, boots, [("txn", "txn_graph")], "val")
    assert (paired["diff"].fillna(0) == 0).all()
    assert (paired.ci_low.fillna(0) == 0).all() and (paired.ci_high.fillna(0) == 0).all()


def test_results_table_reads_eval_output(tmp_path):
    """Contract between R4's files and R3's §4.1 formatter: it must read them unchanged."""
    from src.models import results_table as R
    run_toy(tmp_path)
    t = tmp_path / "tables"
    table = R.run(t / "results_main_test.csv", t / "results_main_paired_test.csv", t / "t.md",
                  split="test", log=lambda *a: None)
    assert list(table.Features) == ["txn", "txn + graph", "Δ (paired)"]
    md = (t / "t.md").read_text()
    assert "Recall @1% (normal-looking)" in md
    main = pd.read_csv(t / "results_main_test.csv")
    ceiling = main.loc[main.metric == "max_recall_at_1pct", "value"].iloc[0]
    assert ceiling < 1 and f"Best possible recall@1% on test is {ceiling:.2%}" in md   # read from the file


def test_budget_curve_and_fp_profile_agree_with_budget_table(tmp_path):
    _, _, budget = run_toy(tmp_path)
    t = tmp_path / "tables"
    curve = pd.read_csv(t / "results_main_budget_curve_test.csv")
    prof = pd.read_csv(t / "results_main_fp_profile_test.csv")
    for fs in ["txn", "txn_graph"]:
        for policy, b in [("budget_1pct", 0.01), ("budget_2pct", 0.02)]:
            row = budget[(budget.policy == policy) & (budget.feature_set == fs)].iloc[0]
            at_b = (curve.feature_set == fs) & ((curve.budget - b).abs() < 1e-9)
            c = curve[at_b & (curve.subset == "all")].iloc[0]
            assert c.tp == row.tp and c.fp == row.fp and c.n_alerts == row.n_alerts
            assert curve[at_b & (curve.subset == "normal_looking")].iloc[0].tp == row.normal_looking_tp
            p = prof[(prof.policy == policy) & (prof.feature_set == fs)]
            assert p.fp.sum() == row.fp                               # every queued legit row counted once
            assert p.fp_share.sum() == pytest.approx(1.0 if row.fp else 0.0)   # empty when no FPs
            assert p.fp_normal_looking.sum() == row.normal_looking_fp
    assert curve.budget.nunique() == 50 and curve.budget.max() == pytest.approx(0.05)
    assert (curve.groupby(["feature_set", "subset"]).recall.apply(lambda r: r.is_monotonic_increasing)).all()


def test_figure_draws_from_tables(tmp_path):
    from src.eval import figures
    run_toy(tmp_path)
    out = figures.run("test", table_dir=tmp_path / "tables", fig_dir=tmp_path / "figs", log=lambda *a: None)
    assert out.exists() and out.stat().st_size > 10_000
