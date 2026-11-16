"""Business-integration numbers for report §4.5 (R1). Read from R4's tables; never computes a metric.

    uv run python -m src.eval.business                # val (interim, not report numbers)
    uv run python -m src.eval.business --split test   # after the freeze
    uv run python -m src.eval.business --minutes 10   # another review-time assumption

Turns each alerting policy (1% and 2% budget, val F1 threshold) into what an operations team would
ask: how many alerts a day, how many analyst hours, how many of those hours go on false alerts, and
how much fraud value is caught.

Writes to reports/tables/:
- business_{split}.csv: one row per model, feature set and policy.
- business_uplift_{split}.csv: txn -> txn + graph for each model and policy, with the paired
  bootstrap CI where R4's paired table has one (it has none for amounts at the val threshold).

Assumptions, stated here and in §4.5, because the data can't supply them:
- 5 minutes per review, the same default as the dashboard's budget tab (`app/tabs/budget.py`).
- An 8-hour analyst shift. Every review takes the same time, fraud or not.

Amounts are in BankSim's own units; the simulator doesn't state a currency, so don't print one.
"""
import argparse

import numpy as np
import pandas as pd

from src.config import TABLE_DIR

MINUTES_PER_REVIEW = 5.0
SHIFT_HOURS = 8.0
PAIRED_METRIC = {"budget_1pct": "{}_at_1pct", "budget_2pct": "{}_at_2pct", "val_f1_threshold": "{}"}


def business_table(budget, minutes=MINUTES_PER_REVIEW, shift_hours=SHIFT_HOURS):
    """One row per model, feature set and policy. `budget` is results_{name}_budget_{split}.csv."""
    b = budget
    return pd.DataFrame({
        "model": b.model, "feature_set": b.feature_set, "split": b.split, "policy": b.policy,
        "n_days": b.n_days,
        "minutes_per_review": minutes,
        "shift_hours": shift_hours,
        "alerts_per_day": b.alerts_per_day,
        "analyst_hours_per_day": b.alerts_per_day * minutes / 60,
        "analysts_needed": b.alerts_per_day * minutes / 60 / shift_hours,
        "frauds_caught_per_day": b.tp / b.n_days,
        "false_alerts_per_day": b.fp_per_day,
        "hours_on_false_alerts_per_day": b.fp_per_day * minutes / 60,
        "reviews_per_fraud_caught": b.n_alerts / b.tp.replace(0, np.nan),
        "recall": b.recall,
        "max_recall": b.max_recall,
        "precision": b.precision,
        "amount_caught": b.fraud_amount_caught,
        "amount_total": b.fraud_amount_total,
        "amount_share": b.amount_share,
        "amount_caught_per_day": b.fraud_amount_caught / b.n_days,
        "amount_missed_per_day": (b.fraud_amount_total - b.fraud_amount_caught) / b.n_days,
    })


def paired_ci(paired, model, policy, base):
    """(diff, ci_low, ci_high) from R4's paired table, or NaNs if it has no such metric."""
    metric = PAIRED_METRIC.get(policy, "").format(base)
    row = paired[(paired.model == model) & (paired.metric == metric) & (paired.subset == "all")]
    if len(row) != 1:
        return np.nan, np.nan, np.nan
    r = row.iloc[0]
    return r["diff"], r.ci_low, r.ci_high


def uplift_table(budget, paired, minutes=MINUTES_PER_REVIEW):
    """txn -> txn_graph per model and policy."""
    idx = ["model", "split", "policy"]
    txn = budget[budget.feature_set == "txn"].set_index(idx)
    graph = budget[budget.feature_set == "txn_graph"].set_index(idx)
    rows = []
    for key in graph.index.intersection(txn.index):
        t, g = txn.loc[key], graph.loc[key]
        extra = g.tp - t.tp
        extra_amount = g.fraud_amount_caught - t.fraud_amount_caught
        _, amt_low, amt_high = paired_ci(paired, key[0], key[2], "amount_caught")
        _, rec_low, rec_high = paired_ci(paired, key[0], key[2], "recall")
        rows.append({
            "model": key[0], "split": key[1], "policy": key[2],
            "extra_alerts_per_day": g.alerts_per_day - t.alerts_per_day,
            "extra_frauds_caught": extra,
            "extra_frauds_per_day": extra / g.n_days,
            "recall_diff": g.recall - t.recall, "recall_diff_ci_low": rec_low, "recall_diff_ci_high": rec_high,
            "false_alerts_avoided_per_day": t.fp_per_day - g.fp_per_day,
            "analyst_hours_freed_per_day": (t.fp_per_day - g.fp_per_day) * minutes / 60,
            "extra_amount_caught": extra_amount,
            "extra_amount_ci_low": amt_low, "extra_amount_ci_high": amt_high,
            "extra_amount_per_day": extra_amount / g.n_days,
            "amount_share_diff": g.amount_share - t.amount_share,
            # Net: frauds gained minus frauds lost, so this isn't the mean amount of any actual set of rows.
            "amount_per_net_extra_fraud": extra_amount / extra if extra > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def run(split="val", minutes=MINUTES_PER_REVIEW, shift_hours=SHIFT_HOURS, tables=TABLE_DIR, log=print):
    budget = pd.read_csv(tables / f"results_main_budget_{split}.csv")
    paired = pd.read_csv(tables / f"results_main_paired_{split}.csv")
    biz = business_table(budget, minutes, shift_hours)
    up = uplift_table(budget, paired, minutes)
    biz.to_csv(tables / f"business_{split}.csv", index=False, float_format="%.6g")
    up.to_csv(tables / f"business_uplift_{split}.csv", index=False, float_format="%.6g")

    show = biz[["model", "feature_set", "policy", "alerts_per_day", "analyst_hours_per_day",
                "frauds_caught_per_day", "false_alerts_per_day", "amount_share"]]
    log(f"Split: {split}. {minutes:g} min per review, {shift_hours:g}-hour shift (assumptions).")
    log(show.round(3).to_string(index=False))
    log("\ntxn -> txn + graph:")
    log(up[["model", "policy", "extra_frauds_caught", "false_alerts_avoided_per_day", "extra_amount_caught",
            "extra_amount_ci_low", "extra_amount_ci_high", "amount_share_diff",
            "amount_per_net_extra_fraud"]].round(3).to_string(index=False))
    return biz, up


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--minutes", type=float, default=MINUTES_PER_REVIEW, help="assumed minutes per review")
    p.add_argument("--shift-hours", type=float, default=SHIFT_HOURS, help="assumed hours per analyst per day")
    a = p.parse_args(argv)
    run(a.split, a.minutes, a.shift_hours)


if __name__ == "__main__":
    main()
