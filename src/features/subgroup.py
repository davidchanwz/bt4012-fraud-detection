"""'Normal-looking fraud' subgroup -- defined in week 1, BEFORE any model results.
See docs/subgroup_definition.md for the reasoning.

Primary definition:
    a transaction is "normal-looking" if its amount is <= the 95th percentile
    of LEGIT amounts in the same category, with percentiles computed on the
    TRAIN split only.

Secondary (stricter) definition, reported as a robustness check:
    amount <= the 95th percentile of ALL legit train amounts (one global cut).

Both are applied to every row (fraud or not), so we can compute recall on the
fraud rows and false positives on the legit rows inside the subgroup.
"""
import pandas as pd

QUANTILE = 0.95


def fit_thresholds(train, quantile=QUANTILE):
    # `quantile` is only changed for the 90th-percentile sensitivity check (R4);
    # the subgroup definition itself stays at QUANTILE = 0.95.
    legit = train[train.fraud == 0]
    by_cat = legit.groupby("category", observed=True)["amount"].quantile(quantile)
    global_cut = float(legit["amount"].quantile(quantile))
    return by_cat, global_cut


def flag_normal_looking(df, by_cat, global_cut):
    cat_cut = df["category"].astype(str).map(by_cat.rename(index=str))
    out = pd.DataFrame(index=df.index)
    out["normal_looking"] = (df["amount"] <= cat_cut).astype("int8")
    out["normal_looking_strict"] = (df["amount"] <= global_cut).astype("int8")
    return out
