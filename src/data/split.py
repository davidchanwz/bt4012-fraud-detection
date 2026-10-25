"""Chronological train / val / test split on `step`.

train: steps 0-119, val: 120-149, test: 150-179 (boundaries in src/config.py)

No shuffling -- the whole point is to only ever use the past to predict the
future. Graph/history features in week 2 should be computed from steps
strictly before the transaction's own step.

Also writes reports/tables/split_summary.csv and split_overlap.csv so we can
quote the numbers in the report.

Usage: python -m src.data.split
"""
import pandas as pd

from src.config import CLEAN_FILE, SPLIT_DIR, TABLE_DIR, TRAIN_END, VAL_END


def assign_split(step):
    if step <= TRAIN_END:
        return "train"
    if step <= VAL_END:
        return "val"
    return "test"


def split(df):
    parts = {
        "train": df[df.step <= TRAIN_END],
        "val": df[(df.step > TRAIN_END) & (df.step <= VAL_END)],
        "test": df[df.step > VAL_END],
    }
    assert sum(len(p) for p in parts.values()) == len(df)
    return parts


def split_summary(parts):
    rows = []
    for name, p in parts.items():
        rows.append({
            "split": name,
            "steps": f"{p.step.min()}-{p.step.max()}",
            "n_days": p.step.nunique(),
            "n_txns": len(p),
            "n_fraud": int(p.fraud.sum()),
            "fraud_rate": round(p.fraud.mean(), 5),
            "n_customers": p.customer.nunique(),
            "n_merchants": p.merchant.nunique(),
            "n_categories_with_fraud": p.loc[p.fraud == 1, "category"].nunique(),
        })
    return pd.DataFrame(rows)


def overlap_summary(parts):
    """How many val/test customers/merchants were never seen in train.
    Matters for the graph features -- unseen entities have no history."""
    train = parts["train"]
    seen_c = set(train.customer)
    seen_m = set(train.merchant)
    rows = []
    for name in ["val", "test"]:
        p = parts[name]
        rows.append({
            "split": name,
            "customers_not_in_train": len(set(p.customer) - seen_c),
            "txns_from_unseen_customers": int((~p.customer.isin(seen_c)).sum()),
            "merchants_not_in_train": len(set(p.merchant) - seen_m),
            "txns_at_unseen_merchants": int((~p.merchant.isin(seen_m)).sum()),
        })
    return pd.DataFrame(rows)


def main():
    df = pd.read_parquet(CLEAN_FILE)
    parts = split(df)

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    for name, p in parts.items():
        p.to_parquet(SPLIT_DIR / f"{name}.parquet", index=False)

    summ = split_summary(parts)
    ov = overlap_summary(parts)
    summ.to_csv(TABLE_DIR / "split_summary.csv", index=False)
    ov.to_csv(TABLE_DIR / "split_overlap.csv", index=False)
    print(summ.to_string(index=False))
    print()
    print(ov.to_string(index=False))


if __name__ == "__main__":
    main()
