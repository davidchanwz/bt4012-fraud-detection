"""Clean the raw BankSim CSV -> data/processed/banksim_clean.parquet

What this does:
- strips the stray single quotes around every string field ('C1093826151' etc.)
- drops zipcodeOri and zipMerchant (each has only one value, 28007)
- keeps age as a category ('0'-'6' plus 'U'); age 'U' rows are all gender 'E'
  (enterprise accounts), so we keep them instead of dropping
- adds txn_id = original row order. Rows inside the same step have no real
  timestamp, so the feature code must NOT use same-step history (we can't tell
  which transaction came first within a day).
- adds a few simple transaction-level columns (log_amount, zero_amount flag)

Usage: python -m src.data.clean
"""
import numpy as np
import pandas as pd

from src.config import CLEAN_FILE, PROCESSED_DIR, RAW_FILE

STRING_COLS = ["customer", "age", "gender", "merchant", "category"]
DROP_COLS = ["zipcodeOri", "zipMerchant"]

AGE_LABELS = {
    "0": "<=18", "1": "19-25", "2": "26-35", "3": "36-45",
    "4": "46-55", "5": "56-65", "6": ">65", "U": "unknown",
}
GENDER_LABELS = {"F": "female", "M": "male", "E": "enterprise", "U": "unknown"}


def load_raw(path=RAW_FILE):
    # read everything as str first so '0' ages etc. don't get turned into ints
    return pd.read_csv(path, dtype=str)


def clean(raw):
    df = raw.copy()

    # every column comes in as text (dtype=str), so strip quotes from all of them.
    # (don't check `dtype == object` -- pandas 3 reads text as the new "str" dtype)
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.strip("'")

    # constant columns -- check before dropping in case a different file is used
    for col in DROP_COLS:
        n = df[col].nunique()
        assert n == 1, f"{col} expected to be constant but has {n} values"
    df = df.drop(columns=DROP_COLS)

    df["step"] = df["step"].astype("int16")
    df["amount"] = df["amount"].astype("float64")
    df["fraud"] = df["fraud"].astype("int8")

    df["category"] = df["category"].str.replace("es_", "", regex=False)

    for col in ["age", "gender", "category"]:
        df[col] = df[col].astype("category")

    df.insert(0, "txn_id", np.arange(len(df), dtype="int32"))

    df["log_amount"] = np.log1p(df["amount"])
    df["zero_amount"] = (df["amount"] == 0).astype("int8")  # 52 rows have amount 0

    return df


def summarise(df):
    print(f"rows: {len(df):,}  cols: {df.shape[1]}")
    print(f"days: {df.step.min()}-{df.step.max()}  customers: {df.customer.nunique():,}  "
          f"merchants: {df.merchant.nunique()}  categories: {df.category.nunique()}")
    print(f"fraud: {df.fraud.sum():,} ({df.fraud.mean():.3%})")
    print(f"missing values: {int(df.isna().sum().sum())}  duplicate rows: {int(df.duplicated().sum())}")


def main():
    raw = load_raw()
    df = clean(raw)
    summarise(df)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLEAN_FILE, index=False)
    print(f"saved {CLEAN_FILE}")


if __name__ == "__main__":
    main()
