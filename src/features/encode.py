"""Transaction-only features for the baseline models.

These are the "no graph" features: amount, category, age, gender.
The encoder is fitted on TRAIN only and reused for val/test, so categories
that only show up later don't leak in (there aren't any in BankSim, but this
is the right habit).

Customer/merchant IDs are deliberately left out here -- anything based on them
(history, counts, fraud rates) goes in the week-2 graph features, computed
from past steps only.

Usage:
    from src.features.encode import TransactionEncoder
    enc = TransactionEncoder().fit(train)
    X_train = enc.transform(train)
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CAT_COLS = ["category", "age", "gender"]
NUM_COLS = ["log_amount"]


class TransactionEncoder:
    def __init__(self):
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.scaler = StandardScaler()

    def fit(self, train):
        self.ohe.fit(train[CAT_COLS].astype(str))
        self.scaler.fit(train[NUM_COLS])
        return self

    def transform(self, df):
        cat = self.ohe.transform(df[CAT_COLS].astype(str))
        num = self.scaler.transform(df[NUM_COLS])
        X = np.hstack([num, cat, df[["zero_amount"]].to_numpy()])
        return pd.DataFrame(X, columns=self.feature_names(), index=df.index)

    def feature_names(self):
        return NUM_COLS + list(self.ohe.get_feature_names_out(CAT_COLS)) + ["zero_amount"]
