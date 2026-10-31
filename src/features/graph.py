"""Graph / behavioural features from the customer-merchant network.

Every feature for a transaction on day t uses ONLY rows from days < t.
Rows on the same day have no real order (see clean.py), so same-day rows
are never used -- not even other transactions by the same customer.
Label-based features (g5_) are stricter still: a fraud label from day s is
only used from day s + label_delay onwards.

Groups (column prefix = group, so ablation can drop a group by prefix):
    g1_  customer history      how this customer has behaved so far
    g2_  merchant history      how busy / what amounts this merchant has seen
    g3_  customer-merchant     history of this exact edge in the graph
    g5_  past labels           known fraud on this customer / merchant / edge
(g4_ network structure is a stretch goal, not built yet.)

How it works: one (entity x day) matrix per statistic, cumulative-summed
along days. "Total up to day t-1" is then just a lookup at column t-1, so
the whole thing is vectorised and takes a few seconds on the full data.
The output never depends on rows after day t-1 (for g5_: day t-label_delay),
which tests/test_features.py checks.

Usage:
    python -m src.features.graph            # -> data/processed/features_graph.parquet
                                            #    (+ features_graph_delay1.parquet)
    from src.features.graph import load_graph_features, GraphScaler
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import GRAPH_FILE, CLEAN_FILE, LABEL_DELAY, TABLE_DIR, TRAIN_END

WINDOW = 7          # days for the "recent velocity" features
GAP_CAP = 30        # days-since features are capped here (also used for "never seen")
Z_CAP = 10          # amount z-scores are clipped to +/- this
PRIOR_WEIGHT = 10   # pseudo-transactions for smoothing fraud rates toward the global rate

GROUPS = {
    "g1": "customer history",
    "g2": "merchant history",
    "g3": "customer-merchant edge",
    "g5": "past labels",
}

# counts get log1p'd before scaling (they're heavy-tailed)
COUNT_COLS = [
    "g1_cust_n_prev", "g1_cust_n_7d", "g1_cust_n_merchants", "g1_cust_n_categories",
    "g2_merch_n_prev", "g2_merch_n_7d", "g2_merch_n_customers",
    "g3_pair_n_prev",
    "g5_cust_fraud_n", "g5_pair_fraud_n",
]


# --- matrix helpers -----------------------------------------------------------

def _daily(codes, steps, n_codes, n_steps, weights=None):
    """(entity x day) matrix of per-day totals."""
    m = np.zeros((n_codes, n_steps))
    np.add.at(m, (codes, steps), 1.0 if weights is None else weights)
    return m


def _at(cum, codes, day, fill=0.0):
    """cum[code, day] per row, or `fill` where day < 0."""
    out = np.full(len(codes), fill, dtype=float)
    ok = day >= 0
    out[ok] = cum[codes[ok], day[ok]]
    return out


def _upto(m, codes, day):
    """Total of m over days 0..day (inclusive) for each row's entity."""
    return _at(m.cumsum(axis=1), codes, day)


def _window(m, codes, t, w):
    """Total over days t-w .. t-1."""
    cum = m.cumsum(axis=1)
    return _at(cum, codes, t - 1) - _at(cum, codes, t - 1 - w)


def _days_since_last(m, codes, t):
    """Days since the entity was last active before day t (capped at GAP_CAP)."""
    n_steps = m.shape[1]
    active_day = np.where(m > 0, np.arange(n_steps), -1)
    last = np.maximum.accumulate(active_day, axis=1)
    prev = _at(last, codes, t - 1, fill=-1)
    gap = np.where(prev < 0, GAP_CAP, t - prev)
    return np.minimum(gap, GAP_CAP)


def _first_seen_counts(owner_codes, pair_codes, steps, n_owner, n_steps):
    """(owner x day) count of pairs the owner had for the first time that day.
    Summed up to t-1 this gives the number of distinct counterparties before t."""
    first = pd.Series(steps).groupby(pair_codes).min()
    owner_of_pair = pd.Series(owner_codes).groupby(pair_codes).first()
    return _daily(owner_of_pair.to_numpy(), first.to_numpy(), n_owner, n_steps)


def _amount_stats(m_n, m_sum, m_sq, codes, t, amount):
    n = _upto(m_n, codes, t - 1)
    s = _upto(m_sum, codes, t - 1)
    sq = _upto(m_sq, codes, t - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, s / n, 0.0)
        var = np.where(n > 0, sq / n - mean ** 2, 0.0)
    std = np.sqrt(np.clip(var, 0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where((n >= 2) & (std > 0), (amount - mean) / std, 0.0)
    return n, mean, std, np.clip(z, -Z_CAP, Z_CAP)


def _smoothed_rate(fraud, n, prior):
    return (fraud + PRIOR_WEIGHT * prior) / (n + PRIOR_WEIGHT)


# --- main builder -------------------------------------------------------------

def build_graph_features(df, label_delay=LABEL_DELAY):
    """One row per txn_id with g1_/g2_/g3_/g5_ columns. No NaNs.

    df: cleaned BankSim rows (any subset of days, any row order).
    """
    assert label_delay >= 1, "label_delay must be >= 1 (same-day labels are never known)"
    df = df.sort_values("txn_id")
    t = df["step"].to_numpy().astype(int)
    n_steps = int(t.max()) + 1
    amount = df["amount"].to_numpy()
    fraud = df["fraud"].to_numpy().astype(float)

    cust, cust_ids = pd.factorize(df["customer"])
    merch, merch_ids = pd.factorize(df["merchant"])
    pair, _ = pd.factorize(df["customer"].astype(str) + "|" + df["merchant"].astype(str))
    cust_cat, _ = pd.factorize(df["customer"].astype(str) + "|" + df["category"].astype(str))
    n_c, n_m, n_p = len(cust_ids), len(merch_ids), pair.max() + 1

    out = pd.DataFrame(index=pd.Index(df["txn_id"].to_numpy(), name="txn_id"))

    # g1: customer history
    c_n = _daily(cust, t, n_c, n_steps)
    c_sum = _daily(cust, t, n_c, n_steps, amount)
    c_sq = _daily(cust, t, n_c, n_steps, amount ** 2)
    n, mean, std, z = _amount_stats(c_n, c_sum, c_sq, cust, t, amount)
    out["g1_cust_n_prev"] = n
    out["g1_cust_is_new"] = (n == 0).astype(float)
    out["g1_cust_n_7d"] = _window(c_n, cust, t, WINDOW)
    out["g1_cust_amt_mean"] = mean
    out["g1_cust_amt_std"] = std
    out["g1_cust_amt_z"] = z
    out["g1_cust_days_since_last"] = _days_since_last(c_n, cust, t)
    c_new_m = _first_seen_counts(cust, pair, t, n_c, n_steps)
    out["g1_cust_n_merchants"] = _upto(c_new_m, cust, t - 1)
    c_new_cat = _first_seen_counts(cust, cust_cat, t, n_c, n_steps)
    out["g1_cust_n_categories"] = _upto(c_new_cat, cust, t - 1)

    # g2: merchant history
    m_n = _daily(merch, t, n_m, n_steps)
    m_sum = _daily(merch, t, n_m, n_steps, amount)
    m_sq = _daily(merch, t, n_m, n_steps, amount ** 2)
    n, mean, std, z = _amount_stats(m_n, m_sum, m_sq, merch, t, amount)
    out["g2_merch_n_prev"] = n
    out["g2_merch_n_7d"] = _window(m_n, merch, t, WINDOW)
    out["g2_merch_amt_mean"] = mean
    out["g2_merch_amt_z"] = z
    out["g2_merch_log_amt_ratio"] = np.where(n > 0, np.log1p(amount) - np.log1p(mean), 0.0)
    m_new_c = _first_seen_counts(merch, pair, t, n_m, n_steps)
    out["g2_merch_n_customers"] = _upto(m_new_c, merch, t - 1)

    # g3: customer-merchant edge
    p_n = _daily(pair, t, n_p, n_steps)
    pair_prev = _upto(p_n, pair, t - 1)
    out["g3_pair_n_prev"] = pair_prev
    out["g3_pair_is_new"] = (pair_prev == 0).astype(float)
    out["g3_pair_days_since_last"] = _days_since_last(p_n, pair, t)
    cust_prev = out["g1_cust_n_prev"].to_numpy()
    out["g3_pair_share"] = np.where(cust_prev > 0, pair_prev / np.maximum(cust_prev, 1), 0.0)

    # g5: past labels, only those known by day t (label from day s known at s + delay)
    known = t - label_delay
    g_n = _daily(np.zeros_like(t), t, 1, n_steps)
    g_f = _daily(np.zeros_like(t), t, 1, n_steps, fraud)
    zeros = np.zeros_like(t)
    g_n_known, g_f_known = _upto(g_n, zeros, known), _upto(g_f, zeros, known)
    prior = np.where(g_n_known > 0, g_f_known / np.maximum(g_n_known, 1), 0.0)

    c_f = _daily(cust, t, n_c, n_steps, fraud)
    c_fraud, c_lab = _upto(c_f, cust, known), _upto(c_n, cust, known)
    out["g5_cust_fraud_n"] = c_fraud
    out["g5_cust_fraud_rate"] = _smoothed_rate(c_fraud, c_lab, prior)

    m_f = _daily(merch, t, n_m, n_steps, fraud)
    out["g5_merch_fraud_rate"] = _smoothed_rate(_upto(m_f, merch, known), _upto(m_n, merch, known), prior)

    p_f = _daily(pair, t, n_p, n_steps, fraud)
    out["g5_pair_fraud_n"] = _upto(p_f, pair, known)

    assert not out.isna().any().any()
    return out


# --- for the models -----------------------------------------------------------

def group_columns(columns, groups):
    """Columns belonging to the given groups, e.g. group_columns(X.columns, ["g1", "g3"])."""
    return [c for c in columns if c.split("_")[0] in groups]


def load_graph_features(path=GRAPH_FILE):
    return pd.read_parquet(path)


class GraphScaler:
    """log1p on count columns, then standardise. Fit on TRAIN rows only.

    Usage (same pattern as TransactionEncoder):
        G = load_graph_features()
        scaler = GraphScaler().fit(G.loc[train.txn_id])
        G_train = scaler.transform(G.loc[train.txn_id])
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def _prep(self, X):
        X = X.copy()
        cols = [c for c in COUNT_COLS if c in X]
        X[cols] = np.log1p(X[cols])
        return X

    def fit(self, X):
        self.columns = list(X.columns)
        self.scaler.fit(self._prep(X))
        return self

    def transform(self, X):
        X = self._prep(X[self.columns])
        return pd.DataFrame(self.scaler.transform(X), columns=self.columns, index=X.index)


# --- script -------------------------------------------------------------------

def feature_summary(df, feats):
    """Mean of each feature for fraud vs legit on TRAIN rows (for the report)."""
    train = df[df.step <= TRAIN_END].set_index("txn_id")
    f = feats.loc[train.index]
    s = f.groupby(train["fraud"]).mean().T
    s.columns = ["legit_mean", "fraud_mean"]
    s.insert(0, "group", [GROUPS[c.split("_")[0]] for c in s.index])
    return s.rename_axis("feature").round(4)


def main():
    df = pd.read_parquet(CLEAN_FILE)

    feats = build_graph_features(df, LABEL_DELAY)
    feats.to_parquet(GRAPH_FILE)
    print(f"saved {GRAPH_FILE}  ({feats.shape[0]:,} rows x {feats.shape[1]} features, "
          f"label delay {LABEL_DELAY} days)")

    # 1-day delay version for the label-delay sensitivity check
    alt = GRAPH_FILE.with_name("features_graph_delay1.parquet")
    build_graph_features(df, 1).to_parquet(alt)
    print(f"saved {alt}")

    summ = feature_summary(df, feats)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summ.to_csv(TABLE_DIR / "graph_feature_summary.csv")
    print()
    print(summ.to_string())


if __name__ == "__main__":
    main()
