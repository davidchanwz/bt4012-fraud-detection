# Data dictionary: BankSim (cleaned)

Source: BankSim synthetic payments data (Lopez-Rojas & Axelsson, 2014), Kaggle `ealaxi/banksim1`, file `bs140513_032310.csv`.
594,643 transactions, 180 days (`step` 0–179), 4,112 customers, 50 merchants, 15 categories, 7,200 fraud (1.21%).
No missing values and no duplicate rows.

This is the table for the "tabular summary of variables" the report guidelines ask for.

| Variable | Type | Values / range | Notes | Used how |
|---|---|---|---|---|
| `txn_id` | int | 0–594,642 | Added by us: original row order | ID only, never a feature |
| `step` | int | 0–179 | Day index. No time of day, so rows on the same day have no reliable order | Split key; history windows use earlier steps only |
| `customer` | str | 4,112 IDs | Quotes stripped | Graph node, not used directly as a feature |
| `age` | categorical | `0` (≤18), `1` (19–25), `2` (26–35), `3` (36–45), `4` (46–55), `5` (56–65), `6` (>65), `U` (unknown) | `U` (1,178 rows) lines up exactly with gender `E` | One-hot |
| `gender` | categorical | `F`, `M`, `E` (enterprise), `U` (unknown) | `U` has 515 rows and 0 fraud | One-hot |
| `merchant` | str | 50 IDs | Each merchant belongs to exactly one category | Graph node, not used directly as a feature |
| `category` | categorical | 15 values (`es_` prefix removed) | 3 categories (transportation, food, contents) never have fraud | One-hot |
| `amount` | float | 0–8,329.96 (median 26.90) | Heavily right-skewed. 52 rows have amount 0 | `log_amount` = log1p(amount), standardised on train |
| `zero_amount` | int | 0/1 | Added by us | Binary flag |
| `fraud` | int | 0/1 | Target. Exactly 40 fraud rows every day | Label |
| ~~`zipcodeOri`~~ | – | constant `28007` | Dropped | – |
| ~~`zipMerchant`~~ | – | constant `28007` | Dropped | – |

## Cleaning steps (`src/data/clean.py`)
1. Stripped the single quotes wrapped around every string value (e.g. `'C1093826151'` becomes `C1093826151`).
2. Dropped `zipcodeOri` and `zipMerchant`. The script asserts they're constant before dropping them.
3. Removed the `es_` prefix from `category`.
4. Added `txn_id`, `log_amount` and `zero_amount`.
5. Kept the unknown age/gender rows as their own categories instead of dropping them.

## Encoding (`src/features/encode.py`)
- One-hot encoding for `category`, `age` and `gender`, fitted on train only, with `handle_unknown="ignore"`.
- `log_amount` is standardised with the train mean and standard deviation.
- Gives 29 transaction-only features for the baseline models.

## Graph / behavioural features (`src/features/graph.py`)
Built with `python -m src.features.graph` into `data/processed/features_graph.parquet`: one row per `txn_id`, 23 features, no missing values.

**Leakage rule.** A feature for a transaction on day *t* only uses transactions from days before *t*. Same-day rows are never used, because rows within a day have no real order. Label features (`g5_`) are stricter: a fraud label from day *s* is only used from day *s* + 7 (`LABEL_DELAY` in `src/config.py`). A 1-day version is saved as `features_graph_delay1.parquet` for the sensitivity check. `tests/test_features.py` checks all of this.

"Before *t*" means all earlier days, unless the name says 7d (days *t*−7 to *t*−1).

| Feature | Definition |
|---|---|
| **g1 customer history** | |
| `g1_cust_n_prev` | Number of earlier transactions by this customer |
| `g1_cust_is_new` | 1 if the customer has no earlier transactions |
| `g1_cust_n_7d` | Customer's transactions in the last 7 days (velocity) |
| `g1_cust_amt_mean` | Mean of the customer's earlier amounts (0 if none) |
| `g1_cust_amt_std` | Standard deviation of the customer's earlier amounts (0 if none) |
| `g1_cust_amt_z` | (amount − customer mean) / customer std. 0 if fewer than 2 earlier transactions; clipped to ±10 |
| `g1_cust_days_since_last` | Days since the customer's last transaction, capped at 30 (30 if none) |
| `g1_cust_n_merchants` | Distinct merchants the customer paid before |
| `g1_cust_n_categories` | Distinct categories the customer spent in before |
| **g2 merchant history** | |
| `g2_merch_n_prev` | Number of earlier transactions at this merchant |
| `g2_merch_n_7d` | Merchant's transactions in the last 7 days |
| `g2_merch_amt_mean` | Mean of the merchant's earlier amounts |
| `g2_merch_amt_z` | (amount − merchant mean) / merchant std, clipped to ±10 |
| `g2_merch_log_amt_ratio` | log1p(amount) − log1p(merchant mean) |
| `g2_merch_n_customers` | Distinct customers the merchant had before (its degree in the graph) |
| **g3 customer–merchant edge** | |
| `g3_pair_n_prev` | Earlier transactions between this customer and this merchant |
| `g3_pair_is_new` | 1 if this is the first time this customer pays this merchant |
| `g3_pair_days_since_last` | Days since this pair last transacted, capped at 30 (30 if never) |
| `g3_pair_share` | Share of the customer's earlier transactions that went to this merchant |
| **g5 past labels** (only labels at least 7 days old) | |
| `g5_cust_fraud_n` | Known fraud transactions by this customer |
| `g5_cust_fraud_rate` | Customer's known fraud rate, smoothed toward the overall known rate (10 pseudo-transactions) |
| `g5_merch_fraud_rate` | Merchant's known fraud rate, smoothed the same way |
| `g5_pair_fraud_n` | Known fraud transactions on this customer–merchant pair |

For the models, `GraphScaler` applies log1p to the count features and then standardises, fitted on train only, the same pattern as `TransactionEncoder`. `group_columns(cols, ["g1", "g3"])` picks feature groups for the ablation.

Train-split means for fraud and legit rows are in `reports/tables/graph_feature_summary.csv`. The biggest gaps: 59% of fraud is a customer's first payment to that merchant (`g3_pair_is_new`), against 9% of legit transactions. Fraud also happens at much quieter merchants (`g2_merch_n_prev`), and the customer's known fraud history is far higher (`g5_cust_fraud_n` 10.1 against 0.4).
