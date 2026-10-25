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
