# BT4012 Group 07: Graph features for fraud detection on BankSim

Does adding historical customer–merchant network features to transaction-level features improve fraud detection? We focus on fraud that looks normal on its own.

Setup: the same classifiers (logistic regression, a small feedforward network), trained with and without graph-derived features. We use a chronological train/val/test split and report PR-AUC, precision/recall/F1, and recall within a 1–2% review budget.

## Setup

We use [uv](https://docs.astral.sh/uv/) for dependencies. `uv.lock` pins every package version, and `.python-version` pins Python 3.12, so everyone (and the markers) get the same environment.

```bash
# install uv once (macOS/Linux; see the uv docs for Windows)
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync          # creates .venv with Python 3.12 + all locked packages
```

uv downloads Python 3.12 itself if you don't have it.

To add a package: `uv add lightgbm` (or `uv add --dev <pkg>` for dev-only tools). Commit both `pyproject.toml` and `uv.lock`. Don't `pip install` into the venv directly, or the lockfile won't know about it.

No uv? `requirements.txt` is exported from the lockfile (`uv export --format requirements-txt --no-hashes --no-dev --no-emit-project > requirements.txt`). So `pip install -r requirements.txt` on Python 3.12 gets the same versions. Re-export it whenever the lockfile changes.

## Reproduce week 1 (data + EDA)

```bash
./run_week1.sh
```

or step by step:

```bash
uv run python -m src.data.download   # gets data/raw/bs140513_032310.csv (48 MB) and checks its hash
uv run python -m src.data.clean      # -> data/processed/banksim_clean.parquet
uv run python -m src.data.split      # -> data/processed/splits/{train,val,test}.parquet
uv run python -m src.eda             # -> reports/figures/*.png, reports/tables/*.csv
uv run pytest -q                     # sanity checks
uv run jupyter lab                   # notebooks
```

The raw CSV isn't committed because it's too big for the repo. `download.py` pulls it from a public GitHub mirror of the Kaggle dataset and checks the file hash. If that fails, download `bs140513_032310.csv` from [Kaggle (ealaxi/banksim1)](https://www.kaggle.com/datasets/ealaxi/banksim1) and put it in `data/raw/`.

## Data split

| split | days (`step`) | txns | fraud |
|---|---|---|---|
| train | 0–119 | 374,914 | 4,800 |
| val | 120–149 | 108,423 | 1,200 |
| test | 150–179 | 111,306 | 1,200 |

The boundaries live in `src/config.py`. **Don't change them.** Any history or graph feature for a transaction on day *t* may only use days < *t*.

## Repo layout

```
src/
  config.py            paths + split boundaries
  data/download.py     get + verify raw CSV
  data/clean.py        strip quotes, drop constant zip cols, add log_amount
  data/split.py        chronological split + summary tables
  features/encode.py   transaction-only features (one-hot + scaled amount), fit on train
  features/subgroup.py "normal-looking fraud" flags (fixed in week 1)
  eda.py               figures + tables
tests/                 pytest sanity checks (cleaning, split leakage, subgroup)
docs/
  data_dictionary.md   variable table for the report
  subgroup_definition.md
  week1_findings.md
notebooks/01_eda.ipynb walkthrough of the EDA
reports/figures, reports/tables
```

## Data source
Lopez-Rojas, E. A. & Axelsson, S. (2014). BankSim: A bank payments simulator for fraud detection research. *26th European Modeling and Simulation Symposium (EMSS)*.
