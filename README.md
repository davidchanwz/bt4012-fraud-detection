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
uv run python -m src.features.graph  # -> data/processed/features_graph.parquet (graph features, week 2)
uv run pytest -q                     # sanity checks
uv run jupyter lab                   # notebooks
```

The raw CSV isn't committed because it's too big for the repo. `download.py` pulls it from a public GitHub mirror of the Kaggle dataset and checks the file hash. If that fails, download `bs140513_032310.csv` from [Kaggle (ealaxi/banksim1)](https://www.kaggle.com/datasets/ealaxi/banksim1) and put it in `data/raw/`.

## Models (week 2+)

Needs the processed data and graph features above. **macOS:** LightGBM needs OpenMP, so run `brew install libomp` once, otherwise `import lightgbm` fails with `Library not loaded: @rpath/libomp.dylib`.

```bash
uv run python -m src.models.run_main                 # 3 models x {txn, txn_graph}, val scores (~4 min)
uv run python -m src.models.train --model lgbm --features txn_graph --groups g1,g3   # one run / ablation
uv run python -m src.models.interpret --model lgbm   # feature contributions (val)
uv run python -m src.models.case_studies --model lgbm --split val   # graph-only catches
uv run python -m src.models.run_main --score-test    # at the freeze only
```

Each run writes `reports/scores/{model}_{feature_set}_{split}.parquet` (`txn_id, score`) and a row in `reports/tables/hyperparams.csv`. Test scores are only written with `--score-test`. Fitted models go to `models/` (gitignored).

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
  features/graph.py    leakage-safe graph / history features (g1 customer, g2 merchant,
                       g3 customer-merchant edge, g5 past labels) + GraphScaler
  models/train.py      train + tune one model on one feature set -> scores, hyperparams
  models/run_main.py   the six main runs in one command
  models/interpret.py  LR coefficients, LightGBM contributions
  models/case_studies.py  fraud caught by the graph model only (shortlist)
  models/results_table.py §4.1 table from R4's results CSVs
  eda.py               figures + tables
tests/                 pytest sanity checks (cleaning, split leakage, subgroup,
                       graph-feature leakage, model training contract)
docs/
  plan.md              4-week plan + owners
  data_dictionary.md   variable table for the report
  subgroup_definition.md
  week1_findings.md
notebooks/01_eda.ipynb walkthrough of the EDA
reports/figures, reports/tables
```

## Data source
Lopez-Rojas, E. A. & Axelsson, S. (2014). BankSim: A bank payments simulator for fraud detection research. *26th European Modeling and Simulation Symposium (EMSS)*.
