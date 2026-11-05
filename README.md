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
uv run python -m src.eda --normal-looking   # -> figure 07: normal-looking fraud vs legit (needs the graph features)
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

## Evaluation (week 2+)

Reads the score files above and never retrains. About 4 minutes for the six main runs (1,000 bootstrap resamples).

```bash
uv run python -m src.eval.metrics                    # main six runs on val
uv run python -m src.eval.metrics --split test       # at the freeze only, after --score-test
uv run python -m src.eval.metrics --models lr,lgbm --feature-sets txn_graph,txn_graph_g1g3 \
    --pairs txn_graph:txn_graph_g1g3 --name ablation # any other comparison, e.g. R2's ablation
```

Writes to `reports/tables/`:
- `results_{name}_{split}.csv`: long format, `model, feature_set, split, metric, value, ci_low, ci_high, subset, n_rows, n_fraud`. PR-AUC, ROC-AUC; precision / recall / F1 / FP at the val-chosen F1 threshold; recall, precision, FP and fraud amount caught at the 1% and 2% review budgets (with the `max_recall` ceiling). Every metric on all rows and on `normal_looking`, `normal_looking_strict` and `normal_looking_q90` (sensitivity check). Subgroup metrics carry the subgroup as a suffix, e.g. `recall_at_1pct_normal_looking`.
- `results_{name}_paired_{split}.csv`: `diff = new - base` (default `txn_graph - txn`) with a paired bootstrap CI.
- `results_{name}_budget_{split}.csv`: one row per run and alerting policy (each budget, plus the val threshold): alerts and false positives per day, fraud amount caught, and how many queued legit rows look normal.
- `results_{name}_budget_curve_{split}.csv`: recall, precision and FP at every budget from 0.1% to 5% (point estimates), per subgroup. Feeds the figure and the UI's budget slider.
- `results_{name}_fp_profile_{split}.csv`: the legit rows in each 1% / 2% queue by category (count, share, how many look normal, median amount, share of that category's legit rows that get queued).

Figures, drawn only from those tables: `uv run python -m src.eval.figures [--split test]` → `reports/figures/budget_curve_{split}.png` (recall against review budget, all vs normal-looking fraud, txn vs txn + graph).

CIs are 95% percentile intervals from a stratified bootstrap (fraud and legit resampled separately, seed from `src/config.py`). The budget queue is the top `floor(budget × n)` rows of the whole split, ties broken by lower `txn_id`; subgroup recall at a budget is the share of that subgroup's fraud that lands in this same queue. Python helper for model selection: `from src.eval.metrics import score_predictions`.

## Run the dashboard (stretch)

An interactive, read-only view of the results tables. Sidebar controls (split, model, features, fraud subset) apply to every tab: review-budget simulator, scoreboard, why graph features, case viewer, false positives, ablation and label delay.

```bash
uv run streamlit run app/main.py      # opens http://localhost:8501
uv run python tests/app_check.py --empty    # headless check, no browser: renders every tab under 48 sidebar
                                              # selections, with and without the tables (~1 min, prints 'All OK')
```

It only reads `reports/tables/` (committed), so it works on a fresh clone without retraining. The case viewer's customer-history panel also reads `data/processed/` and shows a notice if that folder hasn't been rebuilt. It never trains, scores or recomputes a metric; the one derived number is analyst hours (alerts per day × an editable minutes-per-review assumption). Sources that don't exist yet (the ablation, the test split) show a "waiting on" card listing the files that will fill them in. Theme: `.streamlit/config.toml`; shared look: `app/ui.py`; loaders: `app/data.py`.

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
  eval/core.py         metric maths (weighted, so the bootstrap reuses it)
  eval/metrics.py      score files -> results tables with CIs (R4)
  eval/figures.py      budget-curve figure from the results tables
  eda.py               figures + tables
app/                   Streamlit dashboard (read-only on reports/tables): main.py, data.py, ui.py, tabs/
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
