# R3 (Models): execution plan for the work that can start now

Written 1 Nov 2026. Self-contained: a fresh session should be able to execute this from top to bottom. Read `CLAUDE.md` first for the project context, then `docs/plan.md` for dates and owners.

Seven work items, all doable before R4's metrics code exists:

| # | Item | Real data or mock? | Due (plan.md) |
|---|---|---|---|
| 1 | Smoke test: `txn` + v1 graph features (G1, G3) | real | Thu 5 Nov |
| 2 | Baselines: LR, MLP, LightGBM on `txn` | real | Fri 6 Nov |
| 3 | Graph-augmented runs: all 3 models on `txn_graph` | real | Mon 9 – Tue 10 Nov |
| 4 | Interpretation: LR coefficients, LightGBM contributions | real | Fri 13 Nov |
| 5 | Report §3.3 draft (models and justification) | real (no numbers needed) | Fri 13 Nov |
| 6 | Case-study selection script | **mock scores now**, real later | Fri 13 Nov |
| 7 | §4.1 results-table script and section skeleton | **mock table now**, real later | Mon 16 Nov |

**Item 8 (added 4 Nov): Random Forest** as a robustness model; spec in `docs/plan.md`, "Added after proposal feedback", item B. Due: code by Fri 6 Nov, group OK by Sun 8 Nov.

Build order: Phase 0 → Phase 1 (`train.py`, shared by items 1–3) → items 1, 2, 3 → items 4, 6, 7 → item 5 in any gap.

---

## Phase 0. Setup (do first)

1. **Create the data.** `data/processed/` is empty on this machine. In the user's own terminal (the Claude sandbox blocks the GitHub download), run `./run_week1.sh`, then `uv run python -m src.features.graph`. Check these exist:
   - `data/processed/splits/{train,val,test}.parquet`
   - `data/processed/features_graph.parquet` (23 columns, indexed by `txn_id`)
   - `data/processed/features_graph_delay1.parquet`
2. **Branch.** `git checkout -b r3-models` from up-to-date `main`. One PR for the whole work package at the end is fine, or one per phase.
3. **Add LightGBM.** `uv add lightgbm`, then `uv export --format requirements-txt --no-hashes --no-dev --no-emit-project > requirements.txt`. Commit `pyproject.toml`, `uv.lock` and `requirements.txt` together. On macOS LightGBM needs OpenMP: `brew install libomp` if the import fails. This step needs network, so the user may have to run it.
4. **Gitignore.** Add `models/` (fitted artifacts) and `reports/scores_mock/` (mock files, see item 6).
5. **Read before writing code:** `src/config.py`, `src/features/encode.py`, `src/features/graph.py` (especially `GraphScaler`, `group_columns`, `GROUPS`), `src/features/subgroup.py`, `src/data/split.py`, `tests/test_features.py` (for test style).
6. **Kickoff decisions: settled on 1 Nov 2026** (ticked in `docs/plan.md`; roles: R1 David, R2 Jayden, R3 Aayush, R4 Rayson):
   - LightGBM as third model: yes.
   - G5 label delay: 7 days main, 1 day sensitivity check.
   - sklearn `MLPClassifier`, legit undersampled in train.
   - Train on train, tune on val, score test once at freeze.
   Still my own choices, not group decisions (tell Rayson and Jayden):
   - Test scores are **not** written until `--score-test` is passed (see Phase 1).
   - Commit `reports/scores/*.parquet` for the six main runs (float32 keeps each near 1 MB), not for ablation runs. Rayson needs these, so ask whether he'd rather regenerate.

Verify Phase 0: `uv run python -c "import lightgbm, sklearn; print(lightgbm.__version__)"` and `uv run pytest -q` passes.

---

## Phase 1. `src/models/train.py` (shared by items 1–3)

Create `src/models/__init__.py` and `src/models/train.py`. Split helpers into `src/models/data.py` (feature loading) and `src/models/grids.py` (grids and model factories) if `train.py` passes about 250 lines.

### CLI
```
uv run python -m src.models.train --model {lr,mlp,lgbm} --features {txn,txn_graph}
    [--groups g1,g3]      # only with txn_graph: restrict graph columns to these groups (R2's ablation uses this)
    [--delay 1]           # use features_graph_delay1.parquet instead of the 7-day file
    [--score-test]        # off by default; writes test scores only when passed
```

### Feature-set name (used in filenames and `hyperparams.csv`)
- `txn`
- `txn_graph` (all of G1, G2, G3, G5)
- `txn_graph_g1g3` when `--groups g1,g3` (groups sorted, concatenated)
- `txn_graph_delay1` when `--delay 1`; combinations append both
Write one function `feature_set_name(features, groups, delay)` and unit-test it. Filenames are `{model}_{feature_set}_{split}.parquet`. Parse them from the right: the split is the last `_` token, the model is the first.

### Feature matrices
- Load `train`, `val` and `test` from `SPLIT_DIR`. Fit everything on **train only**.
- `txn`: `TransactionEncoder().fit(train).transform(df)`; set the result's index to `txn_id`.
- `txn_graph`: the `txn` columns plus graph columns `G.loc[df.txn_id]`, optionally restricted with `group_columns`.
- LR and MLP use graph columns passed through `GraphScaler` (fit on train rows). LightGBM uses the raw graph columns, since trees don't need scaling.
- Assert after building: no NaNs, same columns in all splits, row counts match the split summary (374,914 / 108,423 / 111,306).

### Models and grids
Keep grids in one dict so the graph runs reuse them exactly (plan.md: "same grids get reused").
- **LR:** `LogisticRegression(class_weight="balanced", max_iter=2000)`, `C` in `[1e-3, 1e-2, 1e-1, 1, 10, 100]`.
- **MLP:** `MLPClassifier(early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, max_iter=200)`, grid `hidden_layer_sizes` in `[(32,), (64, 32)]` × `alpha` in `[1e-4, 1e-2]`. Imbalance is handled by undersampling: for seed *s*, keep all train fraud and a random sample of legit rows at **10 legit per fraud**, drawn from `np.random.default_rng(SEED + s)` over `txn_id`. The sample depends only on the seed, so `txn` and `txn_graph` runs see exactly the same rows. Seeds `0..4`. Select the grid point by **mean val PR-AUC over the 5 seeds**. The score file holds the mean of the 5 seeds' probabilities on the full val/test sets. `hyperparams.csv` also records the per-seed val PR-AUC mean and sd. (sklearn's early-stopping hold-out comes out of the undersampled train, not out of val.)
- **LightGBM:** `LGBMClassifier(scale_pos_weight=n_legit/n_fraud on train, n_estimators=1000, random_state=SEED)`, grid `num_leaves` in `[15, 31, 63]` × `learning_rate` in `[0.05, 0.1]` × `min_child_samples` in `[20, 100]`. Early stopping on val average precision (`eval_metric="average_precision"`, 50 rounds); record the best iteration. This uses val for tuning, which is the allowed protocol.
- Tuning metric everywhere: `sklearn.metrics.average_precision_score` on val. R4's `metrics.py` isn't needed yet.

### Outputs
- `reports/scores/{model}_{feature_set}_val.parquet`, columns `txn_id` (int) and `score` (float32), one row per val transaction.
- `..._test.parquet` **only with `--score-test`**. Without the flag the script must not even compute test features' predictions. Print "test scoring skipped" so it's obvious.
- `reports/tables/hyperparams.csv`: upsert one row per `(model, feature_set)` with columns `model, feature_set, params (JSON string), val_pr_auc, val_pr_auc_sd (MLP only), best_iteration (LGBM only), label_delay, trained_at`. Re-running a combination replaces its row.
- `models/{model}_{feature_set}.joblib` (gitignored): dict with the fitted model(s), encoder, scaler, feature column names and chosen params. For the MLP save all 5 seed models. Interpretation (item 4) loads these.
- Print a one-line summary: val PR-AUC, number of features, runtime.

### Rules the script must enforce
- Never read `fraud` from val/test for anything except the val PR-AUC used in tuning. Never use test labels at all.
- Same seeds give identical score files on a rerun (check this).
- Whole script is re-runnable end to end with one command per model and feature set.

### Tests: `tests/test_models.py`
Use a toy frame like `toy()` in `tests/test_features.py`; do not require the real data.
- `feature_set_name` cases.
- Score file has columns `txn_id, score`, no duplicates, row count equals the split.
- Without `--score-test` no test file is written; with it, one is.
- MLP undersampling: sample for seed *s* is identical across `txn` and `txn_graph`, contains every train fraud row, and has a 10:1 ratio.
- Fitting only sees train rows (monkeypatch the model's `fit` to assert on the `txn_id` range).
- `hyperparams.csv` upsert replaces rather than duplicates.
- Determinism: two runs with the same seed give equal scores.

Verify Phase 1: `uv run pytest -q` is green and a run of `--model lr --features txn` finishes in seconds with a val PR-AUC printed.

---

## Item 1. Smoke test (Thu 5 Nov)

Goal: prove the pipeline end to end, not tune anything.
```
uv run python -m src.models.train --model lr   --features txn
uv run python -m src.models.train --model lr   --features txn_graph --groups g1,g3
uv run python -m src.models.train --model lgbm --features txn_graph --groups g1,g3
```
Check: score files exist and load, val PR-AUC for `txn` is high (week 1 found amount and category separate fraud strongly; a low value means a bug), and `hyperparams.csv` has the rows. Post the three PR-AUC numbers to the group as the smoke-test result. Note: if `txn_graph_g1g3` is not better than `txn`, that is expected and not a bug (see CLAUDE.md, "What week 1 means").

## Item 2. Baselines (Fri 6 Nov)
```
for m in lr mlp lgbm; do uv run python -m src.models.train --model $m --features txn; done
```
Then check: `hyperparams.csv` has three `txn` rows; the MLP row has a sd; val score files exist for all three. No test scores yet. Put a short results table (val PR-AUC per model) in the group chat. Do not claim budget metrics: R4's `metrics.py` doesn't exist yet.

## Item 3. Graph-augmented runs (Mon 9 – Tue 10 Nov)
```
for m in lr mlp lgbm; do uv run python -m src.models.train --model $m --features txn_graph; done
```
Same grids, no new tuning choices. Because R2 may fix feature bugs, write the loop as `scripts/run_models.sh` (or a `--all` flag) so a full rerun is one command. Record which `features_graph.parquet` the run used (git hash of `src/features/graph.py` in `hyperparams.csv` is enough: add a `graph_code_hash` column). At freeze (Sun 15 Nov) rerun everything with `--score-test`, once.

---

## Item 4. Interpretation (by Fri 13 Nov): `src/models/interpret.py`
```
uv run python -m src.models.interpret --model lr   --features txn_graph --split val
uv run python -m src.models.interpret --model lgbm --features txn_graph --split val
```
- Load `models/{model}_{feature_set}.joblib`.
- **LR:** coefficients on the standardised features (so they're comparable), saved to `reports/tables/lr_coefficients_{feature_set}.csv` sorted by absolute value, with the graph group of each feature.
- **LightGBM:** `booster_.predict(X, pred_contrib=True)` on the chosen split (this gives SHAP-style contributions with no extra dependency; the last column is the bias). Save `reports/tables/lgbm_importance_{feature_set}.csv` with mean |contribution| per feature and per group, and a bar chart to `reports/figures/lgbm_importance_{feature_set}.png`. Also save the per-row contributions for the top-K case-study rows (item 6) to `reports/tables/case_study_contribs.csv`.
- Default `--split val`, so interpretation does not touch test before the freeze. Re-run with `--split test` after the freeze.
- Answer in a short note (put in `docs/r3_interpretation_notes.md`): does `g5_merch_fraud_rate` or another G5 feature dominate? Merchant identity is nearly a label (week 1 finding 4), so if it does, that is a finding to report, and it is the reason R2's G1–G4 vs G5 ablation exists.
- Add `shap` only if something needs it that `pred_contrib` can't do; otherwise skip the dependency.

---

## Item 6. Case-study script, mock first: `src/models/case_studies.py`

Goal: "3–4 case studies of fraud caught only by the graph model."

### Logic
```
uv run python -m src.models.case_studies --model lgbm --split val --budget 0.01 --top 10 [--scores-dir reports/scores]
```
1. Load `{model}_txn_{split}` and `{model}_txn_graph_{split}` score files and the split's labels.
2. Flag the top `floor(budget × n)` by score for each model (budget 1% by default; also run 2%), ties broken by lower `txn_id`. This matches R4's tables and the 1,113-alert / 92.75% figures in `CLAUDE.md`. (Changed from `ceil` on 2 Nov: `ceil` gave 1,085 alerts on val and would give 1,114 on test.)
3. Candidates: `fraud == 1` and flagged by the graph model and **not** flagged by the txn model. Also compute the reverse set (flagged by txn only), and the overlap, so the report can say honestly how many there are in each direction.
4. Add the `normal_looking` and `normal_looking_strict` flags using `src/features/subgroup.py` (read it for the function signatures; thresholds come from train only).
5. Rank candidates (prefer normal-looking, then largest rank improvement). Save the top K to `reports/tables/case_study_candidates.csv` with `txn_id, step, customer, merchant, category, amount`, both scores and ranks, the subgroup flags, and key graph features (`g3_pair_is_new`, `g1_cust_amt_z`, `g5_merch_fraud_rate`, ...).
6. Print the counts: graph-only, txn-only, both. If the graph-only set is empty or tiny, say so; that is a legitimate result.
The human picks 3–4 from the CSV; the script does not decide the narrative.

### Mock first
Write `src/models/mock_scores.py`: for a given `--out reports/scores_mock`, generate `{model}_{txn,txn_graph}_{split}.parquet` with the real `txn_id`s and random scores (make `txn_graph` partly correlated with the label so candidates exist). Run `case_studies.py --scores-dir reports/scores_mock` against it. `reports/scores_mock/` is gitignored, and the script's default `--scores-dir` is the real directory, so mock output can't be mistaken for results. Add a test that runs the selection on a tiny toy score set. When real val scores exist (after item 3), run it for real with no code changes. For test-split case studies, wait for the freeze.

---

## Item 7. §4.1 results table, mock first: `src/models/results_table.py`

R4 owns `metrics.py`. R3 only formats the main table. **Before building, agree the schema with R4 (Rayson).** Proposal for `reports/tables/results_main.csv` (long format):
`model, feature_set, split, metric, value, ci_low, ci_high`
with metrics such as `pr_auc, roc_auc, precision, recall, f1, recall_at_1pct, recall_at_2pct, ...`, plus `reports/tables/results_paired.csv` (`model, metric, diff, ci_low, ci_high`) for txn vs txn+graph.

Build `results_table.py` to turn that into a markdown table (`reports/tables/results_main.md`): 3 models × {txn, txn_graph}, point estimate with 95% CI, the recall@1% ceiling (92.75%) printed beside every recall@1%, paired difference column. Test it on a hand-made mock CSV in the same schema (put it in `tests/fixtures/`). When R4's real file lands, rerun. If R4 uses a different schema, adapt only the reader.

Also create the section skeleton `docs/report/4_1_main_results.md`: headings for the main table, the paired difference, the subgroup headline, interpretation and the case studies, with `TBD` markers where numbers go. No invented numbers. Every number in the final text must come from `reports/tables/`.

---

## Item 5. §3.3 draft (models and justification): `docs/report/3_3_models.md`

About 1 page. Write from decisions, with no result numbers:
- Why these three: LR (interpretable baseline, `class_weight`), MLP (the proposal's non-linear model, small by design), LightGBM (strong tabular baseline, cheap).
- Imbalance handling per model and why (class weights, undersampling 10:1 with the same sample for both feature sets, `scale_pos_weight`), and the note that undersampling distorts probabilities but not rankings, which is why threshold-free and budget metrics are primary.
- Tuning protocol: grids, val PR-AUC, MLP 5 seeds (scores averaged; per-seed sd as a stability check), same grids reused for graph versions, no refit on train+val, threshold picked on val.
- The fairness point: the only difference between paired runs is the feature set.
- Limitations: small grids, one split, simulated data.
- Cross-reference R2's §3.1–3.2 and R4's §3.4 instead of repeating them.

---

## Definition of done
- [ ] `uv run pytest -q` green, including `tests/test_models.py`
- [ ] 3 models × {`txn`, `txn_graph`} have val score files; `hyperparams.csv` has 6 rows
- [ ] Interpretation tables/figure for LR and LightGBM, with the G5-dominance note
- [ ] `case_studies.py` works on mock and on real val scores; candidate CSV produced
- [ ] `results_table.py` works on the mock CSV; `4_1_main_results.md` skeleton exists
- [ ] `3_3_models.md` drafted
- [ ] GenAI use logged in `docs/genai_log.md`
- [ ] PR opened, reviewed by one other person

## Not in this plan
- Test-set scoring and final numbers (freeze, Sun 15 Nov, rerun with `--score-test`)
- Bootstrap CIs, subgroup recall and budget metrics (R4)
- Ablation runs (R2 calls `train.py --groups ...`; this plan only makes sure the flag works)
- The UI (stretch, see `CLAUDE.md`)
