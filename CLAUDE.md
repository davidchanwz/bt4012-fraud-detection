# CLAUDE.md: BT4012 Group 07, graph features for fraud detection on BankSim

Team-wide context. This file is for whoever is working in the repo, whatever their role. `docs/plan.md` is the source of truth for dates, owners and checkbox status; don't duplicate its status here. State when this file was written: 1 Nov 2026 (week 1 and R2's graph features are merged to `main`).

## Deadline and deliverables
**Fri 20 Nov 2026, 23:59** (aim to submit by midday). Three files:
- `report07.pdf`: 8–12 pages excluding appendices and references, with the GitHub URL
- `presentation07.pptx`
- `video07.mp4`: at most 15 minutes, every member speaks (aim for 13:30)

A wrong file name costs 2% of that component. Results freeze **Sun 15 Nov** (tag `v-results`); after that, bug fixes only. Every number in the report comes from `reports/tables/`.

Key dates: kickoff Sat 31 Oct · checkpoint Sun 8 Nov · freeze Sun 15 Nov · record Wed 18 Nov · submit Fri 20 Nov.

## The question
Does adding historical customer–merchant ("graph") features to transaction-level features improve fraud detection, especially on fraud that looks normal on its own? Same classifiers, trained with and without the graph features, compared on a chronological test set. A small or null gain is still a valid result if reported honestly with confidence intervals.

## The data
BankSim, a simulated payments dataset (Lopez-Rojas & Axelsson, 2014): 594,643 transactions over 180 days (`step` 0–179), 4,112 customers, 50 merchants, 15 categories, 7,200 fraud (1.21%). Columns are in `docs/data_dictionary.md`. Things that shape the work (details in `docs/week1_findings.md`):
- Exactly 40 fraud rows per day. A simulation artifact: don't build features from fraud counts per day, and mention it as a limitation.
- Amount and category separate fraud strongly (median fraud amount 319 against 27 for legit; leisure is 95% fraud). Baselines will already score high, so the overall graph gain may be small.
- Merchant identity is nearly a label (20 merchants never see fraud, 8 are at least 75% fraud). Past-label features are the main leakage risk.
- Almost no cold start: nearly every test customer and every merchant appears in train. A real bank would see more new customers; say so.

## Fixed design (don't change without telling the group and saying so in the report)
- **Split by time** (`src/config.py`): train days 0–119, val 120–149, test 150–179. Test has 111,306 rows and 1,200 fraud.
- **Normal-looking fraud subgroup** (`docs/subgroup_definition.md`, `src/features/subgroup.py`), defined before any modelling. Primary: amount at or below the 95th percentile of legit amounts in the same category (train only); 400 test cases. Strict: amount at or below 72.81; 164 test cases. A 90th-percentile sensitivity check is allowed, but it is a check, not a way to pick the best number.
- **Review budget.** At a 1% budget the test set gives 1,113 alerts, so the **best possible recall@1% is 92.75%**. Print this ceiling next to every recall@1% number. The queue is the top **floor**(budget × n) rows of the whole split by score, ties broken by lower `txn_id`. Every script that builds a queue (R4's tables, R3's case studies) must use this rule, so counts match across the report.
- **Protocol.** Train on train, tune on val, score test once at freeze. The F1 threshold is picked on val and applied to test unchanged. No refit on train+val.
- **Metrics.** PR-AUC primary, ROC-AUC secondary. Precision/recall/F1 at the val threshold. Recall, precision and false positives at 1% and 2% budgets, and fraud amount caught. All of it on the whole test set and on both subgroups. Stratified bootstrap CIs (1,000 resamples) and a **paired** bootstrap for txn vs txn+graph on the same rows.

## Models (each trained on `txn` and on `txn_graph` features)
The kickoff decisions were agreed on 1 Nov 2026 on the plan's recommended answers: LightGBM is in, G5 label delay is 7 days (1 day as a sensitivity check), the NN is sklearn `MLPClassifier`, and the protocol is train / tune on val / score test once at freeze.
- Logistic regression: `class_weight="balanced"`, grid over C.
- Feedforward NN: sklearn `MLPClassifier`, early stopping, legit rows undersampled in train (same sample for both feature sets), 5 seeds. The score file averages the 5 seeds' probabilities, so every results table reports that 5-seed average. The per-seed sd of val PR-AUC (`hyperparams.csv`) is a separate stability check, not the table's error bar (that's the bootstrap CI).
- LightGBM: `scale_pos_weight`. Cheap on CPU and makes ablations fast.
Tune by val PR-AUC; reuse the same grids for the graph versions. GNNs are out of scope (future work only).

## Features
- `txn`: 29 features from `src/features/encode.py` (log amount, zero-amount flag, one-hot category, age, gender), fitted on train only.
- Graph features from `src/features/graph.py`, written to `data/processed/features_graph.parquet` (one row per `txn_id`, 23 columns):
  - **G1** customer history · **G2** merchant history · **G3** customer–merchant edge · **G5** past labels · **G4** network structure (stretch, not built).
- **Leakage rule.** A feature for a row on day *t* uses only rows from days before *t* (same-day rows have no reliable order). G5 label features use a label only from day *s* + `LABEL_DELAY` (7 days main; a 1-day version `features_graph_delay1.parquet` is the sensitivity check). `tests/test_features.py` enforces this. Never weaken those tests to make something pass.
- Ablation: all groups minus one at a time, plus label-free (G1–G4) vs labels-only (G5), using LR and LightGBM.

## File contracts (so people can work in parallel)
- `src/features/graph.py` → `data/processed/features_graph.parquet`, columns prefixed `g1_` … `g5_`
- `src/models/train.py --model {lr,mlp,lgbm} --features {txn,txn_graph,...}` → `reports/scores/{model}_{feature_set}_{split}.parquet` with columns `txn_id, score`; hyperparameters in `reports/tables/hyperparams.csv`
- `src/eval/metrics.py` reads score files → `reports/tables/results_{name}_{split}.csv`, `results_{name}_paired_{split}.csv`, `results_{name}_budget_{split}.csv` (`name` is `main` by default). **Evaluation never retrains; training never computes final metrics.** Conventions (budget queue, subgroup metrics, bootstrap) are in the README, "Evaluation".
- Detailed R3 plan: `docs/r3_plan.md`.

## Model code (`src/models/`, built 1 Nov 2026)
- `train.py`: one model on one feature set. Fits on train, tunes on val PR-AUC over the grids in `grids.py`, writes val scores. **Test scores are only written with `--score-test`.** Without that flag the test split isn't even loaded; use it once, at the freeze. Other flags: `--groups g1,g3` (graph groups to keep, for the ablation; gives feature set `txn_graph_g1g3`) and `--delay 1` (1-day label-delay file; gives `txn_graph_delay1`).
- `run_main.py`: the six main runs (3 models × {txn, txn_graph}) in one command, about 4 minutes. Rerun it whenever the graph features change. `hyperparams.csv` records a hash of `graph.py`, so stale scores can be spotted.
- `interpret.py`: LR coefficients and LightGBM contributions (TreeSHAP via `pred_contrib`, no `shap` dependency), per feature and per group, plus a figure.
- `case_studies.py`: fraud the graph model flags within the budget and the txn model misses, with the reverse counts. Writes a shortlist CSV; people pick the stories.
- `results_table.py`: formats R4's `results_main.csv` / `results_paired.csv` into the §4.1 markdown table. R4 confirmed the schema on 2 Nov (`model, feature_set, split, metric, value, ci_low, ci_high`, and the paired file with `diff`); R4's files add trailing columns (`subset, n_rows, n_fraud`; paired: `feature_set_base, feature_set_new, subset`) and are named by split, so point the reader at `results_main_{split}.csv` / `results_main_paired_{split}.csv`.
- `mock_scores.py`: fake score files in `reports/scores_mock/` for building downstream code early. Mock output is gitignored and gets a `_mock` suffix; never use it for report numbers.
- Tests: `tests/test_models.py` (fits only on train rows, no test access without the flag, deterministic reruns, MLP sample identical across feature sets) and `tests/test_reporting.py`.

Early val findings, with caveats, are in `docs/r3_interpretation_notes.md`. One affects everyone's write-up. The strongest graph feature is `g1_cust_n_prev` (number of past transactions): low-activity customers are victimised far more in BankSim. It isn't leakage, but it is probably a simulation artifact, so it belongs in the limitations (§5) and is worth an ablation run.

## Roles and deliverables
Each owner writes up and presents their own part.

| Role | Code | Report | Video |
|---|---|---|---|
| R1 Data & delivery (David) | `src/run_all.py` (whole pipeline without bash), README, business-integration numbers | editor; §1, §2, §4.5, §5 | problem + data (~3:10) |
| R2 Graph features (Jayden) | `src/features/graph.py`, `tests/test_features.py`, ablation, label-delay comparison, `docs/data_dictionary.md` | §3.1–3.2, ablation in §4 | features + leakage (~3:10) |
| R3 Models (Aayush) | `src/models/train.py`, interpretation, case studies | §3.3, §4.1 | models + main results (~3:30) |
| R4 Evaluation (Rayson) | `src/eval/metrics.py`, subgroup analysis, bootstrap, false-positive burden | §3.4, §4.2–4.4 | subgroup, workload, conclusion (~3:40) |

## Schedule (see `docs/plan.md` for the full list and checkboxes)
- **Week 2 (2–8 Nov):** eval harness (R4, Tue 3); graph features v1 Thu 5, full Sat 7 (R2); baselines + smoke test (R3, Thu 5 / Fri 6); `run_all` (R1). Checkpoint Sun 8: baseline table reviewed, feature pipeline merged, tests green.
- **Week 3 (9–15 Nov):** graph-augmented runs (R3); main table with CIs and paired differences, subgroup, FP burden (R4); ablation and label delay (R2); interpretation and case studies (R3); business numbers (R1). Report §1–2 by Wed 11, §3 by Fri 13. Slide template by Sat 14. **Freeze Sun 15.**
- **Week 4 (16–20 Nov):** Mon write §4; Tue slides and full report review (everyone reads one other section; GenAI acknowledgement; GitHub link); Wed rehearsal then record by segment; Thu buffer, README polish, fresh-clone test on a second laptop; Fri submit by midday and check file names and repo access.

Report outline (about 11 pages): §1 Problem 1 · §2 Data 1.5 · §3.1–3.2 features and leakage 1.5 · §3.3 models 1 · §3.4 evaluation 0.5 · §4.1 main results 1.5 · §4.2–4.4 ablation, subgroup, FP burden, interpretation 2 · §4.5 business use 0.5 · §5 conclusion and limitations 0.5. Appendix: feature definitions, hyperparameters, extra figures, GenAI acknowledgement.

## Working agreements
- Environment: `uv` only. Add packages with `uv add`, then commit `pyproject.toml`, `uv.lock` and a re-exported `requirements.txt` together:
  `uv export --format requirements-txt --no-hashes --no-dev --no-emit-project > requirements.txt`. Never `pip install` into the venv.
- First-time setup on macOS: `brew install uv libomp`, then `uv sync`. LightGBM needs libomp; without it `import lightgbm` fails with `Library not loaded: @rpath/libomp.dylib`. Don't work around it with `DYLD_LIBRARY_PATH`.
- Run scripts as modules: `uv run python -m src.<package>.<script>`. Tests: `uv run pytest -q`. Week 1 pipeline: `./run_week1.sh`, then `uv run python -m src.features.graph`. `data/processed/` is gitignored and must be rebuilt on each machine.
- One branch per work package; merge through a PR reviewed by one other person. Written standup in the group chat Mon / Wed / Fri; 30-minute call each Sunday.
- Results come from scripts, not notebook cells. Seeds are fixed (`SEED` in `src/config.py`); reruns give identical outputs.
- Don't touch split boundaries or the subgroup definition.
- Log GenAI help in `docs/genai_log.md` as you go. Everyone must be able to explain every part of the submission.
- Report honestly: failing tests, null results and skipped steps get said as they are.

## Stretch scope (only after the base plan is satisfied): UI / webapp
A UI is rewarded, but it must never put the required deliverables at risk. Full spec in `docs/plan.md`, section "Stretch: UI / webapp". Summary:
- **Start only when** results are frozen and tagged, report §3–4 are drafted, slides are finished, tests are green, and an owner with spare time is agreed.
- **Read-only** on frozen outputs (`reports/tables`, `reports/scores`, `data/processed`). It never retrains or computes a number not already in `reports/tables/`.
- Streamlit app in `app/`, run with `uv run streamlit run app/main.py`. Cut-off **Tue 17 Nov evening**; if not working, drop it. Nothing in the report, slides or video may depend on it. If it works: ~1 min demo clip in the video, a screenshot in the appendix, and a "Run the app" section in the README. The fresh-clone test must pass with or without the app.
- Pages, in build order: (1) results explorer with CIs, paired differences and the 92.75% ceiling; (2) review-budget simulator (slider for 0.5–5% budget; recall, false positives, fraud value caught, analyst hours; all fraud vs normal-looking; txn vs txn+graph); (3) alert queue and case viewer with feature contributions and the pre-selected case studies; (4) customer–merchant graph view with a "before day *t*" slider; (5) ablation and label-delay page.
- Design for it now: `train.py` keeps fitted models, case studies are selected by script and saved to a CSV, and tables are tidy long-format CSVs.

## Out of scope
GNNs (future work only). The only stretch items are G4 and the UI/webapp above.
