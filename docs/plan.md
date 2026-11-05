# Project plan (4 weeks)

Deadline: **Fri 20 Nov 2026, 23:59**. Due: `report07.pdf` (8–12 pages excl. appendices/refs, with the GitHub URL), `presentation07.pptx`, `video07.mp4` (≤15 min, everyone speaks).
Results get frozen a week before the deadline, so the report and video aren't written around numbers that keep changing.

Key dates: kickoff **Sat 31 Oct** · checkpoint **Sun 8 Nov** · freeze **Sun 15 Nov** · record **Wed 18 Nov** · submit **Fri 20 Nov by midday**.

## Week 1 · Mon 26 Oct – Sun 1 Nov: Data and setup ✅ (done 25 Oct)
- [x] Repo structure, README, uv environment (`pyproject.toml` + `uv.lock`, Python 3.12)
- [x] Download + hash check of the raw CSV (`src/data/download.py`)
- [x] Cleaning: strip quotes, drop the constant zip columns, encode categoricals (`src/data/clean.py`, `src/features/encode.py`)
- [x] EDA: imbalance, amounts, fraud by category / age / gender / merchant, fraud over `step` (`src/eda.py`, `notebooks/01_eda.ipynb`)
- [x] Chronological split fixed: train 0–119, val 120–149, test 150–179. Every split has ≥1,200 fraud (`src/data/split.py`)
- [x] "Normal-looking fraud" subgroup defined before any modelling (`docs/subgroup_definition.md`)
- [x] Sanity tests (`tests/test_data.py`)

Findings: `docs/week1_findings.md`

### What week 1 means for the rest
- Amount and category already separate fraud well, so the overall gain from graph features may be small. The headline result has to come from the subgroup and review-budget metrics, with confidence intervals. A small or null gain is still a result if we report it honestly.
- Past-label features (merchant fraud rate, customer prior fraud) are the main leakage risk. They go in their own group (G5) with a label delay, so the ablation can separate "graph structure" from "past labels".
- Test has 111,306 rows and 1,200 fraud. A 1% budget is 1,113 alerts, so the **best possible recall@1% is 92.75%**. Print this ceiling next to every recall@1% number. 1% of ~3,700 txns/day is about 37 reviews a day.

## Roles
Each owner writes up their own part and presents it. That keeps commits and speaking time balanced for the peer evaluation.

| Role | Owner | Code | Report | Video |
|---|---|---|---|---|
| R1 Data & delivery | David | `src/run_all.py`, README, business-integration numbers | editor; §1, §2, §4.5, §5 | problem + data (~3:10) |
| R2 Graph features | Jayden | `src/features/graph.py`, `tests/test_features.py`, ablation | §3.1–3.2, ablation in §4 | features + leakage (~3:10) |
| R3 Models | Aayush | `src/models/train.py`, interpretation, case studies | §3.3, §4.1 | models + main results (~3:30) |
| R4 Evaluation | Rayson | `src/eval/metrics.py`, subgroup, bootstrap | §3.4, §4.2–4.4 | subgroup, workload, conclusion (~3:40) |

## Kickoff · Sat 31 Oct – Sun 1 Nov
Decisions (all agreed 1 Nov 2026 on the recommended answers in bold):
- [x] Add LightGBM as a third model? **Yes.** It's the standard strong tabular baseline, cheap on CPU, and makes ablation runs fast. `uv add lightgbm`.
- [x] Label delay for G5 features? **7 days as the main setting, 1 day as a sensitivity check.**
- [x] Feedforward NN: sklearn or PyTorch? **sklearn `MLPClassifier`** with early stopping. Handle imbalance by undersampling legit rows in train (same sample for both feature sets).
- [x] Training protocol? **Train on train, tune on val, score test once at freeze.** F1 threshold picked on val, applied to test unchanged. No refit on train+val.
- [x] Fill in the role owners above.

File contracts, so everyone can work in parallel:
- `src/features/graph.py` → `data/processed/features_graph.parquet`: one row per `txn_id`, columns prefixed `g1_` … `g5_`
- `src/models/train.py --model {lr,mlp,lgbm} --features {txn,txn_graph,...}` → `reports/scores/{model}_{features}_{split}.parquet` with `txn_id, score`
- `src/eval/metrics.py` reads score files → `reports/tables/results_*.csv`. Evaluation never retrains.

Also: everyone runs the week 1 pipeline on their own machine; create `docs/genai_log.md`.

## Added after proposal feedback · 4 Nov 2026
The markers said the topic is appropriate and the dataset, preprocessing and method have no obvious issues. They asked for four things. Two became new work items (A, B); the other two are already planned and are listed in C so they get written up.

### A. EDA: normal-looking fraud vs normal transactions · by Wed 11 Nov
Owner to confirm in the group chat. Suggested: R1 David (owns the EDA and §2), with R4 Rayson on the subgroup side.

Why: it shows that on amount and category alone, normal-looking fraud is hard to tell from legit, and that history features are what separate them. That is the motivation for the whole project. It also feeds tab 1 of the dashboard (see the stretch section).
- [ ] Script, not a notebook cell: `src/eda_subgroup.py`, run with `uv run python -m src.eda_subgroup`. Compare three groups on **train** (val as a check, never test): normal-looking fraud, normal-looking **legit** (the fair control: same amount range), and the rest of the fraud (obvious fraud, for contrast).
- [ ] Transaction-level view: amount distribution, category mix, age/gender. The hypothesis to test is that these look similar for normal-looking fraud and normal-looking legit. If they don't, report that.
- [ ] History view from `features_graph.parquet`, for example `g3_pair_is_new`, `g1_cust_n_prev`, `g1_cust_amt_z`, `g2_merch_amt_z`, `g2_merch_n_customers` and the G5 rates. One effect-size number per feature (standardised mean difference) plus a figure.
- [ ] Outputs: `reports/figures/07_normal_looking_vs_legit.png`, `reports/tables/normal_looking_comparison.csv` (effect size per feature), and `reports/tables/normal_looking_feature_hist.csv` (tidy binned counts per feature and group, so the dashboard can redraw the plot without touching raw data). Cite in §2 (R1) and §4.3 (R4).
- [ ] One line tying it to the model result: the txn-only model's recall on the normal-looking subgroup, from R4's subgroup tables.
- Caveat to state: `g1_cust_n_prev` is probably a BankSim artifact (`docs/r3_interpretation_notes.md`), so don't let the story rest on it alone.

### B. Random Forest as a fourth model (R3) · implemented 4 Nov; group agreed 4 Nov
Why: the markers asked whether graph features still help across model types. We have linear (LR), neural (MLP) and boosted trees (LightGBM); Random Forest adds bagged trees. **XGBoost is deliberately not added**: it is too close to LightGBM to add evidence. Say so in §3.3.

Scope: a robustness check, not a headline. The main table stays 3 models × {txn, txn+graph}. RF gets one appendix table (PR-AUC and recall@1%, txn vs txn+graph, with the paired difference) and one sentence in §4.1. The group agreed on 4 Nov.
- [x] `src/models/grids.py`: add `rf` (sklearn `RandomForestClassifier`, `class_weight="balanced_subsample"`, `n_jobs=-1`, `random_state=SEED`, `max_features="sqrt"`, 300 trees). Grid `max_depth` {20, None} × `min_samples_leaf` {1, 5, 20}: a probe on val showed `min_samples_leaf` matters a lot (PR-AUC about 0.90 at 20 against 0.94 at 1), so the first-planned {5, 20} would have missed the best region. One fit takes about 10 s, so the grid is cheap. Keep the list of all models separate from the six main runs, so `run_main` still defaults to LR, MLP and LightGBM.
- [x] `src/models/train.py`: `--model rf`, raw (unscaled) graph columns like LightGBM, same protocol (tune on val PR-AUC; test only with `--score-test`).
- [x] `tests/test_models.py`: RF on the toy data, same score-file contract (`rf_{feature_set}_{split}.parquet`).
- [x] `results_table.py` knows `rf`. R4's `src/eval/metrics.py` has `MODELS = ("lr", "mlp", "lgbm")`, which is the default of its `--models` flag. That is deliberate: RF is **not** in the main table, and the RF run uses `--models rf --name rf`, which writes separate `results_rf_*` files. No change to R4's code is needed.
- [x] `docs/report/3_3_models.md` (why RF, why not XGBoost) and `docs/r3_plan.md` item 8.
- [x] Val run: `uv run python -m src.models.run_main --models rf`, then `uv run python -m src.eval.metrics --split val --models rf --name rf` (4 Nov). Val PR-AUC 0.791 → 0.944 (txn → txn+graph).
- [ ] At the freeze: `run_main --models rf --score-test`, then `metrics --split test --models rf --name rf`. Appendix table: `uv run python -m src.models.results_table --split test --main reports/tables/results_rf_test.csv --paired reports/tables/results_rf_paired_test.csv --out reports/tables/results_rf_test.md`.

### C. Already planned; tracked here so they get written up
- **GNN ("if time permits")**: stays out of scope. §5 (R1) says why (time, and keeping the comparison fair) and names it as future work.
- **Graph construction, feature engineering and model selection, with reasons**: §3.1–3.2 (R2) must say how the customer–merchant graph is built, why each feature group is included and why the leakage rules matter. §3.3 (R3) justifies the models. §4.1–4.2 explain *why* gains differ by model and feature group, not only that they do.
- **Practical implications, limitations, reproducibility**: §4.5 and §5 (R1). Limitations to name: simulated data, exactly 40 fraud per day, the `g1_cust_n_prev` artifact, almost no cold start, one chronological split. Reproducibility: `src/run_all.py` (R1), fixed seeds, `uv.lock`, fresh-clone test on Thu 19.

## Week 2 · Mon 2 Nov – Sun 8 Nov: Features and baselines

### Evaluation harness (R4) · by Tue 3 Nov
- [x] PR-AUC (primary), ROC-AUC (secondary)
- [x] Precision / recall / F1 at the val-chosen threshold
- [x] Recall, precision, false positives at 1% and 2% budget (+ ceiling); fraud amount caught at each budget
- [x] All of the above on `normal_looking` and `normal_looking_strict` subsets (+ `normal_looking_q90` for the sensitivity check)
- [x] Stratified bootstrap CIs (1,000 resamples) and a **paired** bootstrap for txn vs txn+graph on the same test rows
- [x] Unit tests on toy scores (`tests/test_eval.py`)

Done 2 Nov: `src/eval/metrics.py`, run on val. Output files and conventions in the README, "Evaluation".

### Graph / behavioural features (R2) · v1 Thu 5 Nov, full Sat 7 Nov
Each transaction on day *t* uses only days **before** *t* (same-step rows have no reliable order). Compute on the full cleaned data in step order: per-step aggregates, then cumulative sum shifted by one step.

| Group | Features | Ships in |
|---|---|---|
| G1 customer history | prior txn count; 7-day count (velocity); mean/std past amount; amount z-score vs own history; days since last txn; unique merchants / categories so far | v1 |
| G2 merchant history | prior txn count; unique customers (degree); 7-day volume; mean past amount; amount vs merchant mean | full |
| G3 customer–merchant edge | prior pair count; first-time-pair flag; days since pair last transacted; share of customer's history at this merchant | v1 |
| G4 network structure (stretch) | avg degree of customer's past merchants; weekly-snapshot PageRank | if time |
| G5 past labels | smoothed merchant fraud rate; customer prior fraud count; only labels older than the delay | full |

Leakage tests (`tests/test_features.py`):
- [x] Truncation: recompute using only days ≤ *t*; rows on day *t* must match the full-data result exactly
- [x] Label shuffle: shuffle labels inside the delay window and after; G5 must not change
- [x] Day 0: every row on step 0 has zero history
- [x] One-line definition of every feature in `docs/data_dictionary.md`

### Baselines (R3) · by Fri 6 Nov
- [x] LR (`class_weight="balanced"`, grid over C), MLP (5 seeds, mean ± sd), LightGBM (`scale_pos_weight`)
- [x] Tune by val PR-AUC. Same grids get reused for the graph versions
- [x] Row-level scores for val and test; hyperparameters to `reports/tables/hyperparams.csv` (val done 1 Nov; test written at the freeze with `--score-test`)
- [x] Smoke-test a txn + features-v1 run on Thu 5 Nov (done 1 Nov)

### Reproducibility (R1)
- [x] `python -m src.run_all` so the whole pipeline runs without bash (Windows users, markers) (done 5 Nov; full run on a clean clone took about 11 min, 18 steps. Not yet run on Windows)

**Checkpoint Sun 8 Nov:** baseline results table reviewed, feature pipeline merged, all tests green.

## Week 3 · Mon 9 Nov – Sun 15 Nov: Main experiments, then freeze
- [x] Graph-augmented runs for all 3 models (R3, Mon 9 – Tue 10) (val done 1 Nov; rerun if graph features change)
- [ ] Main table: 3 models × {txn, txn+graph} on test with CIs and paired differences (R4)
- [ ] Ablation: all minus one group at a time, plus label-free (G1–G4) vs labels-only (G5). Use LR + LightGBM (R2)
- [ ] Label delay: 7-day vs 1-day G5 (R2)
- [ ] Subgroup: recall at threshold and at budget on primary (400 test cases) and strict (164), bootstrap CIs, 90th-percentile sensitivity check (R4)
- [ ] False-positive burden: legit rows in each 1% / 2% queue, and how many are normal-looking (R4)
- [ ] Interpretation: LR coefficients, SHAP for LightGBM; 3–4 case studies of fraud caught only by the graph model (R3)
- [x] Business integration: daily alert volume, analyst time (stated minutes-per-review assumption), fraud value caught (R1) (val done 5 Nov, `src/eval/business.py`; test comes with `--score-test` at the freeze)
- [ ] Report drafts in parallel: §1–2 by Wed 11 (R1); §3.1–3.4 by Fri 13 (R2, R3, R4)
- [ ] Slide template + storyline outline by Sat 14 (R1)
- **Freeze results Sun 15 Nov.** Tag `v-results`. After this, bug fixes only. Every number in the report comes from `reports/tables/`.

## Week 4 · Mon 16 Nov – Fri 20 Nov: Write-up and recording
- **Mon 16:** §4.1 (R3), §4.2–4.4 (R4), §4.5 + final figures (R1)
- **Tue 17:** finish slides. Full report review: 8–12 pages, GenAI acknowledgement, GitHub link. Each person reads one other section.
- **Wed 18:** one full rehearsal, then record by segment. Aim for 13:30 (limit is 15). Title + all names on screen at the start for ~8 s.
- **Thu 19:** buffer for a re-record. Polish the README. Fresh-clone reproducibility test on a second laptop.
- **Fri 20:** submit by midday, not at 23:59. Check the file names (a wrong name costs 2% of that component) and that the repo is accessible.

## Stretch: UI / webapp (only after the base plan is satisfied)
A UI is rewarded, but it must never put the required deliverables at risk.

**Started early, 4 Nov (R3 Aayush, with Claude Code).** Originally planned to start only after the freeze. It was started on val data instead because it reads only committed tables, so it can't change any result and costs no one else time. The rules below still hold: nothing in the report depends on it, the Tue 17 Nov cut-off stands, and the test split appears in it automatically once the freeze files exist. Pending sources show a "waiting on" card and fill in when their files land:
- [x] Shell: `app/main.py` (sidebar + six tabs), `app/data.py` (read-only loaders, missing file = empty state), `app/ui.py` (palette, chart style, tiles), `.streamlit/config.toml`, `tests/app_check.py` (renders every tab under every sidebar selection, with and without the tables)
- [x] Tabs 1, 2, 4, 5 on real val data; tabs 3 and 6 with interim content until item A and R2's ablation exist (4 Nov; each tab reviewed against the CSVs; `tests/test_app.py` in the suite)
- [x] Tab 3 full view from R4's item-A tables (`normal_looking_comparison.csv`, `normal_looking_category_mix.csv`, figure 07; merged 4 Nov, PR #8): category mix, SMD against all normal-looking legit and against same-category legit, per-feature detail
- [ ] Tab 6 full view: needs R2's evaluated ablation and label-delay results (`metrics.py ... --name ablation`; the app reads any results family with non-standard feature sets)
- [ ] Test split: appears once `results_main_test.csv` exists (freeze). For the case viewer on test, run in this order after `--score-test`: `uv run python -m src.models.case_studies --model lgbm --split test` (and `--budget 0.02`, and `--model lr`), then `uv run python -m src.models.interpret --model lgbm --split test` (it writes contributions only for shortlists that already exist)
- [ ] Screenshot for the appendix and ~1 min demo clip (after the freeze)

Rules:
- **Read-only on frozen outputs.** The app reads `reports/tables/*.csv` and `data/processed/` (case viewer history only). It never retrains, recomputes metrics or produces a number that isn't already in `reports/tables/`. Otherwise the app and the report could disagree. The one derived number is analyst hours from an editable minutes-per-review assumption.
- Streamlit app in `app/`, started with `uv run streamlit run app/main.py`. Built with Streamlit's own widgets and Plotly (`uv add streamlit plotly`, done 4 Nov); no third-party component libraries (the shadcn wrapper renders in iframes and ignores the theme; the antd and option-menu wrappers are unmaintained since 2024).
- **Cut-off Tue 17 Nov evening.** If it isn't working by then, drop it. Nothing in the report, slides or video may depend on it. If it works, it gets a short demo clip (~1 min) in the video and a screenshot in the report appendix.
- The README gets a "Run the app" section. The fresh-clone test on Thu 19 must still pass with or without the app.

Form: one Streamlit **dashboard**, not a set of static pictures. A sidebar holds controls that apply to every tab; each tab answers one question. Every control only looks up rows that already exist in `reports/tables/` (committed, so a fresh clone works without retraining or the raw data). Nothing is recomputed.

Sidebar controls: split (`val` now; `test` appears only after the freeze), model (LR / MLP / LightGBM / RF), features (`txn` / `txn+graph` / both side by side), subset (all fraud / normal-looking / strict / q90).

Tabs, in build order (stop whenever time runs out):
1. **Review-budget simulator** (build first; the best demo). A slider for the review budget (0.1%–5%, the 50 points in `results_main_budget_curve_{split}.csv`). It updates alerts per day, frauds caught, false positives, precision, share of fraud value caught, and draws the recall-vs-budget curve for txn vs txn+graph with the slider position marked and the best-possible-recall ceiling as a reference line. A box for "minutes per review" turns alerts per day into analyst hours (a stated assumption, matching §4.5). Toggling the subset to "normal-looking" shows where the graph features earn their keep.
2. **Model scoreboard.** Toggle models and feature sets; pick a metric. Bars with bootstrap CIs, the paired txn → txn+graph difference with its interval, and the "CI excludes zero" call. Data: `results_main_{split}.csv`, `results_main_paired_{split}.csv`.
3. **Why graph features? (interactive EDA, from item A).** Category mix of normal-looking fraud vs legit, then the standardised difference on each history feature against all normal-looking legit and against legit in the same categories, and a per-feature detail view. Data: `normal_looking_comparison.csv`, `normal_looking_category_mix.csv` (R4, train only).
4. **Why was it flagged? (case viewer).** Choose one of the shortlisted graph-only catches (`case_study_candidates_*.csv`): see its score under each model, the top feature contributions (`case_study_contribs_*.csv`) and the customer's earlier transactions with the new/old customer–merchant link highlighted. The history panel reads `data/processed/` and shows a notice instead if that folder is missing.
5. **False-positive profile.** Which categories the wasted alerts come from and how many are normal-looking legit. Data: `results_main_fp_profile_{split}.csv`.
6. **Ablation and label delay** (after R2's results exist). Leave-one-group-out and 7-day vs 1-day as a bar chart you can switch between models.
Optional if time remains: a customer–merchant network view with a "before day t" slider (illustrates the leakage rule).

Not planned: a box where you type a transaction and get a score. It would need the fitted models, which are gitignored and absent after a fresh clone, and it would print numbers that are not in `reports/tables/`.

Dependencies: `streamlit` and `plotly` (added 4 Nov; `requirements.txt` re-exported).

## Report outline (aim for 11 pages)
| Section | Pages | Author |
|---|---|---|
| 1 Problem | 1 | R1 |
| 2 Data (variable table, EDA, split, simulation artifacts) | 1.5 | R1 |
| 3.1–3.2 Preprocessing, graph features, leakage controls | 1.5 | R2 |
| 3.3 Models and justification | 1 | R3 |
| 3.4 Evaluation design | 0.5 | R4 |
| 4.1 Main results | 1.5 | R3 |
| 4.2–4.4 Ablation, subgroup, FP burden, interpretation | 2 | R4, R2 |
| 4.5 Business use | 0.5 | R1 |
| 5 Conclusion and limitations | 0.5 | R1 + all |

Appendix: feature definitions, hyperparameters, extra figures, GenAI acknowledgement.

## Working agreements
- Don't change the split boundaries in `src/config.py` or the subgroup definition. If one has to change, the report says so and why.
- Add packages with `uv add` only; commit `pyproject.toml`, `uv.lock` and a re-exported `requirements.txt` together.
- Results come from scripts, not notebook cells.
- One branch per work package; merge through a PR reviewed by one other person.
- Written standup in the group chat Mon / Wed / Fri (done, next, blocked); 30-min call each Sunday.
- Log GenAI help in `docs/genai_log.md` as you go. Everyone must be able to explain every part of the submission.
- Out of scope: GNNs (future work only). G4 and the UI/webapp (see the stretch section) are the only stretch items. The UI was started early (4 Nov) on val data; see the stretch section.
