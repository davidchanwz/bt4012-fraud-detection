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
- [ ] Truncation: recompute using only days ≤ *t*; rows on day *t* must match the full-data result exactly
- [ ] Label shuffle: shuffle labels inside the delay window and after; G5 must not change
- [ ] Day 0: every row on step 0 has zero history
- [ ] One-line definition of every feature in `docs/data_dictionary.md`

### Baselines (R3) · by Fri 6 Nov
- [x] LR (`class_weight="balanced"`, grid over C), MLP (5 seeds, mean ± sd), LightGBM (`scale_pos_weight`)
- [x] Tune by val PR-AUC. Same grids get reused for the graph versions
- [x] Row-level scores for val and test; hyperparameters to `reports/tables/hyperparams.csv` (val done 1 Nov; test written at the freeze with `--score-test`)
- [x] Smoke-test a txn + features-v1 run on Thu 5 Nov (done 1 Nov)

### Reproducibility (R1)
- [ ] `python -m src.run_all` so the whole pipeline runs without bash (Windows users, markers)

**Checkpoint Sun 8 Nov:** baseline results table reviewed, feature pipeline merged, all tests green.

## Week 3 · Mon 9 Nov – Sun 15 Nov: Main experiments, then freeze
- [x] Graph-augmented runs for all 3 models (R3, Mon 9 – Tue 10) (val done 1 Nov; rerun if graph features change)
- [ ] Main table: 3 models × {txn, txn+graph} on test with CIs and paired differences (R4)
- [ ] Ablation: all minus one group at a time, plus label-free (G1–G4) vs labels-only (G5). Use LR + LightGBM (R2)
- [ ] Label delay: 7-day vs 1-day G5 (R2)
- [ ] Subgroup: recall at threshold and at budget on primary (400 test cases) and strict (164), bootstrap CIs, 90th-percentile sensitivity check (R4)
- [ ] False-positive burden: legit rows in each 1% / 2% queue, and how many are normal-looking (R4)
- [ ] Interpretation: LR coefficients, SHAP for LightGBM; 3–4 case studies of fraud caught only by the graph model (R3)
- [ ] Business integration: daily alert volume, analyst time (stated minutes-per-review assumption), fraud value caught (R1)
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
A UI is rewarded, but it must never put the required deliverables at risk. **Do not start until all of these are true:**
- [ ] Results frozen and `v-results` tagged (Sun 15 Nov)
- [ ] Report §3–4 drafted, slides finished, all tests green
- [ ] Owner agreed in the group chat (someone with spare time, not whoever is behind on their report section)

Rules:
- **Read-only on frozen outputs.** The app reads `reports/tables/*.csv`, `reports/scores/*.parquet` and `data/processed/`. It never retrains, recomputes metrics or produces a number that isn't already in `reports/tables/`. Otherwise the app and the report could disagree.
- Single page-per-idea Streamlit app in `app/`, started with `uv run streamlit run app/main.py`. Add it with `uv add streamlit` (plus `plotly` if needed) and re-export `requirements.txt`.
- **Cut-off Tue 17 Nov evening.** If it isn't working by then, drop it. Nothing in the report, slides or video may depend on it. If it works, it gets a short demo clip (~1 min) in the video and a screenshot in the report appendix.
- The README gets a "Run the app" section. The fresh-clone test on Thu 19 must still pass with or without the app.

Candidate pages (build in this order, stop whenever time runs out):
1. **Results explorer.** Pick model × feature set × metric and see test results with bootstrap CIs, the paired difference (txn vs txn+graph), and the recall@1% ceiling (92.75%) drawn as a reference line.
2. **Review-budget simulator.** A slider for the review budget (0.5%–5%) updates recall, precision, false positives, fraud value caught, analyst reviews per day and analyst hours (stated minutes-per-review assumption, matching §4.5). Toggle "all fraud" vs "normal-looking fraud only" and "txn" vs "txn+graph".
3. **Alert queue and case viewer.** A ranked list of the top-scoring test transactions for the chosen model, with the score, amount, category and whether the txn was in the normal-looking subgroup. Clicking one shows the customer's past transactions, whether this customer–merchant pair is new, and the top feature contributions (SHAP for LightGBM, coefficients for LR). Pre-select the 3–4 case studies of fraud only the graph model caught.
4. **Customer–merchant graph view.** For a selected customer: their merchants as a small network, edges sized by prior transaction count, the flagged transaction highlighted. Optional day slider that shows the graph as it looked before that day, which also illustrates the leakage rule.
5. **Ablation and label-delay page.** Bar chart of the leave-one-group-out ablation and the 7-day vs 1-day G5 comparison, so the "graph structure vs past labels" story is interactive.

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
- Out of scope: GNNs (future work only). G4 and the UI/webapp (see the stretch section) are the only stretch items, and the UI is only started after the freeze.
