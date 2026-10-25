# Project plan (4 weeks)

Deadline: **Fri 20 Nov 2026, 23:59**. Due: `report07.pdf`, `presentation07.pptx`, `video07.mp4`.
Results get frozen a week before the deadline, so the report and video aren't written around numbers that keep changing.

## Week 1 · Mon 26 Oct – Sun 1 Nov: Data and setup ✅ (done 25 Oct)
- [x] Repo structure, README, uv environment (`pyproject.toml` + `uv.lock`, Python 3.12)
- [x] Download + hash check of the raw CSV (`src/data/download.py`)
- [x] Cleaning: strip quotes, drop the constant zip columns, encode categoricals (`src/data/clean.py`, `src/features/encode.py`)
- [x] EDA: imbalance, amounts, fraud by category / age / gender / merchant, fraud over `step` (`src/eda.py`, `notebooks/01_eda.ipynb`)
- [x] Chronological split fixed: train 0–119, val 120–149, test 150–179. Every split has ≥1,200 fraud (`src/data/split.py`)
- [x] "Normal-looking fraud" subgroup defined before any modelling (`docs/subgroup_definition.md`)
- [x] Sanity tests (`tests/test_data.py`)

Findings: `docs/week1_findings.md`

## Week 2 · Mon 2 Nov – Sun 8 Nov: Features and baselines
- [ ] Leakage-safe graph / behavioural features: frequency, unique counterparties, repeat customer–merchant pairs, historical volume, velocity. Each transaction only uses steps **before** its own step (same-step rows have no reliable order).
- [ ] Merchant fraud-rate feature: only use labels known at that time (see open decision below)
- [ ] Test that no feature uses same-step or future data
- [ ] Transaction-only baselines: logistic regression, feedforward NN (+ LightGBM if we add it). Tune on val.
- [ ] One shared evaluation function for every model: PR-AUC, precision, recall, F1, recall at 1% and 2% review budget
- **Checkpoint Sun 8 Nov:** baseline results table and feature pipeline merged

## Week 3 · Mon 9 Nov – Sun 15 Nov: Main experiments, then freeze
- [ ] Graph-augmented versions of the same models
- [ ] Ablation: drop one feature group at a time
- [ ] Subgroup analysis on normal-looking fraud (primary + strict). Bootstrap CIs for the strict one (only 164 test cases)
- [ ] Check how many false positives each model sends to the review budget
- [ ] Start the report in parallel: Problem, Data, Methodology don't depend on final numbers
- **Freeze results Sun 15 Nov.** After this, bug fixes only.

## Week 4 · Mon 16 Nov – Fri 20 Nov: Write-up and recording
- **Mon 16:** Results + Conclusion sections, final figures
- **Tue 17:** finish slides. Full report review: 8–12 pages, GenAI acknowledgement, GitHub link
- **Wed 18:** rehearse, then record. Aim for 13–14 min (limit is 15). Everyone speaks. Title + all names on screen at the start for up to ~8 s.
- **Thu 19:** buffer for a re-record. Polish the README. Fresh-clone reproducibility test (`uv sync && ./run_week1.sh` etc.)
- **Fri 20:** submit early in the day, not at 23:59. Check the file names (a wrong name costs 2% of that component).

## Suggested split (4 people)
1. Data, EDA, README / reproducibility
2. Graph feature engineering + leakage checks
3. Baseline + graph-augmented model training
4. Evaluation framework, ablation, subgroup analysis

Everyone writes up their own part and presents it. That keeps commits and speaking time balanced for the peer evaluation.

| Area | Owner |
|---|---|
| Data / EDA / reproducibility | David |
| Graph features | |
| Models | |
| Evaluation / ablation / subgroups | |

## Open decisions (settle before week 2 work starts)
- [ ] Add LightGBM as a third model? It's the standard strong baseline on tabular fraud data, and markers may ask why it's missing. If not, justify it in the report.
- [ ] Label delay for the merchant fraud-rate feature: assume labels are known the next day, or add a delay (e.g. 7 days) to be more realistic?
- [ ] Feedforward NN: sklearn `MLPClassifier` or PyTorch? (Add with `uv add`.)
