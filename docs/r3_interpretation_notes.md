# R3 interpretation notes (val split, before the freeze)

All numbers here are on **val** (days 120–149) and come from `reports/tables/`. They will change slightly at the freeze, and the report must use the frozen test numbers, not these. Written 1 Nov 2026.

## Val PR-AUC (`reports/tables/hyperparams.csv`)

| Model | txn | txn + graph | Δ |
|---|---|---|---|
| LR | 0.729 | 0.909 | +0.181 |
| MLP (per-seed mean ± sd, 5 seeds) | 0.810 ± 0.006 | 0.944 ± 0.003 | +0.134 |
| LightGBM | 0.805 | 0.951 | +0.146 |

The MLP row is the average of the 5 seeds' individual PR-AUCs. The MLP score file averages the 5 seeds' *probabilities* instead, and that combined model scores a little higher on val: 0.816 (txn) and 0.950 (txn + graph) in R4's `results_main_val.csv`. Results tables report the combined model; the ± here is only a stability check.

Smoke test (G1 + G3 only): LR 0.782, LightGBM 0.882. Most of the gain needs G2 and/or G5 as well. R2's ablation will say which.

The graph gain is much bigger than week 1 expected ("may be small"). That is good news for the project, but treat it with care until R4's paired bootstrap and R2's ablation are in. Point estimates on val are not the result.

## Random Forest (robustness check, val)
Val PR-AUC 0.791 (txn) → 0.944 (txn + graph); paired Δ +0.153 [0.136, 0.171] in `results_rf_paired_val.csv`. Recall@1% on normal-looking fraud goes from 0.19 to 0.58. So the graph gain is not specific to the three main models: it shows up in a fourth, different family (bagged trees). The jump is of the same size as LightGBM's. These are val numbers; the appendix table uses test after the freeze. The best settings were deep trees with small leaves (`min_samples_leaf` 5 for txn, 1 for txn + graph).

## What LightGBM relies on (`lgbm_importance_groups_txn_graph_val.csv`)
Share of mean |contribution| by group: customer history 27%, transaction 22%, merchant history 21%, edge 15%, past labels 15%. **G5 does not dominate.** The label-free groups (G1–G3) carry about 63% between them. So the gain is not just "past labels in disguise", though R2's labels-only vs label-free ablation is the proper test.

Top features: `g1_cust_n_prev`, `g2_merch_n_7d`, `category_transportation`, `g5_merch_fraud_rate`, `g5_cust_fraud_rate`, `log_amount`, `g3_pair_share`.

## Caveat: `g1_cust_n_prev` is probably a simulation artifact
The single most useful feature is how many transactions the customer has made before. On val, fraud rate is 5.0% in the lowest decile of `g1_cust_n_prev` and about 0.3% in the middle deciles. Fraud rows average 70 prior transactions against 111 for legit. Across the whole dataset, customers with ≤88 transactions have 11% fraud share against under 1% for customers with 155+.

This is **not leakage**. The feature only uses earlier days, and `tests/test_features.py` checks that. But it reflects how BankSim generates fraud (low-activity customers are victimised more), and there is no reason to expect it in real data. The report should say so in the limitations. Ask R2 whether the ablation should also run "G1 without `g1_cust_n_prev`" so we can see how much of the gain rests on it.

## LR coefficients (`lr_coefficients_txn_graph.csv`)
Mostly category dummies (transportation, food, travel strongly negative; hyper, barsandrestaurants positive), then `g2_merch_n_prev`, `log_amount`, `g2_merch_log_amt_ratio`. The negative travel coefficient looks odd given travel is 79% fraud. It is conditional on the merchant features and amount (travel fraud is high-amount, which `log_amount` already captures), so read coefficients as conditional effects, not as category risk.

## Case-study candidates (`case_study_counts.csv`, `case_study_candidates_lgbm_val_1pct.csv`)
At a 1% budget on val (1,085 alerts, 1,200 fraud):

| | LightGBM | LR |
|---|---|---|
| caught by both | 821 | 689 |
| graph model only | **195** (168 normal-looking) | **252** (165 normal-looking) |
| txn model only | 30 (20 normal-looking) | 60 (7 normal-looking) |
| missed by both | 154 | 199 |

Almost all graph-only catches are normal-looking fraud, which is exactly the subgroup the project is about. The top candidates are small purchases in health, fashion, home, hotel services, at merchants with a high past fraud rate and/or customers with many earlier known frauds (`g5_cust_fraud_n` up to 70). So the strongest case studies so far are "repeat-victim customer" and "risky merchant" stories, which lean on G5. For a label-free story, pick a candidate with low `g5_cust_fraud_n` and look at `case_study_contribs_val.csv` for its top G1–G3 contributions.

Pick 3–4 after the freeze from the **test** candidate file (`--split test`), not from these.

## Still to do
- Rerun everything with `--score-test` at the freeze, then `case_studies --split test` and `interpret --split test`.
- Check case-study candidates against R4's subgroup numbers once `metrics.py` exists.
