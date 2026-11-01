# 4.1 Main results (skeleton, R3)

*Skeleton only. Fill in after the freeze (Sun 15 Nov). Every number must come from `reports/tables/` (frozen at tag `v-results`). Don't use the val numbers in `docs/r3_interpretation_notes.md` here.*

## Main table
Table 4.1 = `reports/tables/results_main.md`, produced by `python -m src.models.results_table` from R4's `results_main.csv` and `results_paired.csv`.

- One paragraph: overall effect of graph features per model. PR-AUC txn → txn+graph: LR TBD → TBD, MLP TBD → TBD (± TBD over seeds), LightGBM TBD → TBD. Paired Δ with CIs: TBD.
- Whether the CIs exclude zero: TBD.
- Ranking of models: TBD. Does the graph gain shrink for the stronger model? TBD.

## At the review budget
- Recall@1% for each model, always next to the 92.75% ceiling: TBD.
- Fraud value caught at 1% / 2%: TBD (link to §4.5 business numbers).

## Normal-looking fraud (headline only; detail in §4.3, R4)
- Recall@1% on the 400 normal-looking test frauds, txn vs txn+graph: TBD.
- Share of graph-only catches that are normal-looking: TBD (from `case_study_counts.csv`, test).

## What the models rely on
- LightGBM contribution share by group: TBD (`lgbm_importance_groups_txn_graph_test.csv`).
- Whether G5 (past labels) dominates: TBD. Point to R2's ablation in §4.2.
- The `g1_cust_n_prev` caveat (low-activity customers are victimised more in BankSim): one sentence, plus limitations in §5.

## Case studies (3–4)
Pick from `case_study_candidates_lgbm_test_1pct.csv`. For each: amount and category (why txn missed it), top contributions from `case_study_contribs_test.csv` (why graph caught it). Aim for at least one label-free story (low `g5_cust_fraud_n`).
1. TBD
2. TBD
3. TBD
4. TBD (optional)

## Also say
- txn-only catches exist too (TBD at 1%); the graph model is not uniformly better.
- MLP seed variation: TBD.
