# R2: ablation and label delay, val findings (14 Nov)

Notes for §4 (ablation) and the R2 video segment. **These are val numbers.** The test numbers come from the same script at the freeze and replace these in the report.

Rebuild: `uv run python -m src.models.run_ablation` (about an hour on a laptop: 12 trainings take about 20 minutes, then two evaluations with 1,000 bootstrap resamples take about 35).
Tables: `reports/tables/results_ablation_{,paired_,budget_,...}val.csv` and `results_delay_*_val.csv`. Every number below comes from those files.

## Set-up
LR and LightGBM, same grids as the main runs, retrained on each variant:

| Feature set | Graph groups | Meaning |
|---|---|---|
| `txn_graph` | G1 G2 G3 G5 | all groups (the main run, reference) |
| `txn_graph_g2g3g5` | minus G1 | no customer history |
| `txn_graph_g1g3g5` | minus G2 | no merchant history |
| `txn_graph_g1g2g5` | minus G3 | no customer–merchant edge |
| `txn_graph_g1g2g3` | minus G5 | **label-free**: graph structure only, no past fraud labels |
| `txn_graph_g5` | G5 only | **labels only** |
| `txn_graph_delay1` | all, 1-day delay | label-delay sensitivity check |

G4 (network structure) wasn't built, so "label-free" is G1–G3.

## Results (val, 1,200 fraud; 400 normal-looking fraud)

| Model | Feature set | PR-AUC | Recall @1% | Recall @1%, normal-looking |
|---|---|---|---|---|
| LR | txn | 0.729 | 62.4% | 22.5% |
| LR | all groups | 0.909 | 78.3% | 62.3% |
| LR | label-free (G1–G3) | 0.900 | 77.4% | 56.2% |
| LR | labels only (G5) | 0.882 | 75.4% | 61.8% |
| LightGBM | txn | 0.805 | 70.8% | 23.5% |
| LightGBM | all groups | 0.951 | 84.6% | 61.3% |
| LightGBM | label-free (G1–G3) | 0.947 | 83.3% | 60.0% |
| LightGBM | labels only (G5) | 0.937 | 81.3% | 52.4% |

Best possible recall at a 1% budget on val is 90.33%.

## What it shows
1. **Graph structure alone carries most of the gain, so the gain isn't just past labels.** Label-free features (G1–G3) add +0.171 PR-AUC for LR and +0.141 for LightGBM over txn only. They add +34 and +36 percentage points of recall@1% on normal-looking fraud. All of these paired CIs are well above zero. Against all groups, label-free loses only 0.009 (LR) and 0.004 (LightGBM) PR-AUC. This answers the leakage worry: the improvement doesn't depend on the label-based features.
2. **Past labels alone are also strong** (+0.153 / +0.132 PR-AUC over txn). The two halves overlap a lot: each alone gets most of the way, and together they're only a little better.
3. **No single group is essential.** Dropping one group costs at most 0.024 PR-AUC (LR without G2). The groups are partly redundant.
   - **G2 merchant history** matters most for LR (−0.024 PR-AUC, CI −0.029 to −0.020). For LightGBM, dropping it barely matters (−0.003, CI crosses zero), probably because trees can recover merchant information from the category and G5 merchant fraud rate.
   - **G1 customer history** matters for LightGBM on normal-looking fraud: −7.6 pp recall@1% (CI −10.6 to −4.0). This fits R3's finding that `g1_cust_n_prev` is the strongest single feature, and probably partly a simulation artifact (see `docs/r3_interpretation_notes.md`).
   - **G3 customer–merchant edge** is redundant given G1 and G2. Dropping it changes nothing for LightGBM (+0.001 PR-AUC, CI crosses zero) and very little for LR (−0.001). The pair information is largely recoverable from customer history plus merchant history.
   - **G5 labels** matter most for LightGBM on normal-looking fraud: −8.9 pp recall@1% with labels only vs all, and −1.3 pp when G5 is dropped (CI crosses zero).
4. **The 7-day label delay costs essentially nothing.** With a 1-day delay, LR's PR-AUC changes by −0.002 (CI −0.004 to −0.000) and LightGBM's by +0.001 (CI crosses zero). Recall@1% changes by under 1 pp either way. So the more realistic 7-day setting doesn't weaken the result.

## Caveats for the write-up
- Val numbers, and the models were tuned on val, so they're slightly optimistic. Use the test numbers in the report.
- Many comparisons. Treat single CIs that barely exclude zero (for example LR minus G3, −0.0013 PR-AUC) as "no practical difference", not as findings.
- The groups overlap, so leave-one-out understates each group's value. That's why the label-free vs labels-only runs against txn are the clearer test.
- BankSim is simulated. Strong history effects (low-activity customers being targeted) may not carry over to real data.
