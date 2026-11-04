# "Normal-looking fraud": subgroup definition

**Fixed on 25 Oct 2026, before any model was trained.** Don't change this after seeing results. If we have to change it, say so in the report and explain why.

## Why we need one
The proposal says graph features should help most on fraud that "looks normal in isolation". The EDA shows most BankSim fraud does *not* look normal. The median fraud amount is 319 against 27 for legit transactions, and fraud sits in a few categories: leisure is 95% fraud and travel is 79% fraud. A transaction-only model will catch the obvious cases easily. The interesting question is the rest.

## Definition (code: `src/features/subgroup.py`)

**Primary:** a transaction counts as *normal-looking* if

> amount ≤ 95th percentile of **legit** amounts in the **same category**, computed on the **train split only** (steps 0–119)

In words, the amount wouldn't stand out to someone who only knows what normal spending in that category looks like.

**Secondary (stricter), for a robustness check:** amount ≤ 72.81, which is the 95th percentile of all legit train amounts, regardless of category.

Both flags are computed for every row, fraud or not. That way we can report:
- recall on normal-looking fraud, and
- how many normal-looking legit transactions get flagged inside the review budget.

## Subgroup sizes

| split | fraud | normal-looking fraud (primary) | % of fraud | strict version | % of fraud |
|---|---|---|---|---|---|
| train | 4,800 | 1,478 | 30.8% | 547 | 11.4% |
| val | 1,200 | 395 | 32.9% | 149 | 12.4% |
| test | 1,200 | **400** | 33.3% | **164** | 13.7% |

400 fraud cases in test is enough for subgroup recall. 164 is getting small, so for the strict version we should report a confidence interval (bootstrap) as well as the point estimate.

## Caveats to put in the report
- **Normal-looking does not mean low-risk category.** The primary subgroup is mostly sportsandtoys (664 of 2,273 across all splits), health (602) and wellnessandbeauty (253). Category alone still gives some of these away. So the subgroup measures "fraud the amount doesn't reveal", not "fraud nothing reveals". Figure 07 (`python -m src.eda --normal-looking`, train only) compares this subgroup with normal-looking legit rows, overall and within the same categories: category differs a lot, amount within category overlaps, and the history differences that survive within category are mostly past labels (G5) and repeat customer–merchant pairs, not first visits or unusual spend.
- **Some thresholds rest on few legit rows.** Leisure has only 22 legit transactions in train, so its threshold (113.21) is noisy. Travel's (1,661.53) is very high. Thresholds are saved in `reports/tables/subgroup_thresholds.csv`.
- **The 95th percentile is a choice.** If time allows, show that the conclusion holds at the 90th percentile too. Treat it as a sensitivity check, not a way to pick the best-looking number.
