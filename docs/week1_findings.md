# Week 1 findings (data + EDA)

Notes for the group. Figures are in `reports/figures/` and tables in `reports/tables/`.

## Split (fixed)

| split | steps | txns | fraud | fraud rate | customers |
|---|---|---|---|---|---|
| train | 0–119 | 374,914 | 4,800 | 1.28% | 4,111 |
| val | 120–149 | 108,423 | 1,200 | 1.11% | 4,070 |
| test | 150–179 | 111,306 | 1,200 | 1.08% | 4,079 |

- Every split has at least 1,200 fraud cases. 12 of the 15 categories have fraud in every split.
- Cold start is basically a non-issue. Only 1 customer in val and 1 in test (2 and 3 transactions) never appear in train, and every merchant appears in train. So graph/history features will exist for almost every test row. That's good for the experiment, but say in the report that a real bank would see more new customers.

## Things we found that affect week 2+

1. **Exactly 40 fraud rows every day** (`01_volume_and_fraud_over_time.png`). Daily volume grows from about 2,400 to 3,700 while fraud stays fixed, so the fraud rate drifts down from 1.65% to 1.08%. This is a simulation artifact. It's also why val and test have slightly lower fraud rates than train. Mention it as a limitation, and don't build a feature like "fraud count today".
2. **Amount separates the classes strongly** (`02_amount_distribution.png`). Median fraud amount is 319 against 27 for legit. Baselines with just amount and category will already score high, which leaves less room for the graph features to show improvement. That's why the subgroup analysis matters.
3. **Fraud is concentrated by category** (`03_fraud_rate_by_category.png`). Leisure is 95% fraud, travel 79% and sportsandtoys 50%. Transportation, food and contents have 0 fraud, even though transportation makes up 85% of all transactions.
4. **Merchant identity is a near-label** (`05_merchant_fraud_rate.png`). 20 of the 50 merchants never see fraud, and 8 merchants are at least 75% fraud. Any merchant fraud-rate feature must use only labels known at that time (earlier steps). Otherwise it leaks badly. This matches what we said in the proposal. It's the main leakage risk for week 2.
5. **Age/gender effects are small** (`04_fraud_rate_by_age_gender.png`). Women have a 1.47% fraud rate against 0.91% for men. Age group 0 (≤18) is at 1.96%, but that group only has 2,452 rows. Gender `U` has 0 fraud.
6. **Fraud is spread across many customers** (`06_customer_activity.png`). 1,483 of the 4,112 customers have at least one fraud, and for 1,090 of those 1,483 it's under 5% of their transactions. Most customers make 150–180 transactions over the 6 months, roughly one a day.

## Open questions for the group
- Add LightGBM as a third model? It's the standard strong baseline on tabular fraud data, and markers may ask about it.
- Merchant fraud-rate feature: do we assume labels are known the next day, or add a delay (e.g. 7 days) to be more realistic? This needs deciding before Feature engineering starts.
