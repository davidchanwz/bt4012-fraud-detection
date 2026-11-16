| Model | Features | PR-AUC | ROC-AUC | F1 @ val thr. | Recall @1% | Recall @2% | Recall @1% (normal-looking) |
|---|---|---|---|---|---|---|---|
| Random forest | txn | 0.784 [0.764, 0.803] | 0.991 [0.990, 0.992] | 0.732 [0.711, 0.751] | 0.697 [0.676, 0.716] | 0.804 [0.782, 0.824] | 0.212 [0.176, 0.251] |
| Random forest | txn + graph | 0.943 [0.934, 0.953] | 0.998 [0.996, 0.999] | 0.877 [0.863, 0.891] | 0.842 [0.830, 0.856] | 0.980 [0.972, 0.988] | 0.585 [0.552, 0.628] |
|  | Δ (paired) | +0.159 [0.142, 0.176] | +0.007 [0.005, 0.008] | +0.145 [0.127, 0.165] | +0.145 [0.128, 0.166] | +0.176 [0.156, 0.198] | +0.372 [0.332, 0.418] |

Split: test. Point estimate [95% bootstrap CI]. Δ = txn+graph minus txn, paired bootstrap.
Best possible recall@1% on test is 92.75% (the 1% queue is smaller than the number of fraud cases).
