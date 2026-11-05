| Model | Features | PR-AUC | ROC-AUC | F1 @ val thr. | Recall @1% | Recall @2% | Recall @1% (normal-looking) |
|---|---|---|---|---|---|---|---|
| Random forest | txn | 0.791 [0.771, 0.811] | 0.992 [0.991, 0.993] | 0.738 [0.715, 0.758] | 0.688 [0.667, 0.708] | 0.816 [0.794, 0.838] | 0.192 [0.155, 0.227] |
| Random forest | txn + graph | 0.944 [0.935, 0.954] | 0.998 [0.996, 0.999] | 0.882 [0.868, 0.896] | 0.834 [0.822, 0.847] | 0.976 [0.967, 0.984] | 0.577 [0.540, 0.614] |
|  | Δ (paired) | +0.153 [0.136, 0.171] | +0.005 [0.004, 0.007] | +0.145 [0.127, 0.166] | +0.147 [0.127, 0.167] | +0.160 [0.139, 0.180] | +0.385 [0.345, 0.427] |

Split: val. Point estimate [95% bootstrap CI]. Δ = txn+graph minus txn, paired bootstrap.
Best possible recall@1% on val is 90.33% (the 1% queue is smaller than the number of fraud cases).
