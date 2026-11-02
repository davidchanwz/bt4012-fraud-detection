"""Metric maths for the evaluation harness (R4). Pure numpy, no file reading.

The trick that keeps this short and fast: every metric takes an integer
weight per row.
- Weight 1 for every row gives the ordinary metric. tests/test_eval.py checks
  this against sklearn.
- A bootstrap resample is the same rows with weight = how many times each row
  was drawn (0, 1, 2, ...). Drawing row i twice is the same as counting it twice.
So point estimates and the 1,000 bootstrap replicates go through exactly the
same code, and the rows only need sorting by score once per model.

Conventions (agreed in CLAUDE.md, "Fixed design"):
- PR-AUC is sklearn's average precision, the same number R3 tuned on.
- Ties in score are one threshold for PR-AUC / ROC-AUC (as in sklearn).
- Review budget b: the analyst reviews the top floor(b * n) rows of the WHOLE
  split, ties broken by lower txn_id first (same rule as R3's case studies).
  For n = 111,306 and b = 1% that is 1,113 alerts, so best recall is 1113/1200.
- Subgroup budget metrics use that same whole-split queue. "Recall@1% on
  normal-looking fraud" = share of normal-looking fraud that lands in the
  analyst's queue, not a re-ranking inside the subgroup.
- Subgroup PR-AUC / ROC-AUC / threshold metrics are computed on the subgroup's
  rows only (its fraud vs its legit).
"""
import math

import numpy as np

RANK_METRICS = ("pr_auc", "roc_auc")
THRESHOLD_METRICS = ("precision", "recall", "f1", "fp")
BUDGET_METRICS = ("recall", "precision", "fp", "amount_caught", "amount_share")


def n_alerts(n_rows, budget):
    """Queue size for a review budget. The epsilon stops 0.29 * 100 = 28.999... flooring to 28."""
    return max(1, math.floor(budget * n_rows + 1e-9))


def budget_tag(budget):
    """0.01 -> '1pct', 0.005 -> '0p5pct' (used in metric names)."""
    return f"{budget * 100:g}pct".replace(".", "p")


def metric_names(budgets, subsets):
    """Every (subset, metric) pair the harness produces, in output order."""
    names = []
    for sub in subsets:
        names += [(sub, m) for m in RANK_METRICS + THRESHOLD_METRICS]
        for b in budgets:
            tag = budget_tag(b)
            names += [(sub, f"{m}_at_{tag}") for m in BUDGET_METRICS]
            if sub == "all":
                names.append((sub, f"max_recall_at_{tag}"))
    return names


def full_name(subset, metric):
    """Flat metric name used in the CSVs: 'recall_at_1pct' or 'recall_at_1pct_normal_looking'."""
    return metric if subset == "all" else f"{metric}_{subset}"


def choose_threshold(y, score):
    """Score cut-off that maximises F1 on these rows. Rows with score >= threshold are flagged.

    Call it on val only; the result is applied to test unchanged.
    """
    y = np.asarray(y, dtype=float)
    s = np.asarray(score, dtype=float)
    order = np.argsort(-s, kind="stable")
    s, y = s[order], y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    last = _run_ends(s)                       # each distinct score is one candidate threshold
    tp, fp, thr = tp[last], fp[last], s[last]
    precision = tp / (tp + fp)
    recall = tp / tp[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(tp > 0, 2 * precision * recall / (precision + recall), 0.0)
    return float(thr[np.argmax(f1)])          # argmax: first (highest) threshold on ties


def stratified_weights(y, b, seed):
    """Bootstrap resample b as row weights. Fraud and legit rows are resampled separately,
    so every resample has exactly the same number of each (stratified bootstrap).

    Seeded by (seed, b), so resample b is the same rows for every model. That is
    what makes the txn vs txn+graph comparison paired.
    """
    y = np.asarray(y)
    rng = np.random.default_rng([seed, b])
    w = np.zeros(len(y))
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        w[idx] = np.bincount(rng.integers(0, len(idx), len(idx)), minlength=len(idx))
    return w


class SortedRun:
    """One model's scores on one split, sorted once: score high to low, then txn_id low to high."""

    def __init__(self, score, y, txn_id, amount, subsets):
        score = np.asarray(score, dtype=float)
        if np.isnan(score).any():
            raise ValueError("scores contain NaN")
        self.order = np.lexsort((np.asarray(txn_id), -score))
        self.s = score[self.order]
        self.y = np.asarray(y, dtype=float)[self.order]
        self.amt = np.asarray(amount, dtype=float)[self.order]
        self.masks = {k: np.asarray(m, dtype=bool)[self.order] for k, m in subsets.items()}

    def evaluate(self, threshold, budgets, weights=None):
        """All metrics for this run. weights are in the ORIGINAL row order (None = all ones).

        Returns {(subset, metric): value}.
        """
        w = np.ones(len(self.s)) if weights is None else np.asarray(weights, dtype=float)[self.order]
        n = w.sum()
        out = {}
        flags = w * (self.s >= threshold)                       # copies flagged at the val threshold
        queues = {b: _queue_copies(w, n_alerts(n, b)) for b in budgets}
        for sub, m in self.masks.items():
            out[(sub, "pr_auc")], out[(sub, "roc_auc")] = _rank_metrics(self.s, self.y, w * m)
            tp, fp, pos, _, _ = self._counts(w, flags, m)
            prec = tp / (tp + fp) if tp + fp > 0 else 0.0     # sklearn's zero_division=0
            rec = tp / pos if pos > 0 else np.nan
            out[(sub, "precision")] = prec
            out[(sub, "recall")] = rec
            out[(sub, "f1")] = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
            out[(sub, "fp")] = fp
            for b, q in queues.items():
                tag = budget_tag(b)
                tp, fp, pos, caught, total = self._counts(w, q, m)
                out[(sub, f"recall_at_{tag}")] = tp / pos if pos > 0 else np.nan
                out[(sub, f"precision_at_{tag}")] = tp / (tp + fp) if tp + fp > 0 else np.nan
                out[(sub, f"fp_at_{tag}")] = fp
                out[(sub, f"amount_caught_at_{tag}")] = caught
                out[(sub, f"amount_share_at_{tag}")] = caught / total if total > 0 else np.nan
                if sub == "all":
                    out[(sub, f"max_recall_at_{tag}")] = min(n_alerts(n, b), pos) / pos
        return out

    def policy_counts(self, policy, threshold=None, budget=None, weights=None):
        """Raw counts for one alerting policy (budget or val threshold), for the budget table."""
        w = np.ones(len(self.s)) if weights is None else np.asarray(weights, dtype=float)[self.order]
        if policy == "threshold":
            q = w * (self.s >= threshold)
        else:
            q = _queue_copies(w, n_alerts(w.sum(), budget))
        out = {"n_alerts": q.sum()}
        for sub, m in self.masks.items():
            tp, fp, pos, caught, total = self._counts(w, q, m)
            out[sub] = {"tp": tp, "fp": fp, "n_fraud": pos, "amount_caught": caught, "amount_total": total}
        return out

    def _counts(self, w, q, m):
        """TP, FP, fraud count, fraud amount caught, total fraud amount inside subset m.
        q = copies of each row that were flagged / in the queue; w = copies present."""
        fraud_q = q * self.y * m
        tp = fraud_q.sum()
        fp = (q * (1 - self.y) * m).sum()
        pos = (w * self.y * m).sum()
        caught = (fraud_q * self.amt).sum()
        total = (w * self.y * m * self.amt).sum()
        return tp, fp, pos, caught, total


def _run_ends(s):
    """Index of the last row in each run of equal scores (s already sorted)."""
    return np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1]


def _rank_metrics(s, y, w):
    """Weighted average precision and ROC-AUC on rows already sorted by score (high first)."""
    keep = w > 0
    s, y, w = s[keep], y[keep], w[keep]
    if len(s) == 0:
        return np.nan, np.nan
    last = _run_ends(s)
    tp = np.cumsum(w * y)[last]
    fp = np.cumsum(w * (1 - y))[last]
    pos, neg = tp[-1], fp[-1]
    if pos == 0 or neg == 0:
        return np.nan, np.nan
    recall = tp / pos
    precision = tp / (tp + fp)
    ap = float(np.sum(np.diff(recall, prepend=0.0) * precision))    # sum of (R_k - R_k-1) * P_k
    roc = float(np.trapezoid(np.r_[0.0, recall], np.r_[0.0, fp / neg]))
    return ap, roc


def _queue_copies(w, k):
    """How many copies of each (sorted) row sit in a queue of the first k copies."""
    before = np.cumsum(w) - w
    return np.clip(k - before, 0, w)
