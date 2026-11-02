"""Candidates for the case studies: fraud the graph model catches and the txn model misses.

For one model and split, flag the top ceil(budget * n) rows by score for the
txn and txn_graph versions. Count fraud caught by both, graph only, txn only.
Rank the graph-only catches (normal-looking first, then by how far the graph
model moved them up) and save the top K with context for a human to pick 3-4.
The script doesn't pick the story; it just makes the shortlist reproducible.

    python -m src.models.case_studies --model lgbm --split val [--budget 0.01] [--top 10]
        [--scores-dir reports/scores_mock]    # build against mock scores first

Outputs (suffix _mock when --scores-dir isn't the real score dir):
    reports/tables/case_study_candidates_{model}_{split}_{pct}pct.csv
    reports/tables/case_study_counts.csv     upserted, one row per (model, split, budget, source)
"""
import argparse
import math

import numpy as np
import pandas as pd

from src.config import GRAPH_FILE, ROOT, SCORE_DIR, TABLE_DIR
from src.features.subgroup import fit_thresholds, flag_normal_looking
from src.models.data import load_splits
from src.models.grids import MODELS

CONTEXT_FEATURES = [
    "g1_cust_n_prev", "g1_cust_amt_z", "g1_cust_days_since_last",
    "g2_merch_amt_z", "g3_pair_n_prev", "g3_pair_is_new",
    "g5_cust_fraud_n", "g5_merch_fraud_rate",
]


def load_score(scores_dir, model, fs, split):
    path = scores_dir / f"{model}_{fs}_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python -m src.models.train --model {model} "
                                f"--features {fs}` first (or src.models.mock_scores)")
    return pd.read_parquet(path).set_index("txn_id")["score"]


def top_k_flags(scores, budget):
    """1 for the top ceil(budget * n) rows by score (ties broken by txn_id, so it's deterministic)."""
    k = math.ceil(budget * len(scores))
    order = pd.DataFrame({"s": scores.to_numpy(), "id": scores.index}).sort_values(
        ["s", "id"], ascending=[False, True])
    flags = pd.Series(0, index=scores.index, dtype="int8")
    flags.loc[order.id.to_numpy()[:k]] = 1
    return flags


def select(df, s_txn, s_graph, budget, top, graph=None):
    """Return (counts dict, candidates frame). df: split rows with subgroup flags, indexed by txn_id."""
    s_txn, s_graph = s_txn.loc[df.index], s_graph.loc[df.index]
    f_txn, f_graph = top_k_flags(s_txn, budget), top_k_flags(s_graph, budget)
    fraud = df.fraud == 1
    both = fraud & (f_txn == 1) & (f_graph == 1)
    graph_only = fraud & (f_graph == 1) & (f_txn == 0)
    txn_only = fraud & (f_txn == 1) & (f_graph == 0)
    counts = {
        "n_rows": len(df), "n_fraud": int(fraud.sum()), "budget_alerts": int(f_graph.sum()),
        "caught_both": int(both.sum()), "caught_graph_only": int(graph_only.sum()),
        "caught_txn_only": int(txn_only.sum()), "missed_both": int((fraud & (f_txn == 0) & (f_graph == 0)).sum()),
        "graph_only_normal_looking": int((graph_only & (df.normal_looking == 1)).sum()),
        "txn_only_normal_looking": int((txn_only & (df.normal_looking == 1)).sum()),
    }
    rank_txn = s_txn.rank(ascending=False, method="first").astype(int)
    rank_graph = s_graph.rank(ascending=False, method="first").astype(int)
    cand = df.loc[graph_only].copy()
    cand["score_txn"], cand["score_graph"] = s_txn[cand.index], s_graph[cand.index]
    cand["rank_txn"], cand["rank_graph"] = rank_txn[cand.index], rank_graph[cand.index]
    cand["rank_gain"] = cand.rank_txn - cand.rank_graph
    if graph is not None:
        cols = [c for c in CONTEXT_FEATURES if c in graph]
        cand = cand.join(graph.loc[cand.index, cols])
    cand = cand.sort_values(["normal_looking", "normal_looking_strict", "rank_gain"],
                            ascending=[False, False, False]).head(top)
    keep = ["step", "customer", "merchant", "category", "amount", "normal_looking", "normal_looking_strict",
            "score_txn", "score_graph", "rank_txn", "rank_graph", "rank_gain"]
    keep += [c for c in CONTEXT_FEATURES if c in cand]
    return counts, cand[keep].reset_index()


def split_with_flags(split):
    parts = load_splits(["train", split])
    by_cat, cut = fit_thresholds(parts["train"])        # thresholds from train only
    df = parts[split]
    df = pd.concat([df, flag_normal_looking(df, by_cat, cut)], axis=1)
    return df.set_index("txn_id")


def upsert_counts(row, path):
    keys = ["model", "split", "budget", "source"]
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        mask = np.logical_and.reduce([old[k] == row[k] for k in keys])
        new = pd.concat([old[~mask], new], ignore_index=True)
    new.sort_values(keys).to_csv(path, index=False)


def run(model, split, budget=0.01, top=10, scores_dir=SCORE_DIR, table_dir=TABLE_DIR, log=print):
    real = scores_dir.resolve() == SCORE_DIR.resolve()
    source = "real" if real else "mock"
    df = split_with_flags(split)
    graph = pd.read_parquet(GRAPH_FILE) if GRAPH_FILE.exists() else None
    counts, cand = select(df, load_score(scores_dir, model, "txn", split),
                          load_score(scores_dir, model, "txn_graph", split), budget, top, graph)
    pct = f"{budget * 100:g}".replace(".", "p")
    suffix = "" if real else "_mock"
    table_dir.mkdir(parents=True, exist_ok=True)
    out = table_dir / f"case_study_candidates_{model}_{split}_{pct}pct{suffix}.csv"
    cand.to_csv(out, index=False)
    upsert_counts({"model": model, "split": split, "budget": budget, "source": source, **counts},
                  table_dir / f"case_study_counts{suffix}.csv")

    log(f"{model} / {split} at {budget:.0%} budget ({counts['budget_alerts']:,} alerts, "
        f"{counts['n_fraud']:,} fraud){' [MOCK SCORES]' if not real else ''}")
    log(f"  caught by both {counts['caught_both']}, graph only {counts['caught_graph_only']} "
        f"({counts['graph_only_normal_looking']} normal-looking), txn only {counts['caught_txn_only']} "
        f"({counts['txn_only_normal_looking']} normal-looking), missed by both {counts['missed_both']}")
    if counts["caught_graph_only"] == 0:
        log("  no graph-only catches at this budget -- report that as the result")
    log(f"  saved {len(cand)} candidates to {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    return counts, cand


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="lgbm", choices=MODELS)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--budget", type=float, default=0.01)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--scores-dir", default=SCORE_DIR, type=lambda s: (ROOT / s) if not s.startswith("/") else s)
    a = p.parse_args(argv)
    if a.split == "test":
        print("note: test case studies are for after the freeze (Sun 15 Nov)")
    run(a.model, a.split, a.budget, a.top, a.scores_dir)


if __name__ == "__main__":
    main()
