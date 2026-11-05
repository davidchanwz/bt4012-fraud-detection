"""Case viewer: a fraud the graph model caught and the txn model missed, and why.

Read-only on R3's case-study outputs (src/models/case_studies.py, src/models/interpret.py):
    case_study_candidates_{model}_{split}_{pct}pct.csv   shortlist of graph-only catches with context
    case_study_counts.csv                                 caught by both / graph only / txn only / missed
    case_study_contribs_{split}.csv                       LightGBM txn + graph TreeSHAP, top 8 per case
plus data/processed/banksim_clean.parquet for the customer's earlier transactions (rebuilt locally).
Nothing here recomputes a score, rank or count: every number is a cell of one of those files.
"""
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import data as D, ui

KEY = "cases_"
CAND_RE = re.compile(r"^case_study_candidates_(?P<model>[a-z0-9]+)_(?P<split>[a-z]+)_(?P<pct>[0-9p]+)pct\.csv$")
HISTORY_N = 40

# Plain-English labels for the context columns in the shortlist (same order as CONTEXT_FEATURES).
CONTEXT = {
    "g1_cust_n_prev": "Past transactions by this customer",
    "g1_cust_amt_z": "Amount against the customer's usual (z-score)",
    "g1_cust_days_since_last": "Days since the customer's last transaction",
    "g2_merch_amt_z": "Amount against the merchant's usual (z-score)",
    "g3_pair_n_prev": "Past payments by this customer to this merchant",
    "g3_pair_is_new": "First time this customer used this merchant",
    "g5_cust_fraud_n": "Known fraud on this customer (labels 7+ days old)",
    "g5_merch_fraud_rate": "Merchant's known fraud rate (smoothed)",
}

# Labels for the contribution bars (docs/data_dictionary.md). Unknown names fall back to the raw name.
FEATURE_TEXT = {
    "log_amount": "amount (log)", "zero_amount": "zero amount",
    "g1_cust_n_prev": "customer: past transactions", "g1_cust_is_new": "customer: first transaction",
    "g1_cust_n_7d": "customer: transactions, last 7 days", "g1_cust_amt_mean": "customer: mean past amount",
    "g1_cust_amt_std": "customer: sd of past amounts", "g1_cust_amt_z": "customer: amount z-score",
    "g1_cust_days_since_last": "customer: days since last", "g1_cust_n_merchants": "customer: merchants used",
    "g1_cust_n_categories": "customer: categories used",
    "g2_merch_n_prev": "merchant: past transactions", "g2_merch_n_7d": "merchant: transactions, last 7 days",
    "g2_merch_amt_mean": "merchant: mean past amount", "g2_merch_amt_z": "merchant: amount z-score",
    "g2_merch_log_amt_ratio": "merchant: amount vs mean (log ratio)",
    "g2_merch_n_customers": "merchant: distinct customers",
    "g3_pair_n_prev": "pair: past payments", "g3_pair_is_new": "pair: first payment",
    "g3_pair_days_since_last": "pair: days since last", "g3_pair_share": "pair: share of customer's payments",
    "g5_cust_fraud_n": "labels: customer's known fraud", "g5_cust_fraud_rate": "labels: customer's fraud rate",
    "g5_merch_fraud_rate": "labels: merchant's fraud rate", "g5_pair_fraud_n": "labels: pair's known fraud",
}


# --- sources ------------------------------------------------------------------

def load(name):
    """D.load, but a zero-byte or header-less CSV (truncated copy, bad checkout) reads as an empty frame."""
    try:
        return D.load(name)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def shortlists(split):
    """[(model, pct, filename)] for the real (non-mock) candidate files of this split, sorted."""
    out = []
    for p in sorted(D.TABLE_DIR.glob(f"case_study_candidates_*_{split}_*pct.csv")):
        m = CAND_RE.match(p.name)
        if m and m["split"] == split and m["model"] in D.MODELS:
            out.append((m["model"], m["pct"], p.name))
    order = {m: i for i, m in enumerate(D.MODELS)}
    return sorted(out, key=lambda t: (order[t[0]], float(t[1].replace("p", "."))))


def pct_budget(pct):
    return float(pct.replace("p", ".")) / 100


def pct_text(pct):
    return f"{pct.replace('p', '.')}%"


def counts_row(model, split, pct):
    df = load("case_study_counts.csv")
    if df is None or not {"model", "split", "budget", "source"} <= set(df.columns):
        return None
    r = df[(df.model == model) & (df.split == split) & (df.source == "real")
           & np.isclose(df.budget.astype(float), pct_budget(pct))]
    return None if r.empty else r.iloc[0].to_dict()


def contribs(split, txn_id):
    """None if the contributions file doesn't exist; an empty frame if it has no rows for this case."""
    df = load(f"case_study_contribs_{split}.csv")
    if df is None:
        return None
    if not {"txn_id", "feature", "contrib"} <= set(df.columns):
        return df.iloc[0:0]
    return df[df.txn_id == txn_id]


# --- formatting ---------------------------------------------------------------

def n(x):
    return "–" if x is None or pd.isna(x) else f"{int(x):,}"


def amount(x):
    return f"{x:,.2f}"


def case_label(r):
    return (f"txn {int(r.txn_id)} · {r.category} · {amount(r.amount)} · "
            f"rank {int(r.rank_txn):,} → {int(r.rank_graph):,}")


def context_value(col, v):
    if pd.isna(v):
        return "–"
    if col == "g3_pair_is_new":
        return "Yes" if v >= 0.5 else "No"
    if col.endswith("_z"):
        return f"{v:+.2f}"
    if col.endswith("_rate"):
        return ui.pct(v)
    return f"{v:,.0f}"


def feature_text(f, category=None):
    if f in FEATURE_TEXT:
        return FEATURE_TEXT[f]
    if f.startswith("category_") and category is not None:
        cat = f[len("category_"):]
        return f"category is {cat}" if cat == str(category) else f"category is not {cat}"
    for prefix in ("category_", "age_", "gender_"):
        if f.startswith(prefix):
            return f"{prefix[:-1]}: {f[len(prefix):]}"
    return f


# --- sections -----------------------------------------------------------------

def pickers(sel, lists):
    labels = {name: f"{D.MODEL_SHORT[m]} · {pct_text(p)} budget" for m, p, name in lists}
    names = [name for _, _, name in lists]
    default = next((i for i, (m, _, _) in enumerate(lists) if m == sel.model), 0)
    c1, c2 = st.columns([1, 2])
    with c1:
        name = st.selectbox("Shortlist", names, index=default, format_func=labels.get,
                            key=f"{KEY}shortlist_{sel.split}_{sel.model}",
                            help="Fraud flagged within the budget by the txn + graph model and missed by the "
                                 "txn model, ranked by how far the graph features moved it up the queue.")
    model, pct = next((m, p) for m, p, nm in lists if nm == name)
    if sel.model not in {m for m, _, _ in lists}:
        have = ", ".join(D.MODEL_LABELS[m] for m in dict.fromkeys(m for m, _, _ in lists))
        st.caption(f"No shortlist for {D.MODEL_LABELS[sel.model]}; shortlists exist only for: {have}.")
    cand = load(name)
    if cand is None or cand.empty:
        with c2:
            st.info("This shortlist is empty: no fraud was caught by the graph model alone at this budget.")
        return model, pct, None
    note = None
    if sel.subset in ("normal_looking", "normal_looking_strict") and sel.subset in cand.columns:
        cand = cand[cand[sel.subset] == 1]
        if cand.empty:
            with c2:
                st.info(f"No case on this shortlist is in the {D.SUBSET_LABELS[sel.subset].lower()} subset. "
                        "Pick 'All fraud' in the sidebar to see the whole shortlist.")
            return model, pct, None
    elif sel.subset == "normal_looking_q90":
        note = "The shortlist flags only the primary and strict subgroups, so every case is listed here."
    with c2:
        rows = {int(r.txn_id): r for r in cand.itertuples(index=False)}
        tid = st.selectbox("Case", list(rows), format_func=lambda t: case_label(rows[t]),
                           key=f"{KEY}case_{name}",
                           help="rank txn → rank txn + graph: position in the review queue (1 = most suspicious).")
    if note:
        st.caption(note)
    return model, pct, cand[cand.txn_id == tid].iloc[0]


def overview(model, split, pct, cnt, sel_subset="all"):
    ui.section(f"How the two {D.MODEL_SHORT[model]} queues split the {split} fraud",
               f"txn and txn + graph get the same {pct_text(pct)} queue"
               + ("" if cnt is None or pd.isna(cnt.get("budget_alerts"))
                  else f" ({n(cnt['budget_alerts'])} alerts)")
               + ". Graph-only catches are the shortlist's pool.")
    if cnt is None:
        st.info(f"No real counts for {D.MODEL_LABELS[model]} at {pct_text(pct)} on {split} in "
                "`case_study_counts.csv`.")
        return
    if sel_subset != "all":
        st.caption("Counts cover all fraud; only the two normal-looking tiles are restricted to the primary "
                   "normal-looking subgroup (the source has no strict or 90th-percentile counts).")
    tiles = [("Caught by both", "caught_both", None),
             ("Graph only", "caught_graph_only", "Caught by txn + graph, missed by txn."),
             ("Txn only", "caught_txn_only", "Caught by txn, missed by txn + graph."),
             ("Missed by both", "missed_both", None),
             ("Graph only, normal-looking", "graph_only_normal_looking",
              "Graph-only catches in the primary normal-looking subgroup."),
             ("Txn only, normal-looking", "txn_only_normal_looking",
              "Txn-only catches in the primary normal-looking subgroup.")]
    tiles = [t for t in tiles if t[1] in cnt]
    cols = st.columns(len(tiles))
    for col, (label, k, h) in zip(cols, tiles):
        with col:
            ui.stat(label, n(cnt.get(k)), help=h)
    c = ui.colors()
    bars = [("Caught by both", "caught_both", c["ink2"]),
            ("Graph only", "caught_graph_only", ui.fs_color("txn_graph")),
            ("Txn only", "caught_txn_only", ui.fs_color("txn")),
            ("Missed by both", "missed_both", c["muted"])]
    bars = [b for b in bars if b[1] in cnt and not pd.isna(cnt[b[1]])]
    if not bars:
        return
    fig = go.Figure(go.Bar(
        x=[cnt[k] for _, k, _ in bars], y=[lab for lab, _, _ in bars], orientation="h",
        marker=dict(color=[col for _, _, col in bars], cornerradius=4),
        text=[n(cnt[k]) for _, k, _ in bars], textposition="outside", textfont=dict(color=c["ink"]),
        hovertemplate="%{y}: %{x:,} fraud cases<extra></extra>", cliponaxis=False))
    ui.style(fig, height=210, legend=False, hover="closest")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(title=f"fraud cases (of {n(cnt.get('n_fraud'))})")
    table = pd.DataFrame({"outcome": [lab for lab, _, _ in bars], "fraud cases": [int(cnt[k]) for _, k, _ in bars]})
    ui.chart(fig, table, key=f"{KEY}overview_{model}_{pct}",
             caption="Orange: the graph features rescued it. Blue: adding them lost it.")


def case_card(r, split):
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown(f"**Transaction {int(r.txn_id)}**")
            if "normal_looking" in r.index and pd.notna(r["normal_looking"]):
                if int(r["normal_looking"]) == 1:
                    st.badge("normal-looking", color="gray", help=D.SUBSET_HELP["normal_looking"])
                else:
                    st.badge("not normal-looking", color="gray")
            if ("normal_looking_strict" in r.index and pd.notna(r["normal_looking_strict"])
                    and int(r["normal_looking_strict"]) == 1):
                st.badge("strict normal-looking", color="gray", help=D.SUBSET_HELP["normal_looking_strict"])
        cols = st.columns(5)
        with cols[0]:
            ui.stat("Day", f"{int(r.step)}", help=f"step in the {split} split")
        with cols[1]:
            ui.stat("Amount", amount(r.amount))
        with cols[2]:
            ui.stat("Category", str(r.category))
        with cols[3]:
            ui.stat("Customer", str(r.customer))
        with cols[4]:
            ui.stat("Merchant", str(r.merchant))


def rank_chart(sel, r, cnt, model, pct):
    ui.section("Where it sits in the review queue")
    cutoff = None if cnt is None or pd.isna(cnt.get("budget_alerts")) else int(cnt["budget_alerts"])
    c = ui.colors()
    fss = list(sel.feature_sets)
    rank = {"txn": int(r.rank_txn), "txn_graph": int(r.rank_graph)}
    score = {"txn": float(r.score_txn), "txn_graph": float(r.score_graph)}
    ys = [ui.fs_label(fs) for fs in fss]
    fig = go.Figure()
    if len(fss) == 2:
        fig.add_trace(go.Scatter(x=[rank[f] for f in fss], y=ys, mode="lines", showlegend=False,
                                 line=dict(color=c["baseline"], width=2), hoverinfo="skip"))
    if cutoff:
        fig.add_vrect(x0=1, x1=cutoff, fillcolor=c["grid"], opacity=0.45, line_width=0, layer="below")
        fig.add_vline(x=cutoff, line=dict(color=c["ink2"], width=2, dash="dash"),
                      annotation=dict(text=f"queue cutoff: {cutoff:,} alerts", font=dict(color=c["ink2"])),
                      annotation_position="top")
    for fs in fss:
        fig.add_trace(go.Scatter(
            x=[rank[fs]], y=[ui.fs_label(fs)], mode="markers", name=ui.fs_label(fs),
            marker=dict(size=14, color=ui.fs_color(fs), line=dict(color=c["surface"], width=2)),
            customdata=[[score[fs]]],
            hovertemplate=f"{ui.fs_label(fs)}: rank %{{x:,}}<br>score %{{customdata[0]:.3f}}<extra></extra>"))
    hi = max([rank[f] for f in fss] + [cutoff or 1])
    ui.style(fig, height=230, legend=True, hover="closest")
    fig.update_xaxes(type="log", range=[0, np.log10(hi * 2.5)], title="rank in the queue (log scale)")
    fig.update_yaxes(showgrid=False, range=[-0.6, len(ys) - 0.4], autorange=False)
    table = pd.DataFrame({"features": ys, "rank": [rank[f] for f in fss], "score": [score[f] for f in fss],
                          "queue cutoff": [cutoff] * len(fss)})
    ui.chart(fig, table, key=f"{KEY}rank_{model}_{pct}_{int(r.txn_id)}_{'_'.join(fss)}",
             caption="Rank 1 = most suspicious; inside the queue (shaded) if rank ≤ cutoff.")
    cols = st.columns(len(fss))
    for col, fs in zip(cols, fss):
        with col:
            where = "" if cutoff is None else (" (inside the queue)" if rank[fs] <= cutoff
                                               else " (outside the queue)")
            ui.stat(f"Score, {ui.fs_label(fs)}", f"{score[fs]:.3f}",
                    help=f"{D.MODEL_LABELS[model]} {ui.fs_label(fs)} probability; rank {rank[fs]:,}{where}.")
    if cutoff is None:
        st.caption("Queue cutoff unknown: no matching row in `case_study_counts.csv`.")


def context_table(r):
    ui.section("What the graph features saw", "Computed from earlier days only (the leakage rule).")
    rows = [{"Feature": label, "Value": context_value(col, r[col])} for col, label in CONTEXT.items() if col in r]
    if not rows:
        st.info("This shortlist has no context columns.")
        return
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"Feature": st.column_config.TextColumn(width="large"),
                                "Value": st.column_config.TextColumn(width="small")})


def contrib_chart(split, r, model):
    ui.section("Why LightGBM (txn + graph) flagged it",
               "TreeSHAP contributions in log-odds, top 8 features. Positive pushes towards fraud. "
               "A category the case is not in can still push, by its absence.")
    df = contribs(split, int(r.txn_id))
    if df is None:
        st.info(f"`case_study_contribs_{split}.csv` is not written yet: run "
                f"`uv run python -m src.models.interpret --split {split}`.")
        return
    if df.empty:
        st.info("No contributions for this case: they were computed for the LightGBM 1% shortlist only "
                f"(`case_study_contribs_{split}.csv`).")
        return
    c = ui.colors()
    df = df.sort_values("contrib").assign(label=lambda d: d.feature.map(lambda f: feature_text(f, r.category)))
    fig = go.Figure()
    for name, mask, color in (("pushes towards fraud", df.contrib > 0, c["ink2"]),
                              ("pushes away from fraud", df.contrib <= 0, c["muted"])):
        part = df[mask]
        if part.empty:
            continue
        fig.add_trace(go.Bar(x=part.contrib, y=part.label, orientation="h", name=name,
                             marker=dict(color=color, cornerradius=4), customdata=part[["feature"]],
                             hovertemplate="%{y}<br><span style='font-size:11px'>%{customdata[0]}</span>"
                                           "<br>%{x:+.2f} log-odds<extra></extra>"))
    ui.style(fig, height=320, legend=True, hover="closest")
    fig.update_yaxes(categoryorder="array", categoryarray=list(df.label), showgrid=False)
    fig.update_xaxes(title="contribution (log-odds)", zeroline=True, zerolinecolor=c["baseline"], zerolinewidth=1)
    note = "" if model == "lgbm" else f" These are LightGBM's, not {D.MODEL_SHORT[model]}'s."
    ui.chart(fig, df[["feature", "label", "contrib"]].sort_values("contrib", ascending=False),
             key=f"{KEY}contrib_{int(r.txn_id)}",
             caption="Longest bars explain most of the score." + note)


def history(r):
    ui.section("The customer's earlier transactions",
               f"Days before the case only (same-day rows have no reliable order), last {HISTORY_N} at most.")
    d = D.clean_data(["txn_id", "step", "customer", "merchant", "category", "amount", "fraud"])
    if d is None:
        st.info("History needs `data/processed/banksim_clean.parquet`, which is gitignored and rebuilt on each "
                "machine: run `./run_week1.sh`.")
        return
    cust = d[d.customer == r.customer]
    past = cust[cust.step < r.step].sort_values(["step", "txn_id"]).tail(HISTORY_N)
    case = cust[cust.txn_id == r.txn_id]
    n_prev = r.get("g1_cust_n_prev")
    n_pair = r.get("g3_pair_n_prev")
    is_new = r.get("g3_pair_is_new")
    if pd.notna(is_new) and is_new >= 0.5:
        first = f"This is the customer's **first** payment to {r.merchant}."
    elif pd.notna(n_pair):
        first = f"Not the first time at {r.merchant}: {int(n_pair):,} earlier payments to it."
    else:
        first = ""
    shown = f"Showing {len(past):,} of {n(n_prev)} earlier transactions." if pd.notna(n_prev) else ""
    st.caption(" ".join(s for s in (first, shown) if s))
    if past.empty and case.empty:
        st.info("No rows for this customer in `data/processed/` (rebuild it with `./run_week1.sh`).")
        return
    c = ui.colors()
    same = past.merchant == r.merchant
    fig = go.Figure()
    for name, part, color, size, symbol in (
            ("other merchants", past[~same], c["muted"], 8, "circle"),
            ("same merchant", past[same], c["ink2"], 10, "circle"),
            ("this case", case, c["ink"], 14, "diamond")):
        if part.empty:
            continue
        fig.add_trace(go.Scatter(
            x=part.step, y=part.amount, mode="markers", name=name,
            marker=dict(size=size, color=color, symbol=symbol, line=dict(color=c["surface"], width=2)),
            customdata=np.column_stack([part.merchant, part.category.astype(str),
                                        np.where(part.fraud == 1, "fraud", "legit")]),
            hovertemplate="day %{x}<br>%{y:,.2f} at %{customdata[0]}<br>%{customdata[1]} · %{customdata[2]}"
                          f"<extra>{name}</extra>"))
    scale = st.segmented_control("Amount axis", ["log", "linear"], default="log",
                                 key=f"{KEY}history_scale_{int(r.txn_id)}", label_visibility="collapsed") or "log"
    ui.style(fig, height=320, legend=True, hover="closest")
    fig.update_xaxes(title="day (step)")
    fig.update_yaxes(title="amount (log scale)" if scale == "log" else "amount", type=scale)
    shown_rows = pd.concat([past, case])
    if scale == "log" and (shown_rows.amount <= 0).any():
        st.caption(f"{int((shown_rows.amount <= 0).sum())} zero-amount rows can't be drawn on a log axis; "
                   "they are in the data table.")
    table = pd.concat([past.assign(row="earlier"), case.assign(row="this case")])
    table = table.assign(same_merchant=table.merchant == r.merchant,
                         fraud=np.where(table.fraud == 1, "fraud", "legit"))
    table = table[["row", "step", "merchant", "same_merchant", "category", "amount", "fraud", "txn_id"]]
    ui.chart(fig, table.sort_values(["step", "txn_id"], ascending=False),
             key=f"{KEY}history_{int(r.txn_id)}",
             caption="Dark markers: earlier payments to the same merchant. Fraud labels are shown for context; "
                     "the model only sees labels at least 7 days old.")


# --- entry point --------------------------------------------------------------

def render(sel):
    ui.section("Case viewer: why was it flagged?",
               "A fraud the txn + graph model put in the review queue and the txn model missed, with what the "
               "history features saw.")
    lists = shortlists(sel.split)
    if not lists:
        if sel.split == "test":
            body = ("Test-split shortlists are written once, at the freeze (Sun 15 Nov): "
                    "`uv run python -m src.models.case_studies --model lgbm --split test`.")
        else:
            body = ("No case-study shortlist yet. Run `uv run python -m src.models.case_studies --model lgbm "
                    f"--split {sel.split}` after the score files exist.")
        ui.empty_state("No case-study shortlist for this split", body,
                       files=[f"case_study_candidates_lgbm_{sel.split}_1pct.csv", "case_study_counts.csv"],
                       owner="R3 Aayush")
        return
    model, pct, r = pickers(sel, lists)
    cnt = counts_row(model, sel.split, pct)
    overview(model, sel.split, pct, cnt, sel.subset)
    if r is None:
        return
    st.divider()
    case_card(r, sel.split)
    left, right = st.columns([3, 2], gap="large")
    with left:
        rank_chart(sel, r, cnt, model, pct)
    with right:
        context_table(r)
    left, right = st.columns(2, gap="large")
    with left:
        contrib_chart(sel.split, r, model)
    with right:
        history(r)
