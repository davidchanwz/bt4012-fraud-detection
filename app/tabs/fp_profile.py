"""False-positive profile: when the review queue is full, which legit transactions fill it?

Read-only. Totals come from results_{family}_budget_{split}.csv (one row per model x feature set x
policy), the per-category breakdown from results_{family}_fp_profile_{split}.csv, and the paired
txn -> txn + graph difference (with its bootstrap CI) from results_{family}_paired_{split}.csv.
Nothing is summed or averaged here: every number shown is a CSV cell.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import data as D, ui

POLICIES = {"budget_1pct": "1% budget", "budget_2pct": "2% budget"}
TAG = {"budget_1pct": "1pct", "budget_2pct": "2pct"}
BADGE = {"txn": "blue", "txn_graph": "orange"}
NL_LABEL = {
    "all": "Normal-looking legit in queue",
    "normal_looking": "Normal-looking legit in queue",
    "normal_looking_strict": "Legit at or below 72.81 in queue",
    "normal_looking_q90": "Normal-looking legit in queue (q90)",
}
NL_HELP = {
    "normal_looking": "Legit rows in the queue whose amount is at or below the 95th percentile of legit "
                      "amounts in the same category (train thresholds): the same rule as the fraud subgroup.",
    "normal_looking_strict": "Legit rows in the queue with an amount at or below 72.81 (the strict rule).",
    "normal_looking_q90": "Same as normal-looking, at the 90th percentile (sensitivity check).",
}


def _files(split):
    return [f"results_main_fp_profile_{split}.csv", f"results_main_budget_{split}.csv",
            f"results_rf_fp_profile_{split}.csv", f"results_rf_budget_{split}.csv"]


def _int(x):
    return "–" if x is None or pd.isna(x) else f"{int(round(float(x))):,}"


def _paired(split, model, policy, subset):
    """Paired difference row for fp_at_{budget}[_{subset}], or None."""
    return D.paired(split, model, f"fp_at_{TAG[policy]}", subset)


# --- tiles --------------------------------------------------------------------

def _delta_stat(label, value, d, help_text):
    """Tile with an optional paired delta. Fewer FP is better (inverse colours). A zero change is
    neutral with no arrow; a change whose bootstrap CI includes 0 is greyed and says so."""
    if d is None:
        ui.stat(label, value, help=help_text)
        return
    ci = f" Paired bootstrap 95% CI of the change: [{_int(d['ci_low'])}, {_int(d['ci_high'])}]."
    diff, lo, hi = float(d["diff"]), float(d["ci_low"]), float(d["ci_high"])
    if round(diff) == 0:
        # st.metric reads "0 vs txn" as a rise; show a neutral delta without an arrow instead.
        st.metric(label, value, delta="±0 vs txn", delta_color="off", delta_arrow="off",
                  help=help_text + ci, border=True)
        return
    crosses = lo <= 0 <= hi
    ui.stat(label, value, delta=f"{_int(diff)} vs txn" + (" (CI crosses 0)" if crosses else ""),
            delta_color="off" if crosses else "inverse", help=help_text + ci)


def _tiles(sel, budget, policy):
    nl_subset = "normal_looking" if sel.subset == "all" else sel.subset
    nl_col = f"{nl_subset}_fp"
    both = len(sel.feature_sets) == 2
    p_all = _paired(sel.split, sel.model, policy, "all") if both else None
    p_nl = _paired(sel.split, sel.model, policy, nl_subset) if both else None

    cols = st.columns(len(sel.feature_sets))
    for col, fs in zip(cols, sel.feature_sets):
        r = budget[(budget.feature_set == fs)]
        with col:
            st.badge(ui.fs_label(fs), color=BADGE[fs])
            if r.empty:
                st.info(f"No {POLICIES[policy]} row for {D.MODEL_LABELS[sel.model]} on {ui.fs_label(fs)}.")
                continue
            r = r.iloc[0]
            is_new = both and fs == "txn_graph"
            a, b, c = st.columns(3)
            with a:
                _delta_stat("False positives in queue", _int(r.fp), p_all if is_new else None,
                            f"Legit rows among the {_int(r.n_alerts)} alerts in the {POLICIES[policy]} queue.")
            with b:
                ui.stat("False positives per day", f"{float(r.fp_per_day):.1f}",
                        help=f"False positives divided by the {_int(r.n_days)} days in the split "
                             f"(from the budget table). Each one is an analyst review that finds nothing."
                             + (" The paired change is shown on the total tile; there is no separate "
                                "per-day paired figure (per day it is the total change divided by the "
                                "days in the split)." if both else ""))
            with c:
                val = r[nl_col] if nl_col in r.index else None
                _delta_stat(NL_LABEL[sel.subset], _int(val), p_nl if is_new else None, NL_HELP[nl_subset])
    if both and (p_all is not None or p_nl is not None):
        st.caption("Deltas are the paired bootstrap difference, txn + graph minus txn, on the same rows. "
                   "Fewer is better, so a drop shows green and a rise red. A grey delta means the 95% CI "
                   "includes zero (no clear change).")
    elif both:
        st.info(f"No paired txn vs txn + graph difference for {D.MODEL_LABELS[sel.model]} at the "
                f"{POLICIES[policy]} (`results_*_paired_{sel.split}.csv`), so the tiles show no change.")


# --- charts -------------------------------------------------------------------

def _order(prof, fss):
    """Categories sorted by the txn false-positive count (or the only feature set shown), largest first."""
    lead = prof[prof.feature_set == fss[0]].set_index("category").fp
    cats = list(dict.fromkeys(prof.category))
    return sorted(cats, key=lambda c: (-lead.get(c, -1), c))


def _hover_data(g):
    return list(zip(g.fp.astype(int), g.fp_normal_looking.astype(int), g.fp_rate, g.fp_median_amount,
                    g.fp_share, g.legit_in_split.astype(int)))


def _fp_bars(prof, fss, cats, c):
    fig = go.Figure()
    for fs in fss:
        g = prof[prof.feature_set == fs].set_index("category").reindex(cats).dropna(subset=["fp"]).reset_index()
        if g.empty:
            continue
        color, name = ui.fs_color(fs), ui.fs_label(fs)
        line = dict(color=c["surface"], width=2)
        fig.add_bar(
            y=g.category, x=g.fp_normal_looking, orientation="h", offsetgroup=fs, legendgroup=fs,
            name=f"{name}: normal-looking amount (95th-pct rule)",
            marker=dict(color=color, opacity=0.55, line=line,
                        pattern=dict(shape="/", fgcolor=c["surface"], size=6, solidity=0.3, fillmode="overlay")),
            hoverinfo="skip")
        fig.add_bar(
            y=g.category, x=g.fp - g.fp_normal_looking, base=g.fp_normal_looking, orientation="h",
            offsetgroup=fs, legendgroup=fs, name=f"{name}: larger amount",
            marker=dict(color=color, line=line), customdata=_hover_data(g),
            hovertemplate=(f"<b>{name}</b>: %{{customdata[0]:,}} false positives"
                           " (%{customdata[1]:,} normal-looking)<br>"
                           "%{customdata[2]:.2%} of the category's %{customdata[5]:,} legit rows flagged · "
                           "median amount %{customdata[3]:,.2f} · %{customdata[4]:.1%} of all false positives"
                           "<extra></extra>"))
    ui.style(fig, height=max(340, 30 * len(cats) * len(fss) + 90), hover="y unified")
    fig.update_layout(barmode="group", legend=dict(traceorder="normal"))
    fig.update_xaxes(title_text="False positives in the queue", rangemode="tozero", tickformat=",d")
    fig.update_yaxes(categoryorder="array", categoryarray=cats, autorange="reversed", showgrid=False)
    return fig


def _rate_dots(prof, fss, cats, c):
    fig = go.Figure()
    if len(fss) == 2:          # muted connector between the two feature sets, no hover
        w = prof.pivot_table(index="category", columns="feature_set", values="fp_rate").reindex(cats)
        xs, ys = [], []
        for cat, row in w.iterrows():
            if row.notna().all():
                xs += [row["txn"], row["txn_graph"], None]
                ys += [cat, cat, None]
        fig.add_scatter(x=xs, y=ys, mode="lines", line=dict(color=c["baseline"], width=2),
                        hoverinfo="skip", showlegend=False)
    for fs in fss:
        g = prof[prof.feature_set == fs].set_index("category").reindex(cats).dropna(subset=["fp"]).reset_index()
        if g.empty:
            continue
        name = ui.fs_label(fs)
        fig.add_scatter(
            x=g.fp_rate, y=g.category, mode="markers", name=name,
            marker=dict(color=ui.fs_color(fs), size=10, line=dict(color=c["surface"], width=2)),
            customdata=list(zip(g.fp.astype(int), g.legit_in_split.astype(int))),
            hovertemplate=(f"<b>{name}</b>: %{{x:.2%}} flagged "
                           "(%{customdata[0]:,} of %{customdata[1]:,} legit rows)<extra></extra>"))
    ui.style(fig, height=max(340, 30 * len(cats) * len(fss) + 90), hover="y unified")
    fig.update_xaxes(type="log", title_text="Share of the category's legit rows flagged (log scale)",
                     tickformat=".1%")
    fig.update_yaxes(categoryorder="array", categoryarray=cats, autorange="reversed")
    return fig


def _table(prof):
    t = prof[["feature_set", "category", "fp", "fp_normal_looking", "fp_share", "fp_rate",
              "legit_in_split", "fp_median_amount"]].copy()
    t["feature_set"] = t.feature_set.map(ui.fs_label)
    t["fp_share"] = (100 * t.fp_share).round(1)
    t["fp_rate"] = (100 * t.fp_rate).round(2)
    return t.rename(columns={"feature_set": "Features", "category": "Category", "fp": "False positives",
                             "fp_normal_looking": "Normal-looking (95th pct)", "fp_share": "% of all FP",
                             "fp_rate": "% of category's legit flagged", "legit_in_split": "Legit rows in split",
                             "fp_median_amount": "Median FP amount"})


# --- entry point ----------------------------------------------------------------

def render(sel):
    ui.section("Who fills the queue for nothing?",
               "When the review queue is full, these are the legit transactions analysts open and close "
               "without finding fraud. Each one costs review time and can mean a blocked card or a call to "
               "an annoyed customer. Fewer false positives at the same budget means more fraud caught per hour.")

    policy = st.segmented_control("Review budget", list(POLICIES), format_func=POLICIES.get, default="budget_1pct",
                                  required=True, key="fp_profile_policy",
                                  help="The queue is the top 1% (or 2%) of the split's transactions by score.")
    policy = policy or "budget_1pct"

    if sel.subset != "all":
        st.info(f"The fraud subset ({D.SUBSET_LABELS[sel.subset]}) doesn't filter this tab: false positives are "
                "legit rows. It only changes which amount rule the third tile uses to count normal-looking "
                "legit rows. The bars always split by the primary 95th-percentile rule.",
                icon=":material/info:")

    prof_all = D.results("fp_profile", sel.split)
    budget_all = D.results("budget", sel.split)
    if prof_all is None and budget_all is None:
        if sel.split == "test":
            ui.empty_state("Test split not scored yet",
                           "The test split is scored once, at the results freeze (Sun 15 Nov). This tab fills in "
                           "when R4's evaluation is run with --split test.", files=_files("test"),
                           owner="the freeze (Sun 15 Nov)")
        else:
            ui.empty_state("No false-positive profile yet",
                           "Run R4's evaluation: `uv run python -m src.eval.metrics` (and `--models rf --name rf`).",
                           files=_files(sel.split), owner="R4 Rayson")
        return

    # Tiles from the budget table.
    budget = None
    if budget_all is not None:
        budget = budget_all[(budget_all.model == sel.model) & (budget_all.policy == policy)
                            & budget_all.feature_set.isin(sel.feature_sets)]
    if budget is None or budget.empty:
        st.info(f"No {POLICIES[policy]} totals for {D.MODEL_LABELS[sel.model]} in the budget table "
                f"(`results_*_budget_{sel.split}.csv`).")
    else:
        _tiles(sel, budget, policy)

    # Per-category charts from the profile table.
    prof = None
    if prof_all is not None:
        prof = prof_all[(prof_all.model == sel.model) & (prof_all.policy == policy)
                        & prof_all.feature_set.isin(sel.feature_sets)]
    if prof is None or prof.empty:
        st.info(f"No per-category false-positive profile for {D.MODEL_LABELS[sel.model]} at the "
                f"{POLICIES[policy]} (`results_*_fp_profile_{sel.split}.csv`).")
        return
    fss = [fs for fs in sel.feature_sets if fs in set(prof.feature_set)]
    missing = [fs for fs in sel.feature_sets if fs not in fss]
    if missing:
        st.info(f"No profile rows for {', '.join(ui.fs_label(f) for f in missing)}; showing the rest.")
    cats = _order(prof, fss)
    c = ui.colors()

    left, right = st.columns([3, 2], gap="large")
    with left:
        ui.section("False positives by category",
                   f"Sorted by the {ui.fs_label(fss[0])} count. The hatched part of each bar is legit spending "
                   "with a normal-looking amount for its category; the solid part is larger amounts. A category "
                   "missing from a bar had no false positives.")
        ui.chart(_fp_bars(prof, fss, cats, c), table=_table(prof), key="fp_profile_bars")
    with right:
        ui.section("How much of each category gets flagged",
                   "Share of the category's legit rows in the split that land in the queue. A category with few "
                   "legit rows can show a high rate from a handful of alerts; hover for the counts.")
        rate = _table(prof)[["Features", "Category", "% of category's legit flagged", "False positives",
                             "Legit rows in split"]]
        ui.chart(_rate_dots(prof, fss, cats, c), table=rate, key="fp_profile_rate")
