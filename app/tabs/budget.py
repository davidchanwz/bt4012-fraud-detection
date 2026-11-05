"""Tab 1, review-budget simulator: if analysts can review only X% of transactions, how much fraud
do we catch, with and without graph features?

Reads R4's results_{main,rf}_budget_curve_{split}.csv (50 budgets, 0.1% to 5%, per model, feature
set and subset). Every number on screen is a value in one of those rows. The only arithmetic is
presentation: percentages, differences between two displayed rows (labelled "vs txn"), and the
analyst-workload conversion (alerts per day x minutes per review / 60), which is a stated assumption.
"""
import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import data as D, ui

DEFAULT_BUDGET = 0.01
DEFAULT_MINUTES = 5.0
SHIFT_HOURS = 8
COMPARE = ("txn", "txn_graph")


# --- formatting -----------------------------------------------------------------

def _ok(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def _pct(x, digits=1):
    return ui.pct(x, digits) if _ok(x) else "–"


def _int(x):
    return f"{int(x):,}" if _ok(x) else "–"


def _budget_label(b):
    return f"{100 * b:.1f}%"


def _pp_delta(new, base):
    """Difference of two displayed shares, in percentage points."""
    if not (_ok(new) and _ok(base)):
        return None
    return f"{100 * (new - base):+.1f} pp vs txn"


def _count_delta(new, base):
    if not (_ok(new) and _ok(base)):
        return None
    return f"{int(new) - int(base):+,} vs txn"


# --- render ---------------------------------------------------------------------

def render(sel):
    ui.section("Review-budget simulator",
               "If analysts can review only a fixed share of transactions, how much fraud lands in the queue, "
               "with and without graph features? Move the budget and watch the gap.")
    curve = D.results("budget_curve", sel.split)
    if curve is None:
        when = (" Test-split files only exist after the results freeze (Sun 15 Nov)."
                if sel.split == "test" else " Run `uv run python -m src.eval.metrics` to write it.")
        ui.empty_state("Budget curve not available for this split",
                       "The simulator draws R4's budget curve (recall, precision and false positives at "
                       "every budget from 0.1% to 5%)." + when,
                       files=[f"results_main_budget_curve_{sel.split}.csv",
                              f"results_rf_budget_curve_{sel.split}.csv"],
                       owner="R4 Rayson" if sel.split == "val" else "the freeze")
        return

    cur = curve[curve.model == sel.model]
    if cur.empty:
        st.info(f"No budget curve for {D.MODEL_LABELS.get(sel.model, sel.model)} on the {sel.split} split yet. "
                "Pick another model in the sidebar.")
        return
    cur = cur[cur.subset == sel.subset]
    if cur.empty:
        st.info(f"The budget curve has no rows for the subset “{D.SUBSET_LABELS.get(sel.subset, sel.subset)}”.")
        return

    present = [fs for fs in sel.feature_sets if fs in set(cur.feature_set)]
    missing = [fs for fs in sel.feature_sets if fs not in present]
    if not present:
        st.info("No budget curve for the selected feature set(s) and this model yet.")
        return
    if missing:
        st.info("Not in the budget curve yet: " + ", ".join(ui.fs_label(f) for f in missing)
                + ". Showing " + ", ".join(ui.fs_label(f) for f in present) + " only.")

    cur = cur[cur.feature_set.isin(present)].copy()
    cur["budget"] = cur["budget"].round(3)
    _simulator(sel, cur, tuple(present))


@st.fragment
def _simulator(sel, cur, present):
    """Slider and everything it drives; moving it reruns only this fragment."""
    common = set.intersection(*(set(cur.budget[cur.feature_set == fs]) for fs in present))
    budgets = sorted(common)
    if not budgets:
        st.info("The selected feature sets have no budget level in common in the curve file, so they cannot be "
                "compared. Pick one feature set in the sidebar.")
        return
    if len(budgets) < cur.budget.nunique():
        st.caption("Some budget levels are missing for one feature set; the slider offers only the levels "
                   "every selected feature set has.")
    cur = cur[cur.budget.isin(common)]
    default = DEFAULT_BUDGET if DEFAULT_BUDGET in budgets else budgets[len(budgets) // 2]
    b = st.select_slider("Review budget (share of all transactions in the split sent to analysts)",
                         options=budgets, value=default, format_func=_budget_label, key="budget_level")

    at = {fs: cur[(cur.feature_set == fs) & (cur.budget == b)].iloc[0].to_dict() for fs in present}
    compare = set(COMPARE) <= set(present)
    head = at["txn_graph"] if compare else at[present[0]]
    base = at["txn"] if compare else None
    is_all = sel.subset == "all"
    ceiling = head.get("max_recall") if is_all else None

    _headline(sel, b, at, present, compare)
    _tiles(head, base, ceiling, compare, sel)

    left, right = st.columns([2, 1], gap="medium")
    with left:
        _recall_chart(sel, cur, present, b, is_all)
    with right:
        _second_chart(sel, cur, present, b)

    _workload(head)

    st.caption("**How the queue works.** At budget *b* the analysts review the top floor(*b* × *n*) "
               "transactions of the whole split, ranked by model score (ties broken by lower `txn_id`), the "
               "same rule as every table in the report. Recall on a subset is the share of that subset's fraud "
               "that lands inside this one queue; on a subset, false positives and precision count only the "
               "subset's rows (legit transactions with normal-looking amounts). Point estimates; bootstrap CIs "
               "at 1% and 2% are on the Scoreboard tab.")


def _headline(sel, b, at, present, compare):
    who = D.SUBSET_LABELS.get(sel.subset, sel.subset).lower()
    first = at[present[0]]
    parts = [f"**{ui.fs_label(fs)}** catches **{_int(at[fs]['tp'])}** ({_pct(at[fs]['recall'])})"
             for fs in (("txn_graph", "txn") if compare else present)]
    ceiling = (f" (best possible {_pct(first['max_recall'], 2)} at this budget)"
               if sel.subset == "all" and _ok(first.get("max_recall")) else "")
    line = (f"At a **{_budget_label(b)}** budget ({first['alerts_per_day']:,.1f} alerts a day), "
            f"{D.MODEL_LABELS.get(sel.model, sel.model)} with " + "; with ".join(parts)
            + f" of the {_int(first['n_fraud'])} cases in “{who}”{ceiling}.")
    if compare and _ok(at["txn_graph"]["tp"]) and _ok(at["txn"]["tp"]):
        gap = int(at["txn_graph"]["tp"]) - int(at["txn"]["tp"])
        line += f" Difference: **{gap:+,}** cases vs txn."
    st.markdown(line)
    if sel.subset == "all":
        st.caption("Tip: switch the sidebar subset to “Normal-looking” to see where the graph features earn "
                   "their keep.")


def _tiles(head, base, ceiling, compare, sel):
    cols = st.columns(6)
    with cols[0]:
        ui.stat("Alerts per day", f"{head['alerts_per_day']:,.1f}",
                help=f"{_int(head['n_alerts'])} alerts in the whole {sel.split} split. The same for every "
                     "feature set, because the queue size depends only on the budget.")
    with cols[1]:
        label = "Recall" + (f" (best possible {_pct(ceiling, 2)})" if _ok(ceiling) else "")
        ui.stat(label, _pct(head["recall"]), delta=_pp_delta(head["recall"], base["recall"]) if compare else None,
                help="Share of the subset's fraud inside the queue. Best possible recall: the queue can hold at "
                     "most this share of all fraud when it is shorter than the fraud count." if _ok(ceiling)
                else "Share of the subset's fraud inside the queue.")
    with cols[2]:
        ui.stat("Frauds caught", f"{_int(head['tp'])} of {_int(head['n_fraud'])}",
                delta=_count_delta(head["tp"], base["tp"]) if compare else None)
    with cols[3]:
        ui.stat("False positives", _int(head["fp"]),
                delta=_count_delta(head["fp"], base["fp"]) if compare else None, delta_color="inverse",
                help=("Legit transactions in the queue (whole split). Fewer is better." if sel.subset == "all" else
                      "Legit transactions in the queue that fall in this subset's amount range. Fewer is better."))
    with cols[4]:
        ui.stat("Precision", _pct(head["precision"]),
                delta=_pp_delta(head["precision"], base["precision"]) if compare else None,
                help=("Share of the queue that is fraud." if sel.subset == "all" else
                      "Precision within the subset: subset fraud / (subset fraud + subset legit) in the queue. "
                      "Not the share of the whole queue that is fraud."))
    with cols[5]:
        ui.stat("Fraud value caught", _pct(head["amount_share"]),
                delta=_pp_delta(head["amount_share"], base["amount_share"]) if compare else None,
                help="Share of the subset's total fraud amount that is inside the queue.")
    if compare:
        st.caption("Tiles show **txn + graph**; the small figures underneath are the change vs **txn** at the same "
                   "budget (pp = percentage points). For false positives, down is better.")
    else:
        st.caption(f"Showing **{ui.fs_label(head['feature_set'])}** only. Pick “Compare” in the sidebar to see "
                   "the change vs txn.")


def _marker_shape(fig, b, c):
    fig.add_vline(x=b, line=dict(color=c["ink2"], width=1, dash="dot"))


def _recall_chart(sel, cur, present, b, is_all):
    c = ui.colors()
    fig = go.Figure()
    sub = cur.sort_values("budget")
    if is_all:
        ref = sub[sub.feature_set == present[0]]
        if ref.max_recall.notna().any():
            fig.add_trace(go.Scatter(
                x=ref.budget, y=ref.max_recall, name="Best possible recall", mode="lines",
                line=dict(color=c["muted"], width=2, dash="dash"),
                hovertemplate="%{y:.1%}<extra>Best possible</extra>"))
    for fs in present:
        d = sub[sub.feature_set == fs]
        fig.add_trace(go.Scatter(
            x=d.budget, y=d.recall, name=ui.fs_label(fs), mode="lines",
            line=dict(color=ui.fs_color(fs), width=2),
            customdata=d[["tp", "n_fraud"]].to_numpy(),
            hovertemplate="%{y:.1%} (%{customdata[0]:,} of %{customdata[1]:,})<extra>" + ui.fs_label(fs)
                          + "</extra>"))
    for fs in present:
        p = sub[(sub.feature_set == fs) & (sub.budget == b)]
        upper = fs == "txn_graph" or len(present) == 1
        fig.add_trace(go.Scatter(
            x=p.budget, y=p.recall, mode="markers+text", showlegend=False, hoverinfo="skip",
            marker=dict(color=ui.fs_color(fs), size=11, line=dict(color=c["surface"], width=2)),
            text=[_pct(v) for v in p.recall], textposition="top center" if upper else "bottom center",
            textfont=dict(color=c["ink"], size=13)))
    _marker_shape(fig, b, c)
    ui.style(fig, height=400, legend=True)
    fig.update_xaxes(title="Review budget (share of transactions reviewed)", tickformat=".0%",
                     hoverformat=".1%", range=[0, 0.0515])
    fig.update_yaxes(title=f"Recall, {D.SUBSET_LABELS.get(sel.subset, sel.subset).lower()}",
                     tickformat=".0%", range=[0, 1.12],
                     tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0])      # headroom so the label above 100% isn't clipped
    table = sub[["budget", "feature_set", "n_alerts", "alerts_per_day", "n_fraud", "tp", "recall", "max_recall"]]
    table = table.assign(budget=table.budget.map(_budget_label), feature_set=table.feature_set.map(ui.fs_label),
                         recall=table.recall.map(_pct), max_recall=table.max_recall.map(lambda v: _pct(v, 2)),
                         alerts_per_day=table.alerts_per_day.round(1))
    table.columns = ["Budget", "Features", "Alerts", "Alerts per day", "Fraud in subset", "Caught", "Recall",
                     "Best possible recall"]
    if not is_all:
        table = table.drop(columns="Best possible recall")
    ui.chart(fig, table, key="budget_recall_chart",
             caption="Recall as the queue grows. The dotted line is the selected budget." + _ceiling_note(sub, is_all))


def _ceiling_note(sub, is_all):
    if not is_all:
        return ""
    full = sub[sub.max_recall >= 1].budget
    where = f"below {_budget_label(full.min())}" if not full.empty else "across this range"
    return (f" The dashed grey line is the best possible recall: {where} the queue is shorter than the "
            "fraud count, so no model can catch everything.")


def _second_chart(sel, cur, present, b):
    c = ui.colors()
    what = st.segmented_control("Second measure", ["precision", "fp"], default="precision", required=True,
                                format_func={"precision": "Precision", "fp": "False positives"}.get,
                                key="budget_second_measure", label_visibility="collapsed")
    col = what or "precision"
    sub = cur.sort_values("budget")
    fig = go.Figure()
    for fs in present:
        d = sub[sub.feature_set == fs]
        fig.add_trace(go.Scatter(
            x=d.budget, y=d[col], name=ui.fs_label(fs), mode="lines", line=dict(color=ui.fs_color(fs), width=2),
            hovertemplate=("%{y:.1%}" if col == "precision" else "%{y:,}") + "<extra>" + ui.fs_label(fs)
                          + "</extra>"))
        p = d[d.budget == b]
        fig.add_trace(go.Scatter(
            x=p.budget, y=p[col], mode="markers", showlegend=False, hoverinfo="skip",
            marker=dict(color=ui.fs_color(fs), size=10, line=dict(color=c["surface"], width=2))))
    _marker_shape(fig, b, c)
    ui.style(fig, height=340, legend=len(present) > 1)
    fig.update_xaxes(title="Review budget", tickformat=".0%", hoverformat=".1%", range=[0, 0.0515])
    is_all = sel.subset == "all"
    if col == "precision":
        fig.update_yaxes(title="Precision (share of queue that is fraud)" if is_all else "Precision within subset",
                         tickformat=".0%", range=[0, 1.04])
    else:
        fig.update_yaxes(title="False positives in queue" if is_all else "Subset legit in queue",
                         tickformat=",", rangemode="tozero")
    table = sub[["budget", "feature_set", "n_alerts", "tp", "fp", "precision"]]
    table = table.assign(budget=table.budget.map(_budget_label), feature_set=table.feature_set.map(ui.fs_label),
                         precision=table.precision.map(_pct))
    table.columns = (["Budget", "Features", "Alerts", "Fraud caught", "False positives", "Precision"] if is_all else
                     ["Budget", "Features", "Alerts (whole split)", "Subset fraud caught", "Subset legit in queue",
                      "Precision within subset"])
    if col == "precision":
        caption = ("Precision falls as the budget grows past the fraud count: the extra alerts are mostly legit."
                   if is_all else
                   "Precision within the subset: subset fraud / (subset fraud + subset legit) in the queue. "
                   "It falls as the budget grows: the extra alerts are mostly legit.")
    else:
        caption = ("Legit transactions the analysts review for nothing, over the whole split." if is_all else
                   "Legit transactions in the queue whose amount falls in this subset's range, not every false "
                   "positive in the queue (switch the subset to “All” for that).")
    ui.chart(fig, table, key="budget_second_chart", caption=caption)


def _workload(head):
    with st.container(border=True):
        st.markdown("**Analyst workload**")
        a, b_, c_ = st.columns([1.2, 1, 1])
        with a:
            minutes = st.number_input("Minutes per review (assumption)", min_value=0.5,
                                      max_value=60.0, value=DEFAULT_MINUTES, step=0.5, key="budget_minutes")
        per_day = head["alerts_per_day"]
        hours = per_day * minutes / 60
        with b_:
            ui.stat("Analyst hours per day", f"{hours:,.1f}")
        with c_:
            ui.stat(f"Analysts needed ({SHIFT_HOURS} h shifts)", f"{hours / SHIFT_HOURS:,.2f}")
        st.caption(f"Analyst hours per day = alerts per day ({per_day:,.1f}) × minutes per review ({minutes:g}) / 60. "
                   f"Analysts = hours / {SHIFT_HOURS}. The same for txn and txn + graph: the queue size depends "
                   "only on the budget; what changes is how much of it is fraud.")
