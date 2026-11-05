"""Model scoreboard: which model does best, and how much do graph features add, with uncertainty?

Reads results_{main,rf}_{split}.csv and results_{main,rf}_paired_{split}.csv through app.data.
Every number shown is a value from one CSV row (value / ci_low / ci_high, or diff / ci_low /
ci_high). Nothing is recomputed; the only conversions are display ones (percent, percentage points).
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import data as D, ui

K = "scoreboard_"
PCT_PREFIXES = ("recall", "precision", "amount_share")
# Shared labels plus fraud amount caught (in the CSVs, not yet in app.data.METRIC_LABELS).
METRICS = {**D.METRIC_LABELS,
           **{k: v for k, v in {"amount_caught_at_1pct": "Fraud amount caught @ 1%",
                                "amount_caught_at_2pct": "Fraud amount caught @ 2%"}.items()
              if k not in D.METRIC_LABELS}}
UNDEFINED = "not defined"


def _nan(x):
    return x is None or pd.isna(x)


def _lower_better(base):
    return base.startswith("fp")


# --- formatting (display only) -------------------------------------------------

def _kind(base):
    if base in D.COUNT_METRICS:
        return "count"
    return "pct" if base.startswith(PCT_PREFIXES) else "score"


def _fmt(base, x):
    if _nan(x):
        return "–"
    k = _kind(base)
    if k == "count":
        return f"{x:,.0f}"
    if k == "pct":
        return f"{100 * x:.1f}%"
    return f"{x:.3f}"


def _fmt_diff(base, x):
    if _nan(x):
        return "–"
    k = _kind(base)
    if k == "count":
        return f"{x:+,.0f}"
    if k == "pct":
        return f"{100 * x:+.1f} pp"
    return f"{x:+.3f}"


def _with_ci(base, row, diff=False):
    if row is None:
        return "–"
    f = _fmt_diff if diff else _fmt
    v = row["diff"] if diff else row["value"]
    if _nan(v):
        return f"– ({UNDEFINED})"
    return f"{f(base, v)} [{f(base, row['ci_low'])}, {f(base, row['ci_high'])}]"


def _tickformat(base, diff=False):
    k = _kind(base)
    if k == "count":
        return "+,d" if diff else ",d"
    if k == "pct":
        return "+.0f" if diff else ".0%"  # differences are plotted in percentage points
    return "+.3f" if diff else ".2f"


def _verdict(base, row):
    """Reads the CI from the paired row: (label, badge colour, icon)."""
    lo, hi = row["ci_low"], row["ci_high"]
    if _nan(row["diff"]) or _nan(lo) or _nan(hi):
        return "Not defined in the results file", "gray", ":material/help:"
    lower_better = _lower_better(base)
    if lo <= 0 <= hi:
        return "CI includes zero: no clear difference", "gray", ":material/remove:"
    graph_higher = lo > 0
    better = graph_higher != lower_better
    return (f"CI excludes zero: graph {'better' if better else 'worse'}",
            "green" if better else "red", ":material/check:" if better else ":material/close:")


def _tick(model, sel_model):
    label = D.MODEL_LABELS[model] + ("<br>(robustness)" if model == "rf" else "")
    return f"<b>{label}</b>" if model == sel_model else label


# --- lookups (filter only) -----------------------------------------------------

def _rows(df, base, subset):
    return df[df.metric == D.metric_name(base, subset)]


def _one(df, **eq):
    if df is None:
        return None
    m = df
    for col, val in eq.items():
        m = m[m[col] == val]
    return None if m.empty else m.iloc[0].to_dict()


def _paired_row(pdf, model, base, subset):
    if pdf is None:
        return None
    return _one(pdf, model=model, metric=D.metric_name(base, subset),
                feature_set_base="txn", feature_set_new="txn_graph")


# --- sections -------------------------------------------------------------------

def _tiles(sel, res, pdf, ceil):
    m, sub = sel.model, sel.subset
    name = D.MODEL_LABELS[m]
    pr = {fs: _one(res, model=m, feature_set=fs, metric=D.metric_name("pr_auc", sub)) for fs in sel.feature_sets}
    rec = {fs: _one(res, model=m, feature_set=fs, metric=D.metric_name("recall_at_1pct", sub))
           for fs in sel.feature_sets}
    both = len(sel.feature_sets) == 2
    p_pr = _paired_row(pdf, m, "pr_auc", sub) if both else None
    p_rec = _paired_row(pdf, m, "recall_at_1pct", sub) if both else None

    if all(v is None for v in pr.values()):
        st.info(f"No {name} rows for {D.SUBSET_LABELS[sub].lower()} on the {sel.split} split.")
        return

    ceil_txt = (f"Best possible recall @ 1% on {sel.split} is {ui.pct(ceil, 2)}: the 1% queue is smaller than "
                "the number of fraud cases." if (sub == "all" and ceil is not None) else
                "No recall ceiling is published for subsets; the ceiling applies to all fraud.")
    cols = st.columns(4 if both else 3)
    i = 0
    for fs in sel.feature_sets:
        with cols[i]:
            r = pr[fs]
            ui.stat(f"PR-AUC, {ui.fs_label(fs)}", _fmt("pr_auc", r and r["value"]),
                    help=f"95% bootstrap CI {_with_ci('pr_auc', r)}." if r else "Not in the results file.")
        i += 1
    if both:
        with cols[i]:
            if p_pr is None:
                ui.stat("Paired gain, PR-AUC", "–", help="Paired results file missing.")
            else:
                ui.stat("Paired gain, PR-AUC", _fmt_diff("pr_auc", p_pr["diff"]),
                        help="txn + graph minus txn on the same rows, paired bootstrap (1,000 resamples). "
                             f"95% CI [{_fmt_diff('pr_auc', p_pr['ci_low'])}, {_fmt_diff('pr_auc', p_pr['ci_high'])}].")
                st.caption(f"95% CI [{_fmt_diff('pr_auc', p_pr['ci_low'])}, {_fmt_diff('pr_auc', p_pr['ci_high'])}]")
        i += 1
    fs_rec = "txn_graph" if "txn_graph" in sel.feature_sets else sel.feature_sets[0]
    with cols[i]:
        r = rec[fs_rec]
        delta = (f"{_fmt_diff('recall_at_1pct', p_rec['diff'])} vs txn"
                 if p_rec and not _nan(p_rec["diff"]) else None)
        ui.stat(f"Recall @ 1%, {ui.fs_label(fs_rec)}", _fmt("recall_at_1pct", r and r["value"]), delta=delta,
                help=(f"95% CI {_with_ci('recall_at_1pct', r)}. " if r else "") + ceil_txt)
        if sub == "all" and ceil is not None:
            st.caption(f"of a best possible {ui.pct(ceil, 2)}")


def _bars(sel, res, base, ceil):
    df = _rows(res, base, sel.subset)
    models = [m for m in D.MODELS if m in set(df.model)]
    if not models:
        st.info(f"No rows for {METRICS[base]} on {D.SUBSET_LABELS[sel.subset].lower()} "
                f"({sel.split} split).")
        return
    fig = go.Figure()
    ticks = [_tick(m, sel.model) for m in models]
    table, undefined = [], []
    for fs in sel.feature_sets:
        sub = df[df.feature_set == fs].drop_duplicates("model").set_index("model")
        for m in models:
            if m in sub.index and (_nan(sub.at[m, "value"]) or _nan(sub.at[m, "ci_low"])
                                   or _nan(sub.at[m, "ci_high"])):
                undefined.append(f"{D.MODEL_LABELS[m]} ({ui.fs_label(fs)})")
        ms = [m for m in models if m in sub.index and not _nan(sub.at[m, "value"])]
        if not ms:
            continue
        s = sub.loc[ms]
        fig.add_bar(
            name=ui.fs_label(fs), x=[_tick(m, sel.model) for m in ms], y=s.value,
            marker=dict(color=ui.fs_color(fs), line=dict(width=0)),
            error_y=dict(type="data", symmetric=False, array=s.ci_high - s.value, arrayminus=s.value - s.ci_low,
                         color=ui.colors()["ink2"], thickness=1.5, width=5),
            customdata=[_with_ci(base, r) for r in s.reset_index().to_dict("records")],
            hovertemplate=f"{ui.fs_label(fs)}: %{{customdata}}<extra></extra>",
        )
        for m, r in s.iterrows():
            table.append({"Model": D.MODEL_LABELS[m], "Features": ui.fs_label(fs), "Value": _fmt(base, r.value),
                          "CI low": _fmt(base, r.ci_low), "CI high": _fmt(base, r.ci_high),
                          "Fraud cases": int(r.n_fraud),
                          "Rows": int(r.n_rows), "Source": f"results_{r.family}_{sel.split}.csv"})
    show_ceiling = base == "recall_at_1pct" and sel.subset == "all" and ceil is not None
    if show_ceiling:
        c = ui.colors()
        fig.add_hline(y=ceil, line=dict(color=c["ink2"], width=1.5, dash="dash"),
                      annotation=dict(text=f"best possible {ui.pct(ceil, 2)}", font=dict(color=c["ink2"]),
                                      xanchor="right", yanchor="bottom"),
                      annotation_position="top right")
    ui.style(fig, height=380, legend=len(sel.feature_sets) > 1)
    fig.update_layout(barcornerradius=4, bargap=0.3, bargroupgap=0.1)
    fig.update_xaxes(categoryorder="array", categoryarray=ticks)
    fig.update_yaxes(tickformat=_tickformat(base), title=METRICS[base], rangemode="tozero")
    if _kind(base) == "pct" or base in ("pr_auc", "roc_auc", "f1"):
        fig.update_yaxes(range=[0, 1.05])
    caps = ["Whiskers are 95% bootstrap CIs. Bars start at zero, so small gaps near the top are easier "
            "to read in the paired chart below."]
    if "rf" in models:
        caps.append("Random forest is a robustness check (appendix), not part of the main table.")
    if _lower_better(base):
        caps.append("Fewer false positives is better.")
    if undefined:
        caps.append(f"No bar for {', '.join(undefined)}: the value is not defined in the results file.")
    if sel.subset == "normal_looking_q90":
        caps.append("The q90 subset is a sensitivity check, not a way to pick the best number.")
    ui.chart(fig, table=table, key=f"{K}bars", caption=" ".join(caps))


def _forest(sel, pdf, base):
    if len(sel.feature_sets) < 2:
        st.info("Pick **Compare** under Features in the sidebar to see the paired difference "
                "(txn + graph minus txn).")
        return
    if pdf is None:
        ui.empty_state("Paired differences not available", "The paired bootstrap file for this split "
                       "hasn't been written yet.", files=[f"results_main_paired_{sel.split}.csv",
                                                          f"results_rf_paired_{sel.split}.csv"], owner="R4 Rayson")
        return
    df = pdf[(pdf.metric == D.metric_name(base, sel.subset)) & (pdf.feature_set_base == "txn")
             & (pdf.feature_set_new == "txn_graph")]
    models = [m for m in D.MODELS if m in set(df.model)]
    if not models:
        st.info(f"No paired rows for {METRICS[base]} on {D.SUBSET_LABELS[sel.subset].lower()}.")
        return
    s_all = df.drop_duplicates("model").set_index("model").loc[models]
    ok = s_all[["diff", "ci_low", "ci_high"]].notna().all(axis=1)
    undefined = [m for m in models if not ok[m]]
    s = s_all[ok]
    badges = list(zip(models, s_all.reset_index().to_dict("records")))
    models = [m for m in models if ok[m]]
    if not models:
        st.info(f"The paired difference for {METRICS[base]} is not defined for any model on "
                f"{D.SUBSET_LABELS[sel.subset].lower()}.")
        _badges(base, badges)
        return
    # Display conversion only: percentage-metric differences are plotted in percentage points.
    scale = 100 if _kind(base) == "pct" else 1
    c = ui.colors()
    ticks = [_tick(m, sel.model) for m in models]
    fig = go.Figure()
    fig.add_vline(x=0, line=dict(color=c["baseline"], width=1.5))
    fig.add_scatter(
        x=s["diff"] * scale, y=ticks, mode="markers", name="txn + graph minus txn",
        marker=dict(color=c["ink2"], size=10, line=dict(color=c["surface"], width=2)),
        error_x=dict(type="data", symmetric=False, array=(s.ci_high - s["diff"]) * scale,
                     arrayminus=(s["diff"] - s.ci_low) * scale,
                     color=c["ink2"], thickness=2, width=6),
        customdata=[f"{D.MODEL_LABELS[m]}<br>{_with_ci(base, r, diff=True)}"
                    for m, r in zip(models, s.reset_index().to_dict("records"))],
        hovertemplate="%{customdata}<extra></extra>",
    )
    lo, hi = min(0.0, s.ci_low.min() * scale), max(0.0, s.ci_high.max() * scale)
    pad = 0.12 * (hi - lo or 1)
    ui.style(fig, height=80 + 62 * len(models), legend=False, hover="closest")
    fig.update_yaxes(categoryorder="array", categoryarray=ticks, autorange="reversed", showgrid=False)
    unit = " (percentage points)" if _kind(base) == "pct" else ""
    fig.update_xaxes(tickformat=_tickformat(base, diff=True), range=[lo - pad, hi + pad],
                     ticksuffix=" pp" if scale == 100 else "",
                     title=f"txn + graph minus txn, {METRICS[base]}{unit}")
    table = [{"Model": D.MODEL_LABELS[m], "Difference": _fmt_diff(base, r["diff"]),
              "CI low": _fmt_diff(base, r["ci_low"]), "CI high": _fmt_diff(base, r["ci_high"]),
              "Source": f"results_{r['family']}_paired_{sel.split}.csv"}
             for m, r in zip(models, s.reset_index().to_dict("records"))]
    note = " Negative means fewer false positives with graph features, which is better." \
        if _lower_better(base) else " Right of zero means graph features help."
    if undefined:
        note += (f" Not plotted: {', '.join(D.MODEL_LABELS[m] for m in undefined)} (difference not defined "
                 "in the results file).")
    ui.chart(fig, table=table, key=f"{K}forest",
             caption="Dot = paired difference on the same rows, whiskers = 95% paired bootstrap CI." + note)
    _badges(base, badges)


def _badges(base, items):
    cols = st.columns(len(items))
    for col, (m, r) in zip(cols, items):
        label, color, icon = _verdict(base, r)
        with col:
            st.markdown(f"**{D.MODEL_SHORT[m]}**")
            st.badge(label, color=color, icon=icon)


def _detail(sel, res, pdf, ceil):
    m, sub = sel.model, sel.subset
    both = len(sel.feature_sets) == 2
    rows, any_row, size = [], False, None
    for base, label in METRICS.items():
        rec = {"Metric": label}
        for fs in sel.feature_sets:
            r = _one(res, model=m, feature_set=fs, metric=D.metric_name(base, sub))
            any_row = any_row or r is not None
            size = size or r
            rec[ui.fs_label(fs)] = _with_ci(base, r)
        if both:
            p = _paired_row(pdf, m, base, sub)
            rec["Paired difference"] = _with_ci(base, p, diff=True)
            rec["CI excludes zero"] = (None if p is None or _nan(p["ci_low"]) or _nan(p["ci_high"])
                                       else not (p["ci_low"] <= 0 <= p["ci_high"]))
        rows.append(rec)
    if not any_row:
        st.info(f"{D.MODEL_LABELS[m]} has no rows for {D.SUBSET_LABELS[sub].lower()} on the {sel.split} split.")
        return
    cfg = {"Metric": st.column_config.TextColumn("Metric", width="medium")}
    for fs in sel.feature_sets:
        cfg[ui.fs_label(fs)] = st.column_config.TextColumn(ui.fs_label(fs), help="Point estimate [95% bootstrap CI]")
    if both:
        cfg["Paired difference"] = st.column_config.TextColumn(
            "Paired difference", help="txn + graph minus txn [95% paired bootstrap CI]. For false positives, "
                                      "negative is better.")
        cfg["CI excludes zero"] = st.column_config.CheckboxColumn("CI excludes zero")
    st.dataframe(rows, hide_index=True, width="stretch", column_config=cfg, key=f"{K}detail")
    caps = []
    if size is not None:
        caps.append(f"{D.SUBSET_LABELS[sub]}: {int(size['n_fraud']):,} fraud cases among {int(size['n_rows']):,} "
                    f"rows ({sel.split} split).")
    if sub != "all":
        caps.append("Subset rows are every transaction, fraud or legit, that passes the same amount rule; "
                    "budget metrics still use the whole-split review queue.")
    if sub == "all" and ceil is not None:
        caps.append(f"Best possible recall @ 1% is {ui.pct(ceil, 2)}.")
    caps.append("Thresholded metrics use the F1 threshold chosen on val.")
    st.caption(" ".join(caps))


# --- entry point -----------------------------------------------------------------

def render(sel):
    ui.section("Model scoreboard", "Which model does best, and how much do graph features add? "
                                    "Every number is a point estimate with its 95% bootstrap CI.")
    res = D.results("", sel.split)
    if res is None:
        if sel.split == "test":
            ui.empty_state("Test results arrive at the freeze",
                           "Test scores are written once, at the results freeze (Sun 15 Nov). Until then "
                           "use the val split.", files=D.pending_files("test"), owner="everyone")
        else:
            ui.empty_state("No results yet", "Run `uv run python -m src.models.run_main`, then "
                           "`uv run python -m src.eval.metrics`.",
                           files=[f"results_main_{sel.split}.csv", f"results_rf_{sel.split}.csv"], owner="R3 / R4")
        return
    pdf = D.results("paired", sel.split)
    ceil = D.ceiling(sel.split, "1pct")

    st.markdown(f"**{D.MODEL_LABELS[sel.model]}** on {D.SUBSET_LABELS[sel.subset].lower()}")
    _tiles(sel, res, pdf, ceil)

    st.write("")
    ui.section("Compare models", "Pick a metric. The sidebar model is in bold.")
    base = st.pills("Metric", list(METRICS), default="pr_auc", required=True,
                    format_func=METRICS.get, key=f"{K}metric", label_visibility="collapsed") or "pr_auc"
    left, right = st.columns([3, 2], gap="large")
    with left:
        _bars(sel, res, base, ceil)
    with right:
        st.markdown("**Gain from graph features**")
        _forest(sel, pdf, base)

    st.write("")
    ui.section(f"All metrics: {D.MODEL_LABELS[sel.model]}",
               "Point estimate [95% bootstrap CI]. Change the model in the sidebar.")
    _detail(sel, res, pdf, ceil)
