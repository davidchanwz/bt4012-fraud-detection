"""Tab 3, "Why graph features?": does fraud that looks ordinary on amount also look ordinary on
customer/merchant history?

FULL MODE: plan item A, built by R4 (`uv run python -m src.eda --normal-looking`, train days 0-119 only):
  normal_looking_comparison.csv    one row per feature: feature, label, group, nl_fraud_mean,
      nl_fraud_median, nl_legit_mean, nl_legit_median, nl_legit_same_category_mean,
      smd_vs_nl_legit, smd_vs_same_category_legit
      (SMD = fraud minus legit in pooled standard deviations; "same category" reweights the legit
      rows to the fraud's category mix, so category differences aren't read as history differences)
  normal_looking_category_mix.csv  one row per category: category, nl_fraud, nl_legit,
      share_of_nl_fraud, share_of_nl_legit, fraud_rate_within_nl
  reports/figures/07_normal_looking_fraud_vs_legit.png   the report's static version
If the columns differ, the tab shows the raw table with a note instead of a chart.

INTERIM MODE (before item A existed; kept behind a toggle): subgroup_thresholds.csv, subgroup_sizes.csv,
category_summary.csv, graph_feature_summary.csv (ALL fraud vs ALL legit), lgbm_importance*_txn_graph_{split}.csv.

Read-only: every number is a CSV value. The only arithmetic is labelled display conversions
(|SMD| for sorting, percentages, a fraud / legit mean ratio in the interim view).
"""
import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import data as D, ui
from src.config import FIG_DIR

COMPARISON = "normal_looking_comparison.csv"
MIX = "normal_looking_category_mix.csv"
FIGURE = FIG_DIR / "07_normal_looking_fraud_vs_legit.png"
CMP_REQUIRED = {"feature", "smd_vs_nl_legit", "smd_vs_same_category_legit"}
MIX_REQUIRED = {"category", "share_of_nl_fraud", "share_of_nl_legit"}
SMALL, LARGE = 0.2, 0.8           # conventional SMD cut-offs, as in src/eda.py

# Colour slots in ui.colors()["series"]. Slots 0 and 1 belong to the feature sets (txn / txn + graph)
# app-wide, so this tab uses others. Slots 5 (pure green) and 7 (red) read as status colours, so they
# are not used. Legit is the neutral reference class and is drawn in the muted ink token (None below).
FEATURE_KIND_SLOT = {"txn": 3, "history": 2}                   # amber / green: transaction vs history feature
CLASS_SLOT = {"nl_fraud": 6, "nl_legit": None, "obvious_fraud": 4,  # violet / neutral grey / pink
              "legit": None, "fraud": 4}                           # interim: all legit / all fraud
CLASS_LABELS = {"nl_fraud": "Normal-looking fraud", "nl_legit": "Normal-looking legit",
                "obvious_fraud": "Obvious fraud (the rest)", "legit": "All legit", "fraud": "All fraud"}
ARTIFACT_NOTE = ("Caveat: `g1_cust_n_prev` (number of past transactions) is probably a BankSim simulation "
                 "artifact (low-activity customers are victimised far more). It is not leakage, but the story "
                 "should not rest on it alone.")

# --- helpers ------------------------------------------------------------------

def _series(slot):
    """Series colour for a slot; None means the neutral reference colour (legit)."""
    return ui.colors()["muted"] if slot is None else ui.colors()["series"][slot]


def _group_of(feature, given=None):
    if isinstance(given, str) and given:
        return given
    head = str(feature).split("_")[0]
    return head if head in D.GROUP_LABELS and head != "txn" else "txn"


def _kind(group):
    return "txn" if group == "txn" else "history"


def _kind_label(kind):
    return "Transaction feature" if kind == "txn" else "History feature (graph)"


def _fmt(x):
    """Readable number for means that range from 0.002 to 68,000."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    a = abs(x)
    if a >= 1000:
        return f"{x:,.0f}"
    if a >= 1:
        return f"{x:,.2f}"
    return f"{x:.4f}"


def _raw(df, name, missing):
    st.warning(f"`{name}` is present but its columns don't match the expected schema "
               f"(missing: {', '.join(sorted(missing))}). Showing the raw table; see the schema at the top "
               "of `app/tabs/eda.py`.")
    st.dataframe(df, hide_index=True, width="stretch")


def _share_tile(label, value, share, help):
    """ui.stat look, but the delta slot holds a static share, so no arrow (ui.stat has no delta_arrow)."""
    st.metric(label, value, delta=share, help=help, border=True, delta_color="off", delta_arrow="off")


def _join(xs):
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def _height(n, per=24, pad=70, lo=220):
    return max(lo, per * n + pad)


# --- full mode (item A) ---------------------------------------------------------

def _short_group(g):
    """'G3 pair' -> 'G3'; 'transaction' -> 'txn'."""
    g = str(g)
    return g.split()[0] if g[:1] == "G" else "txn"


def _full_mode(sel):
    cmp_df, mix = D.load(COMPARISON), D.load(MIX)
    ui.section("Normal-looking fraud vs normal-looking legit",
               "Both groups are under the same per-category amount cut, so amount alone can't separate them. "
               "Train days 0–119 only (item A, `src/eda.py --normal-looking`), so this view ignores the "
               "sidebar's split.")
    miss_c = CMP_REQUIRED - set(cmp_df.columns)
    miss_m = None if mix is None else MIX_REQUIRED - set(mix.columns)
    if miss_c:
        _raw(cmp_df, COMPARISON, miss_c)
        return
    c = cmp_df.copy()
    c["label"] = c["label"] if "label" in c.columns else c.feature
    c["group"] = c["group"] if "group" in c.columns else ["transaction" if not str(f).startswith("g")
                                                          else str(f)[:2].upper() for f in c.feature]
    c["g"] = c.group.map(_short_group)
    _headline(c, mix if mix is not None and not miss_m else None)

    st.markdown("##### 1 · Category is not ordinary")
    if mix is None:
        st.info(f"`{MIX}` not found.")
    elif miss_m:
        _raw(mix, MIX, miss_m)
    else:
        _mix_chart(mix)

    st.markdown("##### 2 · Which history differences survive once category is matched")
    _smd_chart(c)

    st.markdown("##### 3 · One feature in detail")
    _feature_detail(c)

    if "g1_cust_n_prev" in set(c.feature):
        st.caption(ARTIFACT_NOTE)
    if FIGURE.exists():
        with st.expander("The report's static figure (figure 07)", icon=":material/image:"):
            st.image(str(FIGURE), width="stretch")
            st.caption(f"`reports/figures/{FIGURE.name}`, drawn by `src/eda.py` from the same tables.")


def _headline(c, mix):
    """Three tiles that state the finding, read straight from the two tables."""
    cols = st.columns(3)
    with cols[0]:
        if mix is not None and not mix.empty:
            top = mix.sort_values("share_of_nl_fraud", ascending=False).iloc[0]
            ui.stat(f"Normal-looking fraud in {top.category}", ui.pct(top.share_of_nl_fraud),
                    delta=f"{ui.pct(top.share_of_nl_legit)} of normal-looking legit", delta_color="off",
                    help="Category mix differs a lot, so category still gives normal-looking fraud away.")
        else:
            ui.stat("Category mix", "–")
    amt = c[c.feature == "amount"]
    with cols[1]:
        if not amt.empty:
            a = amt.iloc[0]
            ui.stat("Amount difference within category", f"{a.smd_vs_same_category_legit:+.2f} sd",
                    delta=f"{a.smd_vs_nl_legit:+.2f} sd vs all normal-looking legit", delta_color="off",
                    help="Standardised mean difference. Below 0.2 in size is conventionally negligible: within "
                         "the same categories, the amounts overlap.")
        else:
            ui.stat("Amount difference within category", "–")
    hist = c[c.g != "txn"].sort_values("smd_vs_same_category_legit", key=abs, ascending=False)
    with cols[2]:
        if not hist.empty:
            top = hist.iloc[0]
            nxt = hist.iloc[1:3]
            ui.stat("Largest history difference within category", f"{top.smd_vs_same_category_legit:+.2f} sd",
                    delta=f"{top.label} [{top.g}]", delta_color="off",
                    help="After matching legit rows to the fraud's category mix. Next: " + "; ".join(
                        f"{r.label} [{r.g}] {r.smd_vs_same_category_legit:+.2f}" for r in nxt.itertuples())
                         + ". Full ranking in chart 2.")
        else:
            ui.stat("Largest history difference within category", "–")


def _mix_chart(mix):
    m = mix.copy()
    has = m[(m.share_of_nl_fraud > 0)].sort_values("share_of_nl_fraud")
    none = m[m.share_of_nl_fraud == 0]
    left, right = st.columns([3, 2], gap="large")
    with left:
        fig = go.Figure()
        for cls, col, cnt in (("nl_legit", "share_of_nl_legit", "nl_legit"),
                              ("nl_fraud", "share_of_nl_fraud", "nl_fraud")):
            n = has[cnt] if cnt in has.columns else pd.Series([None] * len(has), index=has.index)
            fig.add_bar(y=has.category, x=has[col], orientation="h", name=CLASS_LABELS[cls],
                        marker=dict(color=_series(CLASS_SLOT[cls]), cornerradius=4),
                        customdata=[("–" if v is None or pd.isna(v) else f"{int(v):,}") for v in n],
                        hovertemplate="<b>%{y}</b><br>" + CLASS_LABELS[cls]
                                      + ": %{x:.1%} (%{customdata} txns)<extra></extra>")
        fig.update_layout(barmode="group")
        fig.update_xaxes(title="Share of the group's transactions", tickformat=".0%", rangemode="tozero")
        fig.update_yaxes(categoryorder="array", categoryarray=list(has.category))
        ui.style(fig, height=_height(len(has), per=30), hover="y unified")
        ui.chart(fig, mix, key="eda_mix",
                 caption="Categories with at least one normal-looking fraud. "
                         + (f"{_join(list(none.category))} have none; they hold "
                            f"{ui.pct(none.share_of_nl_legit.sum())} of normal-looking legit rows."
                            if not none.empty else ""))
    with right:
        if "fraud_rate_within_nl" in has.columns:
            r = has.sort_values("fraud_rate_within_nl")
            fig = go.Figure(go.Bar(y=r.category, x=r.fraud_rate_within_nl, orientation="h",
                                   marker=dict(color=_series(CLASS_SLOT["nl_fraud"]), cornerradius=4),
                                   hovertemplate="<b>%{y}</b><br>%{x:.1%} of normal-looking rows are fraud"
                                                 "<extra></extra>"))
            fig.update_xaxes(title="Fraud rate within normal-looking", tickformat=".0%", rangemode="tozero")
            fig.update_yaxes(categoryorder="array", categoryarray=list(r.category))
            ui.style(fig, height=_height(len(r), per=30), legend=False, hover="closest")
            ui.chart(fig, None, key="eda_mix_rate",
                     caption="Among small-amount transactions in each category, the share that are fraud. "
                             "Same table as the left chart.")


def _smd_chart(c):
    view = st.segmented_control(
        "Compare normal-looking fraud with", ["both", "same", "all"], default="both", required=True,
        key="eda_smd_view", format_func={"both": "Both", "same": "Legit in the same categories",
                                         "all": "All normal-looking legit"}.get)
    d = c.sort_values("smd_vs_same_category_legit", key=abs).reset_index(drop=True)   # display order only
    ylab = [f"{r.label}  [{r.g}]" for r in d.itertuples()]
    col = ui.colors()
    fig = go.Figure()
    lo = min(-1.0, d[["smd_vs_nl_legit", "smd_vs_same_category_legit"]].min().min() - 0.2)
    hi = max(1.0, d[["smd_vs_nl_legit", "smd_vs_same_category_legit"]].max().max() + 0.2)
    fig.add_vrect(x0=-SMALL, x1=SMALL, fillcolor=col["grid"], opacity=0.6, line_width=0, layer="below")
    fig.add_vline(x=0, line=dict(color=col["baseline"], width=1))
    if view == "both":
        for r, y in zip(d.itertuples(), ylab):
            fig.add_trace(go.Scatter(x=[r.smd_vs_nl_legit, r.smd_vs_same_category_legit], y=[y, y],
                                     mode="lines", line=dict(color=col["baseline"], width=2),
                                     showlegend=False, hoverinfo="skip"))
    means = list(zip([_fmt(v) for v in d.get("nl_fraud_mean", d.smd_vs_nl_legit * float("nan"))],
                     [_fmt(v) for v in d.get("nl_legit_mean", d.smd_vs_nl_legit * float("nan"))],
                     [_fmt(v) for v in d.get("nl_legit_same_category_mean", d.smd_vs_nl_legit * float("nan"))]))
    if view in ("both", "all"):
        fig.add_trace(go.Scatter(
            x=d.smd_vs_nl_legit, y=ylab, mode="markers", name="vs all normal-looking legit",
            marker=dict(size=10, color=col["surface"], line=dict(color=col["muted"], width=2)),
            customdata=means, hovertemplate="<b>%{y}</b><br>vs all normal-looking legit: %{x:+.2f} sd"
                                            "<br>mean: fraud %{customdata[0]}, legit %{customdata[1]}<extra></extra>"))
    if view in ("both", "same"):
        fig.add_trace(go.Scatter(
            x=d.smd_vs_same_category_legit, y=ylab, mode="markers", name="vs legit in the same categories",
            marker=dict(size=11, color=_series(CLASS_SLOT["nl_fraud"]), line=dict(color=col["surface"], width=2)),
            customdata=means, hovertemplate="<b>%{y}</b><br>vs same-category legit: %{x:+.2f} sd"
                                            "<br>mean: fraud %{customdata[0]}, same-category legit "
                                            "%{customdata[2]}<extra></extra>"))
    fig.update_xaxes(title="Fraud minus legit, in standard deviations", range=[lo, hi], zeroline=False)
    fig.update_yaxes(categoryorder="array", categoryarray=ylab)
    ui.style(fig, height=_height(len(d), per=34, pad=90), hover="closest")
    flips = d[(d.smd_vs_nl_legit * d.smd_vs_same_category_legit < 0)
              & (d.smd_vs_nl_legit.abs() >= SMALL) & (d.smd_vs_same_category_legit.abs() >= SMALL)]
    table = c.drop(columns=["g"]).sort_values("smd_vs_same_category_legit", key=abs, ascending=False)
    ui.chart(fig, table, key="eda_smd",
             caption="Right of zero: fraud has the higher value. Grey band: difference under 0.2 sd (negligible). "
                     "Hollow dot: against every normal-looking legit row. Filled dot: against legit rows "
                     "reweighted to the fraud's category mix, the fairer comparison."
                     + (f" Reverses once category is matched: {_join(list(flips.label))}." if not flips.empty
                        else ""))


def _feature_detail(c):
    order = list(c.sort_values("smd_vs_same_category_legit", key=abs, ascending=False).feature)
    labels = dict(zip(c.feature, c.label))
    feat = st.selectbox("Feature", order, key="eda_feature_pick",
                        format_func=lambda f: f"{labels.get(f, f)}  ({f})",
                        help="Sorted by the size of the same-category difference.")
    r = c[c.feature == feat].iloc[0]
    left, right = st.columns([3, 2], gap="large")
    with left:
        bars = [("nl_fraud_mean", CLASS_LABELS["nl_fraud"], _series(CLASS_SLOT["nl_fraud"])),
                ("nl_legit_mean", CLASS_LABELS["nl_legit"], _series(CLASS_SLOT["nl_legit"])),
                ("nl_legit_same_category_mean", "Legit, same categories", ui.colors()["ink2"])]
        bars = [b for b in bars if b[0] in c.columns]
        meds = {"nl_fraud_mean": r.get("nl_fraud_median"), "nl_legit_mean": r.get("nl_legit_median")}
        fig = go.Figure()
        for key, name, color in bars:
            med = meds.get(key)
            fig.add_bar(x=[name], y=[r[key]], name=name, marker=dict(color=color, cornerradius=4),
                        hovertemplate=f"{name}: mean {_fmt(r[key])}"
                                      + (f", median {_fmt(med)}" if med is not None and pd.notna(med) else "")
                                      + "<extra></extra>")
        fig.update_yaxes(title=f"Mean of {labels.get(feat, feat)}", rangemode="tozero")
        ui.style(fig, height=300, hover="closest")
        ui.chart(fig, c[c.feature == feat].drop(columns=["g"]), key="eda_feature_means",
                 caption="Means on train. The median is in the hover.")
    with right:
        ui.stat("vs legit in the same categories", f"{r.smd_vs_same_category_legit:+.2f} sd",
                help="Standardised mean difference after reweighting legit rows to the fraud's category mix.")
        ui.stat("vs all normal-looking legit", f"{r.smd_vs_nl_legit:+.2f} sd",
                help="Standardised mean difference against every normal-looking legit row.")
        size = abs(r.smd_vs_same_category_legit)
        st.caption(f"{labels.get(feat, feat)} ({r.group}): within the same categories the difference is "
                   + ("large" if size >= LARGE else "moderate" if size >= 0.5 else "small" if size >= SMALL
                      else "negligible") + ".")


# --- link to the model results ----------------------------------------------------

def _model_link(sel):
    subset = sel.subset if sel.subset != "all" else "normal_looking"
    ui.section("Does the model see the difference?",
               f"Recall at a 1% review budget on {D.SUBSET_LABELS[subset].lower()} fraud, "
               f"{D.MODEL_LABELS.get(sel.model, sel.model)}, {sel.split} split."
               + (" Pick a subset in the sidebar to change it." if sel.subset == "all" else ""))
    if sel.subset == "all":
        st.badge("showing normal-looking (sidebar: All fraud)", icon=":material/filter_alt:", color="gray")
    rows = {fs: D.metric(sel.split, sel.model, fs, "recall_at_1pct", subset) for fs in sel.feature_sets}
    if all(r is None for r in rows.values()):
        if sel.split == "test":
            ui.empty_state("Test split not scored yet", "Test results appear after the freeze (Sun 15 Nov).",
                           files=D.pending_files("test", "test"), owner="everyone")
        else:
            st.info(f"No recall@1% rows for {D.MODEL_LABELS.get(sel.model, sel.model)} on this subset yet.")
        return
    pair = D.paired(sel.split, sel.model, "recall_at_1pct", subset) if len(sel.feature_sets) == 2 else None
    cols = st.columns(len(sel.feature_sets) + 1)
    for col, fs in zip(cols, sel.feature_sets):
        r = rows[fs]
        with col:
            delta = None
            if fs == "txn_graph" and pair is not None:
                delta = (f"{100 * pair['diff']:+.1f} pp vs txn "
                         f"[{100 * pair['ci_low']:+.1f}, {100 * pair['ci_high']:+.1f}]")
            ui.stat(f"Recall @ 1%, {ui.fs_label(fs)}", "–" if r is None else ui.pct(r["value"]), delta=delta,
                    help=None if r is None else f"95% bootstrap CI {ui.pct(r['ci_low'])} to "
                                                f"{ui.pct(r['ci_high'])}; {int(r['n_fraud']):,} fraud cases "
                                                "in the subset." + (" Delta: paired bootstrap difference."
                                                                     if delta else ""))
    with cols[-1]:
        r = next((r for r in rows.values() if r is not None), None)
        ui.stat("Fraud cases in subset", f"{int(r['n_fraud']):,}" if r and pd.notna(r.get("n_fraud")) else "–",
                help="Number of fraud rows in this subset on the split (from R4's results table).")


# --- interim mode ----------------------------------------------------------------

def _what_is_normal_looking(sel):
    thr, cat = D.load("subgroup_thresholds.csv"), D.load("category_summary.csv")
    thr_ok = thr is not None and {"category", "legit_p95_amount_train"} <= set(thr.columns)
    cat_ok = cat is not None and {"category", "fraud_rate"} <= set(cat.columns)
    mostly = list(cat[cat.fraud_rate > 0.5].sort_values("fraud_rate", ascending=False).category) if cat_ok else []
    note = (f" {_join(mostly).capitalize()} {'is' if len(mostly) == 1 else 'are'} mostly fraud "
            "(fraud rate above 50%), so category alone gives a lot away; the subgroup removes the amount "
            "signal, not the category one." if mostly else
            " The subgroup removes the amount signal, not the category one.")
    ui.section("What counts as normal-looking",
               "A fraud is normal-looking if its amount is at or below the 95th percentile of legit amounts "
               "in its category (train only)." + note)
    sizes = D.load("subgroup_sizes.csv")
    need = {"split", "n_fraud", "normal_looking_fraud", "pct_of_fraud"}
    if sizes is None:
        ui.empty_state("Subgroup sizes not found", "Written by the week 1 EDA (`src/eda.py`).",
                       files=["subgroup_sizes.csv"], owner="R1 (week 1 pipeline)")
    elif not need <= set(sizes.columns) or sizes[sizes.split == sel.split].empty:
        st.info(f"No subgroup sizes for the {sel.split} split.")
    else:
        r = sizes[sizes.split == sel.split].iloc[0]
        cols = st.columns(4)
        tiles = [("Fraud cases", f"{int(r.n_fraud):,}", None),
                 ("Normal-looking fraud", f"{int(r.normal_looking_fraud):,}", f"{r.pct_of_fraud:.1f}% of fraud"),
                 ("Strict normal-looking", f"{int(r.strict_normal_looking_fraud):,}"
                  if "strict_normal_looking_fraud" in r else "–",
                  f"{r.strict_pct_of_fraud:.1f}% of fraud" if "strict_pct_of_fraud" in r else None),
                 ("Normal-looking legit", f"{int(r.normal_looking_legit):,}" if "normal_looking_legit" in r
                  else "–", None)]
        helps = [f"All fraud rows in the {sel.split} split.", D.SUBSET_HELP["normal_looking"],
                 D.SUBSET_HELP["normal_looking_strict"],
                 "Legit rows under the same per-category thresholds: the fair comparison group."]
        for col, (lab, val, sub), h in zip(cols, tiles, helps):
            with col:
                _share_tile(lab, val, sub, h)
        st.caption(f"Counts for the {sel.split} split, from `subgroup_sizes.csv`.")

    if not (thr_ok or cat_ok):
        ui.empty_state("Category tables not found", "Written by the week 1 EDA (`src/eda.py`).",
                       files=["subgroup_thresholds.csv", "category_summary.csv"], owner="R1 (week 1 pipeline)")
        return
    # Both charts share one category order (by fraud rate) so rows line up side by side.
    order = (list(cat.sort_values("fraud_rate").category) if cat_ok
             else list(thr.sort_values("legit_p95_amount_train").category))
    if thr_ok:
        order += [x for x in thr.category if x not in order]
    h = _height(len(order), per=22)
    c = ui.colors()
    left, right = st.columns(2, gap="large")
    with left:
        if cat_ok:
            k = cat.set_index("category").reindex([x for x in order if x in set(cat.category)]).reset_index()
            extra = [k.get(x, pd.Series([float("nan")] * len(k))) for x in
                     ("n_txns", "n_fraud", "median_amount", "share_of_all_fraud")]
            custom = [("–" if pd.isna(a) else f"{int(a):,}", "–" if pd.isna(b) else f"{int(b):,}",
                       _fmt(m), "–" if pd.isna(s) else f"{100 * s:.1f}%") for a, b, m, s in zip(*extra)]
            fig = go.Figure(go.Bar(y=k.category, x=k.fraud_rate, orientation="h",
                                   marker=dict(color=_series(CLASS_SLOT["fraud"]), cornerradius=4),
                                   customdata=custom,
                                   hovertemplate="<b>%{y}</b><br>fraud rate %{x:.1%}<br>%{customdata[1]} fraud "
                                                 "of %{customdata[0]} txns<br>median amount %{customdata[2]}"
                                                 "<br>%{customdata[3]} of all fraud<extra></extra>"))
            fig.update_xaxes(title="Fraud rate", tickformat=".0%", range=[0, 1.02])
            fig.update_yaxes(categoryorder="array", categoryarray=order)
            ui.style(fig, height=h, legend=False, hover="closest")
            fig.update_layout(title=dict(text="Fraud rate by category", font=dict(color=c["ink"], size=14),
                                         x=0, y=0.99), margin=dict(t=40))
            ui.chart(fig, cat, key="eda_cat_rate", caption="Share of each category's transactions that are fraud. Whole dataset, all 180 days "
                                                "(`category_summary.csv`).")
        else:
            st.info("`category_summary.csv` not found.")
    with right:
        if thr_ok:
            k = thr.set_index("category").reindex(order).dropna(subset=["legit_p95_amount_train"]).reset_index()
            fig = go.Figure(go.Bar(y=k.category, x=k.legit_p95_amount_train, orientation="h",
                                   marker=dict(color=_series(CLASS_SLOT["legit"]), cornerradius=4),
                                   hovertemplate="<b>%{y}</b><br>legit 95th percentile %{x:,.2f}<extra></extra>"))
            fig.update_xaxes(title="Amount threshold (log scale)", type="log")
            fig.update_yaxes(categoryorder="array", categoryarray=order)
            ui.style(fig, height=h, legend=False, hover="closest")
            fig.update_layout(title=dict(text="Normal-looking amount threshold", font=dict(color=c["ink"],
                                                                                          size=14), x=0, y=0.99),
                              margin=dict(t=40))
            top = k.loc[k.legit_p95_amount_train.idxmax()] if not k.empty else None
            ui.chart(fig, thr, key="eda_thresholds",
                     caption="Fraud at or below its category's bar is normal-looking. Log scale"
                             + (f": {top.category}'s threshold ({top.legit_p95_amount_train:,.2f}) is the "
                                "highest." if top is not None else "."))
        else:
            st.info("`subgroup_thresholds.csv` not found.")


def _interim_history():
    ui.section("History features: all fraud vs all legit",
               "Interim view: all fraud vs all legit on train, not the normal-looking subgroup. It shows the "
               "history features carry signal, not that they separate fraud that looks normal (that is item A).")
    st.badge("interim: all fraud vs all legit, not the normal-looking subgroup", icon=":material/info:",
             color="orange")
    s = D.load("graph_feature_summary.csv")
    need = {"feature", "legit_mean", "fraud_mean"}
    if s is None:
        ui.empty_state("Graph feature summary not found", "Written by `uv run python -m src.features.graph`.",
                       files=["graph_feature_summary.csv"], owner="R2 Jayden")
        return
    if not need <= set(s.columns):
        _raw(s, "graph_feature_summary.csv", need - set(s.columns))
        return
    s = s.copy()
    s["group_code"] = [_group_of(f) for f in s.feature]
    pos = (s.fraud_mean > 0) & (s.legit_mean > 0)
    s["fraud_to_legit_ratio"] = (s.fraud_mean / s.legit_mean).where(pos)    # display only, labelled
    left, right = st.columns([6, 5], gap="large")
    with left:
        r = s[pos].sort_values("fraud_to_legit_ratio")
        c = ui.colors()
        fig = go.Figure(go.Scatter(
            x=r.fraud_to_legit_ratio, y=r.feature, mode="markers",
            marker=dict(size=10, color=_series(FEATURE_KIND_SLOT["history"]), line=dict(width=2, color=c["surface"])),
            customdata=list(zip([_fmt(v) for v in r.fraud_mean], [_fmt(v) for v in r.legit_mean],
                                [D.GROUP_LABELS.get(x, x) for x in r.group_code])),
            hovertemplate="<b>%{y}</b> · %{customdata[2]}<br>fraud mean %{customdata[0]}, "
                          "legit mean %{customdata[1]}<br>ratio %{x:.2f}×<extra></extra>"))
        fig.add_vline(x=1, line=dict(color=c["baseline"], width=1))
        fig.update_xaxes(type="log", title="Fraud mean ÷ legit mean (log scale)")
        fig.update_yaxes(categoryorder="array", categoryarray=list(r.feature))
        ui.style(fig, height=_height(len(r), per=22), legend=False, hover="closest")
        ui.chart(fig, s.drop(columns=["group_code"]).rename(
                     columns={"fraud_to_legit_ratio": "fraud_to_legit_ratio (display)"}), key="eda_interim_ratio",
                 caption="A ratio of 1 means no difference. Right of the line: fraud has the larger mean. Features with a zero or negative mean "
                         "(z-scores, log ratios) can't form a ratio and are left out; see the table.")
    with right:
        feats = list(s.feature)
        default = "g3_pair_is_new" if "g3_pair_is_new" in feats else feats[0]
        feat = st.selectbox("Feature", feats, index=feats.index(default), key="eda_interim_feature",
                            format_func=lambda f: f"{f}  ({D.GROUP_LABELS.get(_group_of(f), '')})")
        row = s[s.feature == feat].iloc[0]
        fig = go.Figure()
        for cls, col in (("legit", "legit_mean"), ("fraud", "fraud_mean")):
            fig.add_bar(x=[CLASS_LABELS[cls]], y=[row[col]], name=CLASS_LABELS[cls],
                        marker=dict(color=_series(CLASS_SLOT[cls]), cornerradius=4),
                        hovertemplate=f"{CLASS_LABELS[cls]}: mean {_fmt(row[col])}<extra></extra>")
        fig.update_yaxes(title=f"mean of {feat} (train)")
        ui.style(fig, height=300, hover="closest")
        ui.chart(fig, s[s.feature == feat].drop(columns=["group_code"]), key="eda_interim_bar")
        st.caption("Means on train, from `graph_feature_summary.csv`.")
    st.caption(ARTIFACT_NOTE)


def _model_attention(sel):
    split = sel.split if D.exists(f"lgbm_importance_groups_txn_graph_{sel.split}.csv") else "val"
    groups, feats = (D.load(f"lgbm_importance_groups_txn_graph_{split}.csv"),
                     D.load(f"lgbm_importance_txn_graph_{split}.csv"))
    ui.section("Where the model looks",
               f"From the LightGBM txn + graph model on {split}: share of mean |TreeSHAP contribution| by "
               "feature group. Always LightGBM, whatever the Model control says."
               + (" Test contributions appear after the freeze." if split != sel.split else ""))
    if groups is None and feats is None:
        ui.empty_state("Model contributions not found",
                       "Written by `uv run python -m src.models.interpret --model lgbm --features txn_graph`.",
                       files=[f"lgbm_importance_groups_txn_graph_{split}.csv",
                              f"lgbm_importance_txn_graph_{split}.csv"], owner="R3 Aayush")
        return
    left, right = st.columns([5, 6], gap="large")
    with left:
        if groups is not None and {"group", "share"} <= set(groups.columns):
            g = groups.sort_values("share")
            labels = [D.GROUP_LABELS.get(x, x) for x in g.group]
            fig = go.Figure()
            for kind in ("txn", "history"):
                m = [(_kind(x) == kind) for x in g.group]
                k = g[m]
                if k.empty:
                    continue
                fig.add_bar(y=[lab for lab, keep in zip(labels, m) if keep], x=k.share, orientation="h",
                            name=_kind_label(kind), marker=dict(color=_series(FEATURE_KIND_SLOT[kind]),
                                                               cornerradius=4),
                            customdata=k.get("mean_abs_contrib", k.share),
                            hovertemplate="<b>%{y}</b><br>%{x:.1%} of attention<br>mean |contribution| "
                                          "%{customdata:.3f}<extra></extra>")
            fig.update_yaxes(categoryorder="array", categoryarray=labels)
            fig.update_xaxes(title="share of mean |contribution|", tickformat=".0%", rangemode="tozero")
            ui.style(fig, height=280, hover="closest")
            ui.chart(fig, groups, key="eda_attn_groups",
                     caption="History groups together vs the transaction group: does the model lean on history?")
        else:
            st.info(f"`lgbm_importance_groups_txn_graph_{split}.csv` not found or has unexpected columns.")
    with right:
        if feats is not None and {"feature", "mean_abs_contrib"} <= set(feats.columns):
            n = st.select_slider("Top features", [10, 15, 20, 30], value=15, key="eda_attn_topn")
            top = feats.sort_values("mean_abs_contrib", ascending=False).head(n).iloc[::-1].copy()
            top["grp"] = [_group_of(f, g) for f, g in
                          zip(top.feature, top.group if "group" in top.columns else [None] * len(top))]
            fig = go.Figure()
            for kind in ("txn", "history"):
                k = top[top.grp.map(_kind) == kind]
                if k.empty:
                    continue
                fig.add_bar(y=k.feature, x=k.mean_abs_contrib, orientation="h", name=_kind_label(kind),
                            marker=dict(color=_series(FEATURE_KIND_SLOT[kind]), cornerradius=4),
                            customdata=list(zip([D.GROUP_LABELS.get(x, x) for x in k.grp],
                                                k.get("mean_contrib", k.mean_abs_contrib))),
                            hovertemplate="<b>%{y}</b> · %{customdata[0]}<br>mean |contribution| %{x:.3f}"
                                          "<br>mean signed %{customdata[1]:+.3f}<extra></extra>")
            fig.update_yaxes(categoryorder="array", categoryarray=list(top.feature))
            fig.update_xaxes(title="mean |contribution| (log-odds)", rangemode="tozero")
            ui.style(fig, height=_height(len(top), per=22), hover="closest")
            ui.chart(fig, feats, key="eda_attn_feats",
                     caption="Colour is transaction vs history; the history group (G1, G2, G3, G5) is in "
                             "the hover and the data table.")
        else:
            st.info(f"`lgbm_importance_txn_graph_{split}.csv` not found or has unexpected columns.")
    if feats is not None and {"feature", "mean_abs_contrib"} <= set(feats.columns) \
            and "g1_cust_n_prev" in set(feats.feature):
        is_top = feats.loc[feats.mean_abs_contrib.idxmax(), "feature"] == "g1_cust_n_prev"
        st.caption(("`g1_cust_n_prev` has the largest LightGBM contribution here. " if is_top else "")
                   + ARTIFACT_NOTE)


# --- entry point -------------------------------------------------------------------

def render(sel):
    st.caption("On amount and category alone, normal-looking fraud resembles legit. Do customer and merchant "
               "history features separate them? That is the case for adding graph features.")
    full = D.exists(COMPARISON)
    if full:
        _full_mode(sel)
    else:
        p = D.PENDING["eda"]
        ui.empty_state("Normal-looking fraud vs normal-looking legit (plan item A)",
                       "Compares normal-looking fraud with legit transactions in the same amount range, on "
                       "transaction features and on history features, with effect sizes. Until it exists, the "
                       "sections below use the tables we already have; the history comparison there is all "
                       "fraud vs all legit.", files=p["files"], owner=p["owner"])
    st.divider()
    _model_link(sel)
    st.divider()
    _what_is_normal_looking(sel)
    st.divider()
    # Streamlit can't nest expanders (every chart has a "Data table" one), so a toggle hides the
    # superseded interim view once item A exists.
    if not full or st.toggle("Show the earlier interim view (all fraud vs all legit)",
                             key="eda_show_interim"):
        _interim_history()
    st.divider()
    _model_attention(sel)
