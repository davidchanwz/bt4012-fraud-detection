"""Tab 6: Ablation and label delay. Which graph groups carry the gain, and does the 7-day label delay matter?

Full mode reads R2's evaluated variants (any results family with non-standard feature sets, e.g.
results_ablation_{split}.csv from `metrics.py ... --name ablation`) through app.data.variant_results.
Until those exist the tab shows what is owed, plus the val tuning PR-AUC from hyperparams.csv,
clearly marked as tuning numbers without confidence intervals.

Read-only: every number is a CSV value; the only arithmetic is the error-bar lengths that Plotly
needs (ci_high - value, value - ci_low) to draw the CI that is in the row, and x100 to show paired
differences in recall as percentage points.
"""
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import data as D, ui

BUILT_GROUPS = ("g1", "g2", "g3", "g5")          # G4 (network structure) is a stretch and not built
ABLATION_MODELS = ("lr", "lgbm")                 # plan: the ablation uses LR and LightGBM only
METRICS = {"pr_auc": "PR-AUC", "recall_at_1pct": "Recall @ 1%", "recall_at_2pct": "Recall @ 2%"}
NAME_RE = re.compile(r"txn_graph(?:_((?:g\d)+))?(?:_delay(\d+))?")


# --- naming --------------------------------------------------------------------

def parse(fs):
    """'txn_graph_g1g3_delay1' -> (['g1', 'g3'], 1). Groups None = all built groups; delay None = 7 days."""
    m = NAME_RE.fullmatch(fs)
    if not m:
        return None
    groups = re.findall(r"g\d", m[1]) if m[1] else None
    return groups, (int(m[2]) if m[2] else None)


def variant_label(fs):
    """Human label for a feature-set name (see src/models/data.py feature_set_name)."""
    if fs == "txn":
        return "txn only"
    p = parse(fs)
    if p is None:
        return fs
    groups, delay = p
    gs = set(groups or BUILT_GROUPS)
    built = set(BUILT_GROUPS)
    if gs >= built:
        name = "all groups"
    elif gs - {"g4"} == {"g1", "g2", "g3"}:
        name = "label-free (G1–G4)" if "g4" in gs else "label-free (G1–G3)"
    elif gs == {"g5"}:
        name = "labels only (G5)"
    elif gs <= built and len(built - gs) == 1:
        g = (built - gs).pop()
        name = f"minus {g.upper()} ({D.GROUP_LABELS[g]})"
    else:
        name = "+".join(g.upper() for g in sorted(gs)) + " only"
    if delay is not None:
        name += f" + {delay}-day delay"
    return name


def seven_day(fs):
    """The 7-day counterpart of a delay variant: txn_graph_g5_delay1 -> txn_graph_g5."""
    return re.sub(r"_delay\d+$", "", fs)


def is_delay(fs):
    p = parse(fs)
    return p is not None and p[1] is not None


# --- formatting ----------------------------------------------------------------

def is_rate(base):
    return base.startswith("recall")


def d3(base):
    return ".1%" if is_rate(base) else ".3f"


RESULT_COLS = ["model", "feature_set", "split", "metric", "value", "ci_low", "ci_high", "subset",
               "n_rows", "n_fraud", "family"]


def metric_title(base, subset):
    t = METRICS[base]
    return t if subset == "all" else f"{t}, {D.SUBSET_LABELS[subset].lower()}"


def err(df, scale=1):
    """Asymmetric error bars for the CI in each row (scale=100 when the values are shown in pp)."""
    return dict(type="data", symmetric=False, array=(scale * (df.ci_high - df.value)).tolist(),
                arrayminus=(scale * (df.value - df.ci_low)).tolist(), color=ui.colors()["ink2"], thickness=1.5,
                width=4)


def variant_color():
    return ui.colors()["muted"]          # one neutral colour for every variant; identity is on the axis


# --- full mode -----------------------------------------------------------------

def pick_model(sel, variants):
    have = [m for m in D.MODELS if m in set(variants.model)]
    if sel.model in have:
        return sel.model
    st.info(f"No ablation runs for {D.MODEL_LABELS[sel.model]}: the plan runs the ablation with "
            f"{' and '.join(D.MODEL_LABELS[m] for m in ABLATION_MODELS)} only. Showing a model that has runs.")
    return st.segmented_control("Ablation model", have, default=have[0], required=True,
                                format_func=D.MODEL_LABELS.get, key="ablation_model")


def rows_for(df, model, name, feature_sets):
    if df is None:
        return pd.DataFrame(columns=RESULT_COLS)      # keeps the columns callers read
    r = df[(df.model == model) & (df.metric == name) & df.feature_set.isin(feature_sets)]
    return r.drop_duplicates("feature_set")


def ceiling_note(sel, base):
    """Best possible recall@1% next to any recall@1% number on the whole split."""
    if base != "recall_at_1pct" or sel.subset != "all":
        return None
    return D.ceiling(sel.split, "1pct")


def ablation_chart(sel, model, base, variants, main):
    name = D.metric_name(base, sel.subset)
    var = variants[(variants.model == model) & (variants.metric == name)
                   & ~variants.feature_set.map(is_delay)].drop_duplicates("feature_set")
    if var.empty:
        st.info(f"No ablation rows for {D.MODEL_LABELS[model]} on {metric_title(base, sel.subset)}. "
                "Try another metric or fraud subset.")
        return
    # txn_graph is the ablation base, so it is always shown; txn follows the sidebar.
    want = [fs for fs in D.FEATURE_SETS if fs in sel.feature_sets or fs == "txn_graph"]
    refs = rows_for(main, model, name, want)
    table = pd.concat([refs.assign(kind=refs.feature_set), var.assign(kind="variant")], ignore_index=True)
    table["label"] = [("txn only (reference)" if fs == "txn" else
                       "all groups (reference)" if fs == "txn_graph" else variant_label(fs))
                      for fs in table.feature_set]
    table = table.sort_values("value", kind="stable").reset_index(drop=True)

    fig = go.Figure()
    series = [("txn", "txn", ui.fs_color("txn")), ("txn_graph", "txn + graph, all groups", ui.fs_color("txn_graph")),
              ("variant", "ablation variant", variant_color())]
    for kind, legend, color in series:
        t = table[table.kind == kind]
        if t.empty:
            continue
        fig.add_trace(go.Bar(
            y=t.label, x=t.value, orientation="h", name=legend, marker=dict(color=color, cornerradius=4),
            error_x=err(t), customdata=t[["ci_low", "ci_high", "feature_set"]].values,
            hovertemplate=f"%{{y}}<br>{metric_title(base, sel.subset)} %{{x:{d3(base)}}} "
                          f"[%{{customdata[0]:{d3(base)}}}, %{{customdata[1]:{d3(base)}}}]"
                          "<br><span style='font-size:11px'>%{customdata[2]}</span><extra></extra>"))
    c = ceiling_note(sel, base)
    if c is not None:
        fig.add_vline(x=c, line=dict(color=ui.colors()["ink2"], width=1.5, dash="dash"))
        fig.add_annotation(x=c, y=1, yref="paper", yanchor="bottom", showarrow=False,
                           text=f"best possible {ui.pct(c, 2)}", font=dict(color=ui.colors()["ink2"], size=12))
    ui.style(fig, height=90 + 36 * len(table), hover="closest")
    fig.update_yaxes(categoryorder="array", categoryarray=table.label.tolist(), title=None)
    fig.update_xaxes(title=metric_title(base, sel.subset), range=[0, 1.02], tickformat=d3(base))
    fig.update_layout(barmode="overlay")
    cap = ("Longer bar = better. A variant well below 'all groups' means the dropped groups carried "
           "the gain. Error bars are bootstrap 95% CIs.")
    if c is not None:
        cap += f" Dashed line: best possible recall at a 1% budget ({ui.pct(c, 2)})."
    if "txn_graph" not in sel.feature_sets:
        cap += " The 'all groups' reference is shown even though the sidebar hides txn + graph, " \
               "because every variant is measured against it."
    if len(refs) < len(want):
        cap += " Reference row missing from the main results for this selection."
    ui.chart(fig, table[["label", "feature_set", "family", "metric", "value", "ci_low", "ci_high",
                         "n_rows", "n_fraud"]].rename(columns={"label": "variant"}),
             key="ablation_bars", caption=cap)


def paired_chart(sel, model, base, paired, delay):
    """Paired bootstrap difference vs the base in R4's paired file (dot + CI, zero line)."""
    if paired is None:
        return False
    name = D.metric_name(base, sel.subset)
    p = paired[(paired.model == model) & (paired.metric == name)
               & (paired.feature_set_new.map(is_delay) == delay)].drop_duplicates(
        ["feature_set_base", "feature_set_new"])
    if p.empty:
        return False
    p = p.assign(label=[f"{variant_label(n)} vs {variant_label(b)}"
                        for b, n in zip(p.feature_set_base, p.feature_set_new)]).sort_values("diff")
    p = p.rename(columns={"diff": "value"})
    # Recall differences are shown in percentage points (x100, display only); PR-AUC as is.
    scale, unit, f = (100, " pp", "+.1f") if is_rate(base) else (1, "", "+.3f")
    fig = go.Figure(go.Scatter(
        x=scale * p.value, y=p.label, mode="markers", name="paired difference",
        marker=dict(size=10, color=variant_color(), line=dict(width=2, color=ui.colors()["surface"])),
        error_x=err(p, scale),
        customdata=pd.DataFrame({"lo": scale * p.ci_low, "hi": scale * p.ci_high, "b": p.feature_set_base,
                                 "n": p.feature_set_new}).values,
        hovertemplate=f"%{{y}}<br>difference %{{x:{f}}}{unit} "
                      f"[%{{customdata[0]:{f}}}, %{{customdata[1]:{f}}}]{unit}"
                      "<br>%{customdata[3]} minus %{customdata[2]}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=ui.colors()["baseline"], width=1.5))
    ui.style(fig, height=80 + 40 * len(p), legend=False, hover="closest")
    fig.update_xaxes(title=f"Difference in {metric_title(base, sel.subset)} (variant minus base"
                           + (", percentage points)" if unit else ")"),
                     tickformat=f, ticksuffix=unit)
    fig.update_yaxes(title=None)
    ui.chart(fig, p.rename(columns={"value": "diff"})[
        ["label", "feature_set_base", "feature_set_new", "metric", "diff", "ci_low", "ci_high"]],
        key="ablation_delay_paired" if delay else "ablation_paired",
        caption="Paired bootstrap on the same rows. Left of zero: the variant does worse than its base; "
                "a CI that crosses zero means no clear difference.")
    return True


def delay_section(sel, model, base, variants, main, paired):
    ui.section("Label delay: 7-day vs 1-day",
               "G5 past-label features only use labels at least 7 days old (main setting). The 1-day run "
               "is the sensitivity check: does fresher label feedback help?")
    dvars = None if variants is None else variants[variants.feature_set.map(is_delay)]
    if dvars is None or dvars.empty:
        ui.empty_state("Label delay not evaluated yet",
                       "Needs the 1-day label-delay models (`train.py --delay 1`) scored and evaluated.",
                       files=D.pending_files("delay", sel.split), owner="R2 Jayden")
        return
    if model not in set(dvars.model):
        st.info(f"No label-delay runs for {D.MODEL_LABELS[model]}. Models with runs: "
                + ", ".join(D.MODEL_LABELS[m] for m in D.MODELS if m in set(dvars.model)) + ".")
        return
    name = D.metric_name(base, sel.subset)
    d1 = dvars[(dvars.model == model) & (dvars.metric == name)].drop_duplicates("feature_set")
    if d1.empty:
        st.info(f"No label-delay rows for {metric_title(base, sel.subset)}.")
        return
    pool = pd.concat([x for x in (main, variants) if x is not None], ignore_index=True)
    d7 = rows_for(pool, model, name, [seven_day(fs) for fs in d1.feature_set])
    d1 = d1.assign(setting="1-day delay", group=[variant_label(seven_day(fs)) for fs in d1.feature_set])
    d7 = d7.assign(setting="7-day delay (main)", group=[variant_label(fs) for fs in d7.feature_set])
    order = d1.group.tolist()
    fig = go.Figure()
    # Neither delay setting is a feature-set entity, so both are neutral; the 1-day bar is hatched.
    muted = variant_color()
    for setting, t, pattern in (("7-day delay (main)", d7, dict(shape="")),
                                ("1-day delay", d1, dict(shape="/", fgcolor=muted, bgcolor=ui.colors()["surface"],
                                                         solidity=0.35))):
        if t.empty:
            continue
        fig.add_trace(go.Bar(
            x=t.group, y=t.value, name=setting,
            marker=dict(color=muted, cornerradius=4, pattern=pattern, line=dict(color=muted, width=1.5)),
            error_y=dict(type="data", symmetric=False, array=(t.ci_high - t.value).tolist(),
                         arrayminus=(t.value - t.ci_low).tolist(), color=ui.colors()["ink2"], thickness=1.5,
                         width=4),
            customdata=t[["ci_low", "ci_high", "feature_set"]].values,
            hovertemplate=f"{setting}, %{{x}}<br>{metric_title(base, sel.subset)} %{{y:{d3(base)}}} "
                          f"[%{{customdata[0]:{d3(base)}}}, %{{customdata[1]:{d3(base)}}}]"
                          "<br><span style='font-size:11px'>%{customdata[2]}</span><extra></extra>"))
    c = ceiling_note(sel, base)
    if c is not None:
        fig.add_hline(y=c, line=dict(color=ui.colors()["ink2"], width=1.5, dash="dash"))
    ui.style(fig, height=340, hover="closest")
    fig.update_layout(barmode="group")
    fig.update_xaxes(categoryorder="array", categoryarray=order, title=None)
    fig.update_yaxes(title=metric_title(base, sel.subset), range=[0, 1.02], tickformat=d3(base))
    cap = "Same model and graph groups, only the label delay changes. Similar bars: the 7-day delay costs little."
    if c is not None:
        cap += f" Dashed line: best possible recall at a 1% budget ({ui.pct(c, 2)})."
    ui.chart(fig, pd.concat([d7, d1], ignore_index=True)[
        ["setting", "group", "feature_set", "metric", "value", "ci_low", "ci_high", "n_fraud"]],
        key="ablation_delay", caption=cap)
    paired_chart(sel, model, base, paired, delay=True)


def full_mode(sel, variants):
    paired = D.variant_results("paired", sel.split)
    main = D.results("", sel.split)
    model = pick_model(sel, variants)
    base = st.segmented_control("Metric", list(METRICS), default="pr_auc", required=True,
                                format_func=METRICS.get, key="ablation_metric",
                                help="Fraud subset comes from the sidebar.")
    abl = variants[~variants.feature_set.map(is_delay)]
    ui.section(f"Leave groups out · {D.MODEL_LABELS[model]}",
               "Each variant drops graph groups and is retrained with the same grid. "
               f"Groups: G1 {D.GROUP_LABELS['g1']}, G2 {D.GROUP_LABELS['g2']}, G3 {D.GROUP_LABELS['g3']}, "
               f"G5 {D.GROUP_LABELS['g5']} (G4 not built).")
    if abl.empty:
        ui.empty_state("Ablation not evaluated yet", "Label-delay rows exist but no leave-one-group-out rows.",
                       files=D.pending_files("ablation", sel.split), owner="R2 Jayden")
    else:
        ablation_chart(sel, model, base, abl, main)
        if not paired_chart(sel, model, base, paired, delay=False):
            st.info("No paired differences for these variants yet (R4's paired file, "
                    "`metrics.py --pairs txn_graph:<variant>`).")
    st.divider()
    delay_section(sel, model, base, variants, main, paired)


# --- interim mode --------------------------------------------------------------

def tuning_panel(model, hp, sel):
    t = hp[hp.model == model].copy()
    keep = [fs not in D.FEATURE_SETS or fs in sel.feature_sets or fs == "txn_graph" for fs in t.feature_set]
    t = t[keep]
    if t.empty:
        st.info(f"No tuning rows for {D.MODEL_LABELS[model]}.")
        return None
    t["variant"] = [("txn only (reference)" if fs == "txn" else
                     "all groups (reference)" if fs == "txn_graph" else variant_label(fs))
                    for fs in t.feature_set]
    t["kind"] = [fs if fs in D.FEATURE_SETS else "variant" for fs in t.feature_set]
    t = t.sort_values("val_pr_auc", kind="stable")
    fig = go.Figure()
    for kind, legend, color in (("txn", "txn", ui.fs_color("txn")),
                                ("txn_graph", "txn + graph, all groups", ui.fs_color("txn_graph")),
                                ("variant", "ablation variant", variant_color())):
        s = t[t.kind == kind]
        if s.empty:
            continue
        fig.add_trace(go.Bar(
            y=s.variant, x=s.val_pr_auc, orientation="h", name=legend, marker=dict(color=color, cornerradius=4),
            customdata=s[["feature_set", "n_features", "params"]].values,
            hovertemplate="%{y}<br>tuning val PR-AUC %{x:.3f}<br>%{customdata[1]} features"
                          "<br><span style='font-size:11px'>%{customdata[0]} · %{customdata[2]}</span>"
                          "<extra></extra>"))
    ui.style(fig, height=110 + 38 * len(t), hover="closest")
    fig.update_layout(barmode="overlay", title=dict(text=D.MODEL_LABELS[model], x=0, font=dict(
        color=ui.colors()["ink"], size=14)), margin=dict(t=64))
    fig.update_yaxes(categoryorder="array", categoryarray=t.variant.tolist(), title=None)
    fig.update_xaxes(title="Tuning val PR-AUC", range=[0, 1.02], tickformat=".2f")
    return fig, t


def interim_mode(sel):
    c1, c2 = st.columns(2)
    with c1:
        ui.empty_state("Ablation (leave one group out, label-free vs labels-only)",
                       "R2's ablation runs with LR and LightGBM have not been evaluated yet. "
                       "This view fills in with bootstrap CIs and paired differences once they are.",
                       files=D.pending_files("ablation", sel.split), owner="R2 Jayden")
    with c2:
        delay_file = D.CLEAN_FILE.parent / "features_graph_delay1.parquet"
        hp = D.load("hyperparams.csv")
        trained = hp is not None and hp.feature_set.map(is_delay).any()
        body = ("The 1-day label-delay feature file `features_graph_delay1.parquet` "
                + ("exists" if delay_file.exists() else "is built by `src.features.graph` (not on this machine)")
                + (", and 1-day models have tuning rows, but they are not evaluated yet."
                   if trained else ", but no 1-day models have been trained yet."))
        ui.empty_state("Label delay, 7-day vs 1-day", body, files=D.pending_files("delay", sel.split),
                       owner="R2 Jayden")

    ui.section("Early tuning numbers (no CIs)",
               "Tuning val PR-AUC from `hyperparams.csv`: the best grid point per model and feature set. "
               "Not the evaluated ablation; no confidence intervals, so small gaps mean nothing yet.")
    if sel.split != "val":
        st.info("Tuning numbers exist for the val split only (models are tuned on val). Switch the split to val.")
        return
    if hp is None:
        ui.empty_state("No tuning table", "Written by `src.models.train`.", files=["hyperparams.csv"],
                       owner="R3 Aayush")
        return
    if sel.model not in ABLATION_MODELS:
        st.caption(f"The ablation uses {' and '.join(D.MODEL_LABELS[m] for m in ABLATION_MODELS)} only, "
                   f"so {D.MODEL_LABELS[sel.model]} is not shown here.")
    models = sorted(ABLATION_MODELS, key=lambda m: m != sel.model)
    cols = st.columns(len(models))
    partial = set()
    for col, model in zip(cols, models):
        with col:
            out = tuning_panel(model, hp, sel)
            if out is None:
                continue
            fig, t = out
            partial |= set(t.feature_set) - set(D.FEATURE_SETS)
            ui.chart(fig, t[["model", "variant", "feature_set", "val_pr_auc", "n_features", "label_delay",
                             "params", "graph_code_hash", "trained_at"]].sort_values("val_pr_auc", ascending=False),
                     key=f"ablation_tune_{model}")
    note = " The 'all groups' bar stays even when the sidebar hides txn + graph: it is the ablation base." \
        if "txn_graph" not in sel.feature_sets else ""
    st.caption("Longer bar = better on val. Partial graph sets trained so far: "
               + (", ".join(f"{variant_label(fs)} (`{fs}`)" for fs in sorted(partial)) if partial else "none")
               + "." + note)
    grids = sorted(D.TABLE_DIR.glob("tuning/*.csv"))
    grids = [g for g in grids if g.stem.split("_", 1)[0] in ABLATION_MODELS]
    if grids:
        with st.expander("Full tuning grids (val PR-AUC per grid point)", expanded=False):
            choice = st.selectbox("Grid", [g.stem for g in grids], key="ablation_grid",
                                  format_func=lambda s: f"{D.MODEL_SHORT[s.split('_', 1)[0]]} · "
                                                        f"{variant_label(s.split('_', 1)[1])} ({s.split('_', 1)[1]})")
            g = D.load(f"tuning/{choice}.csv")
            st.dataframe(g, hide_index=True, width="stretch",
                         column_config={"val_pr_auc": st.column_config.NumberColumn("val PR-AUC", format="%.4f")})


def render(sel):
    st.markdown("### Which graph groups carry the gain, and does the label delay matter?")
    st.caption("Leave-one-group-out and label-free vs labels-only runs (LR and LightGBM), plus the 7-day vs "
               "1-day label delay for the past-label features (G5).")
    variants = D.variant_results("", sel.split)
    if variants is None:
        interim_mode(sel)
    else:
        full_mode(sel, variants)
