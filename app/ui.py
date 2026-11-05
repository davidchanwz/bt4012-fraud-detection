"""Small shared building blocks so every tab looks the same.

Built on Streamlit's own widgets (metric, container, badge, expander) and Plotly; this file
only adds the palette, one Plotly style and three helpers. Colours follow the entity across
the whole app: txn is always blue, txn + graph always orange, in light and dark mode.
Palette from the dataviz skill's reference instance, checked with its validator (4 Nov 2026).
"""
import plotly.graph_objects as go
import streamlit as st

from app.data import FEATURE_LABELS

PALETTE = {
    "light": {"series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
              "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "baseline": "#c3c2b7", "surface": "#fcfcfb"},
    "dark": {"series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
             "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
             "baseline": "#383835", "surface": "#1a1a19"},
}
FS_SLOT = {"txn": 0, "txn_graph": 1}
MODEL_SLOT = {"lr": 2, "mlp": 3, "lgbm": 6, "rf": 4}     # only for charts that colour by model


def mode():
    t = getattr(st.context, "theme", None) or {}       # dict-like; empty outside a browser session
    return "dark" if t.get("type") == "dark" else "light"


def colors():
    return PALETTE[mode()]


def fs_color(feature_set):
    return colors()["series"][FS_SLOT[feature_set]]


def model_color(model):
    return colors()["series"][MODEL_SLOT[model]]


def style(fig, height=360, legend=True, hover="x unified"):
    """One look for every chart: hairline grid, recessive axes, legend top-left, animated updates."""
    c = colors()
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=36 if legend else 12, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=c["ink2"], size=13),
        hovermode=hover, hoverlabel=dict(font_size=13),
        showlegend=legend,
        legend=dict(orientation="h", x=0, xanchor="left", y=1.02, yanchor="bottom", title=None,
                    font=dict(color=c["ink"])),
        transition=dict(duration=350, easing="cubic-in-out"),
        bargap=0.25, bargroupgap=0.08,
    )
    axis = dict(gridcolor=c["grid"], gridwidth=1, zeroline=False, linecolor=c["baseline"], automargin=True,
                tickfont=dict(color=c["muted"]), title_font=dict(color=c["ink2"]))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def chart(fig, table=None, key=None, caption=None):
    """A Plotly chart with its data table one click away (every chart needs a table view)."""
    st.plotly_chart(fig, key=key, config={"displaylogo": False, "displayModeBar": False}, theme="streamlit")
    if caption:
        st.caption(caption)
    if table is not None:
        with st.expander("Data table", expanded=False):
            st.dataframe(table, hide_index=True, width="stretch")


def stat(label, value, delta=None, help=None, delta_color="normal"):
    """Headline-number tile (st.metric with a border)."""
    st.metric(label, value, delta=delta, help=help, border=True, delta_color=delta_color)


def empty_state(title, body, files=None, owner=None):
    """Card shown where a source isn't available yet. Lists the files that would fill it in."""
    with st.container(border=True):
        st.markdown(f"**{title}**")
        if owner:
            st.badge(f"waiting on {owner}", icon=":material/schedule:", color="gray")
        st.caption(body)
        if files:
            st.markdown("Fills in automatically when these appear in `reports/tables/`:\n"
                        + "\n".join(f"- `{f}`" for f in files))


def section(title, caption=None):
    st.markdown(f"#### {title}")
    if caption:
        st.caption(caption)


def pct(x, digits=1):
    return "–" if x is None else f"{100 * x:.{digits}f}%"


def num(x, digits=3):
    return "–" if x is None else f"{x:.{digits}f}"


def ci(row, fmt=num):
    """'0.951 [0.94, 0.96]' from a metric() dict."""
    if row is None:
        return "–"
    return f"{fmt(row['value'])} [{fmt(row['ci_low'])}, {fmt(row['ci_high'])}]"


def fs_label(fs):
    return FEATURE_LABELS.get(fs, fs.replace("txn_graph_", "txn + graph: "))


def css():
    """A little spacing polish on top of the theme. Nothing structural."""
    st.html("""<style>
      .block-container {padding-top: 2.2rem; max-width: 1280px;}
      [data-testid="stMetricLabel"] p {font-size: 0.85rem; color: var(--st-secondary-text-color, inherit);}
      [data-testid="stMetricValue"] {font-weight: 600;}
      button[data-baseweb="tab"] p {font-size: 0.95rem;}
    </style>""")


def blank(height=360):
    """An empty figure styled like the others (for layout while a source is missing)."""
    return style(go.Figure(), height=height, legend=False)
