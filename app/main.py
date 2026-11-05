"""BT4012 Group 07 dashboard: do graph features help catch fraud that looks normal?

    uv run streamlit run app/main.py

Read-only on reports/tables/ (plus data/processed/ for the case viewer's history panel).
It never retrains or recomputes a metric; see app/data.py. Sidebar controls apply to every tab.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:                 # `streamlit run app/main.py` puts app/ on the path, not the repo
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app import data as D, ui  # noqa: E402
from app.tabs import ablation, budget, cases, eda, fp_profile, scoreboard  # noqa: E402

TABS = [
    ("Review budget", budget),
    ("Scoreboard", scoreboard),
    ("Why graph features?", eda),
    ("Case viewer", cases),
    ("False positives", fp_profile),
    ("Ablation & label delay", ablation),
]


def sidebar():
    with st.sidebar:
        st.markdown("### Controls")
        st.caption("Apply to every tab.")
        splits = D.available_splits() or ["val"]
        split = st.segmented_control(
            "Split", splits, default="test" if "test" in splits else splits[0], required=True, key="split",
            help="Validation (days 120–149) now. Test (days 150–179) appears after the results freeze on Sun 15 Nov.")
        if "test" not in splits:
            st.caption("Test split appears after the freeze (Sun 15 Nov).")
        models = D.available_models(split) or list(D.MODELS[:3])
        model = st.selectbox("Model", models, index=models.index("lgbm") if "lgbm" in models else 0,
                             format_func=D.MODEL_LABELS.get, key="model",
                             help="Random forest is the robustness check (appendix), not part of the main table.")
        fs = st.segmented_control(
            "Features", ["both", "txn", "txn_graph"], default="both", required=True, key="features",
            format_func={"both": "Compare", "txn": "txn", "txn_graph": "txn + graph"}.get,
            help="txn: 29 transaction-level features. txn + graph: plus 23 customer/merchant history features "
                 "(G1–G3, G5).")
        subset = st.selectbox("Fraud subset", D.SUBSETS, format_func=D.SUBSET_LABELS.get, key="subset",
                              help=D.SUBSET_HELP["normal_looking"])
        st.caption(D.SUBSET_HELP[subset])

        st.divider()
        with st.expander("Data sources", icon=":material/database:"):
            for row in D.status(split):
                icon = ":material/check_circle:" if row["present"] else ":material/schedule:"
                st.markdown(f"{icon} **{row['title']}**  \n"
                            f"<small>{'present' if row['present'] else 'waiting on ' + row['owner']}</small>",
                            unsafe_allow_html=True)
        st.caption("Read-only view of `reports/tables/`. Every number comes from a pipeline CSV.")
    feature_sets = ("txn", "txn_graph") if fs == "both" else (fs,)
    return D.Selection(split=split, model=model, feature_sets=feature_sets, subset=subset)


def main():
    st.set_page_config(page_title="Graph features for fraud detection · BT4012 G07", page_icon="🔎",
                       layout="wide", initial_sidebar_state="expanded")
    ui.css()
    sel = sidebar()
    st.markdown("## Do graph features help catch fraud that looks normal?")
    st.caption("BankSim, 594,643 simulated payments. Same classifiers trained with transaction features only "
               "(txn) and with customer–merchant history added (txn + graph), compared on a chronological "
               f"hold-out. Showing the **{sel.split}** split.")
    if not D.available_splits():
        ui.empty_state("No results yet", "Run the pipeline first: `uv run python -m src.models.run_main`, then "
                       "`uv run python -m src.eval.metrics`.", files=["results_main_val.csv"])
        return
    for tab, (_, module) in zip(st.tabs([t for t, _ in TABS]), TABS):
        with tab:
            module.render(sel)


main()
