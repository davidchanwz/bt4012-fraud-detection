"""R2's ablation set-up: the variants are the ones the plan and the dashboard expect."""
from app.tabs.ablation import variant_label
from src.models.data import feature_set_name
from src.models.run_ablation import ABLATION_GROUPS, DELAY, LABEL_FREE, LABELS_ONLY, ablation_sets


def test_leave_one_out_covers_every_built_group():
    sets = ablation_sets()
    assert sets == ["txn_graph_g2g3g5", "txn_graph_g1g3g5", "txn_graph_g1g2g5",
                    "txn_graph_g1g2g3", "txn_graph_g5"]
    assert len(set(sets)) == len(sets)


def test_label_free_and_labels_only_are_in_the_runs():
    assert LABEL_FREE in ABLATION_GROUPS and LABELS_ONLY in ABLATION_GROUPS


def test_dashboard_labels_match_the_variants():
    labels = [variant_label(fs) for fs in ablation_sets()]
    assert labels == ["minus G1 (customer history)", "minus G2 (merchant history)",
                      "minus G3 (customer-merchant edge)", "label-free (G1–G3)", "labels only (G5)"]
    assert variant_label(feature_set_name("txn_graph", delay=DELAY)) == "all groups + 1-day delay"
