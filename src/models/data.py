"""Feature matrices for the models: `txn` and `txn_graph` (optionally a subset of groups).

Every transformer is fitted on TRAIN rows only and reused for val/test:
    txn        TransactionEncoder (log amount, zero-amount flag, one-hot category/age/gender)
    txn_graph  txn + graph columns from features_graph.parquet (g1_/g2_/g3_/g5_)
LR and MLP get the graph columns through GraphScaler (log1p counts, standardise).
LightGBM gets them raw, since trees don't need scaling.

Feature-set names are used in score filenames and hyperparams.csv:
    txn, txn_graph, txn_graph_g1g3 (--groups g1,g3), txn_graph_delay1 (--delay 1), ...
"""
import hashlib

import pandas as pd

from src.config import GRAPH_FILE, LABEL_DELAY, ROOT, SPLIT_DIR
from src.features.encode import TransactionEncoder
from src.features.graph import GROUPS, GraphScaler, group_columns, load_graph_features

FEATURE_SETS = ["txn", "txn_graph"]
SPLITS = ["train", "val", "test"]
DELAYS = {LABEL_DELAY: GRAPH_FILE, 1: GRAPH_FILE.with_name("features_graph_delay1.parquet")}


def parse_groups(groups):
    """'g3,g1' or ['g3', 'g1'] -> ['g1', 'g3'] (sorted, checked). None -> None."""
    if groups is None:
        return None
    if isinstance(groups, str):
        groups = [g.strip() for g in groups.split(",") if g.strip()]
    unknown = sorted(set(groups) - set(GROUPS))
    if unknown or not groups:
        raise ValueError(f"unknown graph groups {unknown}; available: {sorted(GROUPS)}")
    return sorted(set(groups))


def feature_set_name(features, groups=None, delay=LABEL_DELAY):
    groups = parse_groups(groups)
    if features not in FEATURE_SETS:
        raise ValueError(f"features must be one of {FEATURE_SETS}, got {features!r}")
    if features == "txn":
        if groups is not None or delay != LABEL_DELAY:
            raise ValueError("--groups and --delay only apply to txn_graph")
        return "txn"
    if delay not in DELAYS:
        raise ValueError(f"no graph feature file for label delay {delay}; available: {sorted(DELAYS)}")
    name = "txn_graph"
    if groups is not None and groups != sorted(GROUPS):
        name += "_" + "".join(groups)
    if delay != LABEL_DELAY:
        name += f"_delay{delay}"
    return name


def graph_file(delay=LABEL_DELAY):
    return DELAYS[delay]


def graph_code_hash():
    """Short hash of src/features/graph.py, so a score file can be traced to the feature code."""
    path = ROOT / "src" / "features" / "graph.py"
    return hashlib.sha1(path.read_bytes()).hexdigest()[:10]


def load_splits(names=SPLITS):
    return {n: pd.read_parquet(SPLIT_DIR / f"{n}.parquet") for n in names}


class FeatureBuilder:
    """Fit on the train split, then transform any split into a model matrix indexed by txn_id."""

    def __init__(self, features, groups=None, scale_graph=True):
        self.features = features
        self.groups = parse_groups(groups)
        self.scale_graph = scale_graph
        self.graph_scaler = None

    def fit(self, train, graph=None):
        self.encoder = TransactionEncoder().fit(train)
        if self.features == "txn_graph":
            if graph is None:
                raise ValueError("txn_graph needs the graph feature table")
            self.graph_columns = group_columns(graph.columns, self.groups or list(GROUPS))
            if self.scale_graph:
                self.graph_scaler = GraphScaler().fit(graph.loc[train.txn_id, self.graph_columns])
        return self

    def transform(self, df, graph=None):
        X = self.encoder.transform(df)
        X.index = pd.Index(df["txn_id"].to_numpy(), name="txn_id")
        if self.features == "txn_graph":
            G = graph.loc[df.txn_id, self.graph_columns]
            if self.graph_scaler is not None:
                G = self.graph_scaler.transform(G)
            X = pd.concat([X, G.set_axis(X.index)], axis=1)
        assert not X.isna().any().any(), "NaNs in feature matrix"
        return X

    @property
    def columns(self):
        cols = self.encoder.feature_names()
        return cols + self.graph_columns if self.features == "txn_graph" else cols


def load_graph(features, delay=LABEL_DELAY):
    return load_graph_features(graph_file(delay)) if features == "txn_graph" else None
