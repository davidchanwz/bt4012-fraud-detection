"""What the models rely on: LR coefficients and LightGBM feature contributions.

    python -m src.models.interpret --model lr   --features txn_graph [--split val]
    python -m src.models.interpret --model lgbm --features txn_graph [--split val]

LR: coefficients on the standardised inputs (so their sizes are comparable), sorted by |coef|.
LightGBM: booster.predict(pred_contrib=True) gives per-row SHAP values (TreeSHAP, built into
LightGBM, no extra dependency). We report mean |contribution| per feature and per group, in
log-odds. If a case-study candidate file exists for this split, the per-row contributions for
those rows are saved too.

Defaults to --split val, so nothing here touches test before the freeze.

Outputs:
    reports/tables/lr_coefficients_{feature_set}.csv
    reports/tables/lgbm_importance_{feature_set}_{split}.csv        per feature
    reports/tables/lgbm_importance_groups_{feature_set}_{split}.csv per group (txn, g1, g2, g3, g5)
    reports/figures/lgbm_importance_{feature_set}_{split}.png
    reports/tables/case_study_contribs_{split}.csv                  if candidates exist
"""
import argparse

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIG_DIR, LABEL_DELAY, MODEL_DIR, TABLE_DIR
from src.features.graph import GROUPS
from src.models.data import feature_set_name, load_graph, load_splits

GROUP_LABELS = {"txn": "transaction", **GROUPS}


def feature_group(col):
    g = col.split("_")[0]
    return g if g in GROUPS else "txn"


def load_artifact(model, fs):
    path = MODEL_DIR / f"{model}_{fs}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python -m src.models.train --model {model} ...` first")
    return joblib.load(path)


def lr_coefficients(art):
    m = art["models"][0]
    coef = pd.DataFrame({"feature": art["columns"], "coef": m.coef_[0]})
    coef["group"] = coef.feature.map(feature_group)
    coef["abs_coef"] = coef.coef.abs()
    coef["odds_ratio_per_sd"] = np.exp(coef.coef)
    return coef.sort_values("abs_coef", ascending=False).reset_index(drop=True)


def lgbm_contributions(art, X):
    """Per-row contributions (log-odds), one column per feature plus 'bias'."""
    booster = art["models"][0].booster_
    c = booster.predict(X, pred_contrib=True)
    return pd.DataFrame(c, columns=list(X.columns) + ["bias"], index=X.index)


def importance(contrib):
    feats = contrib.drop(columns="bias")
    per_feat = feats.abs().mean().rename("mean_abs_contrib").to_frame()
    per_feat["mean_contrib"] = feats.mean()
    per_feat["group"] = [feature_group(c) for c in per_feat.index]
    per_feat = per_feat.sort_values("mean_abs_contrib", ascending=False).rename_axis("feature").reset_index()
    # group importance: |sum of the group's contributions| per row, then averaged
    grp = {g: feats[[c for c in feats if feature_group(c) == g]].sum(axis=1).abs().mean()
           for g in per_feat.group.unique()}
    per_group = (pd.Series(grp, name="mean_abs_contrib").rename_axis("group").reset_index()
                 .sort_values("mean_abs_contrib", ascending=False))
    per_group["share"] = per_group.mean_abs_contrib / per_group.mean_abs_contrib.sum()
    per_group["label"] = per_group.group.map(GROUP_LABELS)
    return per_feat, per_group


def plot_importance(per_feat, path, title, top=20):
    order = ["txn", "g1", "g2", "g3", "g5"]
    colors = dict(zip(order, ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2"]))
    d = per_feat.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(d) + 1.2))
    ax.barh(d.feature, d.mean_abs_contrib, color=[colors[g] for g in d.group])
    ax.set_xlabel("mean |contribution| (log-odds)")
    ax.set_title(title, fontsize=10, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    present = [g for g in order if g in set(d.group)]
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=colors[g]) for g in present],
              labels=[GROUP_LABELS[g] for g in present], frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(model, features, split="val", groups=None, delay=LABEL_DELAY, log=print):
    fs = feature_set_name(features, groups, delay)
    art = load_artifact(model, fs)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    if model == "lr":
        coef = lr_coefficients(art)
        coef.to_csv(TABLE_DIR / f"lr_coefficients_{fs}.csv", index=False)
        log(f"LR / {fs}: top coefficients (per sd of the standardised input)")
        log(coef.head(15)[["feature", "group", "coef", "odds_ratio_per_sd"]].round(3).to_string(index=False))
        return coef

    if model != "lgbm":
        raise ValueError("interpretation is implemented for lr and lgbm (the MLP isn't interpreted)")

    df = load_splits([split])[split]
    graph = load_graph(features, delay)
    X = art["builder"].transform(df, graph)
    contrib = lgbm_contributions(art, X)
    per_feat, per_group = importance(contrib)
    per_feat.to_csv(TABLE_DIR / f"lgbm_importance_{fs}_{split}.csv", index=False)
    per_group.to_csv(TABLE_DIR / f"lgbm_importance_groups_{fs}_{split}.csv", index=False)
    plot_importance(per_feat, FIG_DIR / f"lgbm_importance_{fs}_{split}.png",
                    f"LightGBM ({fs}) feature contributions on {split}")

    cand_path = TABLE_DIR / f"case_study_candidates_lgbm_{split}_1pct.csv"
    if cand_path.exists() and fs == "txn_graph":
        ids = pd.read_csv(cand_path).txn_id
        long = contrib.loc[ids].drop(columns="bias").stack().rename("contrib").reset_index()
        long.columns = ["txn_id", "feature", "contrib"]
        long["abs"] = long.contrib.abs()
        long = long.sort_values(["txn_id", "abs"], ascending=[True, False]).groupby("txn_id").head(8)
        long.drop(columns="abs").to_csv(TABLE_DIR / f"case_study_contribs_{split}.csv", index=False)
        log(f"  saved top-8 contributions for {len(ids)} case-study candidates")

    log(f"LightGBM / {fs} on {split}: contribution share by group")
    log(per_group[["group", "label", "mean_abs_contrib", "share"]].round(3).to_string(index=False))
    log("top features:")
    log(per_feat.head(12)[["feature", "group", "mean_abs_contrib", "mean_contrib"]].round(3).to_string(index=False))
    return per_feat, per_group


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=["lr", "lgbm"])
    p.add_argument("--features", default="txn_graph", choices=["txn", "txn_graph"])
    p.add_argument("--split", default="val", choices=["val", "test"])
    a = p.parse_args(argv)
    run(a.model, a.features, a.split)


if __name__ == "__main__":
    main()
