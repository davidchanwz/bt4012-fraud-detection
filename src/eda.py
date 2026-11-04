"""Week-1 EDA: saves figures to reports/figures/ and tables to reports/tables/.

Usage: python -m src.eda
(run src.data.clean and src.data.split first)

The normal-looking fraud vs normal-looking legit comparison (figure 07) also needs the
graph features. Without them it's skipped; after `python -m src.features.graph`, run just
that part with: python -m src.eda --normal-looking
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CLEAN_FILE, FIG_DIR, GRAPH_FILE, SPLIT_DIR, TABLE_DIR, TRAIN_END, VAL_END
from src.data.split import assign_split
from src.features.subgroup import fit_thresholds, flag_normal_looking

LEGIT_C = "#2a78d6"
FRAUD_C = "#eb6834"
GREY = "#8a8985"

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 10,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def split_lines(ax):
    for x in [TRAIN_END + 0.5, VAL_END + 0.5]:
        ax.axvline(x, color=GREY, ls="--", lw=1)
    ymax = ax.get_ylim()[1]
    for x, lab in [(TRAIN_END / 2, "train"), ((TRAIN_END + VAL_END) / 2 + 0.5, "val"),
                   ((VAL_END + 179) / 2 + 0.5, "test")]:
        ax.text(x, ymax * 0.97, lab, ha="center", va="top", color="#52514e")


def fig_over_time(df):
    daily = df.groupby("step").agg(n=("fraud", "size"), n_fraud=("fraud", "sum"))
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    axes[0].plot(daily.index, daily.n, color=LEGIT_C, lw=1.5)
    axes[0].set_ylabel("transactions / day")
    axes[0].set_ylim(0, daily.n.max() * 1.15)
    split_lines(axes[0])
    axes[0].set_title("Daily transaction volume")

    axes[1].plot(daily.index, daily.n_fraud / daily.n * 100, color=FRAUD_C, lw=1.5)
    axes[1].set_ylabel("fraud rate (%)")
    axes[1].set_xlabel("step (day)")
    axes[1].set_ylim(0, (daily.n_fraud / daily.n * 100).max() * 1.2)
    split_lines(axes[1])
    axes[1].set_title(f"Daily fraud rate (fraud count is exactly {int(daily.n_fraud.iloc[0])}/day "
                      f"for all {daily.n_fraud.eq(daily.n_fraud.iloc[0]).sum()} days)")
    save(fig, "01_volume_and_fraud_over_time.png")
    return daily


def fig_amount(df):
    fig, ax = plt.subplots(figsize=(8, 4))
    # log1p(amount) with plain bins -- using density on log-spaced bins
    # made the tiny-amount bins look huge
    bins = np.linspace(0, np.log1p(df.amount.max()), 60)
    for lab, c, name in [(0, LEGIT_C, "legit"), (1, FRAUD_C, "fraud")]:
        a = df.loc[df.fraud == lab, "amount"]
        ax.hist(np.log1p(a), bins=bins, weights=np.full(len(a), 100 / len(a)),
                histtype="step", lw=2, color=c, label=f"{name} (median {a.median():.0f})")
    ticks = [0, 1, 10, 100, 1000, 8000]
    ax.set_xticks(np.log1p(ticks), [str(t) for t in ticks])
    ax.set_xlabel("amount (log1p scale)")
    ax.set_ylabel("% of transactions in class")
    ax.set_title("Transaction amount: fraud vs legit")
    ax.legend(frameon=False)
    save(fig, "02_amount_distribution.png")


def category_table(df):
    t = (df.groupby("category", observed=True)
           .agg(n_txns=("fraud", "size"), n_fraud=("fraud", "sum"), fraud_rate=("fraud", "mean"),
                median_amount=("amount", "median"))
           .sort_values("fraud_rate", ascending=False))
    t["share_of_all_fraud"] = t.n_fraud / t.n_fraud.sum()
    t.round(4).to_csv(TABLE_DIR / "category_summary.csv")
    return t


def fig_category(t):
    t = t.sort_values("fraud_rate")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(t.index.astype(str), t.fraud_rate * 100, color=FRAUD_C, height=0.6)
    for i, (rate, n, nf) in enumerate(zip(t.fraud_rate, t.n_txns, t.n_fraud)):
        ax.text(rate * 100 + 1, i, f"{nf:,} / {n:,}", va="center", fontsize=8, color="#52514e")
    ax.set_xlabel("fraud rate (%)   [labels: fraud / total txns]")
    ax.set_xlim(0, 112)
    ax.set_title("Fraud rate by merchant category")
    ax.grid(axis="y", visible=False)
    save(fig, "03_fraud_rate_by_category.png")


def fig_demographics(df):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, col, title in [(axes[0], "age", "age group"), (axes[1], "gender", "gender")]:
        g = df.groupby(col, observed=True).agg(rate=("fraud", "mean"), n=("fraud", "size"))
        ax.bar(g.index.astype(str), g.rate * 100, color=FRAUD_C, width=0.6)
        for x, (r, n) in enumerate(zip(g.rate, g.n)):
            ax.text(x, r * 100 + 0.05, f"n={n:,}", ha="center", fontsize=7, color="#52514e", rotation=90 if col == "age" else 0, va="bottom")
        ax.set_title(f"Fraud rate by {title}")
        ax.set_xlabel(col)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("fraud rate (%)")
    axes[0].set_ylim(0, 2.6)
    save(fig, "04_fraud_rate_by_age_gender.png")


def fig_merchants(df):
    m = df.groupby("merchant").agg(n=("fraud", "size"), rate=("fraud", "mean")).sort_values("rate")
    m.round(4).to_csv(TABLE_DIR / "merchant_summary.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(m))
    ax.bar(x, m.rate * 100, color=FRAUD_C, width=0.7)
    ax.set_xlabel(f"merchants, sorted by fraud rate (n={len(m)})")
    ax.set_ylabel("fraud rate (%)")
    n_zero = int((m.rate == 0).sum())
    ax.set_title(f"Merchant fraud rate: {n_zero} of {len(m)} merchants never see fraud")
    ax.set_xticks([])
    ax.grid(axis="x", visible=False)
    save(fig, "05_merchant_fraud_rate.png")
    return m


def fig_customers(df):
    c = df.groupby("customer").agg(n=("fraud", "size"), n_fraud=("fraud", "sum"))
    c.to_csv(TABLE_DIR / "customer_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].hist(c.n, bins=50, color=LEGIT_C)
    axes[0].set_xlabel("transactions per customer (6 months)")
    axes[0].set_ylabel("customers")
    axes[0].set_title("Customer activity")
    victims = c[c.n_fraud > 0]
    axes[1].hist(victims.n_fraud / victims.n, bins=40, color=FRAUD_C)
    axes[1].set_xlabel("share of the customer's txns that are fraud")
    axes[1].set_ylabel("customers")
    axes[1].set_title(f"Customers with >=1 fraud: {len(victims):,} of {len(c):,}")
    save(fig, "06_customer_activity.png")
    return c


def subgroup_table(df):
    train = pd.read_parquet(SPLIT_DIR / "train.parquet")
    by_cat, global_cut = fit_thresholds(train)
    flags = flag_normal_looking(df, by_cat, global_cut)
    d = df.join(flags)
    d["split"] = d.step.map(assign_split)
    rows = []
    for s in ["train", "val", "test"]:
        x = d[d.split == s]
        f = x[x.fraud == 1]
        rows.append({
            "split": s,
            "n_fraud": len(f),
            "normal_looking_fraud": int(f.normal_looking.sum()),
            "pct_of_fraud": round(f.normal_looking.mean() * 100, 1),
            "normal_looking_legit": int((x.fraud == 0).mul(x.normal_looking).sum()),
            "strict_normal_looking_fraud": int(f.normal_looking_strict.sum()),
            "strict_pct_of_fraud": round(f.normal_looking_strict.mean() * 100, 1),
        })
    t = pd.DataFrame(rows)
    t.to_csv(TABLE_DIR / "subgroup_sizes.csv", index=False)
    by_cat.rename("legit_p95_amount_train").round(2).to_csv(TABLE_DIR / "subgroup_thresholds.csv")
    print(f"global legit p95 (train): {global_cut:.2f}")
    print(t.to_string(index=False))

    # which categories the normal-looking fraud comes from
    nl = d[(d.fraud == 1) & (d.normal_looking == 1)]
    print(nl.category.value_counts().head(8).to_string())
    return t


# --- normal-looking fraud vs normal-looking legit --------------------------------
# Does fraud that looks ordinary on amount also look ordinary on history? Train days only,
# so nothing here looks at val or test. Legit rows are also reweighted to the fraud's
# category mix ("same-category legit"), because normal-looking fraud sits in a few
# categories that are rare among legit rows; without that, category differences would be
# read as history differences.

NL_FEATURES = [   # (column, label, group)
    ("amount", "amount", "transaction"),
    ("g1_cust_n_prev", "customer's past transactions", "G1 customer"),
    ("g1_cust_amt_z", "amount vs customer's usual (z)", "G1 customer"),
    ("g1_cust_days_since_last", "days since customer's last txn", "G1 customer"),
    ("g1_cust_n_merchants", "merchants customer paid before", "G1 customer"),
    ("g2_merch_amt_z", "amount vs merchant's usual (z)", "G2 merchant"),
    ("g3_pair_is_new", "first time at this merchant", "G3 pair"),
    ("g3_pair_n_prev", "past txns with this merchant", "G3 pair"),
    ("g3_pair_days_since_last", "days since last visit to merchant", "G3 pair"),
    ("g5_merch_fraud_rate", "merchant's known fraud rate", "G5 past labels"),
    ("g5_cust_fraud_n", "customer's known past fraud", "G5 past labels"),
]


def category_weights(legit_cat, fraud_cat):
    """Weights that give the legit rows the same category mix as the fraud rows.
    Categories with no fraud get weight 0."""
    f = fraud_cat.value_counts(normalize=True)
    l = legit_cat.value_counts(normalize=True)
    return legit_cat.map((f / l).fillna(0)).astype(float).to_numpy()


def smd(x_fraud, x_legit, w=None):
    """Standardised mean difference: (fraud mean - legit mean) / pooled sd.
    w weights the legit rows (None = unweighted). 0.2 is conventionally 'small', 0.8 'large'."""
    x_fraud, x_legit = np.asarray(x_fraud, float), np.asarray(x_legit, float)
    w = np.ones(len(x_legit)) if w is None else np.asarray(w, float)
    m_l = np.average(x_legit, weights=w)
    var_l = np.average((x_legit - m_l) ** 2, weights=w)
    sd = np.sqrt((x_fraud.var() + var_l) / 2)
    return (x_fraud.mean() - m_l) / sd if sd > 0 else np.nan


def normal_looking_comparison(train, graph):
    """Feature table and category mix for normal-looking fraud vs normal-looking legit (train only).
    train: train split with normal_looking flags; graph: graph features indexed by txn_id."""
    nl = train[train.normal_looking == 1].join(graph, on="txn_id")
    f, l = nl[nl.fraud == 1], nl[nl.fraud == 0]
    w = category_weights(l.category.astype(str), f.category.astype(str))
    rows = []
    for col, label, group in NL_FEATURES:
        rows.append({
            "feature": col, "label": label, "group": group,
            "nl_fraud_mean": f[col].mean(), "nl_fraud_median": f[col].median(),
            "nl_legit_mean": l[col].mean(), "nl_legit_median": l[col].median(),
            "nl_legit_same_category_mean": np.average(l[col], weights=w),
            "smd_vs_nl_legit": smd(f[col], l[col]),
            "smd_vs_same_category_legit": smd(f[col], l[col], w),
        })
    feat = pd.DataFrame(rows)
    counts = nl.groupby(["category", "fraud"], observed=True).size().unstack(fill_value=0)
    mix = pd.DataFrame({"nl_fraud": counts.get(1, 0), "nl_legit": counts.get(0, 0)})
    mix["share_of_nl_fraud"] = mix.nl_fraud / mix.nl_fraud.sum()
    mix["share_of_nl_legit"] = mix.nl_legit / mix.nl_legit.sum()
    mix["fraud_rate_within_nl"] = mix.nl_fraud / (mix.nl_fraud + mix.nl_legit)
    mix = mix.sort_values(["nl_fraud", "nl_legit"], ascending=False).rename_axis("category").reset_index()
    mix["category"] = mix.category.astype(str)
    return feat, mix, (f, l, w)


def fig_normal_looking(feat, mix, rows, by_cat):
    f, l, w = rows
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1, 0.9, 1.35]},
                             layout="constrained")   # per-panel spacing; (c) has long labels

    # (a) category mix: fraud's categories are rare among legit rows
    ax = axes[0]
    has_fraud = mix[mix.nl_fraud > 0]
    rest = mix[mix.nl_fraud == 0]
    labels = has_fraud.category.tolist() + [" / ".join(rest.category) + "\n(no fraud)"]
    sf = has_fraud.share_of_nl_fraud.tolist() + [0.0]
    sl = has_fraud.share_of_nl_legit.tolist() + [rest.share_of_nl_legit.sum()]
    y = np.arange(len(labels))[::-1]
    ax.barh(y + 0.2, np.array(sf) * 100, height=0.38, color=FRAUD_C, label="normal-looking fraud")
    ax.barh(y - 0.2, np.array(sl) * 100, height=0.38, color=LEGIT_C, label="normal-looking legit")
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlabel("% of the group's transactions")
    ax.set_title("(a) Category: not ordinary", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(axis="y", visible=False)

    # (b) amount within its category, as a share of the normal-looking cut
    ax = axes[1]
    cut = lambda d: d.amount / d.category.astype(str).map(by_cat.rename(index=str))
    bins = np.linspace(0, 1, 26)
    for d, wt, c, name in [(f, None, FRAUD_C, "fraud"), (l, w, LEGIT_C, "legit, same categories")]:
        x = cut(d)
        wt = np.ones(len(x)) if wt is None else wt
        ax.hist(x, bins=bins, weights=wt / wt.sum() * 100, histtype="step", lw=2, color=c, label=name)
    ax.set_xlabel("amount ÷ the category's normal-looking cut (train legit P95)")
    ax.set_ylabel("% of the group's transactions")
    ax.set_title("(b) Amount within category: overlaps, a bit higher", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    # (c) history: which differences survive once category is matched
    ax = axes[2]
    t = feat.iloc[::-1].reset_index(drop=True)
    yy = np.arange(len(t))
    ax.axvspan(-0.2, 0.2, color="#efeeea", zorder=0)
    ax.axvline(0, color=GREY, lw=1, zorder=1)
    ax.hlines(yy, t.smd_vs_nl_legit, t.smd_vs_same_category_legit, color="#c3c2b7", lw=1.5, zorder=2)
    ax.scatter(t.smd_vs_nl_legit, yy, s=36, facecolor="white", edgecolor=GREY, lw=1.5, zorder=3,
               label="vs all normal-looking legit")
    ax.scatter(t.smd_vs_same_category_legit, yy, s=44, color=FRAUD_C, zorder=4,
               label="vs legit in the same categories")
    ax.set_yticks(yy, [f"{lab}  [{g.split()[0]}]" for lab, g in zip(t.label, t.group)], fontsize=8)
    ax.set_xlabel("fraud minus legit, in standard deviations (grey band: |difference| < 0.2)")
    ax.set_title("(c) History: what still differs within category", loc="left", fontsize=10)
    ax.set_ylim(-0.6, len(t) + 0.9)                 # headroom so the legend clears the top row
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    ax.grid(axis="y", visible=False)

    fig.suptitle(f"Normal-looking fraud ({len(f):,}) vs normal-looking legit ({len(l):,}), "
                 "train days 0–119", x=0.01, ha="left", fontsize=12)
    fig.savefig(FIG_DIR / "07_normal_looking_fraud_vs_legit.png", bbox_inches="tight")   # not save(): no tight_layout
    plt.close(fig)
    print("saved 07_normal_looking_fraud_vs_legit.png")


def normal_looking_section():
    if not GRAPH_FILE.exists():
        print(f"skipped figure 07: {GRAPH_FILE.name} missing (run src.features.graph, "
              "then python -m src.eda --normal-looking)")
        return None
    train = pd.read_parquet(SPLIT_DIR / "train.parquet")
    by_cat, global_cut = fit_thresholds(train)
    train = train.join(flag_normal_looking(train, by_cat, global_cut))
    feat, mix, rows = normal_looking_comparison(train, pd.read_parquet(GRAPH_FILE))
    feat.round(4).to_csv(TABLE_DIR / "normal_looking_comparison.csv", index=False)
    mix.round(4).to_csv(TABLE_DIR / "normal_looking_category_mix.csv", index=False)
    fig_normal_looking(feat, mix, rows, by_cat)
    print(feat[["label", "nl_fraud_mean", "nl_legit_mean", "nl_legit_same_category_mean",
                "smd_vs_nl_legit", "smd_vs_same_category_legit"]].round(3).to_string(index=False))
    return feat, mix


def main(argv=None):
    p = argparse.ArgumentParser(description="Week-1 EDA figures and tables")
    p.add_argument("--normal-looking", action="store_true",
                   help="only the normal-looking fraud vs legit comparison (figure 07)")
    a = p.parse_args(argv)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    if a.normal_looking:
        normal_looking_section()
        return
    df = pd.read_parquet(CLEAN_FILE)

    fig_over_time(df)
    fig_amount(df)
    t = category_table(df)
    fig_category(t)
    fig_demographics(df)
    fig_merchants(df)
    fig_customers(df)
    subgroup_table(df)
    normal_looking_section()

    print()
    print(t.round(4).to_string())


if __name__ == "__main__":
    main()
