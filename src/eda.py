"""Week-1 EDA: saves figures to reports/figures/ and tables to reports/tables/.

Usage: python -m src.eda
(run src.data.clean and src.data.split first)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CLEAN_FILE, FIG_DIR, SPLIT_DIR, TABLE_DIR, TRAIN_END, VAL_END
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


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(CLEAN_FILE)

    fig_over_time(df)
    fig_amount(df)
    t = category_table(df)
    fig_category(t)
    fig_demographics(df)
    fig_merchants(df)
    fig_customers(df)
    subgroup_table(df)

    print()
    print(t.round(4).to_string())


if __name__ == "__main__":
    main()
