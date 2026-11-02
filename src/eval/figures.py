"""Figures for report §4.3 and the slides (R4). Drawn only from reports/tables; never computes a metric.

    uv run python -m src.eval.figures                 # val (labelled "not report numbers")
    uv run python -m src.eval.figures --split test    # after the freeze

Writes reports/figures/budget_curve_{split}.png: recall against the review budget, one panel per
model, all fraud on top and normal-looking fraud below, txn vs txn + graph in each panel.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import FIG_DIR, TABLE_DIR  # noqa: E402

# Reference palette (dataviz skill): categorical slots 1-2 validated as a pair; text and grid tokens.
SERIES = {"txn_graph": ("#2a78d6", "txn + graph"), "txn": ("#eb6834", "txn only")}
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
CEILING = "#a3a29c"   # reference line: quieter than the data, louder than the grid
MODEL_LABELS = {"lr": "Logistic regression", "mlp": "MLP", "lgbm": "LightGBM"}
ROWS = [("all", "All fraud"), ("normal_looking", "Normal-looking fraud")]
MARK_BUDGETS = (0.01, 0.02)


def budget_curve_figure(curve, split, out):
    models = [m for m in MODEL_LABELS if m in set(curve.model)]
    fig, axes = plt.subplots(len(ROWS), len(models), figsize=(3.6 * len(models), 6.4),
                             sharex=True, sharey=True, squeeze=False, facecolor=SURFACE)
    for i, (sub, row_label) in enumerate(ROWS):
        for j, model in enumerate(models):
            ax = axes[i, j]
            ax.set_facecolor(SURFACE)
            for b in MARK_BUDGETS:                                       # 1% and 2% guides
                ax.axvline(b * 100, color=GRID, lw=1, zorder=0)
            d = curve[(curve.model == model) & (curve.subset == sub)]
            if sub == "all":                                             # best possible recall
                c = d[d.feature_set == d.feature_set.iloc[0]].sort_values("budget")
                ax.plot(c.budget * 100, c.max_recall, color=CEILING, lw=1.2, zorder=1)
                if j == len(models) - 1:
                    ax.text(4.9, 1.015, "best possible recall", ha="right", va="bottom", fontsize=8, color=TEXT_2)
            for fs, (color, label) in SERIES.items():
                c = d[d.feature_set == fs].sort_values("budget")
                if c.empty:
                    continue
                ax.plot(c.budget * 100, c.recall, color=color, lw=2, solid_capstyle="round",
                        label=label, zorder=3)
                at = c[(c.budget - 0.01).abs() < 1e-9]                   # dot + value at 1%
                if len(at):
                    r = float(at.recall.iloc[0])
                    ax.plot([1.0], [r], "o", ms=7, color=color, mec=SURFACE, mew=2, zorder=4)
                    ax.annotate(f"{r:.0%}", (1.0, r), xytext=(6, -3 if fs == "txn" else 3),
                                textcoords="offset points", fontsize=8.5, color=TEXT,
                                va="top" if fs == "txn" else "bottom", zorder=5,
                                bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none"))
            ax.set_ylim(0, 1.05)
            ax.set_xlim(0, 5)
            ax.set_xticks([0, 1, 2, 3, 4, 5])
            ax.set_xticklabels([f"{t}%" for t in range(6)])
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
            ax.grid(axis="y", color=GRID, lw=1)
            ax.set_axisbelow(True)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
            ax.tick_params(colors=TEXT_2, labelsize=8.5, length=0)
            if i == 0:
                ax.set_title(MODEL_LABELS[model], fontsize=10.5, color=TEXT, loc="left")
            if j == 0:
                ax.set_ylabel(f"{row_label}\nrecall", fontsize=9.5, color=TEXT)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    title = "Fraud caught against the review budget"
    note = ("VAL split: for building the report, not report numbers." if split == "val"
            else "Test split (frozen results).")
    fig.text(0.01, 0.985, title, fontsize=13, color=TEXT, va="top", weight="bold")
    fig.text(0.01, 0.945, f"{note} Dots mark the 1% budget; grey lines at 1% and 2%.",
             fontsize=9, color=TEXT_2, va="top")
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.995, 0.99), ncol=2,
               frameon=False, fontsize=9, labelcolor=TEXT, handlelength=1.6)
    fig.supxlabel("Review budget (share of all transactions an analyst reviews)", fontsize=9.5, color=TEXT_2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out


def run(split="val", name="main", table_dir=TABLE_DIR, fig_dir=FIG_DIR, log=print):
    curve = pd.read_csv(table_dir / f"results_{name}_budget_curve_{split}.csv")
    out = budget_curve_figure(curve, split, fig_dir / f"budget_curve_{split}.png")
    log(f"saved {out}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--name", default="main")
    a = p.parse_args(argv)
    run(a.split, a.name)


if __name__ == "__main__":
    main()
