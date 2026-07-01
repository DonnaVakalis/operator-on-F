"""Figure 2: operator-on-F vs. the Bellman (value-equivalence) residual across
the five model sizes. The rank correlation is only modest, so operator-on-F
orders the models differently from a value-equivalence view.

Reads results/size_sweep.json and writes figures/metric_disagreement.pdf.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "size_sweep.json"

WONG_BLUE = "#0072B2"
WONG_VERMILLION = "#D55E00"
WONG_BLACK = "#000000"

SIZES = ["1", "5", "19", "48", "317"]
SIZE_LABELS = ["1M", "5M", "19M", "48M", "317M"]


def main() -> None:
    data = json.loads(RESULTS.read_text())
    per = data["per_size"]
    bellman = np.array([per[s]["bellman_residual"] for s in SIZES])
    operator = np.array([per[s]["operator_error_full_F"] for s in SIZES])
    sp = data["spearman_across_sizes"]
    sp_bellman = sp["operator_vs_bellman_residual"]["point"]
    sp_kernel = sp["operator_vs_kernel_divergence"]["point"]

    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "pdf.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    i317 = SIZES.index("317")
    other = [i for i in range(len(SIZES)) if i != i317]

    ax.scatter(bellman[other], operator[other], s=55, marker="o",
               facecolor=WONG_BLUE, edgecolor=WONG_BLACK, linewidths=0.8,
               zorder=3, label="1M / 5M / 19M / 48M")
    ax.scatter(bellman[i317], operator[i317], s=130, marker="D",
               facecolor=WONG_VERMILLION, edgecolor=WONG_BLACK, linewidths=1.0,
               zorder=4, label="317M")

    offsets = {"1": (8, 4), "5": (8, -12), "19": (8, 4), "48": (8, -12), "317": (-10, -16)}
    for i, s in enumerate(SIZES):
        dx, dy = offsets[s]
        ax.annotate(SIZE_LABELS[i], xy=(bellman[i], operator[i]),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=7.5, ha="right" if dx < 0 else "left", va="center")

    ax.set_xlabel("Bellman residual")
    ax.set_ylabel(r"operator-on-$F$ error (full $F$)")
    ax.grid(alpha=0.25, linewidth=0.5)
    xpad = (bellman.max() - bellman.min()) * 0.12
    ypad = (operator.max() - operator.min()) * 0.12
    ax.set_xlim(bellman.min() - xpad, bellman.max() + xpad)
    ax.set_ylim(0.0, operator.max() + ypad)

    ax.text(0.02, 0.98,
            rf"Spearman(op-on-$F$, Bellman)$={sp_bellman:.2f}$" + "\n"
            rf"Spearman(op-on-$F$, kernel div.)$={sp_kernel:.2f}$" + "\n"
            "value-only $\\to$ Bellman: Spearman $=1.00$",
            transform=ax.transAxes, fontsize=7.5, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.5))
    ax.legend(loc="lower right", frameon=False, fontsize=7.5)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"metric_disagreement.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote figures/metric_disagreement.pdf (+ .png)")


if __name__ == "__main__":
    main()
