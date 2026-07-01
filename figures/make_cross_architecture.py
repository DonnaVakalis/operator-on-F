"""Figure 3: cross-architecture operator error on a shared observable family.

Pure-SSL LeWM (no anchor; 3 seeds) vs. the released single-task TD-MPC2, both
at a matched 5-step horizon and probed identically. Bars show the operator
error with seed SD / 95% CI; the persistence baseline is a dashed reference.

Reads results/cross_architecture.json and writes figures/cross_architecture.pdf.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "cross_architecture.json"

WONG_BLUE = "#0072B2"
WONG_VERMILLION = "#D55E00"
WONG_BLACK = "#000000"
WONG_GREY = "#555555"


def main() -> None:
    data = json.loads(RESULTS.read_text())
    lewm_seeds = np.array(data["lewm_operator_error_shared_F"]["per_seed"])
    lewm_mean = float(lewm_seeds.mean())
    lewm_sd = float(lewm_seeds.std(ddof=1))
    tdmpc2_val = float(data["tdmpc2_single_task_operator_error_shared_F"]["point"])
    tdmpc2_lo, tdmpc2_hi = data["tdmpc2_single_task_operator_error_shared_F"]["ci95"]
    tdmpc2_err = np.array([[tdmpc2_val - tdmpc2_lo], [tdmpc2_hi - tdmpc2_val]])
    persistence = float(data["persistence_baseline"])

    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5,
        "ytick.labelsize": 8, "legend.fontsize": 8, "pdf.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    x = np.array([0, 1])
    labels = ["LeWM\n(no anchor)", "TD-MPC2\n(single-task)"]
    heights = [lewm_mean, tdmpc2_val]

    bars = ax.bar(x, heights, width=0.55, color=[WONG_BLUE, WONG_VERMILLION],
                  edgecolor=WONG_BLACK, linewidth=0.8)
    for b, h in zip(bars, ["", "//"]):
        b.set_hatch(h)

    ax.errorbar(x[0], lewm_mean, yerr=lewm_sd, fmt="none", ecolor=WONG_BLACK,
                elinewidth=1.0, capsize=4)
    ax.errorbar(x[1], tdmpc2_val, yerr=tdmpc2_err, fmt="none", ecolor=WONG_BLACK,
                elinewidth=1.0, capsize=4)

    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.10, 0.10, size=lewm_seeds.size)
    ax.scatter(x[0] + jitter, lewm_seeds, s=22, marker="o", facecolor="white",
               edgecolor=WONG_BLACK, linewidths=0.9, zorder=5)

    ax.axhline(persistence, color=WONG_GREY, linestyle="--", linewidth=1.0, zorder=1)
    ax.text(1.45, persistence, "persistence", fontsize=7.5, color=WONG_GREY,
            ha="right", va="bottom")

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("operator-on-$F$ error on shared $F$"
                  "\n(lower = closer 5-step pushforward)", fontsize=8.5)
    ax.set_ylim(0.0, max(persistence, tdmpc2_hi) * 1.12)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)

    ax.text(x[0], lewm_mean + lewm_sd + 0.04, f"{lewm_mean:.3f} $\\pm$ {lewm_sd:.3f}",
            ha="center", va="bottom", fontsize=7.5)
    ax.text(x[1], tdmpc2_hi + 0.04, f"{tdmpc2_val:.3f} [{tdmpc2_lo:.2f}, {tdmpc2_hi:.2f}]",
            ha="center", va="bottom", fontsize=7.5)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"cross_architecture.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote figures/cross_architecture.pdf (+ .png)")


if __name__ == "__main__":
    main()
