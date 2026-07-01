"""Figure 1 (scissors): operator-on-F error and reward-prediction error share the
left "rollout error" axis; executed return is on the right axis. At the largest
model size operator-on-F error spikes and return collapses, while reward error
stays flat -- so operator-on-F tracks the return loss where a reward-fit check
does not.

Reads results/size_sweep.json and writes figures/scissors.pdf (+ .png).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "size_sweep.json"

WONG_BLUE = "#0072B2"        # operator-on-F error
WONG_VERMILLION = "#D55E00"  # return
WONG_GREEN = "#009E73"       # reward-prediction error
WONG_BLACK = "#000000"

SIZES = ["1", "5", "19", "48", "317"]
SIZE_LABELS = ["1M", "5M", "19M", "48M", "317M"]


def main() -> None:
    per = json.loads(RESULTS.read_text())["per_size"]
    operator = np.array([per[s]["operator_error_full_F"] for s in SIZES])
    reward = np.array([per[s]["reward_error"] for s in SIZES])
    returns = np.array([per[s]["return"] for s in SIZES])
    x = np.arange(len(SIZES))
    i317 = len(SIZES) - 1

    plt.rcParams.update({
        "font.size": 10.5, "axes.labelsize": 10.5, "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5, "pdf.fonttype": 42,
    })
    fig, axL = plt.subplots(figsize=(6.6, 3.4))
    axR = axL.twinx()

    # right axis first (drawn under), so the error curves sit on top
    l_ret, = axR.plot(x, returns, color=WONG_VERMILLION, marker="s", markersize=6,
                      linestyle="--", linewidth=1.6, markerfacecolor="white",
                      markeredgecolor=WONG_VERMILLION, markeredgewidth=1.4, zorder=2)
    axR.set_ylabel("planning return", color=WONG_VERMILLION)
    axR.tick_params(axis="y", colors=WONG_VERMILLION)
    axR.spines["right"].set_color(WONG_VERMILLION)
    axR.set_ylim(-0.05 * max(returns) * 1.15, max(returns) * 1.15)

    # left axis: both errors, normalized to the rollout-error scale
    l_op, = axL.plot(x, operator, color=WONG_BLUE, marker="o", markersize=6,
                     linestyle="-", linewidth=1.7, zorder=4)
    l_rw, = axL.plot(x, reward, color=WONG_GREEN, marker="^", markersize=6,
                     linestyle=":", linewidth=1.7, zorder=4)
    axL.set_ylabel("normalized rollout error")
    axL.set_ylim(-0.05 * max(operator) * 1.12, max(operator) * 1.12)
    axL.set_zorder(axR.get_zorder() + 1)   # error curves above the return line
    axL.patch.set_visible(False)

    # highlight the largest size on each series
    axL.plot(x[i317], operator[i317], "o", markersize=12, color=WONG_BLUE,
             markeredgecolor=WONG_BLACK, markeredgewidth=1.3, zorder=5)
    axR.plot(x[i317], returns[i317], "s", markersize=12, markerfacecolor="white",
             markeredgecolor=WONG_VERMILLION, markeredgewidth=2.0, zorder=3)
    axL.plot(x[i317], reward[i317], "^", markersize=12, color=WONG_GREEN,
             markeredgecolor=WONG_BLACK, markeredgewidth=1.3, zorder=5)

    # combined legend (operator + reward on the left axis; return on the right)
    axL.legend([l_op, l_ret, l_rw],
               ["operator-on-$F$ error", "return", "reward-prediction error"],
               loc="upper left", bbox_to_anchor=(0.01, 0.99),
               frameon=True, framealpha=0.93, facecolor="white", edgecolor="0.8",
               fontsize=8.5, borderpad=0.55, labelspacing=0.5, handlelength=2.2)

    axL.set_xticks(x)
    axL.set_xticklabels(SIZE_LABELS)
    axL.set_xlabel("TD-MPC2 model size (mt80, cheetah-run)")
    axL.set_xlim(-0.35, len(SIZES) - 0.55)
    axL.grid(alpha=0.22, linewidth=0.5)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"scissors.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote figures/scissors.pdf (+ .png)")


if __name__ == "__main__":
    main()
