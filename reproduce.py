"""Reproduce the paper's headline correlation numbers from results/.

Loads the released per-size and cross-architecture results and recomputes the
across-model statistics with this package, checking they match the values
reported in the paper. Run:

    python3 reproduce.py
"""
from __future__ import annotations

import json
from pathlib import Path

from operator_on_F import spearman, leave_one_out_spearman

HERE = Path(__file__).resolve().parent
SIZES = ["1", "5", "19", "48", "317"]


def _check(name, got, expected, tol=0.02):
    ok = abs(got - expected) <= tol
    print(f"  [{'OK ' if ok else 'XX '}] {name}: computed {got:+.2f}  (paper {expected:+.2f})")
    return ok


def main():
    sweep = json.loads((HERE / "results" / "size_sweep.json").read_text())
    per = sweep["per_size"]
    operator = [per[s]["operator_error_full_F"] for s in SIZES]
    reward = [per[s]["reward_error"] for s in SIZES]
    bellman = [per[s]["bellman_residual"] for s in SIZES]
    value_only = [per[s]["value_only_error"] for s in SIZES]
    returns = [per[s]["return"] for s in SIZES]

    ok = []
    print("Size sweep (n=5 TD-MPC2 sizes on cheetah-run):")
    ok.append(_check("Spearman(operator-on-F, return)", spearman(operator, returns), -0.90))
    ok.append(_check("Spearman(reward error, return)", spearman(reward, returns), -0.30))
    ok.append(_check("Spearman(Bellman residual, return)", spearman(bellman, returns), -0.10))
    ok.append(_check("Spearman(operator-on-F, Bellman)", spearman(operator, bellman), +0.30))
    ok.append(_check("Spearman(value-only, Bellman)", spearman(value_only, bellman), +1.00))

    print("\nLeave-one-out Spearman(operator-on-F, return):")
    loo = leave_one_out_spearman(operator, returns)
    expected_loo = sweep["leave_one_out_operator_vs_return"]
    for (i, val), size in zip(loo, SIZES):
        exp = expected_loo[f"drop_{size}M"]
        ok.append(_check(f"drop {size}M", val, exp))

    print("\nCross-architecture (LeWM vs single-task TD-MPC2 on cheetah-run):")
    cross = json.loads((HERE / "results" / "cross_architecture.json").read_text())
    lewm = cross["lewm_operator_error_shared_F"]["per_seed"]
    tdmpc2 = cross["tdmpc2_single_task_operator_error_shared_F"]["point"]
    ratio = sum(lewm) / len(lewm) / tdmpc2
    ok.append(_check("ratio LeWM / TD-MPC2 (shared F)", ratio,
                     cross["ratio_lewm_over_tdmpc2"]["mean"]))

    print()
    if all(ok):
        print(f"All {len(ok)} checks passed: results reproduce the paper's reported numbers.")
        return 0
    print(f"{sum(ok)}/{len(ok)} checks passed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
