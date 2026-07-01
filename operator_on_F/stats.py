"""Across-model statistics used in the paper: Spearman rank correlation,
leave-one-out stability, and an anchor-bootstrap helper. NumPy only.
"""
from __future__ import annotations

import numpy as np


def _rankdata(a):
    """Average ranks (1-based), handling ties like scipy.stats.rankdata."""
    a = np.asarray(a, dtype=float)
    order = a.argsort()
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1)
    # average tied ranks
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def spearman(x, y):
    """Spearman rank correlation = Pearson correlation of the ranks."""
    rx, ry = _rankdata(x), _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def leave_one_out_spearman(x, y):
    """Spearman with each single observation removed in turn.

    Returns a list of ``(dropped_index, spearman)`` over the reduced samples --
    a stability check that no single point drives the correlation.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = []
    for i in range(len(x)):
        keep = np.arange(len(x)) != i
        out.append((i, round(spearman(x[keep], y[keep]), 2)))
    return out


def bootstrap_spearman_ci(x, y, n_boot=1000, ci=95, seed=0):
    """Percentile bootstrap CI for a Spearman correlation (resampling points).

    Note: the paper's per-anchor 95% CIs are computed by resampling the
    underlying *anchors* before the operator error is aggregated; that requires
    the raw anchor arrays (not shipped). This point-resampling variant is a
    lightweight stand-in for the correlation itself.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(spearman(x[idx], y[idx]))
    lo = (100 - ci) / 2
    return float(np.nanpercentile(vals, lo)), float(np.nanpercentile(vals, 100 - lo))
