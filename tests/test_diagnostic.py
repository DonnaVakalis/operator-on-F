"""Synthetic sanity tests for the operator-on-F reference implementation.

These do not use the released model data; they check the estimator's core
properties on constructed inputs. Run: ``python3 -m pytest`` or
``python3 tests/test_diagnostic.py``.
"""
from __future__ import annotations

import numpy as np

from operator_on_F import (
    operator_error,
    per_functional_error,
    geometry_basis,
    project,
    held_out_split,
    kernel_divergence,
)


def test_perfect_pushforward_is_zero():
    rng = np.random.default_rng(0)
    target = rng.normal(size=(500, 4))
    assert operator_error(target, target) == 0.0


def test_error_grows_with_perturbation():
    rng = np.random.default_rng(1)
    target = rng.normal(size=(1000, 3))
    small = operator_error(target + 0.1 * rng.normal(size=target.shape), target)
    large = operator_error(target + 0.5 * rng.normal(size=target.shape), target)
    assert 0 < small < large


def test_scale_invariance_per_functional():
    # Normalizing by the target std makes a functional's error invariant to its
    # units: scaling one functional by 100x leaves the operator error unchanged.
    rng = np.random.default_rng(2)
    target = rng.normal(size=(2000, 2))
    pred = target + 0.2 * rng.normal(size=target.shape)
    base = operator_error(pred, target)
    scale = np.array([1.0, 100.0])
    scaled = operator_error(pred * scale, target * scale)
    assert abs(base - scaled) < 1e-6


def test_geometry_basis_held_out():
    rng = np.random.default_rng(3)
    latents = rng.normal(size=(400, 8))
    fit_idx, eval_idx = held_out_split(len(latents), seed=3)
    assert set(fit_idx).isdisjoint(set(eval_idx))
    basis = geometry_basis(latents[fit_idx], n_components=4, seed=3)
    assert basis.shape == (4, 8)
    proj = project(latents[eval_idx], basis, center=latents[fit_idx].mean(axis=0, keepdims=True))
    assert proj.shape == (len(eval_idx), 4)


def test_per_functional_matches_full():
    rng = np.random.default_rng(4)
    target = rng.normal(size=(300, 5))
    pred = target + 0.3 * rng.normal(size=target.shape)
    pf = per_functional_error(pred, target)
    full = operator_error(pred, target)
    assert abs(np.sqrt(np.mean(pf ** 2)) - full) < 1e-9


def test_kernel_divergence_zero_on_identity():
    rng = np.random.default_rng(5)
    z = rng.normal(size=(100, 6))
    assert kernel_divergence(z, z) == 0.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"All {len(fns)} tests passed.")
