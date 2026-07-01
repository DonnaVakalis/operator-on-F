"""Operator-on-F: a planning-time diagnostic for latent world models.

The diagnostic compares a model's k-step latent *pushforward* to the
environment's, read out on a chosen set of observable functionals F.

For an anchor ``(s_t, a_{t:t+k-1}, s_{t+k})`` the model produces a predicted
latent ``z_hat = O_k(enc(s_t), a)`` (its own k-step rollout) and the environment
provides the encoded true next state ``z_prime = enc(s_{t+k})``. A functional
``phi in F`` (a probe from latent space to an observable, e.g. reward, value, or
a geometry direction) is evaluated on both. The per-anchor operator error is

    e_phi(t) = | phi(z_hat) - phi(z_prime) | / sigma_phi ,

where ``sigma_phi`` is the standard deviation of ``phi(z_prime)`` across the
anchor pool (so heterogeneous observables are comparable). Errors are aggregated
by root-mean-square over ``(phi, anchor)``.

Two aggregations are reported:
  * **value slice**  F = {reward, value}  -- what value-equivalence checks see.
  * **full F**       value slice + a held-out PCA basis on the encoded
                     next-state geometry -- the planning-relevant signal that a
                     reward/value-only check can miss.

This module is a self-contained reference implementation; it operates on arrays
of functional values and has no dependency on any particular world model.
"""
from __future__ import annotations

import numpy as np


def operator_error(predicted, target):
    """Full-F operator-on-F error.

    Parameters
    ----------
    predicted, target : array-like, shape (n_anchors, n_functionals)
        ``predicted[i, j] = phi_j(z_hat_i)`` (model pushforward) and
        ``target[i, j] = phi_j(z_prime_i)`` (encoded true next state), for each
        functional ``phi_j in F`` and anchor ``i``.

    Returns
    -------
    float
        RMS over ``(functional, anchor)`` of the per-anchor errors, each
        functional normalized by the across-anchor std of its target values.
    """
    e = _per_anchor_errors(predicted, target)
    return float(np.sqrt(np.mean(e ** 2)))


def per_functional_error(predicted, target):
    """RMS operator error for each functional separately (same normalization).

    Returns an array of shape ``(n_functionals,)``.
    """
    e = _per_anchor_errors(predicted, target)
    return np.sqrt(np.mean(e ** 2, axis=0))


def _per_anchor_errors(predicted, target):
    predicted = np.asarray(predicted, dtype=float)
    target = np.asarray(target, dtype=float)
    if predicted.shape != target.shape:
        raise ValueError(f"shape mismatch: {predicted.shape} vs {target.shape}")
    if predicted.ndim != 2:
        raise ValueError("expected arrays of shape (n_anchors, n_functionals)")
    sigma = target.std(axis=0, ddof=0)
    sigma = np.where(sigma > 0, sigma, 1.0)  # guard constant functionals
    return np.abs(predicted - target) / sigma


def geometry_basis(latents, n_components=16, seed=0):
    """Held-out PCA basis on the encoded next-state geometry.

    Fits the top ``n_components`` principal directions of ``latents`` and whitens
    them by their singular values, so each direction is a unit-variance
    functional ``phi_k(z) = u_k . z / sigma_k``. Fit this on one half of the
    anchors and evaluate the operator error on the other half (see
    :func:`held_out_split`) so the basis cannot be adapted to the quantity being
    measured.

    Returns a ``(n_components, latent_dim)`` matrix of whitened directions;
    apply it with :func:`project`.
    """
    latents = np.asarray(latents, dtype=float)
    centered = latents - latents.mean(axis=0, keepdims=True)
    # economy SVD; columns of Vt are principal directions
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    k = min(n_components, vt.shape[0])
    n = latents.shape[0]
    singular = np.where(s[:k] > 0, s[:k], 1.0)
    # sigma_k = s_k / sqrt(n-1) is the std along direction k; whiten by it
    scale = np.sqrt(max(n - 1, 1)) / singular
    return vt[:k] * scale[:, None]


def project(latents, basis, center=None):
    """Evaluate the geometry functionals on ``latents``: ``(z - center) @ basis.T``."""
    latents = np.asarray(latents, dtype=float)
    if center is None:
        center = latents.mean(axis=0, keepdims=True)
    return (latents - center) @ np.asarray(basis, dtype=float).T


def held_out_split(n_anchors, seed=0):
    """Return two disjoint index halves (fit / evaluate) of the anchor pool."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_anchors)
    mid = n_anchors // 2
    return perm[:mid], perm[mid:]


# --- value-equivalence comparators (for context alongside operator-on-F) ---

def value_only_error(value_predicted, value_target):
    """Unnormalized RMS of ``|V(z_hat) - V(z_prime)|`` across anchors."""
    vp = np.asarray(value_predicted, dtype=float)
    vt = np.asarray(value_target, dtype=float)
    return float(np.sqrt(np.mean((vp - vt) ** 2)))


def bellman_residual(reward_predicted, reward_target,
                     value_next_predicted, value_next_target, gamma):
    """Value-equivalence (Bellman) residual with a common value function.

    ``|| (r_hat + gamma V(z_hat)) - (r_true + gamma V(z_prime)) ||`` (RMS over
    anchors). Because the same V is used on both sides, this measures the error
    in the value-relevant direction only; a model wrong *orthogonal* to value
    can have a small Bellman residual yet a large operator-on-F error.
    """
    rp = np.asarray(reward_predicted, dtype=float)
    rt = np.asarray(reward_target, dtype=float)
    vp = np.asarray(value_next_predicted, dtype=float)
    vt = np.asarray(value_next_target, dtype=float)
    resid = (rp + gamma * vp) - (rt + gamma * vt)
    return float(np.sqrt(np.mean(resid ** 2)))


def kernel_divergence(predicted_latents, target_latents):
    """Full-kernel divergence: RMS Euclidean distance ``||z_hat - z_prime||``
    across anchors, normalized by the across-anchor spread of the target.

    A geometry-level (kernel) discrepancy that ignores which directions matter
    for the task; contrast with the value-equivalence comparators above.
    """
    zp = np.asarray(predicted_latents, dtype=float)
    zt = np.asarray(target_latents, dtype=float)
    spread = zt.std(axis=0, ddof=0)
    spread = np.where(spread > 0, spread, 1.0)
    d = (zp - zt) / spread
    return float(np.sqrt(np.mean(np.sum(d ** 2, axis=1))))
