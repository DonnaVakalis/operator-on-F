"""operator-on-F: a planning-time diagnostic for latent world models.

Reference implementation accompanying the paper
"Operator-on-F complements value-equivalence: a planning-time diagnostic for
latent world models" (RLC 2026 World Model Workshop).
"""
from .diagnostic import (
    operator_error,
    per_functional_error,
    geometry_basis,
    project,
    held_out_split,
    value_only_error,
    bellman_residual,
    kernel_divergence,
)
from .stats import spearman, leave_one_out_spearman, bootstrap_spearman_ci

__all__ = [
    "operator_error",
    "per_functional_error",
    "geometry_basis",
    "project",
    "held_out_split",
    "value_only_error",
    "bellman_residual",
    "kernel_divergence",
    "spearman",
    "leave_one_out_spearman",
    "bootstrap_spearman_ci",
]
