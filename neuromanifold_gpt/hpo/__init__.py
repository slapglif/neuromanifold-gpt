"""Hyperparameter Optimization (HPO) package for NeuroManifoldGPT.

This package provides automated hyperparameter optimization using Optuna,
enabling efficient search over learning rates, architecture sizes, and
novel component parameters.
"""

from neuromanifold_gpt.hpo.search_space import SearchSpace

__all__ = ["SearchSpace"]
