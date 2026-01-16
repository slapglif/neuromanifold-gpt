"""Hyperparameter Optimization (HPO) package for NeuroManifoldGPT.

This package provides automated hyperparameter optimization using Optuna,
enabling efficient search over learning rates, architecture sizes, and
novel component parameters.
"""

from neuromanifold_gpt.hpo.search_space import SearchSpace
from neuromanifold_gpt.hpo.optuna_search import OptunaHPO
from neuromanifold_gpt.hpo.pruning import OptunaPruningCallback
from neuromanifold_gpt.hpo.visualize import (
    plot_optimization_history,
    plot_param_importances,
    plot_all_visualizations,
)

__all__ = [
    "SearchSpace",
    "OptunaHPO",
    "OptunaPruningCallback",
    "plot_optimization_history",
    "plot_param_importances",
    "plot_all_visualizations",
]
