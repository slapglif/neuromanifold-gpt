"""Search strategies for Neural Architecture Search.

This package contains various NAS search strategies:
- RandomSearch: Random sampling of architectures
- EvolutionarySearch: Genetic algorithm-based search
- BayesianSearch: Bayesian optimization (future)
- GradientSearch: Gradient-based NAS like DARTS (future)

Each strategy implements the Searcher interface defined in searcher.py.
"""

from neuromanifold_gpt.nas.strategies.evolutionary import EvolutionarySearch
from neuromanifold_gpt.nas.strategies.random_search import RandomSearch

__all__ = ["RandomSearch", "EvolutionarySearch"]
