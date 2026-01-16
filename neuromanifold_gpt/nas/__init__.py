"""Neural Architecture Search (NAS) for NeuroManifoldGPT.

This package implements neural architecture search capabilities for discovering
optimal architectural configurations of the NeuroManifoldGPT model. It provides:
- Architecture search space definition
- Architecture configuration sampling and validation
- Integration with external NAS algorithms and search strategies
"""

from neuromanifold_gpt.nas.search_space import ArchitectureConfig, SearchSpace

__all__ = ["ArchitectureConfig", "SearchSpace"]
