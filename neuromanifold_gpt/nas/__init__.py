"""Neural Architecture Search (NAS) for NeuroManifoldGPT.

This package implements neural architecture search capabilities for discovering
optimal architectural configurations of the NeuroManifoldGPT model. It provides:
- Architecture search space definition and sampling
- Architecture evaluation with configurable training budget
- Search strategies (random, evolutionary, etc.)
- Export utilities for discovered architectures

Example:
    from neuromanifold_gpt.nas import (
        SearchSpace,
        ArchitectureEvaluator,
        RandomSearch,
        ComputeBudget,
        export_config,
    )

    # Define search space and evaluator
    search_space = SearchSpace()
    evaluator = ArchitectureEvaluator(vocab_size=65, device="cuda")
    budget = ComputeBudget(max_evaluations=100)

    # Run architecture search
    searcher = RandomSearch(search_space, evaluator, budget)
    result = searcher.search(data=train_data, n_iters=200)

    # Export best architecture
    export_config(
        result.best_architecture,
        "config/nas_discovered/best_arch.py",
        metrics={"perplexity": result.best_result.perplexity}
    )
"""

# Search space and configuration
# Evaluation
from neuromanifold_gpt.nas.evaluator import (
    ArchitectureEvaluator,
    ComputeBudget,
    EvaluationResult,
)

# Export utilities
from neuromanifold_gpt.nas.export import (
    export_config,
    export_to_json,
    generate_config_summary,
    generate_summary_report,
)
from neuromanifold_gpt.nas.search_space import ArchitectureConfig, SearchSpace

# Base searcher and results
from neuromanifold_gpt.nas.searcher import Searcher, SearchResult
from neuromanifold_gpt.nas.strategies.evolutionary import EvolutionarySearch

# Search strategies
from neuromanifold_gpt.nas.strategies.random_search import RandomSearch

__all__ = [
    # Search space
    "ArchitectureConfig",
    "SearchSpace",
    # Evaluation
    "ArchitectureEvaluator",
    "ComputeBudget",
    "EvaluationResult",
    # Search strategies
    "RandomSearch",
    "EvolutionarySearch",
    # Base classes
    "Searcher",
    "SearchResult",
    # Export utilities
    "export_config",
    "export_to_json",
    "generate_config_summary",
    "generate_summary_report",
]
