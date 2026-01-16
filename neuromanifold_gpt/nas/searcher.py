"""Base interface for Neural Architecture Search strategies.

This module defines the abstract base class for NAS search strategies.
All search strategies (random, evolutionary, Bayesian, etc.) should inherit
from the Searcher class and implement the required methods.

The searcher interface provides a consistent API for:
- Running architecture search
- Tracking search progress
- Managing search results
- Resuming interrupted searches
- Exporting top architectures
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path
import json
import time

from neuromanifold_gpt.nas.search_space import SearchSpace, ArchitectureConfig
from neuromanifold_gpt.nas.evaluator import ArchitectureEvaluator, EvaluationResult, ComputeBudget


@dataclass
class SearchResult:
    """Result from a complete architecture search.

    Attributes:
        architectures: List of evaluated architectures (ArchitectureConfig)
        results: List of evaluation results (EvaluationResult)
        best_architecture: Best architecture found (ArchitectureConfig)
        best_result: Evaluation result for best architecture
        search_time: Total search time in seconds
        n_evaluations: Total number of architectures evaluated
        search_space_size: Size of the search space
        strategy_name: Name of the search strategy used
        metadata: Additional strategy-specific metadata
    """
    architectures: List[ArchitectureConfig]
    results: List[EvaluationResult]
    best_architecture: ArchitectureConfig
    best_result: EvaluationResult
    search_time: float
    n_evaluations: int
    search_space_size: int
    strategy_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_top_k(self, k: int = 5) -> List[tuple[ArchitectureConfig, EvaluationResult]]:
        """Get top-k architectures by perplexity.

        Args:
            k: Number of top architectures to return

        Returns:
            List of (architecture, result) tuples, sorted by perplexity
        """
        # Pair architectures with results
        pairs = list(zip(self.architectures, self.results))

        # Filter successful results and sort by perplexity
        successful_pairs = [(arch, res) for arch, res in pairs if res.success]
        successful_pairs.sort(key=lambda x: x[1].perplexity)

        return successful_pairs[:k]

    def to_dict(self) -> Dict[str, Any]:
        """Convert search result to dictionary for serialization.

        Returns:
            Dictionary representation of the search result
        """
        return {
            "architectures": [arch.to_dict() for arch in self.architectures],
            "results": [
                {
                    "architecture_id": res.architecture_id,
                    "final_loss": res.final_loss,
                    "perplexity": res.perplexity,
                    "n_params": res.n_params,
                    "time_per_iter_ms": res.time_per_iter_ms,
                    "n_iters": res.n_iters,
                    "success": res.success,
                    "error_message": res.error_message,
                }
                for res in self.results
            ],
            "best_architecture": self.best_architecture.to_dict(),
            "best_result": {
                "architecture_id": self.best_result.architecture_id,
                "final_loss": self.best_result.final_loss,
                "perplexity": self.best_result.perplexity,
                "n_params": self.best_result.n_params,
                "time_per_iter_ms": self.best_result.time_per_iter_ms,
                "n_iters": self.best_result.n_iters,
                "success": self.best_result.success,
                "error_message": self.best_result.error_message,
            },
            "search_time": self.search_time,
            "n_evaluations": self.n_evaluations,
            "search_space_size": self.search_space_size,
            "strategy_name": self.strategy_name,
            "metadata": self.metadata,
        }

    def save(self, filepath: Path) -> None:
        """Save search result to JSON file.

        Args:
            filepath: Path to save the search result
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> "SearchResult":
        """Load search result from JSON file.

        Args:
            filepath: Path to the saved search result

        Returns:
            SearchResult instance
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        # Reconstruct architectures
        architectures = [
            ArchitectureConfig.from_dict(arch_dict)
            for arch_dict in data["architectures"]
        ]

        # Reconstruct results
        results = [
            EvaluationResult(**res_dict)
            for res_dict in data["results"]
        ]

        # Reconstruct best architecture and result
        best_architecture = ArchitectureConfig.from_dict(data["best_architecture"])
        best_result = EvaluationResult(**data["best_result"])

        return cls(
            architectures=architectures,
            results=results,
            best_architecture=best_architecture,
            best_result=best_result,
            search_time=data["search_time"],
            n_evaluations=data["n_evaluations"],
            search_space_size=data["search_space_size"],
            strategy_name=data["strategy_name"],
            metadata=data.get("metadata", {}),
        )


class Searcher(ABC):
    """Abstract base class for Neural Architecture Search strategies.

    All NAS strategies should inherit from this class and implement the
    required abstract methods. The base class provides common functionality
    for search progress tracking, result management, and checkpointing.

    Args:
        search_space: SearchSpace defining the architecture search space
        evaluator: ArchitectureEvaluator for scoring architectures
        budget: Optional ComputeBudget for limiting search

    Attributes:
        search_space: The search space being explored
        evaluator: The evaluator for scoring architectures
        budget: Optional compute budget for search
        evaluated_architectures: List of evaluated architectures so far
        evaluation_results: List of evaluation results so far
        start_time: When the search started (None if not started)
    """

    def __init__(
        self,
        search_space: Optional[SearchSpace] = None,
        evaluator: Optional[ArchitectureEvaluator] = None,
        budget: Optional[ComputeBudget] = None,
    ):
        """Initialize the searcher."""
        self.search_space = search_space if search_space is not None else SearchSpace()
        self.evaluator = evaluator
        self.budget = budget

        # Search state
        self.evaluated_architectures: List[ArchitectureConfig] = []
        self.evaluation_results: List[EvaluationResult] = []
        self.start_time: Optional[float] = None

    @abstractmethod
    def search(
        self,
        data: Any,
        n_iters: int = 200,
        batch_size: int = 32,
    ) -> SearchResult:
        """Run the architecture search.

        This is the main method that implements the search strategy.
        Each strategy should implement this method to define how architectures
        are sampled, evaluated, and selected.

        Args:
            data: Training data for evaluation
            n_iters: Number of training iterations per evaluation
            batch_size: Batch size for training

        Returns:
            SearchResult with all evaluated architectures and the best one
        """
        pass

    def get_best_architecture(self) -> Optional[tuple[ArchitectureConfig, EvaluationResult]]:
        """Get the best architecture found so far.

        Returns:
            Tuple of (best_architecture, best_result) or None if no successful evaluations
        """
        # Filter successful results
        successful_pairs = [
            (arch, res)
            for arch, res in zip(self.evaluated_architectures, self.evaluation_results)
            if res.success
        ]

        if not successful_pairs:
            return None

        # Find best by perplexity
        best_pair = min(successful_pairs, key=lambda x: x[1].perplexity)
        return best_pair

    def get_search_progress(self) -> Dict[str, Any]:
        """Get current search progress statistics.

        Returns:
            Dictionary with search progress information
        """
        progress = {
            "n_evaluations": len(self.evaluation_results),
            "n_successful": sum(1 for res in self.evaluation_results if res.success),
            "n_failed": sum(1 for res in self.evaluation_results if not res.success),
        }

        if self.start_time is not None:
            progress["elapsed_time"] = time.time() - self.start_time

        best = self.get_best_architecture()
        if best is not None:
            _, best_result = best
            progress["best_perplexity"] = best_result.perplexity
            progress["best_loss"] = best_result.final_loss

        return progress

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this search strategy.

        Returns:
            Human-readable name of the strategy (e.g., "Random Search")
        """
        pass
