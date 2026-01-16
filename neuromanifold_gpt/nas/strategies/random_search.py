"""Random search strategy for Neural Architecture Search.

This module implements a simple but effective random search strategy for NAS.
Random search samples architectures uniformly from the search space and evaluates
them, tracking the best performers.

Despite its simplicity, random search is often a strong baseline and can be
surprisingly effective, especially in high-dimensional search spaces where more
sophisticated methods may struggle.

Example:
    >>> from neuromanifold_gpt.nas.search_space import SearchSpace
    >>> from neuromanifold_gpt.nas.evaluator import ArchitectureEvaluator, ComputeBudget
    >>> from neuromanifold_gpt.nas.strategies.random_search import RandomSearch
    >>>
    >>> search_space = SearchSpace()
    >>> evaluator = ArchitectureEvaluator(vocab_size=65, device="cuda")
    >>> budget = ComputeBudget(max_evaluations=50)
    >>>
    >>> searcher = RandomSearch(search_space, evaluator, budget)
    >>> result = searcher.search(data=train_data, n_iters=200)
    >>>
    >>> print(f"Best architecture: {result.best_result.perplexity:.2f} perplexity")
    >>> print(f"Evaluated {result.n_evaluations} architectures in {result.search_time:.1f}s")
"""

import time
import random
from typing import Any, Optional, List, Dict
from dataclasses import dataclass, field
from pathlib import Path
import json
from loguru import logger

from neuromanifold_gpt.nas.search_space import SearchSpace, ArchitectureConfig
from neuromanifold_gpt.nas.evaluator import ArchitectureEvaluator, ComputeBudget, EvaluationResult
from neuromanifold_gpt.nas.searcher import Searcher, SearchResult


@dataclass
class SearchResults:
    """Container for tracking architecture search results with top-K selection.

    This class provides incremental result tracking during search, with
    efficient top-K selection and checkpoint capability. It's designed to
    work alongside the search process, maintaining statistics and enabling
    periodic checkpointing.

    Attributes:
        architectures: List of evaluated architecture configurations
        results: List of evaluation results (parallel to architectures)
        search_time: Total time spent searching (seconds)
        n_evaluations: Total number of evaluations performed
        metadata: Additional tracking information

    Example:
        >>> results = SearchResults()
        >>> results.add_result(architecture, evaluation_result)
        >>> top_3 = results.get_top_k(3)
        >>> results.save_checkpoint("search_checkpoint.json")
    """
    architectures: List[ArchitectureConfig] = field(default_factory=list)
    results: List[EvaluationResult] = field(default_factory=list)
    search_time: float = 0.0
    n_evaluations: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_result(
        self,
        architecture: ArchitectureConfig,
        result: EvaluationResult,
    ) -> None:
        """Add a new evaluation result to the tracking.

        Args:
            architecture: The evaluated architecture configuration
            result: The evaluation result for this architecture
        """
        self.architectures.append(architecture)
        self.results.append(result)
        self.n_evaluations += 1

    def get_top_k(
        self,
        k: int = 5,
        metric: str = "perplexity",
    ) -> List[tuple[ArchitectureConfig, EvaluationResult]]:
        """Get top-K architectures by specified metric.

        Args:
            k: Number of top architectures to return
            metric: Metric to sort by ("perplexity", "loss", "params")

        Returns:
            List of (architecture, result) tuples, sorted by metric

        Example:
            >>> top_5 = results.get_top_k(5, metric="perplexity")
            >>> for arch, res in top_5:
            ...     print(f"{arch.architecture_id}: {res.perplexity:.2f}")
        """
        # Pair architectures with results
        pairs = list(zip(self.architectures, self.results))

        # Filter successful results
        successful_pairs = [(arch, res) for arch, res in pairs if res.success]

        if not successful_pairs:
            return []

        # Sort by specified metric
        if metric == "perplexity":
            successful_pairs.sort(key=lambda x: x[1].perplexity)
        elif metric == "loss":
            successful_pairs.sort(key=lambda x: x[1].final_loss)
        elif metric == "params":
            successful_pairs.sort(key=lambda x: x[1].n_params)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return successful_pairs[:k]

    def get_statistics(self) -> Dict[str, Any]:
        """Get current search statistics.

        Returns:
            Dictionary with search statistics including:
            - n_evaluations: Total number of evaluations
            - n_successful: Number of successful evaluations
            - n_failed: Number of failed evaluations
            - search_time: Total search time in seconds
            - best_perplexity: Best perplexity found (if any)
            - best_loss: Best loss found (if any)
        """
        stats = {
            "n_evaluations": self.n_evaluations,
            "n_successful": sum(1 for res in self.results if res.success),
            "n_failed": sum(1 for res in self.results if not res.success),
            "search_time": self.search_time,
        }

        # Add best metrics if we have successful results
        top_1 = self.get_top_k(1)
        if top_1:
            _, best_result = top_1[0]
            stats["best_perplexity"] = best_result.perplexity
            stats["best_loss"] = best_result.final_loss
            stats["best_params"] = best_result.n_params

        return stats

    def save_checkpoint(self, filepath: Path) -> None:
        """Save current search results to checkpoint file.

        Args:
            filepath: Path to save the checkpoint

        Example:
            >>> results.save_checkpoint("checkpoints/search_iter_100.json")
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
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
            "search_time": self.search_time,
            "n_evaluations": self.n_evaluations,
            "metadata": self.metadata,
            "statistics": self.get_statistics(),
        }

        with open(filepath, "w") as f:
            json.dump(checkpoint, f, indent=2)

        logger.info(f"Saved checkpoint to {filepath}")

    @classmethod
    def load_checkpoint(cls, filepath: Path) -> "SearchResults":
        """Load search results from checkpoint file.

        Args:
            filepath: Path to the checkpoint file

        Returns:
            SearchResults instance loaded from checkpoint

        Example:
            >>> results = SearchResults.load_checkpoint("checkpoints/search_iter_100.json")
            >>> print(f"Resuming from {results.n_evaluations} evaluations")
        """
        with open(filepath, "r") as f:
            checkpoint = json.load(f)

        # Reconstruct architectures
        architectures = [
            ArchitectureConfig.from_dict(arch_dict)
            for arch_dict in checkpoint["architectures"]
        ]

        # Reconstruct results
        results = [
            EvaluationResult(**res_dict)
            for res_dict in checkpoint["results"]
        ]

        instance = cls(
            architectures=architectures,
            results=results,
            search_time=checkpoint["search_time"],
            n_evaluations=checkpoint["n_evaluations"],
            metadata=checkpoint.get("metadata", {}),
        )

        logger.info(f"Loaded checkpoint from {filepath}")
        logger.info(f"Resumed {instance.n_evaluations} evaluations")

        return instance

    def __len__(self) -> int:
        """Get number of evaluations tracked.

        Returns:
            Number of evaluation results
        """
        return len(self.results)


class RandomSearch(Searcher):
    """Random search strategy for Neural Architecture Search.

    This strategy samples architectures uniformly at random from the search space
    and evaluates each one. It's simple, parallelizable, and often surprisingly
    effective as a baseline.

    The random search:
    1. Samples a random architecture from the search space
    2. Evaluates it using the evaluator
    3. Tracks all results and identifies the best architecture
    4. Continues until budget is exhausted

    Args:
        search_space: SearchSpace defining valid architectures (default: SearchSpace())
        evaluator: ArchitectureEvaluator for scoring architectures (default: None)
        budget: Optional ComputeBudget for limiting search (default: None)
        seed: Random seed for reproducibility (default: None)

    Attributes:
        seed: Random seed used for reproducibility

    Example:
        >>> searcher = RandomSearch(seed=42)
        >>> result = searcher.search(data, n_iters=200)
        >>> top_3 = result.get_top_k(3)
    """

    def __init__(
        self,
        search_space: Optional[SearchSpace] = None,
        evaluator: Optional[ArchitectureEvaluator] = None,
        budget: Optional[ComputeBudget] = None,
        seed: Optional[int] = None,
    ):
        """Initialize random search strategy."""
        super().__init__(search_space, evaluator, budget)
        self.seed = seed

        # Set random seed if provided
        if seed is not None:
            random.seed(seed)

    def search(
        self,
        data: Any,
        n_iters: int = 200,
        batch_size: int = 32,
    ) -> SearchResult:
        """Run random architecture search.

        Samples architectures uniformly at random from the search space and
        evaluates each one until the budget is exhausted or no more architectures
        can be evaluated.

        Args:
            data: Training data tensor for evaluation
            n_iters: Number of training iterations per architecture evaluation
            batch_size: Batch size for training

        Returns:
            SearchResult containing all evaluated architectures and the best one

        Raises:
            ValueError: If evaluator is not set

        Example:
            >>> from neuromanifold_gpt.nas.evaluator import ArchitectureEvaluator, ComputeBudget
            >>> evaluator = ArchitectureEvaluator(vocab_size=65, device="cuda")
            >>> budget = ComputeBudget(max_evaluations=20)
            >>> searcher = RandomSearch(evaluator=evaluator, budget=budget)
            >>> result = searcher.search(train_data, n_iters=200)
        """
        if self.evaluator is None:
            raise ValueError("Evaluator must be set before calling search()")

        logger.info("Starting random architecture search")
        logger.info(f"Search space size: {self.search_space.get_search_space_size():,}")

        # Initialize search state
        self.start_time = time.time()
        self.evaluated_architectures = []
        self.evaluation_results = []

        # Start budget tracking if provided
        if self.budget is not None:
            self.budget.start()
            logger.info(f"Budget: {self.budget.max_evaluations} max evaluations")

        # Determine number of architectures to evaluate
        if self.budget is not None and self.budget.max_evaluations is not None:
            n_architectures = self.budget.max_evaluations
        else:
            # Default to 50 random samples if no budget specified
            n_architectures = 50
            logger.info(f"No budget specified, defaulting to {n_architectures} evaluations")

        # Sample and evaluate architectures
        for i in range(n_architectures):
            # Check budget before evaluation
            if self.budget is not None:
                should_stop, reason = self.budget.should_stop()
                if should_stop:
                    logger.info(f"Early stopping: {reason}")
                    break

            # Sample random architecture
            architecture = self.search_space.sample()
            architecture.architecture_id = f"random_{i:04d}"
            architecture.search_iteration = i

            logger.info(f"Evaluating architecture {i+1}/{n_architectures}: {architecture.architecture_id}")

            # Evaluate architecture
            result = self.evaluator.evaluate(
                architecture,
                data,
                n_iters=n_iters,
                batch_size=batch_size,
            )

            # Store results
            self.evaluated_architectures.append(architecture)
            self.evaluation_results.append(result)

            # Log result
            if result.success:
                logger.info(
                    f"  Loss: {result.final_loss:.4f}, "
                    f"PPL: {result.perplexity:.2f}, "
                    f"Params: {result.n_params:,}"
                )

                # Update budget
                if self.budget is not None:
                    self.budget.update(result.perplexity)
            else:
                logger.warning(f"  Evaluation failed: {result.error_message}")

                # Update budget even on failure
                if self.budget is not None:
                    self.budget.update(float('inf'))

        # Calculate search time
        search_time = time.time() - self.start_time

        # Find best architecture
        best = self.get_best_architecture()
        if best is None:
            raise RuntimeError("No successful architecture evaluations")

        best_architecture, best_result = best

        logger.info(f"\n=== Random Search Complete ===")
        logger.info(f"Total evaluations: {len(self.evaluation_results)}")
        logger.info(f"Successful: {sum(1 for r in self.evaluation_results if r.success)}")
        logger.info(f"Failed: {sum(1 for r in self.evaluation_results if not r.success)}")
        logger.info(f"Search time: {search_time:.1f}s")
        logger.info(f"Best architecture: {best_architecture.architecture_id}")
        logger.info(f"  Perplexity: {best_result.perplexity:.2f}")
        logger.info(f"  Loss: {best_result.final_loss:.4f}")
        logger.info(f"  Parameters: {best_result.n_params:,}")

        # Create search result
        return SearchResult(
            architectures=self.evaluated_architectures,
            results=self.evaluation_results,
            best_architecture=best_architecture,
            best_result=best_result,
            search_time=search_time,
            n_evaluations=len(self.evaluation_results),
            search_space_size=self.search_space.get_search_space_size(),
            strategy_name=self.get_strategy_name(),
            metadata={
                "seed": self.seed,
                "n_successful": sum(1 for r in self.evaluation_results if r.success),
                "n_failed": sum(1 for r in self.evaluation_results if not r.success),
            },
        )

    def get_strategy_name(self) -> str:
        """Get the name of this search strategy.

        Returns:
            "Random Search"
        """
        return "Random Search"
