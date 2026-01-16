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
from typing import Any, Optional
from loguru import logger

from neuromanifold_gpt.nas.search_space import SearchSpace, ArchitectureConfig
from neuromanifold_gpt.nas.evaluator import ArchitectureEvaluator, ComputeBudget
from neuromanifold_gpt.nas.searcher import Searcher, SearchResult


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
