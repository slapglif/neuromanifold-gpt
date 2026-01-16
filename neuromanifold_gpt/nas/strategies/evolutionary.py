"""Evolutionary search strategy for Neural Architecture Search.

This module implements an evolutionary algorithm for NAS using mutation and
crossover operations. The evolutionary approach maintains a population of
architectures and evolves them over generations using genetic operators.

The evolutionary search algorithm:
1. Initialize a population of random architectures
2. Evaluate all architectures in the population
3. Select the best architectures as parents
4. Generate offspring through mutation and crossover
5. Evaluate offspring and update population
6. Repeat until budget is exhausted

Mutation operations include:
- Changing layer count, embedding dimensions, or attention heads
- Toggling component flags (MHC, MLA, MoE, KAN)
- Adjusting hyperparameters (dropout, thresholds, etc.)

Crossover operations:
- Uniform crossover: randomly select parameters from two parents
- Single-point crossover: split at a random point and swap

Example:
    >>> from neuromanifold_gpt.nas.search_space import SearchSpace
    >>> from neuromanifold_gpt.nas.evaluator import ArchitectureEvaluator, ComputeBudget
    >>> from neuromanifold_gpt.nas.strategies.evolutionary import EvolutionarySearch
    >>>
    >>> search_space = SearchSpace()
    >>> evaluator = ArchitectureEvaluator(vocab_size=65, device="cuda")
    >>> budget = ComputeBudget(max_evaluations=100)
    >>>
    >>> searcher = EvolutionarySearch(
    ...     search_space=search_space,
    ...     evaluator=evaluator,
    ...     budget=budget,
    ...     population_size=20,
    ...     tournament_size=3,
    ...     mutation_rate=0.3,
    ... )
    >>> result = searcher.search(data=train_data, n_iters=200)
    >>>
    >>> print(f"Best architecture: {result.best_result.perplexity:.2f} perplexity")
    >>> print(f"Evolved through {result.metadata['n_generations']} generations")
"""

import time
import random
from typing import Any, Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
from loguru import logger

from neuromanifold_gpt.nas.search_space import SearchSpace, ArchitectureConfig
from neuromanifold_gpt.nas.evaluator import ArchitectureEvaluator, ComputeBudget, EvaluationResult
from neuromanifold_gpt.nas.searcher import Searcher, SearchResult


class EvolutionarySearch(Searcher):
    """Evolutionary search strategy for Neural Architecture Search.

    This strategy uses evolutionary algorithms with mutation and crossover
    to explore the architecture search space. It maintains a population of
    architectures and evolves them over generations using selection, mutation,
    and crossover operations.

    The evolutionary approach is effective for:
    - Exploring complex search spaces with dependencies
    - Balancing exploration and exploitation
    - Finding multiple good architectures (population diversity)
    - Incremental improvement through generations

    Args:
        search_space: SearchSpace defining valid architectures (default: SearchSpace())
        evaluator: ArchitectureEvaluator for scoring architectures (default: None)
        budget: Optional ComputeBudget for limiting search (default: None)
        population_size: Size of the population (default: 20)
        tournament_size: Number of individuals in tournament selection (default: 3)
        mutation_rate: Probability of mutation (default: 0.3)
        crossover_rate: Probability of crossover (default: 0.5)
        elitism_ratio: Fraction of top individuals to keep unchanged (default: 0.1)
        seed: Random seed for reproducibility (default: None)

    Attributes:
        population_size: Size of the architecture population
        tournament_size: Size for tournament selection
        mutation_rate: Probability of applying mutation
        crossover_rate: Probability of applying crossover
        elitism_ratio: Fraction of elite individuals preserved
        seed: Random seed used for reproducibility
        population: Current population of architectures
        population_results: Evaluation results for current population

    Example:
        >>> searcher = EvolutionarySearch(
        ...     population_size=30,
        ...     mutation_rate=0.4,
        ...     seed=42
        ... )
        >>> result = searcher.search(data, n_iters=200)
        >>> print(f"Evolved {result.metadata['n_generations']} generations")
    """

    def __init__(
        self,
        search_space: Optional[SearchSpace] = None,
        evaluator: Optional[ArchitectureEvaluator] = None,
        budget: Optional[ComputeBudget] = None,
        population_size: int = 20,
        tournament_size: int = 3,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.5,
        elitism_ratio: float = 0.1,
        seed: Optional[int] = None,
    ):
        """Initialize evolutionary search strategy."""
        super().__init__(search_space, evaluator, budget)

        self.population_size = population_size
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_ratio = elitism_ratio
        self.seed = seed

        # Evolution state
        self.population: List[ArchitectureConfig] = []
        self.population_results: List[EvaluationResult] = []
        self.generation = 0

        # Set random seed if provided
        if seed is not None:
            random.seed(seed)

    def search(
        self,
        data: Any,
        n_iters: int = 200,
        batch_size: int = 32,
    ) -> SearchResult:
        """Run evolutionary architecture search.

        Evolves a population of architectures over multiple generations using
        selection, mutation, and crossover operations. The algorithm maintains
        population diversity while gradually improving architecture quality.

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
            >>> budget = ComputeBudget(max_evaluations=100)
            >>> searcher = EvolutionarySearch(
            ...     evaluator=evaluator,
            ...     budget=budget,
            ...     population_size=20
            ... )
            >>> result = searcher.search(train_data, n_iters=200)
        """
        if self.evaluator is None:
            raise ValueError("Evaluator must be set before calling search()")

        logger.info("Starting evolutionary architecture search")
        logger.info(f"Search space size: {self.search_space.get_search_space_size():,}")
        logger.info(f"Population size: {self.population_size}")
        logger.info(f"Mutation rate: {self.mutation_rate:.2f}")
        logger.info(f"Crossover rate: {self.crossover_rate:.2f}")

        # Initialize search state
        self.start_time = time.time()
        self.evaluated_architectures = []
        self.evaluation_results = []
        self.generation = 0

        # Start budget tracking if provided
        if self.budget is not None:
            self.budget.start()
            logger.info(f"Budget: {self.budget.max_evaluations} max evaluations")

        # Initialize population with random architectures
        logger.info(f"Initializing population with {self.population_size} random architectures")
        self._initialize_population()

        # Evaluate initial population
        self._evaluate_population(data, n_iters, batch_size)

        # Evolve population until budget is exhausted
        while True:
            # Check budget
            if self.budget is not None:
                should_stop, reason = self.budget.should_stop()
                if should_stop:
                    logger.info(f"Early stopping: {reason}")
                    break

            # Check if we can evaluate at least one more offspring
            if self.budget is not None:
                remaining = self.budget.max_evaluations - len(self.evaluated_architectures)
                if remaining <= 0:
                    logger.info("Budget exhausted")
                    break

            # Evolve to next generation
            self.generation += 1
            logger.info(f"\n=== Generation {self.generation} ===")

            # Create new population through evolution
            offspring = self._create_offspring()

            # Evaluate offspring
            self._evaluate_offspring(offspring, data, n_iters, batch_size)

            # Update population with best individuals (elitism + offspring)
            self._update_population()

            # Log generation statistics
            self._log_generation_stats()

        # Calculate search time
        search_time = time.time() - self.start_time

        # Find best architecture
        best = self.get_best_architecture()
        if best is None:
            raise RuntimeError("No successful architecture evaluations")

        best_architecture, best_result = best

        logger.info(f"\n=== Evolutionary Search Complete ===")
        logger.info(f"Total generations: {self.generation}")
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
                "population_size": self.population_size,
                "n_generations": self.generation,
                "mutation_rate": self.mutation_rate,
                "crossover_rate": self.crossover_rate,
                "elitism_ratio": self.elitism_ratio,
                "n_successful": sum(1 for r in self.evaluation_results if r.success),
                "n_failed": sum(1 for r in self.evaluation_results if not r.success),
            },
        )

    def _initialize_population(self) -> None:
        """Initialize population with random architectures."""
        self.population = []
        for i in range(self.population_size):
            architecture = self.search_space.sample()
            architecture.architecture_id = f"gen0_ind{i:03d}"
            architecture.search_iteration = 0
            self.population.append(architecture)

    def _evaluate_population(
        self,
        data: Any,
        n_iters: int,
        batch_size: int,
    ) -> None:
        """Evaluate all architectures in the current population.

        Args:
            data: Training data for evaluation
            n_iters: Number of training iterations
            batch_size: Batch size for training
        """
        self.population_results = []

        for i, architecture in enumerate(self.population):
            # Check budget before evaluation
            if self.budget is not None:
                should_stop, reason = self.budget.should_stop()
                if should_stop:
                    logger.info(f"Budget exhausted during population evaluation")
                    # Remove unevaluated architectures
                    self.population = self.population[:i]
                    break

            logger.info(f"Evaluating {i+1}/{len(self.population)}: {architecture.architecture_id}")

            # Evaluate architecture
            result = self.evaluator.evaluate(
                architecture,
                data,
                n_iters=n_iters,
                batch_size=batch_size,
            )

            # Store results
            self.population_results.append(result)
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

    def _create_offspring(self) -> List[ArchitectureConfig]:
        """Create offspring through selection, crossover, and mutation.

        Returns:
            List of offspring architectures
        """
        # Calculate number of elite individuals to preserve
        n_elite = max(1, int(self.population_size * self.elitism_ratio))

        # Calculate number of offspring to generate
        n_offspring = self.population_size - n_elite

        offspring = []
        offspring_id = 0

        while len(offspring) < n_offspring:
            # Check budget
            if self.budget is not None:
                remaining = self.budget.max_evaluations - len(self.evaluated_architectures)
                if remaining <= len(offspring):
                    # Can't evaluate more offspring
                    break

            # Select parents using tournament selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()

            # Apply crossover with probability
            if random.random() < self.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                # Just copy parent1 if no crossover
                child = self._copy_architecture(parent1)

            # Apply mutation with probability
            if random.random() < self.mutation_rate:
                child = self._mutate(child)

            # Set metadata
            child.architecture_id = f"gen{self.generation}_off{offspring_id:03d}"
            child.search_iteration = self.generation
            child.parent_id = f"{parent1.architecture_id}_{parent2.architecture_id}"

            offspring.append(child)
            offspring_id += 1

        return offspring

    def _tournament_selection(self) -> ArchitectureConfig:
        """Select an architecture using tournament selection.

        Returns:
            Selected architecture
        """
        # Filter successful evaluations
        successful_indices = [
            i for i, res in enumerate(self.population_results)
            if res.success
        ]

        if not successful_indices:
            # If no successful evaluations, return random architecture
            return random.choice(self.population)

        # Randomly select tournament_size individuals
        tournament_size = min(self.tournament_size, len(successful_indices))
        tournament_indices = random.sample(successful_indices, tournament_size)

        # Select best individual from tournament (lowest perplexity)
        best_idx = min(
            tournament_indices,
            key=lambda i: self.population_results[i].perplexity
        )

        return self.population[best_idx]

    def _crossover(
        self,
        parent1: ArchitectureConfig,
        parent2: ArchitectureConfig,
    ) -> ArchitectureConfig:
        """Perform uniform crossover between two parent architectures.

        Args:
            parent1: First parent architecture
            parent2: Second parent architecture

        Returns:
            Child architecture with mixed parameters
        """
        # Create child by randomly selecting parameters from parents
        child = ArchitectureConfig()

        # Randomly select each parameter from one of the parents
        child.n_layer = random.choice([parent1.n_layer, parent2.n_layer])
        child.n_embd = random.choice([parent1.n_embd, parent2.n_embd])

        # Ensure n_heads is compatible with n_embd
        valid_n_heads = [h for h in [parent1.n_heads, parent2.n_heads] if child.n_embd % h == 0]
        if valid_n_heads:
            child.n_heads = random.choice(valid_n_heads)
        else:
            # Fall back to valid choice from search space
            valid_heads = [h for h in self.search_space.n_heads_choices if child.n_embd % h == 0]
            child.n_heads = random.choice(valid_heads)

        # Attention configuration
        child.attention_type = random.choice([parent1.attention_type, parent2.attention_type])
        child.use_qk_norm = random.choice([parent1.use_qk_norm, parent2.use_qk_norm])

        # Component choices
        child.use_mhc = random.choice([parent1.use_mhc, parent2.use_mhc])
        child.use_mla = random.choice([parent1.use_mla, parent2.use_mla])
        child.use_moe = random.choice([parent1.use_moe, parent2.use_moe])
        child.use_kan = random.choice([parent1.use_kan, parent2.use_kan])

        # KAN configuration
        child.kan_type = random.choice([parent1.kan_type, parent2.kan_type])
        child.kan_num_centers = random.choice([parent1.kan_num_centers, parent2.kan_num_centers])

        # FHN dynamics
        child.fhn_threshold = random.choice([parent1.fhn_threshold, parent2.fhn_threshold])
        child.fhn_tau = random.choice([parent1.fhn_tau, parent2.fhn_tau])
        child.use_fhn_parallel = random.choice([parent1.use_fhn_parallel, parent2.use_fhn_parallel])

        # Manifold projection
        child.manifold_dim = random.choice([parent1.manifold_dim, parent2.manifold_dim])
        child.n_eigenvectors = random.choice([parent1.n_eigenvectors, parent2.n_eigenvectors])
        child.use_multiscale_manifold = random.choice([parent1.use_multiscale_manifold, parent2.use_multiscale_manifold])

        # Regularization
        child.dropout = random.choice([parent1.dropout, parent2.dropout])

        return child

    def _mutate(self, architecture: ArchitectureConfig) -> ArchitectureConfig:
        """Apply mutation to an architecture.

        Randomly modifies one or more parameters of the architecture.

        Args:
            architecture: Architecture to mutate

        Returns:
            Mutated architecture
        """
        # Create a copy to mutate
        mutated = self._copy_architecture(architecture)

        # Randomly select which parameter to mutate
        mutation_choices = [
            "n_layer", "n_embd", "n_heads", "attention_type", "use_qk_norm",
            "use_mhc", "use_mla", "use_moe", "use_kan",
            "kan_type", "kan_num_centers",
            "fhn_threshold", "fhn_tau", "use_fhn_parallel",
            "manifold_dim", "n_eigenvectors", "use_multiscale_manifold",
            "dropout",
            "use_sdr", "sdr_size", "sdr_sparsity", "engram_capacity", "engram_threshold"
        ]

        # Mutate a random parameter
        param_to_mutate = random.choice(mutation_choices)

        if param_to_mutate == "n_layer":
            mutated.n_layer = random.choice(self.search_space.n_layer_choices)
        elif param_to_mutate == "n_embd":
            mutated.n_embd = random.choice(self.search_space.n_embd_choices)
            # Ensure n_heads is still valid
            valid_heads = [h for h in self.search_space.n_heads_choices if mutated.n_embd % h == 0]
            mutated.n_heads = random.choice(valid_heads)
        elif param_to_mutate == "n_heads":
            valid_heads = [h for h in self.search_space.n_heads_choices if mutated.n_embd % h == 0]
            mutated.n_heads = random.choice(valid_heads)
        elif param_to_mutate == "attention_type":
            mutated.attention_type = random.choice(self.search_space.attention_type_choices)
        elif param_to_mutate == "use_qk_norm":
            mutated.use_qk_norm = random.choice(self.search_space.use_qk_norm_choices)
        elif param_to_mutate == "use_mhc":
            mutated.use_mhc = random.choice(self.search_space.use_mhc_choices)
        elif param_to_mutate == "use_mla":
            mutated.use_mla = random.choice(self.search_space.use_mla_choices)
        elif param_to_mutate == "use_moe":
            mutated.use_moe = random.choice(self.search_space.use_moe_choices)
        elif param_to_mutate == "use_kan":
            mutated.use_kan = random.choice(self.search_space.use_kan_choices)
        elif param_to_mutate == "kan_type":
            mutated.kan_type = random.choice(self.search_space.kan_type_choices)
        elif param_to_mutate == "kan_num_centers":
            mutated.kan_num_centers = random.choice(self.search_space.kan_num_centers_choices)
        elif param_to_mutate == "fhn_threshold":
            mutated.fhn_threshold = random.uniform(*self.search_space.fhn_threshold_range)
        elif param_to_mutate == "fhn_tau":
            mutated.fhn_tau = random.choice(self.search_space.fhn_tau_choices)
        elif param_to_mutate == "use_fhn_parallel":
            mutated.use_fhn_parallel = random.choice(self.search_space.use_fhn_parallel_choices)
        elif param_to_mutate == "manifold_dim":
            mutated.manifold_dim = random.choice(self.search_space.manifold_dim_choices)
        elif param_to_mutate == "n_eigenvectors":
            mutated.n_eigenvectors = random.choice(self.search_space.n_eigenvectors_choices)
        elif param_to_mutate == "use_multiscale_manifold":
            mutated.use_multiscale_manifold = random.choice(self.search_space.use_multiscale_manifold_choices)
        elif param_to_mutate == "dropout":
            mutated.dropout = random.uniform(*self.search_space.dropout_range)
        elif param_to_mutate == "use_sdr":
            mutated.use_sdr = random.choice(self.search_space.use_sdr_choices)
        elif param_to_mutate == "sdr_size":
            mutated.sdr_size = random.choice(self.search_space.sdr_size_choices)
        elif param_to_mutate == "sdr_sparsity":
            mutated.sdr_sparsity = random.choice(self.search_space.sdr_sparsity_choices)
        elif param_to_mutate == "engram_capacity":
            mutated.engram_capacity = random.choice(self.search_space.engram_capacity_choices)
        elif param_to_mutate == "engram_threshold":
            mutated.engram_threshold = random.choice(self.search_space.engram_threshold_choices)

        return mutated

    def _copy_architecture(self, architecture: ArchitectureConfig) -> ArchitectureConfig:
        """Create a deep copy of an architecture.

        Args:
            architecture: Architecture to copy

        Returns:
            Copy of the architecture
        """
        return ArchitectureConfig(
            n_layer=architecture.n_layer,
            n_embd=architecture.n_embd,
            n_heads=architecture.n_heads,
            attention_type=architecture.attention_type,
            use_qk_norm=architecture.use_qk_norm,
            use_mhc=architecture.use_mhc,
            use_mla=architecture.use_mla,
            use_moe=architecture.use_moe,
            use_kan=architecture.use_kan,
            kan_type=architecture.kan_type,
            kan_num_centers=architecture.kan_num_centers,
            fhn_threshold=architecture.fhn_threshold,
            fhn_tau=architecture.fhn_tau,
            use_fhn_parallel=architecture.use_fhn_parallel,
            manifold_dim=architecture.manifold_dim,
            n_eigenvectors=architecture.n_eigenvectors,
            use_multiscale_manifold=architecture.use_multiscale_manifold,
            dropout=architecture.dropout,
            use_sdr=architecture.use_sdr,
            sdr_size=architecture.sdr_size,
            sdr_sparsity=architecture.sdr_sparsity,
            engram_capacity=architecture.engram_capacity,
            engram_threshold=architecture.engram_threshold,
        )

    def _evaluate_offspring(
        self,
        offspring: List[ArchitectureConfig],
        data: Any,
        n_iters: int,
        batch_size: int,
    ) -> List[EvaluationResult]:
        """Evaluate offspring architectures.

        Args:
            offspring: List of offspring architectures to evaluate
            data: Training data for evaluation
            n_iters: Number of training iterations
            batch_size: Batch size for training

        Returns:
            List of evaluation results
        """
        offspring_results = []

        for i, architecture in enumerate(offspring):
            # Check budget before evaluation
            if self.budget is not None:
                should_stop, reason = self.budget.should_stop()
                if should_stop:
                    logger.info(f"Budget exhausted during offspring evaluation")
                    break

            logger.info(f"Evaluating offspring {i+1}/{len(offspring)}: {architecture.architecture_id}")

            # Evaluate architecture
            result = self.evaluator.evaluate(
                architecture,
                data,
                n_iters=n_iters,
                batch_size=batch_size,
            )

            # Store results
            offspring_results.append(result)
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

        return offspring_results

    def _update_population(self) -> None:
        """Update population with best individuals (elitism + offspring selection).

        Uses elitism to preserve top performers and diversity-based selection
        to maintain population diversity.
        """
        # Calculate number of elite individuals to preserve
        n_elite = max(1, int(self.population_size * self.elitism_ratio))

        # Separate current population and new offspring
        offspring = [
            arch for arch in self.evaluated_architectures
            if arch.search_iteration == self.generation
        ]
        offspring_results = [
            res for arch, res in zip(
                self.evaluated_architectures,
                self.evaluation_results
            )
            if arch.search_iteration == self.generation
        ]

        # Sort current population by perplexity (best first)
        pop_pairs = list(zip(self.population, self.population_results))
        successful_pop = [(arch, res) for arch, res in pop_pairs if res.success]
        failed_pop = [(arch, res) for arch, res in pop_pairs if not res.success]
        successful_pop.sort(key=lambda x: x[1].perplexity)

        # Preserve elite individuals from current population
        elite_pairs = successful_pop[:n_elite]
        logger.info(f"Preserving {len(elite_pairs)} elite individuals")

        # Combine remaining population with offspring
        remaining_pop = successful_pop[n_elite:] + failed_pop
        offspring_pairs = list(zip(offspring, offspring_results))

        # Sort offspring by fitness
        successful_offspring = [(arch, res) for arch, res in offspring_pairs if res.success]
        failed_offspring = [(arch, res) for arch, res in offspring_pairs if not res.success]
        successful_offspring.sort(key=lambda x: x[1].perplexity)

        # Combine non-elite candidates
        candidate_pairs = successful_offspring + remaining_pop + failed_offspring

        # Select remaining individuals with diversity consideration
        n_remaining = self.population_size - len(elite_pairs)
        selected_pairs = self._select_diverse_individuals(
            candidate_pairs,
            elite_pairs,
            n_remaining
        )

        # Update population with elite + diverse selected
        all_selected = elite_pairs + selected_pairs
        self.population = [arch for arch, _ in all_selected]
        self.population_results = [res for _, res in all_selected]

    def _select_diverse_individuals(
        self,
        candidates: List[Tuple[ArchitectureConfig, EvaluationResult]],
        elite: List[Tuple[ArchitectureConfig, EvaluationResult]],
        n_select: int,
    ) -> List[Tuple[ArchitectureConfig, EvaluationResult]]:
        """Select individuals balancing fitness and diversity.

        Args:
            candidates: List of candidate (architecture, result) pairs
            elite: List of elite (architecture, result) pairs already selected
            n_select: Number of individuals to select

        Returns:
            List of selected (architecture, result) pairs
        """
        if len(candidates) <= n_select:
            return candidates

        selected = []
        remaining = candidates.copy()

        # Select individuals iteratively, balancing fitness and diversity
        for _ in range(n_select):
            if not remaining:
                break

            # Calculate diversity score for each candidate
            best_candidate_idx = 0
            best_score = float('-inf')

            for i, (cand_arch, cand_res) in enumerate(remaining):
                # Fitness score (lower perplexity is better, use negative)
                if cand_res.success:
                    fitness_score = -cand_res.perplexity
                else:
                    fitness_score = float('-inf')

                # Diversity score (distance from already selected)
                diversity_score = self._calculate_diversity_score(
                    cand_arch,
                    [arch for arch, _ in elite + selected]
                )

                # Combined score (weighted sum)
                # Weight fitness more heavily (0.7) vs diversity (0.3)
                combined_score = 0.7 * fitness_score + 0.3 * diversity_score

                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate_idx = i

            # Add best candidate to selected
            selected.append(remaining.pop(best_candidate_idx))

        return selected

    def _calculate_diversity_score(
        self,
        architecture: ArchitectureConfig,
        population: List[ArchitectureConfig],
    ) -> float:
        """Calculate diversity score for an architecture relative to a population.

        Higher score means more diverse (further from existing population).

        Args:
            architecture: Architecture to score
            population: Population to compare against

        Returns:
            Diversity score (higher = more diverse)
        """
        if not population:
            return 1.0

        # Calculate minimum distance to any individual in population
        distances = [
            self._architecture_distance(architecture, other)
            for other in population
        ]

        # Return minimum distance (most similar individual)
        return min(distances)

    def _architecture_distance(
        self,
        arch1: ArchitectureConfig,
        arch2: ArchitectureConfig,
    ) -> float:
        """Calculate normalized distance between two architectures.

        Uses weighted Hamming distance for discrete parameters and
        normalized Euclidean distance for continuous parameters.

        Args:
            arch1: First architecture
            arch2: Second architecture

        Returns:
            Normalized distance in [0, 1]
        """
        differences = 0
        total_params = 0

        # Discrete parameters (use Hamming distance)
        discrete_params = [
            ('n_layer', self.search_space.n_layer_choices),
            ('n_embd', self.search_space.n_embd_choices),
            ('n_heads', self.search_space.n_heads_choices),
            ('attention_type', self.search_space.attention_type_choices),
            ('use_qk_norm', self.search_space.use_qk_norm_choices),
            ('use_mhc', self.search_space.use_mhc_choices),
            ('use_mla', self.search_space.use_mla_choices),
            ('use_moe', self.search_space.use_moe_choices),
            ('use_kan', self.search_space.use_kan_choices),
            ('kan_type', self.search_space.kan_type_choices),
            ('kan_num_centers', self.search_space.kan_num_centers_choices),
            ('fhn_tau', self.search_space.fhn_tau_choices),
            ('use_fhn_parallel', self.search_space.use_fhn_parallel_choices),
            ('manifold_dim', self.search_space.manifold_dim_choices),
            ('n_eigenvectors', self.search_space.n_eigenvectors_choices),
            ('use_multiscale_manifold', self.search_space.use_multiscale_manifold_choices),
        ]

        for param_name, _ in discrete_params:
            val1 = getattr(arch1, param_name)
            val2 = getattr(arch2, param_name)
            if val1 != val2:
                differences += 1
            total_params += 1

        # Continuous parameters (use normalized absolute difference)
        continuous_params = [
            ('fhn_threshold', self.search_space.fhn_threshold_range),
            ('dropout', self.search_space.dropout_range),
        ]

        for param_name, value_range in continuous_params:
            val1 = getattr(arch1, param_name)
            val2 = getattr(arch2, param_name)
            range_size = value_range[1] - value_range[0]
            if range_size > 0:
                normalized_diff = abs(val1 - val2) / range_size
                differences += normalized_diff
            total_params += 1

        # Return normalized distance
        if total_params == 0:
            return 0.0
        return differences / total_params

    def _calculate_population_diversity(self) -> float:
        """Calculate average diversity of the current population.

        Returns:
            Average pairwise distance between all individuals
        """
        if len(self.population) < 2:
            return 0.0

        total_distance = 0.0
        n_pairs = 0

        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                total_distance += self._architecture_distance(
                    self.population[i],
                    self.population[j]
                )
                n_pairs += 1

        return total_distance / n_pairs if n_pairs > 0 else 0.0

    def _log_generation_stats(self) -> None:
        """Log statistics for the current generation."""
        # Get best architecture in current population
        successful_results = [res for res in self.population_results if res.success]

        if successful_results:
            best_ppl = min(res.perplexity for res in successful_results)
            avg_ppl = sum(res.perplexity for res in successful_results) / len(successful_results)

            # Calculate population diversity
            diversity = self._calculate_population_diversity()

            logger.info(f"Generation {self.generation} statistics:")
            logger.info(f"  Best perplexity: {best_ppl:.2f}")
            logger.info(f"  Average perplexity: {avg_ppl:.2f}")
            logger.info(f"  Population diversity: {diversity:.3f}")
            logger.info(f"  Successful evaluations: {len(successful_results)}/{len(self.population_results)}")
        else:
            logger.warning(f"Generation {self.generation}: No successful evaluations")

    def get_strategy_name(self) -> str:
        """Get the name of this search strategy.

        Returns:
            "Evolutionary Search"
        """
        return "Evolutionary Search"
