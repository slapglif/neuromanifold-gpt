"""Architecture evaluation for Neural Architecture Search.

This module provides quick training and evaluation of neural architectures for NAS.
The evaluator runs short training sessions (configurable iterations) to estimate
architecture quality using metrics like loss, perplexity, and training speed.

Key features:
- Quick training for fast architecture evaluation
- Perplexity-based scoring
- Parameter counting and speed measurement
- Memory-efficient cleanup after evaluation
- Configurable training budget and batch size
- Compute budget tracking with early stopping

Example:
    >>> from neuromanifold_gpt.nas.search_space import SearchSpace
    >>> from neuromanifold_gpt.nas.evaluator import ArchitectureEvaluator, ComputeBudget
    >>>
    >>> search_space = SearchSpace()
    >>> evaluator = ArchitectureEvaluator(vocab_size=65, device="cuda")
    >>>
    >>> # Single architecture evaluation
    >>> arch = search_space.sample()
    >>> result = evaluator.evaluate(arch, n_iters=200)
    >>> print(f"Perplexity: {result.perplexity:.2f}")
    >>>
    >>> # Multiple architectures with budget
    >>> architectures = [search_space.sample() for _ in range(100)]
    >>> budget = ComputeBudget(max_evaluations=20, patience=5)
    >>> results = evaluator.compare_architectures(architectures, data, budget=budget)
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Tuple
import time
import math
from loguru import logger


@dataclass
class ComputeBudget:
    """Compute budget for Neural Architecture Search.

    Tracks evaluation budget and enables early stopping based on various criteria:
    - Maximum number of evaluations
    - Maximum wall-clock time
    - Target performance threshold
    - Patience for no improvement

    Attributes:
        max_evaluations: Maximum number of architectures to evaluate (None = unlimited)
        max_time_seconds: Maximum wall-clock time in seconds (None = unlimited)
        min_perplexity_target: Stop if perplexity reaches this target (None = no target)
        patience: Number of evaluations without improvement before stopping (None = no patience)
        evaluations_done: Number of evaluations completed so far
        time_spent: Total time spent in seconds
        best_perplexity: Best perplexity seen so far
        iterations_since_improvement: Iterations since last improvement
        start_time: When the search started (set automatically)

    Example:
        >>> budget = ComputeBudget(max_evaluations=100, max_time_seconds=3600)
        >>> budget.start()
        >>> # ... evaluate architectures ...
        >>> if budget.should_stop():
        ...     print("Budget exhausted")
    """
    max_evaluations: Optional[int] = None
    max_time_seconds: Optional[float] = None
    min_perplexity_target: Optional[float] = None
    patience: Optional[int] = None

    # Tracking fields (not set in constructor)
    evaluations_done: int = field(default=0, init=False)
    time_spent: float = field(default=0.0, init=False)
    best_perplexity: float = field(default=float('inf'), init=False)
    iterations_since_improvement: int = field(default=0, init=False)
    start_time: Optional[float] = field(default=None, init=False)

    def start(self) -> None:
        """Start the budget timer."""
        self.start_time = time.time()
        self.evaluations_done = 0
        self.time_spent = 0.0
        self.best_perplexity = float('inf')
        self.iterations_since_improvement = 0

    def update(self, perplexity: float) -> None:
        """Update budget after an evaluation.

        Args:
            perplexity: Perplexity from the latest evaluation
        """
        self.evaluations_done += 1

        if self.start_time is not None:
            self.time_spent = time.time() - self.start_time

        # Track improvement
        if perplexity < self.best_perplexity:
            self.best_perplexity = perplexity
            self.iterations_since_improvement = 0
        else:
            self.iterations_since_improvement += 1

    def should_stop(self) -> Tuple[bool, Optional[str]]:
        """Check if search should stop based on budget.

        Returns:
            Tuple of (should_stop, reason)
            - should_stop: True if any stopping criterion is met
            - reason: Human-readable reason for stopping (None if continuing)
        """
        # Check max evaluations
        if self.max_evaluations is not None and self.evaluations_done >= self.max_evaluations:
            return True, f"Reached max evaluations ({self.max_evaluations})"

        # Check max time
        if self.max_time_seconds is not None and self.time_spent >= self.max_time_seconds:
            return True, f"Reached max time ({self.max_time_seconds:.1f}s)"

        # Check target perplexity
        if self.min_perplexity_target is not None and self.best_perplexity <= self.min_perplexity_target:
            return True, f"Reached target perplexity ({self.min_perplexity_target:.2f})"

        # Check patience
        if self.patience is not None and self.iterations_since_improvement >= self.patience:
            return True, f"No improvement for {self.patience} evaluations"

        return False, None

    def get_status(self) -> str:
        """Get a human-readable status string.

        Returns:
            Status string with evaluation count, time, and best perplexity
        """
        status_parts = [f"Evaluations: {self.evaluations_done}"]

        if self.max_evaluations is not None:
            status_parts[0] += f"/{self.max_evaluations}"

        if self.start_time is not None:
            time_str = f"Time: {self.time_spent:.1f}s"
            if self.max_time_seconds is not None:
                time_str += f"/{self.max_time_seconds:.1f}s"
            status_parts.append(time_str)

        if self.best_perplexity < float('inf'):
            status_parts.append(f"Best PPL: {self.best_perplexity:.2f}")

        return ", ".join(status_parts)


@dataclass
class EvaluationResult:
    """Result from evaluating a single architecture.

    Attributes:
        architecture_id: Unique identifier for the architecture
        final_loss: Final training loss after n_iters
        perplexity: Perplexity score (exp(loss))
        n_params: Total number of trainable parameters
        time_per_iter_ms: Average time per training iteration in milliseconds
        n_iters: Number of training iterations completed
        success: Whether evaluation completed successfully
        error_message: Error message if evaluation failed (None if success)
    """
    architecture_id: Optional[str]
    final_loss: float
    perplexity: float
    n_params: int
    time_per_iter_ms: float
    n_iters: int
    success: bool = True
    error_message: Optional[str] = None


class ArchitectureEvaluator:
    """Evaluator for neural architectures in NAS.

    This class provides quick training and evaluation of architectures to estimate
    their quality. It's designed to be fast enough for NAS while providing reliable
    performance estimates.

    The evaluator:
    - Creates a model from an ArchitectureConfig
    - Trains it for a configurable number of iterations
    - Tracks loss, perplexity, parameters, and speed
    - Cleans up GPU memory after evaluation

    Args:
        vocab_size: Vocabulary size for the dataset
        block_size: Maximum sequence length (default: 256, shorter for faster eval)
        device: Device to train on ("cuda" or "cpu")
        learning_rate: Learning rate for optimizer (default: 3e-4)
        weight_decay: Weight decay for optimizer (default: 0.1)
        grad_clip_norm: Gradient clipping norm (default: 1.0)

    Example:
        >>> evaluator = ArchitectureEvaluator(vocab_size=65, device="cuda")
        >>> result = evaluator.evaluate(arch, data=train_data, n_iters=200)
        >>> if result.success:
        ...     print(f"Perplexity: {result.perplexity:.2f}")
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int = 256,
        device: str = "cuda",
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        grad_clip_norm: float = 1.0,
    ):
        """Initialize the architecture evaluator."""
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip_norm = grad_clip_norm

    def evaluate(
        self,
        architecture: "ArchitectureConfig",
        data: torch.Tensor,
        n_iters: int = 200,
        batch_size: int = 32,
    ) -> EvaluationResult:
        """Evaluate an architecture by quick training.

        Args:
            architecture: ArchitectureConfig to evaluate
            data: Training data tensor (tokenized text)
            n_iters: Number of training iterations (default: 200)
            batch_size: Batch size for training (default: 32)

        Returns:
            EvaluationResult with performance metrics

        Note:
            This method handles errors gracefully and returns an EvaluationResult
            with success=False if evaluation fails.
        """
        try:
            # Validate architecture
            is_valid, error_msg = architecture.validate()
            if not is_valid:
                return EvaluationResult(
                    architecture_id=architecture.architecture_id,
                    final_loss=float('inf'),
                    perplexity=float('inf'),
                    n_params=0,
                    time_per_iter_ms=0.0,
                    n_iters=0,
                    success=False,
                    error_message=f"Invalid architecture: {error_msg}",
                )

            # Convert to NeuroManifoldConfig
            config = architecture.to_config(
                vocab_size=self.vocab_size,
                block_size=self.block_size,
            )

            # Import here to avoid circular dependency
            from neuromanifold_gpt.model.gpt import NeuroManifoldGPT

            # Create model
            model = NeuroManifoldGPT(config).to(self.device)
            n_params = sum(p.numel() for p in model.parameters())

            # Configure optimizer
            optimizer = model.configure_optimizers(
                weight_decay=self.weight_decay,
                learning_rate=self.learning_rate,
                device_type=self.device
            )

            # Training loop
            model.train()
            start_time = time.time()
            final_loss = 0.0

            for it in range(n_iters):
                # Get batch
                x, y = self._get_batch(data, batch_size)

                # Forward pass
                logits, loss, _ = model(x, y)

                # Backward pass
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip_norm)
                optimizer.step()

                # Track final loss
                if it == n_iters - 1:
                    final_loss = loss.item()

            # Calculate metrics
            elapsed = time.time() - start_time
            time_per_iter = elapsed / n_iters * 1000  # Convert to ms

            # Cap perplexity to avoid overflow
            perplexity = math.exp(min(final_loss, 10))

            # Cleanup
            del model
            del optimizer
            if self.device == "cuda":
                torch.cuda.empty_cache()

            return EvaluationResult(
                architecture_id=architecture.architecture_id,
                final_loss=final_loss,
                perplexity=perplexity,
                n_params=n_params,
                time_per_iter_ms=time_per_iter,
                n_iters=n_iters,
                success=True,
                error_message=None,
            )

        except Exception as e:
            # Log error and return failure result
            logger.error(f"Evaluation failed for architecture {architecture.architecture_id}: {e}")

            # Cleanup on error
            if self.device == "cuda":
                torch.cuda.empty_cache()

            return EvaluationResult(
                architecture_id=architecture.architecture_id,
                final_loss=float('inf'),
                perplexity=float('inf'),
                n_params=0,
                time_per_iter_ms=0.0,
                n_iters=0,
                success=False,
                error_message=str(e),
            )

    def _get_batch(
        self,
        data: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a random batch from the data.

        Args:
            data: Full dataset tensor
            batch_size: Number of sequences in batch

        Returns:
            Tuple of (input_ids, target_ids) tensors
        """
        # Random starting indices
        ix = torch.randint(len(data) - self.block_size, (batch_size,))

        # Stack sequences
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])

        return x.to(self.device), y.to(self.device)

    def compare_architectures(
        self,
        architectures: list,
        data: torch.Tensor,
        n_iters: int = 200,
        batch_size: int = 32,
        budget: Optional[ComputeBudget] = None,
    ) -> list[EvaluationResult]:
        """Evaluate multiple architectures and return results.

        Args:
            architectures: List of ArchitectureConfig objects to evaluate
            data: Training data tensor
            n_iters: Number of training iterations per architecture
            batch_size: Batch size for training
            budget: Optional ComputeBudget for early stopping

        Returns:
            List of EvaluationResult objects, sorted by perplexity (best first)

        Note:
            If a budget is provided, evaluation may stop early based on:
            - Maximum number of evaluations
            - Maximum wall-clock time
            - Target perplexity reached
            - No improvement patience

        Example:
            >>> architectures = [search_space.sample() for _ in range(10)]
            >>> budget = ComputeBudget(max_evaluations=5, patience=3)
            >>> results = evaluator.compare_architectures(architectures, data, budget=budget)
            >>> best = results[0]
            >>> print(f"Best architecture: {best.architecture_id}, PPL: {best.perplexity:.2f}")
        """
        results = []

        # Start budget tracking if provided
        if budget is not None:
            budget.start()

        for i, arch in enumerate(architectures):
            # Check budget before evaluation
            if budget is not None:
                should_stop, reason = budget.should_stop()
                if should_stop:
                    logger.info(f"Early stopping: {reason}")
                    logger.info(f"Budget status: {budget.get_status()}")
                    break

            logger.info(f"Evaluating architecture {i+1}/{len(architectures)}: {arch.architecture_id}")

            result = self.evaluate(arch, data, n_iters, batch_size)
            results.append(result)

            if result.success:
                logger.info(
                    f"  Loss: {result.final_loss:.4f}, "
                    f"PPL: {result.perplexity:.2f}, "
                    f"Params: {result.n_params:,}, "
                    f"Speed: {result.time_per_iter_ms:.1f}ms/iter"
                )

                # Update budget with result
                if budget is not None:
                    budget.update(result.perplexity)
                    if i < len(architectures) - 1:  # Don't log on last iteration
                        logger.debug(f"Budget status: {budget.get_status()}")
            else:
                logger.warning(f"  Failed: {result.error_message}")

                # Update budget even on failure (count as infinite perplexity)
                if budget is not None:
                    budget.update(float('inf'))

        # Final budget status
        if budget is not None:
            logger.info(f"Final budget status: {budget.get_status()}")

        # Sort by perplexity (lower is better), failed architectures at the end
        results.sort(key=lambda r: (not r.success, r.perplexity))

        return results
