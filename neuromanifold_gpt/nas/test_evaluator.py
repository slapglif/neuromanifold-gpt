"""Unit tests for architecture evaluator.

Tests cover:
- ComputeBudget tracking and stop conditions
- Evaluation result structure
- Budget update logic
- Early stopping triggers
"""

import time

from neuromanifold_gpt.nas.evaluator import ComputeBudget, EvaluationResult


class TestComputeBudgetInitialization:
    """Test ComputeBudget initialization."""

    def test_default_initialization(self):
        """Test ComputeBudget initializes with defaults."""
        budget = ComputeBudget()
        assert budget.max_evaluations is None
        assert budget.max_time_seconds is None
        assert budget.min_perplexity_target is None
        assert budget.patience is None
        assert budget.evaluations_done == 0
        assert budget.best_perplexity == float("inf")

    def test_initialization_with_max_evaluations(self):
        """Test ComputeBudget initializes correctly with max_evaluations."""
        budget = ComputeBudget(max_evaluations=10)
        assert budget.max_evaluations == 10
        assert budget.evaluations_done == 0
        assert budget.best_perplexity == float("inf")

    def test_initialization_with_time_limit(self):
        """Test ComputeBudget initializes correctly with time limit."""
        budget = ComputeBudget(max_time_seconds=300)
        assert budget.max_time_seconds == 300
        assert budget.start_time is None  # Not started yet

    def test_initialization_with_patience(self):
        """Test ComputeBudget initializes correctly with patience."""
        budget = ComputeBudget(patience=5)
        assert budget.patience == 5
        assert budget.patience_counter == 0


class TestComputeBudgetTracking:
    """Test ComputeBudget tracking functionality."""

    def test_start_sets_start_time(self):
        """Test that start() sets the start time."""
        budget = ComputeBudget()
        budget.start()
        assert budget.start_time is not None
        assert isinstance(budget.start_time, float)

    def test_update_increments_counter(self):
        """Test that update increments evaluation counter."""
        budget = ComputeBudget()
        budget.start()
        assert budget.evaluations_done == 0
        budget.update(perplexity=10.0)
        assert budget.evaluations_done == 1
        budget.update(perplexity=9.0)
        assert budget.evaluations_done == 2

    def test_update_tracks_best_perplexity(self):
        """Test that update tracks best perplexity."""
        budget = ComputeBudget()
        budget.start()

        budget.update(perplexity=10.0)
        assert budget.best_perplexity == 10.0

        budget.update(perplexity=8.0)
        assert budget.best_perplexity == 8.0

        budget.update(perplexity=12.0)
        assert budget.best_perplexity == 8.0  # Should not increase

    def test_update_resets_patience_on_improvement(self):
        """Test that patience counter resets on improvement."""
        budget = ComputeBudget(patience=3)
        budget.start()

        budget.update(perplexity=10.0)
        assert budget.patience_counter == 0

        budget.update(perplexity=11.0)
        assert budget.patience_counter == 1

        budget.update(perplexity=9.0)  # Improvement
        assert budget.patience_counter == 0


class TestComputeBudgetStopConditions:
    """Test ComputeBudget stop conditions."""

    def test_should_stop_max_evaluations(self):
        """Test budget stops after max evaluations."""
        budget = ComputeBudget(max_evaluations=5)
        budget.start()

        for i in range(5):
            budget.update(perplexity=10.0)
            if i < 4:
                should_stop, _ = budget.should_stop()
                assert not should_stop

        should_stop, reason = budget.should_stop()
        assert should_stop
        assert "evaluations" in reason.lower()

    def test_should_stop_max_time(self):
        """Test budget stops after max time."""
        budget = ComputeBudget(max_time_seconds=0.1)
        budget.start()

        # Wait a bit
        time.sleep(0.15)

        should_stop, reason = budget.should_stop()
        assert should_stop
        assert "time" in reason.lower()

    def test_should_stop_target_perplexity(self):
        """Test budget stops when target perplexity reached."""
        budget = ComputeBudget(min_perplexity_target=5.0)
        budget.start()

        budget.update(perplexity=10.0)
        should_stop, _ = budget.should_stop()
        assert not should_stop

        budget.update(perplexity=4.5)
        should_stop, reason = budget.should_stop()
        assert should_stop
        assert "target" in reason.lower()

    def test_should_stop_patience(self):
        """Test patience-based early stopping."""
        budget = ComputeBudget(patience=3)
        budget.start()

        budget.update(perplexity=10.0)  # Best
        should_stop, _ = budget.should_stop()
        assert not should_stop

        budget.update(perplexity=11.0)  # No improvement (1)
        should_stop, _ = budget.should_stop()
        assert not should_stop

        budget.update(perplexity=12.0)  # No improvement (2)
        should_stop, _ = budget.should_stop()
        assert not should_stop

        budget.update(perplexity=13.0)  # No improvement (3)
        should_stop, reason = budget.should_stop()
        assert should_stop
        assert "patience" in reason.lower()

    def test_should_not_stop_with_no_limits(self):
        """Test that budget without limits never stops."""
        budget = ComputeBudget()
        budget.start()

        for _ in range(100):
            budget.update(perplexity=10.0)
            should_stop, _ = budget.should_stop()
            assert not should_stop


class TestComputeBudgetEdgeCases:
    """Test ComputeBudget edge cases."""

    def test_update_before_start(self):
        """Test that update works before start is called."""
        budget = ComputeBudget()
        # Should not crash
        budget.update(perplexity=10.0)
        assert budget.evaluations_done == 1

    def test_should_stop_before_start(self):
        """Test should_stop before start."""
        budget = ComputeBudget(max_evaluations=10)
        # Should return False before any evaluations
        should_stop, _ = budget.should_stop()
        assert not should_stop

    def test_multiple_stop_conditions(self):
        """Test behavior with multiple stop conditions."""
        budget = ComputeBudget(max_evaluations=100, patience=5)
        budget.start()

        # Patience should trigger first
        budget.update(perplexity=10.0)
        for _ in range(5):
            budget.update(perplexity=11.0)

        should_stop, reason = budget.should_stop()
        assert should_stop
        # Either patience or evaluations could trigger
        assert "patience" in reason.lower() or "evaluation" in reason.lower()


class TestEvaluationResult:
    """Test EvaluationResult dataclass."""

    def test_evaluation_result_creation(self):
        """Test EvaluationResult can be created."""
        result = EvaluationResult(
            success=True,
            final_loss=2.5,
            perplexity=12.18,
            n_params=10_000_000,
            training_time=60.0,
            tokens_per_second=1000.0,
            architecture_id="test_arch_001",
            error_message=None,
        )

        assert result.success
        assert result.final_loss == 2.5
        assert result.perplexity == 12.18
        assert result.n_params == 10_000_000
        assert result.training_time == 60.0

    def test_evaluation_result_failure(self):
        """Test EvaluationResult for failed evaluation."""
        result = EvaluationResult(
            success=False,
            final_loss=None,
            perplexity=None,
            n_params=None,
            training_time=0.0,
            tokens_per_second=None,
            architecture_id="test_arch_002",
            error_message="Out of memory",
        )

        assert not result.success
        assert result.error_message == "Out of memory"
        assert result.final_loss is None

    def test_evaluation_result_fields(self):
        """Test all expected fields exist."""
        result = EvaluationResult(
            success=True,
            final_loss=1.0,
            perplexity=2.72,
            n_params=1000,
            training_time=10.0,
            tokens_per_second=500.0,
            architecture_id="arch_test",
            error_message=None,
        )

        # Check all fields are accessible
        assert hasattr(result, "success")
        assert hasattr(result, "final_loss")
        assert hasattr(result, "perplexity")
        assert hasattr(result, "n_params")
        assert hasattr(result, "training_time")
        assert hasattr(result, "tokens_per_second")
        assert hasattr(result, "architecture_id")
        assert hasattr(result, "error_message")


class TestComputeBudgetIntegration:
    """Test ComputeBudget in realistic scenarios."""

    def test_typical_search_scenario(self):
        """Test budget in a typical search scenario."""
        budget = ComputeBudget(max_evaluations=20, patience=5)
        budget.start()

        # Simulate improving architectures
        perplexities = [15.0, 12.0, 10.0, 9.5, 9.4, 9.35, 9.35, 9.4, 9.5, 9.6]

        for i, ppl in enumerate(perplexities):
            budget.update(perplexity=ppl)
            should_stop, reason = budget.should_stop()

            if should_stop:
                # Should stop due to patience (no improvement after 9.35)
                assert budget.patience_counter >= budget.patience
                break

        # Should have stopped before max evaluations
        assert budget.evaluations_done < budget.max_evaluations

    def test_search_with_no_improvement(self):
        """Test budget when no improvement happens."""
        budget = ComputeBudget(patience=3)
        budget.start()

        # No improvement scenario
        budget.update(perplexity=10.0)
        budget.update(perplexity=10.1)
        budget.update(perplexity=10.2)
        budget.update(perplexity=10.3)

        should_stop, reason = budget.should_stop()
        assert should_stop
        assert "patience" in reason.lower()
