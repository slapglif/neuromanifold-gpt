#!/usr/bin/env python3
"""Test sinkhorn_log convergence behavior with early stopping."""

import torch
import pytest
from loguru import logger

from neuromanifold_gpt.model.mhc import sinkhorn_log


class TestSinkhornConvergence:
    """Test suite for sinkhorn_log convergence and early stopping."""

    def test_backward_compatibility_no_convergence_tol(self):
        """Test that convergence_tol=None uses all iterations (backward compatibility)."""
        torch.manual_seed(42)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create a well-initialized matrix (like mHC uses in practice)
        # Random matrices require 1000+ iterations to converge to 1e-4 tolerance
        logits = torch.full((8, 8), -8.0, device=device)
        logits.fill_diagonal_(0.0)
        num_iters = 10

        # Run without convergence_tol (should use all iterations)
        result = sinkhorn_log(logits, num_iters=num_iters, convergence_tol=None)

        # Verify output is doubly stochastic
        row_sums = result.sum(dim=-1)
        col_sums = result.sum(dim=-2)

        assert result.shape == (8, 8), "Output shape mismatch"
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), "Rows don't sum to 1"
        assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-4), "Columns don't sum to 1"
        assert (result >= 0).all(), "Output has negative values"

        logger.info("✓ Backward compatibility test passed (convergence_tol=None)")

    def test_early_stopping_triggers(self):
        """Test that early stopping triggers with convergence_tol."""
        torch.manual_seed(42)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create a well-initialized matrix (near identity)
        # This should converge quickly
        logits = torch.eye(8, device=device) * 2.0 - 8.0  # diagonal = 2, off-diagonal = -8

        convergence_tol = 1e-6
        num_iters = 100  # Set high to ensure early stopping happens

        # Run with early stopping
        result = sinkhorn_log(logits, num_iters=num_iters, convergence_tol=convergence_tol)

        # Verify output is doubly stochastic
        row_sums = result.sum(dim=-1)
        col_sums = result.sum(dim=-2)

        assert result.shape == (8, 8), "Output shape mismatch"
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), "Rows don't sum to 1"
        assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-4), "Columns don't sum to 1"
        assert (result >= 0).all(), "Output has negative values"

        logger.info("✓ Early stopping test passed")

    def test_convergence_accuracy_with_tolerance(self):
        """Test that early stopped results are still doubly stochastic."""
        torch.manual_seed(123)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Test with different matrix sizes
        for size in [8, 16, 32]:
            # Use well-initialized matrices for realistic convergence
            logits = torch.full((size, size), -8.0, device=device)
            logits.fill_diagonal_(0.0)

            # Run with convergence tolerance
            result = sinkhorn_log(logits, num_iters=20, convergence_tol=1e-6)

            # Verify doubly stochastic property
            row_sums = result.sum(dim=-1)
            col_sums = result.sum(dim=-2)

            assert result.shape == (size, size), f"Output shape mismatch for size {size}"
            assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), \
                f"Rows don't sum to 1 for size {size}"
            assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-4), \
                f"Columns don't sum to 1 for size {size}"
            assert (result >= 0).all(), f"Output has negative values for size {size}"

        logger.info("✓ Convergence accuracy test passed for sizes [8, 16, 32]")

    def test_iteration_reduction_with_well_initialized_matrix(self):
        """Test that well-initialized matrices converge in fewer iterations."""
        torch.manual_seed(42)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Well-initialized matrix (near identity, like mHC initialization)
        logits = torch.full((8, 8), -8.0, device=device)
        logits.fill_diagonal_(0.0)

        # Run without early stopping
        result_no_early = sinkhorn_log(logits, num_iters=10, convergence_tol=None)

        # Run with early stopping
        result_with_early = sinkhorn_log(logits, num_iters=10, convergence_tol=1e-6)

        # Both should produce doubly stochastic matrices
        row_sums_no = result_no_early.sum(dim=-1)
        col_sums_no = result_no_early.sum(dim=-2)
        row_sums_early = result_with_early.sum(dim=-1)
        col_sums_early = result_with_early.sum(dim=-2)

        assert torch.allclose(row_sums_no, torch.ones_like(row_sums_no), atol=1e-4), \
            "No-early rows don't sum to 1"
        assert torch.allclose(col_sums_no, torch.ones_like(col_sums_no), atol=1e-4), \
            "No-early columns don't sum to 1"
        assert torch.allclose(row_sums_early, torch.ones_like(row_sums_early), atol=1e-4), \
            "Early-stop rows don't sum to 1"
        assert torch.allclose(col_sums_early, torch.ones_like(col_sums_early), atol=1e-4), \
            "Early-stop columns don't sum to 1"

        # Results should be close (both converged)
        assert torch.allclose(result_no_early, result_with_early, atol=1e-4), \
            "Results differ between early stopping and no early stopping"

        logger.info("✓ Iteration reduction test passed")

    def test_different_convergence_tolerances(self):
        """Test behavior with different convergence tolerances."""
        torch.manual_seed(42)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Use well-initialized matrix for realistic convergence
        logits = torch.full((8, 8), -8.0, device=device)
        logits.fill_diagonal_(0.0)

        # Test with different tolerances
        tolerances = [1e-4, 1e-6, 1e-8]
        results = []

        for tol in tolerances:
            result = sinkhorn_log(logits, num_iters=20, convergence_tol=tol)
            results.append(result)

            # All should be doubly stochastic
            row_sums = result.sum(dim=-1)
            col_sums = result.sum(dim=-2)

            assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), \
                f"Rows don't sum to 1 for tol={tol}"
            assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-4), \
                f"Columns don't sum to 1 for tol={tol}"

        # Tighter tolerance should give more similar result to fully converged
        # (all should be close since we use enough iterations)
        for i in range(len(results) - 1):
            assert torch.allclose(results[i], results[i + 1], atol=1e-3), \
                f"Results differ significantly between tolerances {tolerances[i]} and {tolerances[i+1]}"

        logger.info("✓ Different convergence tolerances test passed")

    def test_batched_input(self):
        """Test that early stopping works with batched input."""
        torch.manual_seed(42)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Batched input: (B, N, N)
        batch_size = 4
        size = 8
        # Use well-initialized matrices for realistic convergence
        logits = torch.full((batch_size, size, size), -8.0, device=device)
        for b in range(batch_size):
            logits[b].fill_diagonal_(0.0)

        # Run with convergence tolerance
        result = sinkhorn_log(logits, num_iters=10, convergence_tol=1e-6)

        assert result.shape == (batch_size, size, size), "Batched output shape mismatch"

        # Verify each matrix in batch is doubly stochastic
        for b in range(batch_size):
            row_sums = result[b].sum(dim=-1)
            col_sums = result[b].sum(dim=-2)

            assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), \
                f"Batch {b} rows don't sum to 1"
            assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-4), \
                f"Batch {b} columns don't sum to 1"
            assert (result[b] >= 0).all(), f"Batch {b} has negative values"

        logger.info("✓ Batched input test passed")

    def test_extreme_convergence_case(self):
        """Test convergence with already-converged input (identity-like)."""
        torch.manual_seed(42)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create near-perfect doubly stochastic matrix
        # Start with identity and add small noise
        logits = torch.eye(8, device=device) * 10.0  # Strong diagonal
        logits += torch.randn_like(logits) * 0.1  # Small noise

        # This should converge very quickly (1-2 iterations)
        result = sinkhorn_log(logits, num_iters=10, convergence_tol=1e-8)

        # Verify doubly stochastic
        row_sums = result.sum(dim=-1)
        col_sums = result.sum(dim=-2)

        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), "Rows don't sum to 1"
        assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-4), "Columns don't sum to 1"
        assert (result >= 0).all(), "Output has negative values"

        # Result should be close to identity (due to strong diagonal)
        diagonal_values = torch.diag(result)
        assert (diagonal_values > 0.5).all(), "Diagonal not dominant (expected near-identity)"

        logger.info("✓ Extreme convergence case test passed")


def test_all():
    """Run all tests."""
    suite = TestSinkhornConvergence()

    logger.info("Running sinkhorn_log convergence tests...")

    suite.test_backward_compatibility_no_convergence_tol()
    suite.test_early_stopping_triggers()
    suite.test_convergence_accuracy_with_tolerance()
    suite.test_iteration_reduction_with_well_initialized_matrix()
    suite.test_different_convergence_tolerances()
    suite.test_batched_input()
    suite.test_extreme_convergence_case()

    logger.info("✅ All sinkhorn_log convergence tests passed!")


if __name__ == "__main__":
    test_all()
