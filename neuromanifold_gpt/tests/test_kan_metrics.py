"""Tests for KAN metrics evaluation.

These tests verify the correctness of KAN quality metrics:
- compute_activation_statistics: KAN layer activation analysis
- compute_grid_utilization: basis function utilization statistics
- compute_spline_statistics: learnable spline weight analysis
- compute_all: aggregated metrics computation from info dict
"""
import pytest
import torch

from neuromanifold_gpt.evaluation.kan_metrics import KANMetrics


class TestComputeActivationStatistics:
    """Test KAN activation statistics computation."""

    def test_activation_statistics_keys(self):
        """Verify all expected keys are present."""
        kan_activations = torch.randn(2, 10, 8)
        stats = KANMetrics.compute_activation_statistics(kan_activations)

        expected_keys = {
            "activation_mean",
            "activation_std",
            "activation_min",
            "activation_max",
            "activation_range",
            "activation_abs_mean",
        }
        assert set(stats.keys()) == expected_keys

        # All values should be floats
        for key, value in stats.items():
            assert isinstance(value, float)

    def test_activation_statistics_constant_activations(self):
        """Constant activations should have zero std and zero range."""
        kan_activations = torch.full((2, 10, 8), 0.5)
        stats = KANMetrics.compute_activation_statistics(kan_activations)

        assert stats["activation_mean"] == 0.5
        assert stats["activation_std"] == 0.0
        assert stats["activation_min"] == 0.5
        assert stats["activation_max"] == 0.5
        assert stats["activation_range"] == 0.0
        assert stats["activation_abs_mean"] == 0.5

    def test_activation_statistics_zero_activations(self):
        """Zero activations should have all zero statistics."""
        kan_activations = torch.zeros(100)
        stats = KANMetrics.compute_activation_statistics(kan_activations)

        assert stats["activation_mean"] == 0.0
        assert stats["activation_std"] == 0.0
        assert stats["activation_min"] == 0.0
        assert stats["activation_max"] == 0.0
        assert stats["activation_range"] == 0.0
        assert stats["activation_abs_mean"] == 0.0

    def test_activation_statistics_positive_std(self):
        """Non-constant activations should have positive std."""
        kan_activations = torch.randn(100)
        stats = KANMetrics.compute_activation_statistics(kan_activations)

        assert stats["activation_std"] > 0

    def test_activation_statistics_range_consistency(self):
        """Range should equal max - min."""
        kan_activations = torch.randn(2, 10, 8)
        stats = KANMetrics.compute_activation_statistics(kan_activations)

        expected_range = stats["activation_max"] - stats["activation_min"]
        assert abs(stats["activation_range"] - expected_range) < 1e-6

    def test_activation_statistics_abs_mean_nonnegative(self):
        """Absolute mean should always be non-negative."""
        kan_activations = torch.randn(100)
        stats = KANMetrics.compute_activation_statistics(kan_activations)

        assert stats["activation_abs_mean"] >= 0

    def test_activation_statistics_symmetric_values(self):
        """Symmetric positive and negative values should have same abs mean."""
        kan_activations_pos = torch.tensor([1.0, 2.0, 3.0])
        kan_activations_neg = torch.tensor([-1.0, -2.0, -3.0])

        stats_pos = KANMetrics.compute_activation_statistics(kan_activations_pos)
        stats_neg = KANMetrics.compute_activation_statistics(kan_activations_neg)

        assert stats_pos["activation_abs_mean"] == stats_neg["activation_abs_mean"]
        assert stats_pos["activation_mean"] == -stats_neg["activation_mean"]

    def test_activation_statistics_known_values(self):
        """Test with known values for accuracy."""
        kan_activations = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        stats = KANMetrics.compute_activation_statistics(kan_activations)

        assert stats["activation_mean"] == 0.0
        assert stats["activation_min"] == -2.0
        assert stats["activation_max"] == 2.0
        assert stats["activation_range"] == 4.0
        assert abs(stats["activation_abs_mean"] - 1.2) < 1e-6  # (2+1+0+1+2)/5

    def test_activation_statistics_multidimensional(self):
        """Test with various tensor shapes."""
        shapes = [(10,), (5, 10), (2, 5, 10), (2, 3, 4, 5)]
        for shape in shapes:
            kan_activations = torch.randn(*shape)
            stats = KANMetrics.compute_activation_statistics(kan_activations)

            # Should work with any shape
            assert isinstance(stats["activation_mean"], float)
            assert isinstance(stats["activation_std"], float)


class TestComputeGridUtilization:
    """Test KAN grid utilization statistics."""

    def test_grid_utilization_keys(self):
        """Verify all expected keys are present."""
        basis_output = torch.randn(2, 10, 8)
        stats = KANMetrics.compute_grid_utilization(basis_output)

        expected_keys = {
            "grid_utilization_mean",
            "grid_utilization_std",
            "grid_basis_variance",
            "grid_basis_mean_abs",
        }
        assert set(stats.keys()) == expected_keys

        # All values should be floats
        for key, value in stats.items():
            assert isinstance(value, float)

    def test_grid_utilization_constant_basis(self):
        """Constant basis outputs should have zero variance."""
        basis_output = torch.full((2, 10, 8), 0.5)
        stats = KANMetrics.compute_grid_utilization(basis_output)

        assert stats["grid_utilization_mean"] == 0.5
        assert stats["grid_utilization_std"] == 0.0
        assert stats["grid_basis_variance"] == 0.0
        assert stats["grid_basis_mean_abs"] == 0.5

    def test_grid_utilization_zero_basis(self):
        """Zero basis outputs should have all zero statistics."""
        basis_output = torch.zeros(2, 10, 8)
        stats = KANMetrics.compute_grid_utilization(basis_output)

        assert stats["grid_utilization_mean"] == 0.0
        assert stats["grid_utilization_std"] == 0.0
        assert stats["grid_basis_variance"] == 0.0
        assert stats["grid_basis_mean_abs"] == 0.0

    def test_grid_utilization_diverse_basis(self):
        """Diverse basis outputs should have positive variance."""
        # Create basis with varying values across last dimension
        basis_output = torch.randn(10, 5, 8)
        stats = KANMetrics.compute_grid_utilization(basis_output)

        # Should have positive basis variance (diversity across basis dimension)
        assert stats["grid_basis_variance"] >= 0

    def test_grid_utilization_uniform_basis_dimension(self):
        """Uniform values across basis dimension should have low variance."""
        # All basis functions have same value
        basis_output = torch.ones(2, 10, 8) * 2.0
        stats = KANMetrics.compute_grid_utilization(basis_output)

        assert stats["grid_basis_variance"] == 0.0

    def test_grid_utilization_varying_basis_dimension(self):
        """Varying values across basis dimension should have positive variance."""
        # Different values for each basis function
        basis_output = torch.zeros(1, 1, 5)
        basis_output[0, 0, :] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = KANMetrics.compute_grid_utilization(basis_output)

        assert stats["grid_basis_variance"] > 0

    def test_grid_utilization_abs_mean_nonnegative(self):
        """Absolute mean should always be non-negative."""
        basis_output = torch.randn(2, 10, 8)
        stats = KANMetrics.compute_grid_utilization(basis_output)

        assert stats["grid_basis_mean_abs"] >= 0

    def test_grid_utilization_symmetric_values(self):
        """Symmetric positive and negative values should have same abs mean."""
        basis_output_pos = torch.tensor([[[1.0, 2.0, 3.0]]])
        basis_output_neg = torch.tensor([[[-1.0, -2.0, -3.0]]])

        stats_pos = KANMetrics.compute_grid_utilization(basis_output_pos)
        stats_neg = KANMetrics.compute_grid_utilization(basis_output_neg)

        assert stats_pos["grid_basis_mean_abs"] == stats_neg["grid_basis_mean_abs"]

    def test_grid_utilization_multidimensional(self):
        """Test with various tensor shapes."""
        shapes = [(10,), (5, 10), (2, 5, 10), (2, 3, 4, 5)]
        for shape in shapes:
            basis_output = torch.randn(*shape)
            stats = KANMetrics.compute_grid_utilization(basis_output)

            # Should work with any shape
            assert isinstance(stats["grid_utilization_mean"], float)
            assert isinstance(stats["grid_basis_variance"], float)


class TestComputeSplineStatistics:
    """Test KAN spline weight statistics."""

    def test_spline_statistics_keys(self):
        """Verify all expected keys are present."""
        spline_weights = torch.randn(2, 10, 8)
        stats = KANMetrics.compute_spline_statistics(spline_weights)

        expected_keys = {
            "spline_weight_mean",
            "spline_weight_std",
            "spline_weight_min",
            "spline_weight_max",
            "spline_weight_abs_mean",
        }
        assert set(stats.keys()) == expected_keys

        # All values should be floats
        for key, value in stats.items():
            assert isinstance(value, float)

    def test_spline_statistics_constant_weights(self):
        """Constant weights should have zero std."""
        spline_weights = torch.full((2, 10, 8), 0.5)
        stats = KANMetrics.compute_spline_statistics(spline_weights)

        assert stats["spline_weight_mean"] == 0.5
        assert stats["spline_weight_std"] == 0.0
        assert stats["spline_weight_min"] == 0.5
        assert stats["spline_weight_max"] == 0.5
        assert stats["spline_weight_abs_mean"] == 0.5

    def test_spline_statistics_zero_weights(self):
        """Zero weights should have all zero statistics except std."""
        spline_weights = torch.zeros(100)
        stats = KANMetrics.compute_spline_statistics(spline_weights)

        assert stats["spline_weight_mean"] == 0.0
        assert stats["spline_weight_std"] == 0.0
        assert stats["spline_weight_min"] == 0.0
        assert stats["spline_weight_max"] == 0.0
        assert stats["spline_weight_abs_mean"] == 0.0

    def test_spline_statistics_positive_std(self):
        """Non-constant weights should have positive std."""
        spline_weights = torch.randn(100)
        stats = KANMetrics.compute_spline_statistics(spline_weights)

        assert stats["spline_weight_std"] > 0

    def test_spline_statistics_abs_mean_nonnegative(self):
        """Absolute mean should always be non-negative."""
        spline_weights = torch.randn(100)
        stats = KANMetrics.compute_spline_statistics(spline_weights)

        assert stats["spline_weight_abs_mean"] >= 0

    def test_spline_statistics_symmetric_values(self):
        """Symmetric positive and negative values should have same abs mean."""
        spline_weights_pos = torch.tensor([1.0, 2.0, 3.0])
        spline_weights_neg = torch.tensor([-1.0, -2.0, -3.0])

        stats_pos = KANMetrics.compute_spline_statistics(spline_weights_pos)
        stats_neg = KANMetrics.compute_spline_statistics(spline_weights_neg)

        assert (
            stats_pos["spline_weight_abs_mean"] == stats_neg["spline_weight_abs_mean"]
        )
        assert stats_pos["spline_weight_mean"] == -stats_neg["spline_weight_mean"]

    def test_spline_statistics_known_values(self):
        """Test with known values for accuracy."""
        spline_weights = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        stats = KANMetrics.compute_spline_statistics(spline_weights)

        assert stats["spline_weight_mean"] == 0.0
        assert stats["spline_weight_min"] == -2.0
        assert stats["spline_weight_max"] == 2.0
        assert abs(stats["spline_weight_abs_mean"] - 1.2) < 1e-6  # (2+1+0+1+2)/5

    def test_spline_statistics_bounded_weights(self):
        """Weights within expected range should be correctly characterized."""
        spline_weights = torch.randn(100).clamp(-5.0, 5.0)
        stats = KANMetrics.compute_spline_statistics(spline_weights)

        assert -5.0 <= stats["spline_weight_min"] <= 5.0
        assert -5.0 <= stats["spline_weight_max"] <= 5.0
        assert stats["spline_weight_abs_mean"] >= 0

    def test_spline_statistics_multidimensional(self):
        """Test with various tensor shapes."""
        shapes = [(10,), (5, 10), (2, 5, 10), (2, 3, 4, 5)]
        for shape in shapes:
            spline_weights = torch.randn(*shape)
            stats = KANMetrics.compute_spline_statistics(spline_weights)

            # Should work with any shape
            assert isinstance(stats["spline_weight_mean"], float)
            assert isinstance(stats["spline_weight_std"], float)


class TestComputeAll:
    """Test aggregated metrics computation from info dict."""

    def test_compute_all_complete_info(self):
        """Verify compute_all with complete info dict."""
        info = {
            "kan_activations": torch.randn(2, 10, 8),
            "kan_basis_output": torch.randn(2, 10, 8),
            "kan_spline_weights": torch.randn(2, 10, 8),
        }
        metrics = KANMetrics.compute_all(info)

        # Should contain all activation metrics
        assert "activation_mean" in metrics
        assert "activation_std" in metrics
        assert "activation_min" in metrics
        assert "activation_max" in metrics
        assert "activation_range" in metrics
        assert "activation_abs_mean" in metrics

        # Should contain all grid utilization metrics
        assert "grid_utilization_mean" in metrics
        assert "grid_utilization_std" in metrics
        assert "grid_basis_variance" in metrics
        assert "grid_basis_mean_abs" in metrics

        # Should contain all spline weight metrics
        assert "spline_weight_mean" in metrics
        assert "spline_weight_std" in metrics
        assert "spline_weight_min" in metrics
        assert "spline_weight_max" in metrics
        assert "spline_weight_abs_mean" in metrics

    def test_compute_all_activation_only(self):
        """Verify compute_all with only activations."""
        info = {"kan_activations": torch.randn(2, 10, 8)}
        metrics = KANMetrics.compute_all(info)

        # Should contain activation metrics
        assert "activation_mean" in metrics
        assert "activation_std" in metrics

        # Should not contain grid or spline metrics
        assert "grid_utilization_mean" not in metrics
        assert "spline_weight_mean" not in metrics

    def test_compute_all_basis_only(self):
        """Verify compute_all with only basis output."""
        info = {"kan_basis_output": torch.randn(2, 10, 8)}
        metrics = KANMetrics.compute_all(info)

        # Should contain grid utilization metrics
        assert "grid_utilization_mean" in metrics
        assert "grid_basis_variance" in metrics

        # Should not contain activation or spline metrics
        assert "activation_mean" not in metrics
        assert "spline_weight_mean" not in metrics

    def test_compute_all_spline_only(self):
        """Verify compute_all with only spline weights."""
        info = {"kan_spline_weights": torch.randn(2, 10, 8)}
        metrics = KANMetrics.compute_all(info)

        # Should contain spline weight metrics
        assert "spline_weight_mean" in metrics
        assert "spline_weight_std" in metrics

        # Should not contain activation or grid metrics
        assert "activation_mean" not in metrics
        assert "grid_utilization_mean" not in metrics

    def test_compute_all_empty_info(self):
        """Empty info dict should return empty metrics."""
        info = {}
        metrics = KANMetrics.compute_all(info)

        assert metrics == {}

    def test_compute_all_none_values(self):
        """None values in info dict should be skipped."""
        info = {
            "kan_activations": None,
            "kan_basis_output": torch.randn(2, 10, 8),
            "kan_spline_weights": None,
        }
        metrics = KANMetrics.compute_all(info)

        # Should only contain grid metrics
        assert "grid_utilization_mean" in metrics
        assert "activation_mean" not in metrics
        assert "spline_weight_mean" not in metrics

    def test_compute_all_scalar_tensors(self):
        """Verify compute_all handles scalar tensors correctly."""
        info = {
            "kan_activations": torch.tensor(1.0),
            "kan_basis_output": torch.tensor(2.0),
            "kan_spline_weights": torch.tensor(3.0),
        }
        metrics = KANMetrics.compute_all(info)

        # Should convert scalars and compute metrics
        assert "activation_mean" in metrics
        assert "grid_utilization_mean" in metrics
        assert "spline_weight_mean" in metrics

        # Scalar values should equal their mean
        assert metrics["activation_mean"] == 1.0
        assert metrics["grid_utilization_mean"] == 2.0
        assert metrics["spline_weight_mean"] == 3.0

    def test_compute_all_values_are_floats(self):
        """All returned metrics should be Python floats."""
        info = {
            "kan_activations": torch.randn(2, 10, 8),
            "kan_basis_output": torch.randn(2, 10, 8),
            "kan_spline_weights": torch.randn(2, 10, 8),
        }
        metrics = KANMetrics.compute_all(info)

        for key, value in metrics.items():
            assert isinstance(value, float), f"{key} is not a float: {type(value)}"

    def test_compute_all_consistency(self):
        """Compute_all should match individual method results."""
        kan_activations = torch.randn(2, 10, 8)
        kan_basis_output = torch.randn(2, 10, 8)
        kan_spline_weights = torch.randn(2, 10, 8)

        # Compute individually
        activation_stats = KANMetrics.compute_activation_statistics(kan_activations)
        grid_stats = KANMetrics.compute_grid_utilization(kan_basis_output)
        spline_stats = KANMetrics.compute_spline_statistics(kan_spline_weights)

        # Compute all together
        info = {
            "kan_activations": kan_activations,
            "kan_basis_output": kan_basis_output,
            "kan_spline_weights": kan_spline_weights,
        }
        all_metrics = KANMetrics.compute_all(info)

        # Should match
        for key, value in activation_stats.items():
            assert abs(all_metrics[key] - value) < 1e-6

        for key, value in grid_stats.items():
            assert abs(all_metrics[key] - value) < 1e-6

        for key, value in spline_stats.items():
            assert abs(all_metrics[key] - value) < 1e-6

    def test_compute_all_realistic_values(self):
        """Test with realistic KAN layer values."""
        # Simulate realistic KAN layer outputs
        batch_size, seq_len, hidden_dim = 4, 32, 64
        n_basis = 8

        info = {
            "kan_activations": torch.randn(batch_size, seq_len, hidden_dim) * 0.5,
            "kan_basis_output": torch.randn(batch_size, seq_len, n_basis) * 0.3,
            "kan_spline_weights": torch.randn(hidden_dim, n_basis) * 0.1,
        }

        metrics = KANMetrics.compute_all(info)

        # Metrics should be reasonable
        assert -2.0 < metrics["activation_mean"] < 2.0
        assert metrics["activation_std"] > 0
        assert metrics["grid_utilization_std"] > 0
        assert metrics["spline_weight_std"] > 0
