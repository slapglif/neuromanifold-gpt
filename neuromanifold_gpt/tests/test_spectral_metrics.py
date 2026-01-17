"""Tests for Spectral decomposition metrics evaluation.

These tests verify the correctness of spectral quality metrics:
- compute_eigenvalue_statistics: distribution of learned frequencies
- compute_basis_statistics: quality and properties of learned basis
- compute_orthogonality_metrics: basis orthonormality evaluation
- compute_all: aggregated metrics computation from info dict
"""
import pytest
import torch

from neuromanifold_gpt.evaluation.spectral_metrics import SpectralMetrics


class TestComputeEigenvalueStatistics:
    """Test spectral eigenvalue/frequency statistics."""

    def test_eigenvalue_statistics_keys(self):
        """Verify all expected keys are present."""
        spectral_freqs = torch.randn(2, 8)
        stats = SpectralMetrics.compute_eigenvalue_statistics(spectral_freqs)

        expected_keys = {
            "eigenvalue_mean",
            "eigenvalue_std",
            "eigenvalue_min",
            "eigenvalue_max",
            "eigenvalue_range",
        }
        assert set(stats.keys()) == expected_keys

        # All values should be floats
        for key, value in stats.items():
            assert isinstance(value, float)

    def test_eigenvalue_statistics_constant_freqs(self):
        """Constant frequencies should have zero std and zero range."""
        spectral_freqs = torch.full((2, 8), 1.5)
        stats = SpectralMetrics.compute_eigenvalue_statistics(spectral_freqs)

        assert stats["eigenvalue_mean"] == 1.5
        assert stats["eigenvalue_std"] == 0.0
        assert stats["eigenvalue_min"] == 1.5
        assert stats["eigenvalue_max"] == 1.5
        assert stats["eigenvalue_range"] == 0.0

    def test_eigenvalue_statistics_positive_values(self):
        """Positive eigenvalues should be correctly characterized."""
        spectral_freqs = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5])
        stats = SpectralMetrics.compute_eigenvalue_statistics(spectral_freqs)

        assert stats["eigenvalue_mean"] == 1.5
        assert stats["eigenvalue_min"] == 0.5
        assert stats["eigenvalue_max"] == 2.5
        assert stats["eigenvalue_range"] == 2.0

    def test_eigenvalue_statistics_varying_freqs(self):
        """Non-constant frequencies should have positive std."""
        spectral_freqs = torch.randn(100)
        stats = SpectralMetrics.compute_eigenvalue_statistics(spectral_freqs)

        assert stats["eigenvalue_std"] > 0

    def test_eigenvalue_statistics_range_consistency(self):
        """Range should equal max - min."""
        spectral_freqs = torch.randn(2, 8)
        stats = SpectralMetrics.compute_eigenvalue_statistics(spectral_freqs)

        expected_range = stats["eigenvalue_max"] - stats["eigenvalue_min"]
        assert abs(stats["eigenvalue_range"] - expected_range) < 1e-6

    def test_eigenvalue_statistics_batch_dimension(self):
        """Test with batch dimension."""
        spectral_freqs = torch.randn(4, 16)
        stats = SpectralMetrics.compute_eigenvalue_statistics(spectral_freqs)

        # Should aggregate across all dimensions
        assert isinstance(stats["eigenvalue_mean"], float)
        assert isinstance(stats["eigenvalue_std"], float)

    def test_eigenvalue_statistics_zero_freqs(self):
        """Zero frequencies should be handled correctly."""
        spectral_freqs = torch.zeros(2, 8)
        stats = SpectralMetrics.compute_eigenvalue_statistics(spectral_freqs)

        assert stats["eigenvalue_mean"] == 0.0
        assert stats["eigenvalue_std"] == 0.0
        assert stats["eigenvalue_min"] == 0.0
        assert stats["eigenvalue_max"] == 0.0
        assert stats["eigenvalue_range"] == 0.0


class TestComputeBasisStatistics:
    """Test spectral basis statistics."""

    def test_basis_statistics_keys(self):
        """Verify all expected keys are present."""
        spectral_basis = torch.randn(2, 10, 8)
        stats = SpectralMetrics.compute_basis_statistics(spectral_basis)

        expected_keys = {
            "basis_mean",
            "basis_std",
            "basis_abs_mean",
            "basis_min",
            "basis_max",
        }
        assert set(stats.keys()) == expected_keys

        # All values should be floats
        for key, value in stats.items():
            assert isinstance(value, float)

    def test_basis_statistics_constant_basis(self):
        """Constant basis should have zero std."""
        spectral_basis = torch.full((2, 10, 8), 0.5)
        stats = SpectralMetrics.compute_basis_statistics(spectral_basis)

        assert stats["basis_mean"] == 0.5
        assert stats["basis_std"] == 0.0
        assert stats["basis_abs_mean"] == 0.5
        assert stats["basis_min"] == 0.5
        assert stats["basis_max"] == 0.5

    def test_basis_statistics_zero_basis(self):
        """Zero basis should be handled correctly."""
        spectral_basis = torch.zeros(2, 10, 8)
        stats = SpectralMetrics.compute_basis_statistics(spectral_basis)

        assert stats["basis_mean"] == 0.0
        assert stats["basis_std"] == 0.0
        assert stats["basis_abs_mean"] == 0.0
        assert stats["basis_min"] == 0.0
        assert stats["basis_max"] == 0.0

    def test_basis_statistics_positive_std(self):
        """Non-constant basis should have positive std."""
        spectral_basis = torch.randn(2, 10, 8)
        stats = SpectralMetrics.compute_basis_statistics(spectral_basis)

        assert stats["basis_std"] > 0

    def test_basis_statistics_abs_mean_positive(self):
        """Absolute mean should always be non-negative."""
        spectral_basis = torch.randn(2, 10, 8)
        stats = SpectralMetrics.compute_basis_statistics(spectral_basis)

        assert stats["basis_abs_mean"] >= 0

    def test_basis_statistics_abs_mean_symmetric(self):
        """Symmetric bases should have same abs mean."""
        spectral_basis_pos = torch.tensor([1.0, 2.0, 3.0])
        spectral_basis_neg = torch.tensor([-1.0, -2.0, -3.0])

        stats_pos = SpectralMetrics.compute_basis_statistics(spectral_basis_pos)
        stats_neg = SpectralMetrics.compute_basis_statistics(spectral_basis_neg)

        assert stats_pos["basis_abs_mean"] == stats_neg["basis_abs_mean"]

    def test_basis_statistics_bounded_values(self):
        """Bounded basis values should be correctly characterized."""
        spectral_basis = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        stats = SpectralMetrics.compute_basis_statistics(spectral_basis)

        assert stats["basis_mean"] == 0.0
        assert stats["basis_min"] == -2.0
        assert stats["basis_max"] == 2.0

    def test_basis_statistics_batch_and_sequence(self):
        """Test with batch and sequence dimensions."""
        spectral_basis = torch.randn(4, 20, 16)
        stats = SpectralMetrics.compute_basis_statistics(spectral_basis)

        # Should aggregate across all dimensions
        assert isinstance(stats["basis_mean"], float)
        assert isinstance(stats["basis_std"], float)


class TestComputeOrthogonalityMetrics:
    """Test basis orthogonality quality metrics."""

    def test_orthogonality_metrics_keys(self):
        """Verify all expected keys are present."""
        spectral_basis = torch.randn(2, 10, 8)
        ortho_loss = torch.tensor(0.1)
        stats = SpectralMetrics.compute_orthogonality_metrics(
            spectral_basis, ortho_loss
        )

        expected_keys = {"ortho_loss", "basis_norm_mean", "basis_norm_std"}
        assert set(stats.keys()) == expected_keys

        # All values should be floats
        for key, value in stats.items():
            assert isinstance(value, float)

    def test_orthogonality_metrics_scalar_loss(self):
        """Scalar ortho_loss should be handled correctly."""
        spectral_basis = torch.randn(2, 10, 8)
        ortho_loss = torch.tensor(0.05)
        stats = SpectralMetrics.compute_orthogonality_metrics(
            spectral_basis, ortho_loss
        )

        assert abs(stats["ortho_loss"] - 0.05) < 1e-6

    def test_orthogonality_metrics_tensor_loss(self):
        """Tensor ortho_loss should be averaged."""
        spectral_basis = torch.randn(2, 10, 8)
        ortho_loss = torch.tensor([0.1, 0.2, 0.3])
        stats = SpectralMetrics.compute_orthogonality_metrics(
            spectral_basis, ortho_loss
        )

        assert abs(stats["ortho_loss"] - 0.2) < 1e-6

    def test_orthogonality_metrics_zero_loss(self):
        """Zero ortho_loss indicates perfect orthogonality."""
        spectral_basis = torch.randn(2, 10, 8)
        ortho_loss = torch.tensor(0.0)
        stats = SpectralMetrics.compute_orthogonality_metrics(
            spectral_basis, ortho_loss
        )

        assert stats["ortho_loss"] == 0.0

    def test_orthogonality_metrics_norm_positive(self):
        """Basis norms should be non-negative."""
        spectral_basis = torch.randn(2, 10, 8)
        ortho_loss = torch.tensor(0.1)
        stats = SpectralMetrics.compute_orthogonality_metrics(
            spectral_basis, ortho_loss
        )

        assert stats["basis_norm_mean"] >= 0
        assert stats["basis_norm_std"] >= 0

    def test_orthogonality_metrics_unit_norm_basis(self):
        """Unit norm basis vectors should have norm mean close to 1."""
        # Create basis with unit norm vectors
        spectral_basis = torch.randn(2, 10, 8)
        # Normalize along sequence dimension
        spectral_basis = spectral_basis / torch.norm(
            spectral_basis, dim=1, keepdim=True
        )
        ortho_loss = torch.tensor(0.0)
        stats = SpectralMetrics.compute_orthogonality_metrics(
            spectral_basis, ortho_loss
        )

        # All norms should be 1.0
        assert abs(stats["basis_norm_mean"] - 1.0) < 1e-5
        assert stats["basis_norm_std"] < 1e-5

    def test_orthogonality_metrics_zero_basis(self):
        """Zero basis should have zero norms."""
        spectral_basis = torch.zeros(2, 10, 8)
        ortho_loss = torch.tensor(0.1)
        stats = SpectralMetrics.compute_orthogonality_metrics(
            spectral_basis, ortho_loss
        )

        assert stats["basis_norm_mean"] == 0.0
        assert stats["basis_norm_std"] == 0.0

    def test_orthogonality_metrics_varying_norms(self):
        """Varying basis norms should have positive std."""
        # Create basis with varying norms
        spectral_basis = torch.zeros(3, 10, 8)
        spectral_basis[0, :, 0] = 1.0  # Norm = sqrt(10)
        spectral_basis[1, :, 0] = 2.0  # Norm = 2*sqrt(10)
        spectral_basis[2, :, 0] = 3.0  # Norm = 3*sqrt(10)
        ortho_loss = torch.tensor(0.1)
        stats = SpectralMetrics.compute_orthogonality_metrics(
            spectral_basis, ortho_loss
        )

        assert stats["basis_norm_std"] > 0


class TestComputeAll:
    """Test aggregated metrics computation from info dict."""

    def test_compute_all_complete_info(self):
        """Verify compute_all with complete info dict."""
        info = {
            "spectral_basis": torch.randn(2, 10, 8),
            "spectral_freqs": torch.randn(2, 8),
            "ortho_loss": torch.tensor(0.1),
        }
        metrics = SpectralMetrics.compute_all(info)

        # Should contain all metrics from all three methods
        expected_keys = {
            # Eigenvalue statistics
            "eigenvalue_mean",
            "eigenvalue_std",
            "eigenvalue_min",
            "eigenvalue_max",
            "eigenvalue_range",
            # Basis statistics
            "basis_mean",
            "basis_std",
            "basis_abs_mean",
            "basis_min",
            "basis_max",
            # Orthogonality metrics
            "ortho_loss",
            "basis_norm_mean",
            "basis_norm_std",
        }
        assert set(metrics.keys()) == expected_keys

        # All values should be floats
        for key, value in metrics.items():
            assert isinstance(value, float)

    def test_compute_all_missing_ortho_loss(self):
        """Verify compute_all with missing ortho_loss."""
        info = {
            "spectral_basis": torch.randn(2, 10, 8),
            "spectral_freqs": torch.randn(2, 8),
        }
        metrics = SpectralMetrics.compute_all(info)

        # Should have eigenvalue and basis stats, but not orthogonality
        assert "eigenvalue_mean" in metrics
        assert "basis_mean" in metrics
        assert "ortho_loss" not in metrics
        assert "basis_norm_mean" not in metrics

    def test_compute_all_missing_basis(self):
        """Verify compute_all with missing spectral_basis."""
        info = {"spectral_freqs": torch.randn(2, 8), "ortho_loss": torch.tensor(0.1)}
        metrics = SpectralMetrics.compute_all(info)

        # Should only have eigenvalue stats
        assert "eigenvalue_mean" in metrics
        assert "basis_mean" not in metrics
        assert "ortho_loss" not in metrics

    def test_compute_all_missing_freqs(self):
        """Verify compute_all with missing spectral_freqs."""
        info = {
            "spectral_basis": torch.randn(2, 10, 8),
            "ortho_loss": torch.tensor(0.1),
        }
        metrics = SpectralMetrics.compute_all(info)

        # Should have basis and orthogonality stats, but not eigenvalue
        assert "eigenvalue_mean" not in metrics
        assert "basis_mean" in metrics
        assert "ortho_loss" in metrics

    def test_compute_all_empty_info(self):
        """Verify compute_all with empty info dict."""
        info = {}
        metrics = SpectralMetrics.compute_all(info)

        # Should return empty dict
        assert metrics == {}

    def test_compute_all_scalar_freqs(self):
        """Verify compute_all handles scalar spectral_freqs."""
        info = {"spectral_freqs": torch.tensor(1.5)}
        metrics = SpectralMetrics.compute_all(info)

        # Should handle scalar and compute stats
        assert "eigenvalue_mean" in metrics
        assert metrics["eigenvalue_mean"] == 1.5

    def test_compute_all_scalar_basis(self):
        """Verify compute_all handles scalar spectral_basis."""
        info = {"spectral_basis": torch.tensor(0.5), "ortho_loss": torch.tensor(0.1)}
        metrics = SpectralMetrics.compute_all(info)

        # Should handle scalar and compute stats
        assert "basis_mean" in metrics
        assert metrics["basis_mean"] == 0.5

    def test_compute_all_1d_basis(self):
        """Verify compute_all handles 1D spectral_basis."""
        info = {"spectral_basis": torch.randn(8)}
        metrics = SpectralMetrics.compute_all(info)

        # Should handle 1D and compute stats
        assert "basis_mean" in metrics
        assert isinstance(metrics["basis_mean"], float)

    def test_compute_all_realistic_values(self):
        """Test with realistic spectral decomposition values."""
        # Simulate realistic spectral decomposition output
        B, T, n_eig = 4, 20, 16

        # Eigenvalues typically in [0, 2] range for normalized Laplacian
        spectral_freqs = torch.rand(B, n_eig) * 2.0

        # Basis coefficients typically normalized
        spectral_basis = torch.randn(B, T, n_eig)
        spectral_basis = spectral_basis / torch.norm(
            spectral_basis, dim=1, keepdim=True
        )

        # Ortho loss typically small for well-trained model
        ortho_loss = torch.rand(1) * 0.1

        info = {
            "spectral_basis": spectral_basis,
            "spectral_freqs": spectral_freqs,
            "ortho_loss": ortho_loss,
        }
        metrics = SpectralMetrics.compute_all(info)

        # Verify reasonable ranges
        assert 0 <= metrics["eigenvalue_mean"] <= 2.0
        assert metrics["eigenvalue_min"] >= 0
        assert metrics["eigenvalue_max"] <= 2.0
        assert abs(metrics["basis_norm_mean"] - 1.0) < 0.1  # Should be close to 1
        assert metrics["ortho_loss"] < 0.1  # Should be small

    def test_compute_all_consistency(self):
        """Verify compute_all produces consistent results."""
        info = {
            "spectral_basis": torch.randn(2, 10, 8),
            "spectral_freqs": torch.randn(2, 8),
            "ortho_loss": torch.tensor(0.1),
        }

        # Compute twice
        metrics1 = SpectralMetrics.compute_all(info)
        metrics2 = SpectralMetrics.compute_all(info)

        # Should be identical
        assert metrics1.keys() == metrics2.keys()
        for key in metrics1:
            assert metrics1[key] == metrics2[key]

    def test_compute_all_values_are_finite(self):
        """All computed metrics should be finite."""
        info = {
            "spectral_basis": torch.randn(2, 10, 8),
            "spectral_freqs": torch.randn(2, 8),
            "ortho_loss": torch.tensor(0.1),
        }
        metrics = SpectralMetrics.compute_all(info)

        for key, value in metrics.items():
            assert torch.isfinite(torch.tensor(value)).item(), f"{key} is not finite"
