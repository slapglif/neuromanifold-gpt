"""Tests for Manifold Projection Module.

The ManifoldProjection takes SDRs (2048-bit) and projects them onto a learned
Riemannian manifold with:
1. Low-dimensional coordinates (64-dim from 2048-bit SDRs)
2. Learned position-dependent metric tensor G(x)
3. Geodesic distance computation using the metric

Tests verify:
- Output shape correctness (coords and metric)
- Metric tensor is positive definite (all eigenvalues > 0)
- Geodesic distance is symmetric: d(i,j) = d(j,i)
- Geodesic distance to self is zero: d(i,i) = 0
"""
import pytest
import torch

from neuromanifold_gpt.model.manifold import ManifoldProjection


class TestManifoldProjectionShape:
    """Test output shapes from ManifoldProjection."""

    def test_manifold_projection_shape(self):
        """Coordinates should have manifold_dim dimensions."""
        proj = ManifoldProjection(sdr_size=2048, manifold_dim=64)
        sdr = torch.randn(2, 10, 2048)

        coords, metric = proj(sdr)

        assert coords.shape == (2, 10, 64)
        assert metric.shape == (2, 10, 64, 64)

    def test_manifold_projection_single_token(self):
        """Handles single token input."""
        proj = ManifoldProjection(sdr_size=2048, manifold_dim=64)
        sdr = torch.randn(1, 1, 2048)

        coords, metric = proj(sdr)

        assert coords.shape == (1, 1, 64)
        assert metric.shape == (1, 1, 64, 64)

    def test_manifold_projection_various_dims(self):
        """Works with various manifold dimensions."""
        for manifold_dim in [16, 32, 64, 128]:
            proj = ManifoldProjection(sdr_size=2048, manifold_dim=manifold_dim)
            sdr = torch.randn(2, 5, 2048)

            coords, metric = proj(sdr)

            assert coords.shape == (2, 5, manifold_dim)
            assert metric.shape == (2, 5, manifold_dim, manifold_dim)


class TestMetricTensor:
    """Test Riemannian metric tensor properties."""

    def test_metric_positive_definite(self):
        """Metric tensor should be positive definite."""
        proj = ManifoldProjection(sdr_size=2048, manifold_dim=64)
        sdr = torch.randn(2, 10, 2048)

        _, metric = proj(sdr)

        # Check eigenvalues are positive
        eigenvalues = torch.linalg.eigvalsh(metric)
        assert (eigenvalues > 0).all()

    def test_metric_positive_definite_various_inputs(self):
        """Metric remains positive definite for various inputs."""
        proj = ManifoldProjection(sdr_size=2048, manifold_dim=32)

        # Test with different input distributions
        for _ in range(5):
            sdr = torch.randn(4, 8, 2048) * torch.rand(1).item() * 10
            _, metric = proj(sdr)

            eigenvalues = torch.linalg.eigvalsh(metric)
            assert (eigenvalues > 0).all(), "Metric must be positive definite"

    def test_metric_symmetric(self):
        """Metric tensor should be symmetric."""
        proj = ManifoldProjection(sdr_size=2048, manifold_dim=64)
        sdr = torch.randn(2, 10, 2048)

        _, metric = proj(sdr)

        # G should equal G^T
        metric_t = metric.transpose(-2, -1)
        assert torch.allclose(metric, metric_t, atol=1e-6)


class TestGeodesicDistance:
    """Test geodesic distance computation."""

    def test_geodesic_distance_symmetric(self):
        """Distance should be symmetric."""
        proj = ManifoldProjection(sdr_size=2048, manifold_dim=64)
        sdr = torch.randn(1, 5, 2048)

        coords, metric = proj(sdr)
        dist = proj.geodesic_distance(coords, metric)

        # d(i,j) = d(j,i)
        assert torch.allclose(dist, dist.transpose(1, 2), atol=1e-5)

    def test_geodesic_distance_zero_diagonal(self):
        """Distance to self should be zero."""
        proj = ManifoldProjection(sdr_size=2048, manifold_dim=64)
        sdr = torch.randn(1, 5, 2048)

        coords, metric = proj(sdr)
        dist = proj.geodesic_distance(coords, metric)

        # d(i,i) = 0
        diagonal = torch.diagonal(dist, dim1=1, dim2=2)
        assert torch.allclose(diagonal, torch.zeros_like(diagonal), atol=1e-5)

    def test_geodesic_distance_non_negative(self):
        """Distance squared should be non-negative."""
        proj = ManifoldProjection(sdr_size=2048, manifold_dim=64)
        sdr = torch.randn(2, 8, 2048)

        coords, metric = proj(sdr)
        dist_sq = proj.geodesic_distance(coords, metric)

        # d² >= 0
        assert (dist_sq >= -1e-6).all()

    def test_geodesic_distance_shape(self):
        """Distance matrix has correct shape."""
        proj = ManifoldProjection(sdr_size=2048, manifold_dim=64)
        sdr = torch.randn(3, 7, 2048)

        coords, metric = proj(sdr)
        dist = proj.geodesic_distance(coords, metric)

        assert dist.shape == (3, 7, 7)


class TestGradientFlow:
    """Test gradient flow through ManifoldProjection."""

    def test_gradient_through_coords(self):
        """Gradients should flow through coordinate computation."""
        proj = ManifoldProjection(sdr_size=2048, manifold_dim=64)
        proj.train()
        sdr = torch.randn(2, 10, 2048, requires_grad=True)

        coords, _ = proj(sdr)
        loss = coords.sum()
        loss.backward()

        assert sdr.grad is not None
        assert sdr.grad.abs().sum() > 0

    def test_gradient_through_metric(self):
        """Gradients should flow through metric computation."""
        proj = ManifoldProjection(sdr_size=2048, manifold_dim=64)
        proj.train()
        sdr = torch.randn(2, 10, 2048, requires_grad=True)

        _, metric = proj(sdr)
        loss = metric.sum()
        loss.backward()

        assert sdr.grad is not None
        assert sdr.grad.abs().sum() > 0

    def test_gradient_through_distance(self):
        """Gradients should flow through geodesic distance."""
        proj = ManifoldProjection(sdr_size=2048, manifold_dim=64)
        proj.train()
        sdr = torch.randn(2, 5, 2048, requires_grad=True)

        coords, metric = proj(sdr)
        dist = proj.geodesic_distance(coords, metric)
        loss = dist.sum()
        loss.backward()

        assert sdr.grad is not None
        assert sdr.grad.abs().sum() > 0


class TestDeterminism:
    """Test deterministic behavior."""

    def test_eval_mode_deterministic(self):
        """Same input produces same output in eval mode."""
        proj = ManifoldProjection(sdr_size=2048, manifold_dim=64)
        proj.eval()
        sdr = torch.randn(2, 10, 2048)

        coords1, metric1 = proj(sdr)
        coords2, metric2 = proj(sdr)

        assert torch.equal(coords1, coords2)
        assert torch.equal(metric1, metric2)


class TestCustomHiddenDim:
    """Test custom hidden dimension parameter."""

    def test_custom_hidden_dim(self):
        """Custom hidden_dim should work correctly."""
        proj = ManifoldProjection(sdr_size=2048, manifold_dim=64, hidden_dim=512)
        sdr = torch.randn(2, 10, 2048)

        coords, metric = proj(sdr)

        assert coords.shape == (2, 10, 64)
        assert metric.shape == (2, 10, 64, 64)

    def test_default_hidden_dim(self):
        """Default hidden_dim should be sdr_size // 2."""
        proj = ManifoldProjection(sdr_size=2048, manifold_dim=64)
        # Verify internal structure uses default
        # First layer of encoder should have sdr_size -> hidden_dim
        first_layer = proj.encoder[0]
        assert first_layer.in_features == 2048
        assert first_layer.out_features == 1024  # 2048 // 2
