"""Tests for SemanticRetina topographic feature map."""

import pytest
import torch


class TestSemanticRetina:
    """Test suite for the SemanticRetina module."""

    def test_retina_output_shape(self):
        """Test that output shape matches input shape."""
        from neuromanifold_gpt.model.semantic_retina import SemanticRetina

        retina = SemanticRetina(grid_size=64, n_features=4096)
        activation = torch.randn(2, 10, 64, 64)
        out = retina(activation)
        assert out.shape == activation.shape

    def test_retina_smoothing(self):
        """Test Gaussian smoothing spreads activation locally."""
        from neuromanifold_gpt.model.semantic_retina import SemanticRetina

        retina = SemanticRetina(grid_size=64, n_features=4096)
        activation = torch.zeros(1, 1, 64, 64)
        activation[0, 0, 32, 32] = 1.0
        out = retina(activation)
        # Center should retain significant activation
        assert out[0, 0, 32, 32] > 0.1
        # Adjacent cells should receive spread activation
        assert out[0, 0, 31, 32] > 0
        # Far corners should receive minimal activation
        assert out[0, 0, 0, 0] < 0.01

    def test_gaussian_kernel_normalized(self):
        """Test that Gaussian kernel sums to 1.0."""
        from neuromanifold_gpt.model.semantic_retina import SemanticRetina

        retina = SemanticRetina(grid_size=64, n_features=4096)
        assert abs(retina.gaussian_kernel.sum() - 1.0) < 1e-5

    def test_retina_different_grid_sizes(self):
        """Test retina works with different grid sizes."""
        from neuromanifold_gpt.model.semantic_retina import SemanticRetina

        for grid_size in [32, 64, 128]:
            retina = SemanticRetina(grid_size=grid_size, n_features=grid_size * grid_size)
            activation = torch.randn(1, 5, grid_size, grid_size)
            out = retina(activation)
            assert out.shape == activation.shape

    def test_retina_kernel_size_affects_spread(self):
        """Test that larger kernel spreads activation further."""
        from neuromanifold_gpt.model.semantic_retina import SemanticRetina

        activation = torch.zeros(1, 1, 64, 64)
        activation[0, 0, 32, 32] = 1.0

        retina_small = SemanticRetina(grid_size=64, n_features=4096, kernel_size=3, sigma=1.0)
        retina_large = SemanticRetina(grid_size=64, n_features=4096, kernel_size=7, sigma=2.0)

        out_small = retina_small(activation)
        out_large = retina_large(activation)

        # Larger kernel should spread more to distant cells
        assert out_large[0, 0, 29, 32] > out_small[0, 0, 29, 32]

    def test_retina_batch_independence(self):
        """Test that batches are processed independently."""
        from neuromanifold_gpt.model.semantic_retina import SemanticRetina

        retina = SemanticRetina(grid_size=64, n_features=4096)
        activation = torch.zeros(2, 1, 64, 64)
        activation[0, 0, 32, 32] = 1.0
        activation[1, 0, 10, 10] = 1.0

        out = retina(activation)

        # Check that batch 0's peak is at (32, 32)
        assert out[0, 0, 32, 32] > out[0, 0, 10, 10]
        # Check that batch 1's peak is at (10, 10)
        assert out[1, 0, 10, 10] > out[1, 0, 32, 32]

    def test_retina_gradient_flow(self):
        """Test that gradients flow through the retina."""
        from neuromanifold_gpt.model.semantic_retina import SemanticRetina

        retina = SemanticRetina(grid_size=64, n_features=4096)
        activation = torch.randn(2, 10, 64, 64, requires_grad=True)
        out = retina(activation)
        loss = out.sum()
        loss.backward()
        assert activation.grad is not None
        assert activation.grad.shape == activation.shape
