# neuromanifold_gpt/tests/test_faster_kan.py
"""
Tests for FasterKAN layers (RSWAF basis).

Verifies:
- RSWAF basis function properties
- Forward pass shapes
- Backward pass (gradients)
- Numerical stability
- Causality preservation (position independence)
"""

import pytest
import torch
import torch.nn as nn

from neuromanifold_gpt.model.kan.faster import (
    RSWAFBasis,
    FasterKANLayer,
    FasterKANFFN,
    FasterKANLinear,
    replace_linear_with_fasterkan,
)


class TestRSWAFBasis:
    """Tests for RSWAF basis functions."""

    @pytest.fixture
    def basis(self):
        """Standard RSWAF basis for testing."""
        return RSWAFBasis(num_centers=8, h=1.0, grid_range=(-1.0, 1.0))

    @pytest.fixture
    def input_tensor(self):
        """Standard input tensor (B=2, T=8, D=64)."""
        torch.manual_seed(42)
        return torch.randn(2, 8, 64)

    def test_output_shape(self, basis, input_tensor):
        """Test RSWAF output shape."""
        output = basis(input_tensor)
        # Output adds num_centers dimension: (B, T, D, num_centers)
        assert output.shape == (2, 8, 64, 8), f"Got shape {output.shape}"

    def test_output_range(self, basis, input_tensor):
        """Test RSWAF output is in [0, 1] range."""
        output = basis(input_tensor)
        # RSWAF: 1 - tanh(x)^2 is in [0, 1]
        assert output.min() >= 0.0, f"Min={output.min()}"
        assert output.max() <= 1.0, f"Max={output.max()}"

    def test_peak_at_center(self, basis):
        """Test RSWAF peaks at grid center."""
        # At center, (u - u_i) / h = 0, so tanh(0) = 0, basis = 1 - 0 = 1
        x = torch.tensor([[[0.0]]])  # Input at grid center
        output = basis(x)
        # Find center index (grid is [-1, 1] with 8 points)
        # Centers: -1, -0.71, -0.43, -0.14, 0.14, 0.43, 0.71, 1
        # Point 0.0 is between center 3 and 4
        # Let's test at actual grid point
        basis_at_center = RSWAFBasis(num_centers=3, h=1.0, grid_range=(-1.0, 1.0))
        x_at_center = torch.tensor([[[0.0]]])  # Middle grid point
        out = basis_at_center(x_at_center)
        # At center, output should be 1.0
        assert out[0, 0, 0, 1].item() == pytest.approx(1.0, abs=1e-5)

    def test_gradient_flow(self, basis, input_tensor):
        """Test gradients flow through RSWAF basis."""
        input_tensor.requires_grad_(True)
        output = basis(input_tensor)
        loss = output.sum()
        loss.backward()
        assert input_tensor.grad is not None
        assert not torch.allclose(input_tensor.grad, torch.zeros_like(input_tensor.grad))

    def test_learnable_parameters(self):
        """Test learnable grid and h parameters."""
        basis = RSWAFBasis(num_centers=8, h=1.0, learnable_h=True, learnable_grid=True)
        assert isinstance(basis.h, nn.Parameter)
        assert isinstance(basis.grid, nn.Parameter)

        # Non-learnable version
        basis_fixed = RSWAFBasis(num_centers=8, h=1.0, learnable_h=False, learnable_grid=False)
        assert not isinstance(basis_fixed.h, nn.Parameter)
        assert not isinstance(basis_fixed.grid, nn.Parameter)


class TestFasterKANLayer:
    """Tests for FasterKANLayer."""

    @pytest.fixture
    def layer(self):
        """Standard FasterKAN layer for testing."""
        return FasterKANLayer(in_features=64, out_features=128, num_centers=8, bias=True)

    @pytest.fixture
    def input_tensor(self):
        """Standard input tensor (B=2, T=8, D=64)."""
        torch.manual_seed(42)
        return torch.randn(2, 8, 64)

    def test_forward_shape(self, layer, input_tensor):
        """Test forward pass produces correct output shape."""
        output = layer(input_tensor)
        assert output.shape == (2, 8, 128), f"Got shape {output.shape}"

    def test_backward_pass(self, layer, input_tensor):
        """Test backward pass computes gradients."""
        input_tensor.requires_grad = True
        output = layer(input_tensor)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert input_tensor.grad is not None
        assert not torch.allclose(input_tensor.grad, torch.zeros_like(input_tensor.grad))

        # Check parameter gradients
        assert layer.spline_weight.grad is not None
        assert layer.base_weight.grad is not None
        assert layer.bias.grad is not None

    def test_numerical_stability(self, layer):
        """Test layer handles extreme inputs without NaN/Inf."""
        # Large inputs
        x_large = torch.randn(2, 8, 64) * 100
        out_large = layer(x_large)
        assert not torch.isnan(out_large).any(), "NaN detected with large inputs"
        assert not torch.isinf(out_large).any(), "Inf detected with large inputs"

        # Small inputs
        x_small = torch.randn(2, 8, 64) * 0.01
        out_small = layer(x_small)
        assert not torch.isnan(out_small).any(), "NaN detected with small inputs"

        # Zero inputs
        x_zero = torch.zeros(2, 8, 64)
        out_zero = layer(x_zero)
        assert not torch.isnan(out_zero).any(), "NaN detected with zero inputs"

    def test_no_base_weight(self):
        """Test layer without base weight."""
        layer = FasterKANLayer(64, 128, num_centers=8, use_base=False)
        assert layer.base_weight is None

        x = torch.randn(2, 4, 64)
        output = layer(x)
        assert output.shape == (2, 4, 128)

    def test_no_bias(self):
        """Test layer without bias."""
        layer = FasterKANLayer(64, 128, num_centers=8, bias=False)
        assert layer.bias is None

        x = torch.randn(2, 4, 64)
        output = layer(x)
        assert output.shape == (2, 4, 128)

    @pytest.mark.parametrize("num_centers", [4, 8, 16])
    def test_different_centers(self, num_centers):
        """Test layer with different number of centers."""
        layer = FasterKANLayer(32, 64, num_centers=num_centers)
        x = torch.randn(2, 4, 32)
        output = layer(x)
        assert output.shape == (2, 4, 64)
        assert not torch.isnan(output).any()


class TestFasterKANFFN:
    """Tests for FasterKANFFN (2-layer network)."""

    @pytest.fixture
    def ffn(self):
        """Standard FasterKAN FFN."""
        return FasterKANFFN(dim=128, hidden_dim=512, num_centers=8, dropout=0.1)

    @pytest.fixture
    def input_tensor(self):
        """Standard input tensor."""
        torch.manual_seed(42)
        return torch.randn(2, 8, 128)

    def test_forward_shape(self, ffn, input_tensor):
        """Test FFN preserves input shape."""
        output = ffn(input_tensor)
        assert output.shape == input_tensor.shape, (
            f"Expected {input_tensor.shape}, got {output.shape}"
        )

    def test_backward_pass(self, ffn, input_tensor):
        """Test FFN backward pass."""
        input_tensor.requires_grad = True
        output = ffn(input_tensor)
        loss = output.sum()
        loss.backward()

        assert input_tensor.grad is not None
        assert not torch.allclose(input_tensor.grad, torch.zeros_like(input_tensor.grad))

    def test_dropout_behavior(self):
        """Test dropout is applied during training but not eval."""
        ffn = FasterKANFFN(dim=64, hidden_dim=256, num_centers=8, dropout=0.5)
        x = torch.randn(4, 16, 64)

        # Training mode: outputs should vary
        ffn.train()
        out1 = ffn(x)
        out2 = ffn(x)
        assert not torch.allclose(out1, out2, atol=1e-5)

        # Eval mode: outputs should be deterministic
        ffn.eval()
        out3 = ffn(x)
        out4 = ffn(x)
        assert torch.allclose(out3, out4)

    def test_residual_compatibility(self, ffn, input_tensor):
        """Test FFN output can be added to input (residual connection)."""
        output = ffn(input_tensor)
        residual = input_tensor + output

        assert residual.shape == input_tensor.shape
        assert not torch.isnan(residual).any()

    def test_numerical_stability(self, ffn):
        """Test FFN handles extreme inputs."""
        x_large = torch.randn(2, 8, 128) * 100
        out = ffn(x_large)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


class TestFasterKANLinear:
    """Tests for FasterKANLinear (drop-in replacement)."""

    def test_forward_shape(self):
        """Test FasterKANLinear preserves Linear interface."""
        layer = FasterKANLinear(64, 128)
        x = torch.randn(2, 8, 64)
        output = layer(x)
        assert output.shape == (2, 8, 128)

    def test_backward_pass(self):
        """Test FasterKANLinear backward pass."""
        layer = FasterKANLinear(64, 128)
        x = torch.randn(2, 8, 64)
        x.requires_grad = True
        output = layer(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None


class TestCausalityPreservation:
    """Test that FasterKAN preserves causality in autoregressive setting."""

    def test_position_independence(self):
        """Test that each position is processed independently."""
        layer = FasterKANLayer(in_features=32, out_features=32, num_centers=8)
        layer.eval()

        # Create input and modify position 5
        x1 = torch.randn(1, 10, 32)
        x2 = x1.clone()
        x2[:, 5, :] = torch.randn(1, 32)

        out1 = layer(x1)
        out2 = layer(x2)

        # LayerNorm operates per-position (over features), so positions before 5
        # should be identical if causality is preserved
        # NOTE: LayerNorm doesn't look at other positions, so this should pass
        # Check positions 0-4 are identical
        assert torch.allclose(out1[:, :5, :], out2[:, :5, :], atol=1e-5), (
            "Causality violation: position 5 change affected earlier positions"
        )

        # Position 5+ can differ
        assert not torch.allclose(out1[:, 5, :], out2[:, 5, :])


class TestReplaceLinearWithFasterKAN:
    """Test the replace_linear_with_fasterkan utility."""

    def test_replace_linear(self):
        """Test replacing nn.Linear with FasterKANLinear."""
        # Create a simple module with Linear layers
        module = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        # Replace
        module = replace_linear_with_fasterkan(module, num_centers=4)

        # Check replacement
        assert isinstance(module[0], FasterKANLinear)
        assert isinstance(module[2], FasterKANLinear)

        # Test forward pass still works
        x = torch.randn(2, 4, 32)
        output = module(x)
        assert output.shape == (2, 4, 32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestFasterKANCUDA:
    """Test FasterKAN on CUDA."""

    def test_cuda_forward(self):
        """Test forward pass on CUDA."""
        layer = FasterKANLayer(64, 128, num_centers=8).cuda()
        x = torch.randn(2, 8, 64).cuda()

        output = layer(x)
        assert output.is_cuda
        assert output.shape == (2, 8, 128)

    def test_cuda_backward(self):
        """Test backward pass on CUDA."""
        layer = FasterKANLayer(64, 128, num_centers=8).cuda()
        x = torch.randn(2, 8, 64).cuda()
        x.requires_grad_(True)

        output = layer(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.is_cuda

    def test_cuda_ffn(self):
        """Test FasterKANFFN on CUDA."""
        ffn = FasterKANFFN(dim=128, hidden_dim=512, num_centers=8).cuda()
        x = torch.randn(2, 8, 128).cuda()

        output = ffn(x)
        assert output.is_cuda
        assert output.shape == (2, 8, 128)
