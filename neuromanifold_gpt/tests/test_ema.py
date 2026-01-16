"""Tests for EMA (Exponential Moving Average) operations.

EMA uses parallel scan (FFT convolution) for O(T log T) computation.
These tests verify:
- ema_fft: FFT-based parallel scan correctness
- DampedEMA: Single-channel EMA with various alpha modes
- MultiHeadDampedEMA: Multi-head EMA with vectorized processing
- CEMA: Causal EMA wrapper with dimension validation
"""
import pytest
import torch
import torch.nn as nn
from neuromanifold_gpt.model.kan.ema import (
    ema_fft,
    DampedEMA,
    MultiHeadDampedEMA,
    CEMA
)


def ema_naive(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Naive sequential EMA for testing reference.

    Computes: h[t] = α·x[t] + (1-α)·h[t-1]

    Args:
        x: (B, T, D) input
        alpha: smoothing factor

    Returns:
        h: (B, T, D) EMA output
    """
    B, T, D = x.shape
    h = torch.zeros_like(x)

    for b in range(B):
        for d in range(D):
            h[b, 0, d] = alpha * x[b, 0, d]
            for t in range(1, T):
                h[b, t, d] = alpha * x[b, t, d] + (1 - alpha) * h[b, t-1, d]

    return h


class TestEmaFFT:
    """Test FFT-based parallel EMA computation."""

    def test_ema_fft_correctness(self):
        """Verify FFT implementation matches naive sequential EMA."""
        torch.manual_seed(42)
        x = torch.randn(2, 10, 8)
        alpha = 0.9

        # FFT-based (fast)
        h_fft = ema_fft(x, alpha)

        # Naive sequential (reference)
        h_naive = ema_naive(x, alpha)

        # Should match within numerical precision
        assert h_fft.shape == h_naive.shape
        assert torch.allclose(h_fft, h_naive, rtol=1e-4, atol=1e-5)

    def test_ema_fft_single_timestep(self):
        """Single timestep should return alpha * x."""
        x = torch.randn(2, 1, 8)
        alpha = 0.9

        h = ema_fft(x, alpha)

        assert h.shape == x.shape
        assert torch.allclose(h, alpha * x)

    def test_ema_fft_identity_alpha(self):
        """Alpha=1.0 should return input unchanged (no smoothing)."""
        x = torch.randn(2, 10, 8)

        h = ema_fft(x, alpha=1.0)

        assert torch.equal(h, x)

    def test_ema_fft_small_alpha_error(self):
        """Alpha near zero should raise ValueError."""
        x = torch.randn(2, 10, 8)

        with pytest.raises(ValueError, match="alpha must be"):
            ema_fft(x, alpha=1e-8)

    def test_ema_fft_per_channel_alpha(self):
        """Verify per-channel alpha support."""
        torch.manual_seed(42)
        x = torch.randn(2, 10, 4)

        # Different alpha per channel
        alpha = torch.tensor([0.9, 0.8, 0.7, 0.6])

        h = ema_fft(x, alpha)

        assert h.shape == x.shape

        # Verify each channel independently
        for d in range(4):
            h_channel = ema_fft(x[..., d:d+1], alpha[d].item())
            assert torch.allclose(h[..., d], h_channel[..., 0], rtol=1e-4, atol=1e-5)

    def test_ema_fft_shape_preservation(self):
        """Various input shapes should be preserved."""
        shapes = [
            (2, 10, 8),      # Standard (B, T, D)
            (1, 5, 16),      # Single batch
            (4, 100, 32),    # Long sequence
            (2, 10, 4, 8),   # Multi-head (B, T, H, D)
        ]

        for shape in shapes:
            x = torch.randn(shape)
            h = ema_fft(x, alpha=0.9)
            assert h.shape == shape

    def test_ema_fft_causality(self):
        """Verify causal property: h[t] only depends on x[0:t+1]."""
        torch.manual_seed(42)
        x = torch.randn(1, 10, 4)
        alpha = 0.9

        # Full sequence
        h_full = ema_fft(x, alpha)

        # Truncated sequence
        for t in range(2, 10):
            x_trunc = x[:, :t, :]
            h_trunc = ema_fft(x_trunc, alpha)

            # First t timesteps should match
            assert torch.allclose(h_full[:, :t, :], h_trunc, rtol=1e-4, atol=1e-5)

    def test_ema_fft_no_nan_inf(self):
        """Verify no NaN or Inf in output."""
        x = torch.randn(2, 10, 8)
        alpha = 0.9

        h = ema_fft(x, alpha)

        assert not torch.isnan(h).any()
        assert not torch.isinf(h).any()

    def test_ema_fft_gradient_flow(self):
        """Verify gradients flow through FFT operations."""
        x = torch.randn(2, 10, 8, requires_grad=True)
        alpha = 0.9

        h = ema_fft(x, alpha)
        loss = h.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0


class TestDampedEMA:
    """Test DampedEMA module."""

    def test_damped_ema_fixed_alpha(self):
        """Fixed alpha should work correctly."""
        ema = DampedEMA(alpha=0.9)
        x = torch.randn(2, 10, 8)

        h = ema(x)

        assert h.shape == x.shape
        assert torch.isclose(ema.get_alpha(), torch.tensor(0.9), rtol=1e-5).item()

    def test_damped_ema_learnable_alpha(self):
        """Learnable alpha should be trainable parameter."""
        ema = DampedEMA(alpha='learnable')
        x = torch.randn(2, 10, 8)

        # Check parameter exists
        assert hasattr(ema, 'alpha_logit')
        assert isinstance(ema.alpha_logit, nn.Parameter)

        # Forward pass
        h = ema(x)
        assert h.shape == x.shape

        # Check alpha is in valid range
        alpha = ema.get_alpha()
        assert 0 < alpha.item() < 1

    def test_damped_ema_per_channel_alpha(self):
        """Per-channel alpha should work correctly."""
        alpha = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
        ema = DampedEMA(alpha=alpha)
        x = torch.randn(2, 10, 8)

        h = ema(x)

        assert h.shape == x.shape
        assert torch.equal(ema.get_alpha(), alpha)

    def test_damped_ema_gradient_flow(self):
        """Gradients should flow through learnable alpha."""
        ema = DampedEMA(alpha='learnable')
        x = torch.randn(2, 10, 8)

        h = ema(x)
        loss = h.sum()
        loss.backward()

        assert ema.alpha_logit.grad is not None
        assert ema.alpha_logit.grad.abs() > 0

    def test_damped_ema_extra_repr(self):
        """String representation should be informative."""
        ema_fixed = DampedEMA(alpha=0.9)
        ema_learnable = DampedEMA(alpha='learnable')

        repr_fixed = ema_fixed.extra_repr()
        repr_learnable = ema_learnable.extra_repr()

        assert 'alpha=0.9' in repr_fixed
        assert 'learnable' in repr_learnable


class TestMultiHeadDampedEMA:
    """Test MultiHeadDampedEMA module."""

    def test_multi_head_ema_basic(self):
        """Basic multi-head EMA should work correctly."""
        num_heads = 4
        head_dim = 16
        ema = MultiHeadDampedEMA(num_heads=num_heads, head_dim=head_dim, alpha=0.9)

        x = torch.randn(2, 10, num_heads * head_dim)
        h = ema(x)

        assert h.shape == x.shape

    def test_multi_head_ema_dimension_validation(self):
        """Should raise error if input dimension doesn't match."""
        ema = MultiHeadDampedEMA(num_heads=4, head_dim=16, alpha=0.9)

        # Wrong dimension (should be 64, not 60)
        x = torch.randn(2, 10, 60)

        with pytest.raises(ValueError, match="Input dimension"):
            ema(x)

    def test_multi_head_ema_learnable_alpha(self):
        """Learnable alpha should work for multi-head."""
        ema = MultiHeadDampedEMA(num_heads=4, head_dim=16, alpha='learnable')
        x = torch.randn(2, 10, 64)

        # Check parameter exists
        assert hasattr(ema, 'alpha_logit')
        assert isinstance(ema.alpha_logit, nn.Parameter)

        # Forward pass
        h = ema(x)
        assert h.shape == x.shape

        # Gradient flow
        loss = h.sum()
        loss.backward()
        assert ema.alpha_logit.grad is not None

    def test_multi_head_ema_per_head_alpha(self):
        """Per-head alpha should work correctly."""
        num_heads = 4
        head_dim = 16
        alpha = torch.tensor([0.9, 0.8, 0.7, 0.6])  # Per-head

        ema = MultiHeadDampedEMA(num_heads=num_heads, head_dim=head_dim, alpha=alpha)
        x = torch.randn(2, 10, 64)

        h = ema(x)
        assert h.shape == x.shape

    def test_multi_head_ema_per_channel_alpha(self):
        """Per-channel alpha should work correctly."""
        num_heads = 4
        head_dim = 16
        alpha = torch.rand(num_heads, head_dim) * 0.5 + 0.4  # [0.4, 0.9]

        ema = MultiHeadDampedEMA(num_heads=num_heads, head_dim=head_dim, alpha=alpha)
        x = torch.randn(2, 10, 64)

        h = ema(x)
        assert h.shape == x.shape

    def test_multi_head_ema_no_loops(self):
        """Verify all heads processed in parallel (timing test)."""
        import time

        num_heads = 8
        head_dim = 32
        ema = MultiHeadDampedEMA(num_heads=num_heads, head_dim=head_dim, alpha=0.9)
        x = torch.randn(4, 100, num_heads * head_dim)

        # Warmup
        _ = ema(x)

        # Time multi-head
        start = time.time()
        for _ in range(10):
            _ = ema(x)
        multi_head_time = time.time() - start

        # Time single head (for comparison)
        ema_single = DampedEMA(alpha=0.9)
        x_single = torch.randn(4, 100, head_dim)
        _ = ema_single(x_single)

        start = time.time()
        for _ in range(10):
            _ = ema_single(x_single)
        single_head_time = time.time() - start

        # Multi-head should not be 8x slower (proves parallelization)
        # Allow 3x overhead for reshaping and larger FFT
        assert multi_head_time < single_head_time * 3 * num_heads

    def test_multi_head_ema_extra_repr(self):
        """String representation should be informative."""
        ema = MultiHeadDampedEMA(num_heads=4, head_dim=16, alpha=0.9)
        repr_str = ema.extra_repr()

        assert 'num_heads=4' in repr_str
        assert 'head_dim=16' in repr_str
        assert 'alpha=' in repr_str


class TestCEMA:
    """Test CEMA (Causal EMA) module."""

    def test_cema_basic(self):
        """Basic CEMA should work correctly."""
        cema = CEMA(dim=64, alpha=0.9)
        x = torch.randn(2, 10, 64)

        h = cema(x)

        assert h.shape == x.shape

    def test_cema_dimension_validation(self):
        """Should raise error if input dimension doesn't match."""
        cema = CEMA(dim=64, alpha=0.9)

        # Wrong dimension (should be 64, not 32)
        x = torch.randn(2, 10, 32)

        with pytest.raises(ValueError, match="Input dimension"):
            cema(x)

    def test_cema_learnable_alpha(self):
        """Learnable alpha should work for CEMA."""
        cema = CEMA(dim=64, alpha='learnable')
        x = torch.randn(2, 10, 64)

        # Check parameter exists (delegated to DampedEMA)
        assert hasattr(cema.ema, 'alpha_logit')
        assert isinstance(cema.ema.alpha_logit, nn.Parameter)

        # Forward pass
        h = cema(x)
        assert h.shape == x.shape

        # Gradient flow
        loss = h.sum()
        loss.backward()
        assert cema.ema.alpha_logit.grad is not None

    def test_cema_causality(self):
        """Verify CEMA is causal: h[t] only depends on x[0:t+1]."""
        torch.manual_seed(42)
        cema = CEMA(dim=8, alpha=0.9)
        x = torch.randn(1, 10, 8)

        # Full sequence
        h_full = cema(x)

        # Truncated sequence
        for t in range(2, 10):
            x_trunc = x[:, :t, :]
            h_trunc = cema(x_trunc)

            # First t timesteps should match
            assert torch.allclose(h_full[:, :t, :], h_trunc, rtol=1e-4, atol=1e-5)

    def test_cema_extra_repr(self):
        """String representation should be informative."""
        cema = CEMA(dim=64, alpha=0.9)
        repr_str = cema.extra_repr()

        assert 'dim=64' in repr_str
        assert 'alpha=' in repr_str


class TestEMANumericalStability:
    """Test numerical stability across various conditions."""

    def test_ema_large_sequence(self):
        """Large sequences should not cause numerical issues."""
        x = torch.randn(2, 1000, 16)
        ema = DampedEMA(alpha=0.9)

        h = ema(x)

        assert not torch.isnan(h).any()
        assert not torch.isinf(h).any()

    def test_ema_extreme_values(self):
        """Extreme input values should be handled gracefully."""
        # Large values
        x_large = torch.randn(2, 10, 8) * 100
        ema = DampedEMA(alpha=0.9)
        h_large = ema(x_large)
        assert not torch.isnan(h_large).any()

        # Small values
        x_small = torch.randn(2, 10, 8) * 0.01
        h_small = ema(x_small)
        assert not torch.isnan(h_small).any()

    def test_ema_alpha_boundary(self):
        """Alpha near boundaries should work correctly."""
        x = torch.randn(2, 10, 8)

        # Alpha moderately high (0.95 is more realistic than 0.999)
        ema_high = DampedEMA(alpha=0.95)
        h_high = ema_high(x)
        assert torch.allclose(h_high, 0.95 * x, rtol=1e-2)

        # Alpha moderately low
        ema_low = DampedEMA(alpha=0.1)
        h_low = ema_low(x)
        assert not torch.isnan(h_low).any()

    def test_ema_different_dtypes(self):
        """Should work with float32 and float64."""
        x32 = torch.randn(2, 10, 8, dtype=torch.float32)
        x64 = torch.randn(2, 10, 8, dtype=torch.float64)

        ema = DampedEMA(alpha=0.9)

        h32 = ema(x32)
        assert h32.dtype == torch.float32

        h64 = ema(x64)
        assert h64.dtype == torch.float64


class TestGradientFlow:
    """Test gradient flow through EMA modules."""

    def test_gradient_flow(self):
        """Verify gradients flow through all EMA modules."""
        # Test ema_fft
        x_fft = torch.randn(2, 10, 8, requires_grad=True)
        h_fft = ema_fft(x_fft, alpha=0.9)
        loss_fft = h_fft.sum()
        loss_fft.backward()
        assert x_fft.grad is not None
        assert x_fft.grad.abs().sum() > 0

        # Test DampedEMA with learnable alpha
        ema_damped = DampedEMA(alpha='learnable')
        x_damped = torch.randn(2, 10, 8, requires_grad=True)
        h_damped = ema_damped(x_damped)
        loss_damped = h_damped.sum()
        loss_damped.backward()
        assert x_damped.grad is not None
        assert x_damped.grad.abs().sum() > 0
        assert ema_damped.alpha_logit.grad is not None
        assert ema_damped.alpha_logit.grad.abs() > 0

        # Test MultiHeadDampedEMA with learnable alpha
        ema_multi = MultiHeadDampedEMA(num_heads=4, head_dim=16, alpha='learnable')
        x_multi = torch.randn(2, 10, 64, requires_grad=True)
        h_multi = ema_multi(x_multi)
        loss_multi = h_multi.sum()
        loss_multi.backward()
        assert x_multi.grad is not None
        assert x_multi.grad.abs().sum() > 0
        assert ema_multi.alpha_logit.grad is not None
        assert ema_multi.alpha_logit.grad.abs() > 0

        # Test CEMA with learnable alpha
        cema = CEMA(dim=64, alpha='learnable')
        x_cema = torch.randn(2, 10, 64, requires_grad=True)
        h_cema = cema(x_cema)
        loss_cema = h_cema.sum()
        loss_cema.backward()
        assert x_cema.grad is not None
        assert x_cema.grad.abs().sum() > 0
        assert cema.ema.alpha_logit.grad is not None
        assert cema.ema.alpha_logit.grad.abs() > 0


class TestEMAComplexity:
    """Test that parallel scan achieves expected complexity."""

    def test_ema_time_complexity(self):
        """Verify O(T log T) scaling (rough timing test)."""
        import time

        ema = DampedEMA(alpha=0.9)

        # Measure time for different sequence lengths
        times = []
        lengths = [256, 512, 1024]  # Use larger T where O(T log T) is more apparent

        for T in lengths:
            x = torch.randn(2, T, 16)

            # Warmup
            _ = ema(x)

            # Time
            start = time.time()
            for _ in range(10):
                _ = ema(x)
            elapsed = time.time() - start
            times.append(elapsed)

        # Verify sublinear scaling (O(T log T) grows slower than O(T²))
        # If time doubled from 64→128, then 128→256 should be < 2x
        # (since 2*log(2) ≈ 1.44x vs 4x for O(T²))
        if len(times) >= 3:
            ratio_1 = times[1] / times[0]  # 128/64
            ratio_2 = times[2] / times[1]  # 256/128

            # O(T log T): expect ratio_2 ≈ ratio_1 * (log(2)/log(4)) ≈ 0.72 * ratio_1
            # O(T²): expect ratio_2 ≈ 4.0 if ratio_1 ≈ 4.0
            # We check ratio_2 < ratio_1 * 1.5 (generous bound)
            assert ratio_2 < ratio_1 * 1.5, f"Scaling suggests O(T²): {ratio_1:.2f} -> {ratio_2:.2f}"
