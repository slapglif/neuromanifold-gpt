"""Tests for Attention with Linear Biases (ALiBi)."""
import pytest
import torch
import math
from neuromanifold_gpt.model.embeddings.alibi import ALiBiPositionalBias


def test_alibi_initialization():
    """ALiBi should initialize with correct parameters."""
    n_heads = 8
    embed_dim = 512
    max_seq_len = 2048

    alibi = ALiBiPositionalBias(n_heads, embed_dim, max_seq_len)

    assert alibi.n_heads == n_heads
    assert alibi.embed_dim == embed_dim
    assert alibi.max_seq_len == max_seq_len
    assert alibi.cached_seq_len == max_seq_len


def test_alibi_forward_shape():
    """Forward pass should return correct bias shape."""
    n_heads = 8
    seq_len = 16

    alibi = ALiBiPositionalBias(n_heads, 512)

    bias = alibi(seq_len)

    # Shape should be [n_heads, 1, seq_len, seq_len]
    assert bias.shape == (n_heads, 1, seq_len, seq_len)


def test_alibi_different_seq_lengths():
    """ALiBi should work with different sequence lengths."""
    n_heads = 4
    alibi = ALiBiPositionalBias(n_heads, 512)

    for seq_len in [1, 8, 16, 32, 64]:
        bias = alibi(seq_len)
        assert bias.shape == (n_heads, 1, seq_len, seq_len)
        assert torch.isfinite(bias).all()


def test_alibi_cache_extension():
    """Cache should extend automatically for longer sequences."""
    n_heads = 4
    initial_max = 128

    alibi = ALiBiPositionalBias(n_heads, 512, max_seq_len=initial_max)

    assert alibi.cached_seq_len == initial_max

    # Use sequence longer than cache
    long_seq_len = 256
    bias = alibi(long_seq_len)

    # Cache should have been extended
    assert alibi.cached_seq_len >= long_seq_len
    assert bias.shape == (n_heads, 1, long_seq_len, long_seq_len)


def test_alibi_slopes_power_of_2():
    """Verify slopes computation for power-of-2 heads."""
    n_heads = 8
    alibi = ALiBiPositionalBias(n_heads, 512)

    # For power of 2, slopes should follow: 2^(-8/n), 2^(-16/n), ..., 2^(-8)
    expected_slopes = torch.tensor([
        2 ** (-8 * (i + 1) / n_heads) for i in range(n_heads)
    ])

    assert torch.allclose(alibi.slopes, expected_slopes, rtol=1e-5)


def test_alibi_slopes_non_power_of_2():
    """ALiBi should handle non-power-of-2 head counts."""
    n_heads = 6  # Not a power of 2
    alibi = ALiBiPositionalBias(n_heads, 512)

    # Should have n_heads slopes
    assert alibi.slopes.shape == (n_heads,)

    # All slopes should be positive and decreasing
    assert (alibi.slopes > 0).all()
    # Slopes should be in descending order (first head has largest slope)
    assert (alibi.slopes[:-1] >= alibi.slopes[1:]).all()


def test_alibi_bias_formula():
    """Verify bias follows formula: -slope * |i - j|."""
    n_heads = 4
    seq_len = 5
    alibi = ALiBiPositionalBias(n_heads, 512)

    bias = alibi(seq_len)

    # Check a few specific positions for head 0
    slope_0 = alibi.slopes[0].item()

    # bias[h, 0, i, j] = -slope[h] * |i - j|
    # Position [0, 0]: distance = 0
    assert abs(bias[0, 0, 0, 0].item()) < 1e-6

    # Position [0, 2]: distance = 2
    expected_bias_02 = -slope_0 * 2
    assert abs(bias[0, 0, 0, 2].item() - expected_bias_02) < 1e-5

    # Position [2, 4]: distance = 2
    expected_bias_24 = -slope_0 * 2
    assert abs(bias[0, 0, 2, 4].item() - expected_bias_24) < 1e-5

    # Position [4, 1]: distance = 3
    expected_bias_41 = -slope_0 * 3
    assert abs(bias[0, 0, 4, 1].item() - expected_bias_41) < 1e-5


def test_alibi_diagonal_is_zero():
    """Diagonal of bias matrix should be zero (distance = 0)."""
    n_heads = 8
    seq_len = 16
    alibi = ALiBiPositionalBias(n_heads, 512)

    bias = alibi(seq_len)

    # Extract diagonal for each head
    for h in range(n_heads):
        diagonal = torch.diagonal(bias[h, 0])
        assert torch.allclose(diagonal, torch.zeros_like(diagonal), atol=1e-6)


def test_alibi_symmetry():
    """Bias should be symmetric: |i-j| = |j-i|."""
    n_heads = 4
    seq_len = 10
    alibi = ALiBiPositionalBias(n_heads, 512)

    bias = alibi(seq_len)

    # Check symmetry for all heads
    for h in range(n_heads):
        bias_matrix = bias[h, 0]
        assert torch.allclose(bias_matrix, bias_matrix.T, atol=1e-6)


def test_alibi_negative_values():
    """All bias values (except diagonal) should be negative."""
    n_heads = 8
    seq_len = 16
    alibi = ALiBiPositionalBias(n_heads, 512)

    bias = alibi(seq_len)

    # Create mask for off-diagonal elements
    mask = ~torch.eye(seq_len, dtype=bool)

    # All off-diagonal elements should be negative
    for h in range(n_heads):
        off_diagonal_values = bias[h, 0][mask]
        assert (off_diagonal_values < 0).all()


def test_alibi_increasing_penalty_with_distance():
    """Bias penalty should increase (become more negative) with distance."""
    n_heads = 4
    seq_len = 10
    alibi = ALiBiPositionalBias(n_heads, 512)

    bias = alibi(seq_len)

    # For each head, check that bias gets more negative as distance increases
    for h in range(n_heads):
        # Look at first row (position 0 attending to others)
        first_row = bias[h, 0, 0, :]

        # Values should decrease (become more negative) with index
        # first_row[0] = 0, first_row[1] < 0, first_row[2] < first_row[1], etc.
        for i in range(seq_len - 1):
            assert first_row[i] > first_row[i + 1] or abs(first_row[i] - first_row[i + 1]) < 1e-6


def test_alibi_different_slopes_per_head():
    """Each head should have a different slope."""
    n_heads = 8
    alibi = ALiBiPositionalBias(n_heads, 512)

    # All slopes should be unique
    slopes_list = alibi.slopes.tolist()
    assert len(slopes_list) == len(set(slopes_list))


def test_alibi_single_token():
    """ALiBi should work with single token sequences."""
    n_heads = 4
    alibi = ALiBiPositionalBias(n_heads, 512)

    bias = alibi(1)

    assert bias.shape == (n_heads, 1, 1, 1)
    # Single token has no distance, bias should be 0
    assert torch.allclose(bias, torch.zeros_like(bias), atol=1e-6)


def test_alibi_no_learnable_parameters():
    """ALiBi should have no learnable parameters."""
    n_heads = 8
    alibi = ALiBiPositionalBias(n_heads, 512)

    # Count learnable parameters
    num_params = sum(p.numel() for p in alibi.parameters() if p.requires_grad)
    assert num_params == 0


def test_alibi_deterministic():
    """ALiBi should produce deterministic outputs."""
    n_heads = 4
    seq_len = 16
    alibi = ALiBiPositionalBias(n_heads, 512)

    bias1 = alibi(seq_len)
    bias2 = alibi(seq_len)

    assert torch.allclose(bias1, bias2)


def test_alibi_device_consistency():
    """ALiBi bias should be on the same device as the module."""
    n_heads = 4
    seq_len = 16
    alibi = ALiBiPositionalBias(n_heads, 512)

    bias = alibi(seq_len)

    # Slopes and bias should be on the same device
    assert bias.device == alibi.slopes.device


def test_alibi_dtype_consistency():
    """ALiBi bias should maintain consistent dtype."""
    n_heads = 4
    seq_len = 16
    alibi = ALiBiPositionalBias(n_heads, 512)

    bias = alibi(seq_len)

    # Bias should have same dtype as slopes
    assert bias.dtype == alibi.slopes.dtype


def test_alibi_broadcast_compatibility():
    """ALiBi bias shape should be compatible with attention score broadcasting."""
    batch_size = 2
    n_heads = 8
    seq_len = 16

    alibi = ALiBiPositionalBias(n_heads, 512)
    bias = alibi(seq_len)

    # Simulate attention scores: [batch, n_heads, seq_len, seq_len]
    attn_scores = torch.randn(batch_size, n_heads, seq_len, seq_len)

    # Bias should broadcast correctly when added to attention scores
    biased_scores = attn_scores + bias

    assert biased_scores.shape == (batch_size, n_heads, seq_len, seq_len)


def test_alibi_extrapolation():
    """ALiBi should support sequences longer than training length."""
    n_heads = 4
    train_seq_len = 64
    test_seq_len = 128

    alibi = ALiBiPositionalBias(n_heads, 512, max_seq_len=train_seq_len)

    # Should work with longer sequence
    bias = alibi(test_seq_len)

    assert bias.shape == (n_heads, 1, test_seq_len, test_seq_len)
    assert torch.isfinite(bias).all()

    # Verify formula still holds at longer distances
    slope_0 = alibi.slopes[0].item()
    # Check position [0, test_seq_len-1]: maximum distance
    max_distance = test_seq_len - 1
    expected_bias = -slope_0 * max_distance
    actual_bias = bias[0, 0, 0, test_seq_len - 1].item()

    assert abs(actual_bias - expected_bias) < 1e-4


def test_alibi_head_specialization():
    """Different heads should have different bias patterns."""
    n_heads = 8
    seq_len = 16
    alibi = ALiBiPositionalBias(n_heads, 512)

    bias = alibi(seq_len)

    # Compare bias patterns across heads
    # Head 0 should have strongest penalties (largest slope)
    # Head n-1 should have weakest penalties (smallest slope)

    # At distance 10, compare penalty magnitude
    penalty_head0 = abs(bias[0, 0, 0, 10].item())
    penalty_head_last = abs(bias[n_heads - 1, 0, 0, 10].item())

    # First head should have larger absolute penalty
    assert penalty_head0 > penalty_head_last


def test_alibi_linear_scaling():
    """Bias should scale linearly with distance."""
    n_heads = 4
    seq_len = 20
    alibi = ALiBiPositionalBias(n_heads, 512)

    bias = alibi(seq_len)

    # For head 0, check linearity
    slope = alibi.slopes[0].item()

    # bias[0, 0, 0, d] should equal -slope * d
    for distance in [1, 5, 10, 15]:
        expected = -slope * distance
        actual = bias[0, 0, 0, distance].item()
        assert abs(actual - expected) < 1e-5


def test_alibi_small_head_count():
    """ALiBi should work with very small head counts."""
    for n_heads in [1, 2, 3]:
        alibi = ALiBiPositionalBias(n_heads, 512)
        bias = alibi(10)

        assert bias.shape == (n_heads, 1, 10, 10)
        assert torch.isfinite(bias).all()


def test_alibi_large_head_count():
    """ALiBi should work with large head counts."""
    n_heads = 32
    alibi = ALiBiPositionalBias(n_heads, 512)

    bias = alibi(16)

    assert bias.shape == (n_heads, 1, 16, 16)
    assert torch.isfinite(bias).all()
    assert alibi.slopes.shape == (n_heads,)
    # All slopes should be positive
    assert (alibi.slopes > 0).all()
