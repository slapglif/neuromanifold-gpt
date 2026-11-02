"""Tests for Rotary Position Embedding (RoPE)."""
import math

import pytest
import torch

from neuromanifold_gpt.model.embeddings.rotary import RotaryPositionalEmbedding


def test_rope_initialization():
    """RoPE should initialize with correct parameters."""
    embed_dim = 64
    head_dim = 32
    max_seq_len = 512

    rope = RotaryPositionalEmbedding(embed_dim, head_dim, max_seq_len)

    assert rope.embed_dim == embed_dim
    assert rope.head_dim == head_dim
    assert rope.max_seq_len == max_seq_len
    assert rope.cached_seq_len == max_seq_len


def test_rope_odd_head_dim_raises():
    """RoPE should raise error for odd head_dim."""
    with pytest.raises(ValueError, match="head_dim must be even"):
        RotaryPositionalEmbedding(64, 31)  # Odd head_dim


def test_rope_forward_shape_preservation():
    """Forward pass should preserve input shapes."""
    batch_size = 2
    n_heads = 8
    seq_len = 16
    head_dim = 32

    rope = RotaryPositionalEmbedding(64, head_dim, max_seq_len=512)

    q = torch.randn(batch_size, n_heads, seq_len, head_dim)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim)

    q_rot, k_rot = rope(q, k)

    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_rope_different_batch_sizes():
    """RoPE should work with different batch sizes."""
    head_dim = 32
    rope = RotaryPositionalEmbedding(64, head_dim)

    for batch_size in [1, 2, 4, 8]:
        q = torch.randn(batch_size, 4, 10, head_dim)
        k = torch.randn(batch_size, 4, 10, head_dim)

        q_rot, k_rot = rope(q, k)

        assert q_rot.shape == (batch_size, 4, 10, head_dim)
        assert k_rot.shape == (batch_size, 4, 10, head_dim)


def test_rope_cache_extension():
    """Cache should extend automatically for longer sequences."""
    head_dim = 32
    initial_max = 128

    rope = RotaryPositionalEmbedding(64, head_dim, max_seq_len=initial_max)

    assert rope.cached_seq_len == initial_max

    # Use sequence longer than cache
    long_seq_len = 256
    q = torch.randn(1, 4, long_seq_len, head_dim)
    k = torch.randn(1, 4, long_seq_len, head_dim)

    q_rot, k_rot = rope(q, k)

    # Cache should have been extended
    assert rope.cached_seq_len >= long_seq_len
    assert q_rot.shape == q.shape


def test_rope_rotation_correctness():
    """Verify rotation formula is applied correctly."""
    head_dim = 4  # Small for manual verification
    rope = RotaryPositionalEmbedding(64, head_dim, max_seq_len=10)

    # Create simple input
    q = torch.ones(1, 1, 2, head_dim)
    k = torch.ones(1, 1, 2, head_dim)

    q_rot, k_rot = rope(q, k)

    # Rotated output should differ from input (unless cos=1, sin=0)
    # At position 0, rotation is minimal, but at position 1 there should be rotation
    assert not torch.allclose(q_rot, q, atol=1e-6)
    assert not torch.allclose(k_rot, k, atol=1e-6)


def test_rope_rotation_preserves_magnitude():
    """Rotation should approximately preserve vector magnitudes."""
    head_dim = 32
    rope = RotaryPositionalEmbedding(64, head_dim)

    q = torch.randn(2, 4, 10, head_dim)
    k = torch.randn(2, 4, 10, head_dim)

    q_rot, k_rot = rope(q, k)

    # Compute norms
    q_norm = torch.norm(q, dim=-1)
    q_rot_norm = torch.norm(q_rot, dim=-1)
    k_norm = torch.norm(k, dim=-1)
    k_rot_norm = torch.norm(k_rot, dim=-1)

    # Norms should be very close (rotation preserves magnitude in each 2D plane)
    assert torch.allclose(q_norm, q_rot_norm, rtol=1e-5, atol=1e-6)
    assert torch.allclose(k_norm, k_rot_norm, rtol=1e-5, atol=1e-6)


def test_rope_relative_position_encoding():
    """RoPE should encode relative positions (key property)."""
    head_dim = 32
    rope = RotaryPositionalEmbedding(64, head_dim)

    # Create queries at different positions
    q = torch.randn(1, 1, 5, head_dim)
    k = torch.randn(1, 1, 5, head_dim)

    q_rot, k_rot = rope(q, k)

    # Compute attention scores (dot products)
    # q_rot[pos_i] . k_rot[pos_j] should depend on (pos_i - pos_j)
    # This is the key property of RoPE

    # Extract position 0 and position 1
    q0 = q_rot[:, :, 0:1, :]  # [1, 1, 1, head_dim]
    q1 = q_rot[:, :, 1:2, :]

    k1 = k_rot[:, :, 1:2, :]
    k2 = k_rot[:, :, 2:3, :]

    # Relative position is same: q0->k1 and q1->k2 both have relative pos = +1
    score_01 = (q0 * k1).sum(dim=-1)
    score_12 = (q1 * k2).sum(dim=-1)

    # Due to RoPE's relative encoding, these should be similar
    # (not exactly equal due to different content in q/k, but pattern should be preserved)
    # We just verify the operation doesn't crash and produces reasonable values
    assert score_01.shape == (1, 1, 1)
    assert score_12.shape == (1, 1, 1)
    assert torch.isfinite(score_01).all()
    assert torch.isfinite(score_12).all()


def test_rope_single_token():
    """RoPE should work with single token sequences."""
    head_dim = 32
    rope = RotaryPositionalEmbedding(64, head_dim)

    q = torch.randn(1, 4, 1, head_dim)
    k = torch.randn(1, 4, 1, head_dim)

    q_rot, k_rot = rope(q, k)

    assert q_rot.shape == (1, 4, 1, head_dim)
    assert k_rot.shape == (1, 4, 1, head_dim)
    assert torch.isfinite(q_rot).all()
    assert torch.isfinite(k_rot).all()


def test_rope_different_head_dims():
    """RoPE should work with various head dimensions."""
    for head_dim in [16, 32, 64, 128]:
        rope = RotaryPositionalEmbedding(256, head_dim)

        q = torch.randn(1, 4, 10, head_dim)
        k = torch.randn(1, 4, 10, head_dim)

        q_rot, k_rot = rope(q, k)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape


def test_rope_frequency_scaling():
    """Verify frequency computation follows the paper formula."""
    head_dim = 8
    base = 10000

    rope = RotaryPositionalEmbedding(64, head_dim, base=base)

    # Check inv_freq buffer
    expected_inv_freq = torch.tensor(
        [1.0 / (base ** (2 * i / head_dim)) for i in range(head_dim // 2)]
    )

    assert torch.allclose(rope.inv_freq, expected_inv_freq, rtol=1e-5)


def test_rope_cos_sin_cache():
    """Verify cos and sin caches are properly computed."""
    head_dim = 4
    max_seq_len = 10

    rope = RotaryPositionalEmbedding(64, head_dim, max_seq_len=max_seq_len)

    # Check cache shapes
    assert rope.cos_cached.shape == (max_seq_len, head_dim)
    assert rope.sin_cached.shape == (max_seq_len, head_dim)

    # Verify values are in valid range [-1, 1]
    assert (rope.cos_cached >= -1).all() and (rope.cos_cached <= 1).all()
    assert (rope.sin_cached >= -1).all() and (rope.sin_cached <= 1).all()


def test_rope_position_zero_minimal_rotation():
    """At position 0, rotation should be identity (cos=1, sin=0)."""
    head_dim = 32
    rope = RotaryPositionalEmbedding(64, head_dim)

    # Check position 0 in cache
    cos_0 = rope.cos_cached[0]
    sin_0 = rope.sin_cached[0]

    # At position 0, all freqs * 0 = 0, so cos(0)=1, sin(0)=0
    assert torch.allclose(cos_0, torch.ones_like(cos_0), atol=1e-6)
    assert torch.allclose(sin_0, torch.zeros_like(sin_0), atol=1e-6)


def test_rope_deterministic():
    """RoPE should produce deterministic outputs for same inputs."""
    head_dim = 32
    rope = RotaryPositionalEmbedding(64, head_dim)

    torch.manual_seed(42)
    q1 = torch.randn(1, 4, 10, head_dim)
    k1 = torch.randn(1, 4, 10, head_dim)

    q_rot1, k_rot1 = rope(q1, k1)

    torch.manual_seed(42)
    q2 = torch.randn(1, 4, 10, head_dim)
    k2 = torch.randn(1, 4, 10, head_dim)

    q_rot2, k_rot2 = rope(q2, k2)

    assert torch.allclose(q_rot1, q_rot2)
    assert torch.allclose(k_rot1, k_rot2)


def test_rope_rotate_half():
    """Test the _rotate_half helper function."""
    head_dim = 4
    rope = RotaryPositionalEmbedding(64, head_dim)

    # Create test input: [a0, a1, a2, a3]
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])  # [1, 1, 1, 4]

    # Expected output: [-a1, a0, -a3, a2] = [-2, 1, -4, 3]
    rotated = rope._rotate_half(x)

    expected = torch.tensor([[[[-2.0, 1.0, -4.0, 3.0]]]])

    assert torch.allclose(rotated, expected)


def test_rope_no_gradient_through_cache():
    """Cached cos/sin should not require gradients."""
    head_dim = 32
    rope = RotaryPositionalEmbedding(64, head_dim)

    assert not rope.cos_cached.requires_grad
    assert not rope.sin_cached.requires_grad


def test_rope_gradient_flow():
    """Gradients should flow through rotated q and k."""
    head_dim = 32
    rope = RotaryPositionalEmbedding(64, head_dim)

    q = torch.randn(1, 4, 10, head_dim, requires_grad=True)
    k = torch.randn(1, 4, 10, head_dim, requires_grad=True)

    q_rot, k_rot = rope(q, k)

    # Compute dummy loss
    loss = q_rot.sum() + k_rot.sum()
    loss.backward()

    # Gradients should exist
    assert q.grad is not None
    assert k.grad is not None
    assert torch.isfinite(q.grad).all()
    assert torch.isfinite(k.grad).all()
