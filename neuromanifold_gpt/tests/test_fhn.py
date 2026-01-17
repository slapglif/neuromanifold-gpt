# neuromanifold_gpt/tests/test_fhn.py
"""Tests for FHN Excitable Attention mechanism."""
import pytest
import torch

from neuromanifold_gpt.model.attention.fhn import FHNAttention


def test_fhn_output_shape():
    """Output should match input dimensions."""
    attn = FHNAttention(embed_dim=384, n_heads=8)
    x = torch.randn(2, 20, 384)
    spectral_basis = torch.randn(2, 20, 32)

    out, info = attn(x, spectral_basis)

    assert out.shape == (2, 20, 384)
    assert "pulse_widths" in info
    assert "fhn_state" in info


def test_fhn_pulse_width_positive():
    """Pulse widths should be positive."""
    attn = FHNAttention(embed_dim=384, n_heads=8)
    x = torch.randn(2, 20, 384)
    spectral_basis = torch.randn(2, 20, 32)

    _, info = attn(x, spectral_basis)

    assert (info["pulse_widths"] > 0).all()


def test_fhn_threshold_behavior():
    """Sub-threshold inputs should produce weaker response."""
    attn = FHNAttention(embed_dim=384, n_heads=8, threshold=0.5)

    # Small input
    x_small = torch.randn(1, 10, 384) * 0.1
    spectral_basis = torch.randn(1, 10, 32)

    out_small, _ = attn(x_small, spectral_basis)

    # Large input
    x_large = torch.randn(1, 10, 384) * 2.0
    out_large, _ = attn(x_large, spectral_basis)

    # Large should have stronger response (on average)
    assert out_large.abs().mean() > out_small.abs().mean()


def test_fhn_buffer_preallocation():
    """Buffers should be pre-allocated and reused across forward passes."""
    attn = FHNAttention(embed_dim=384, n_heads=8)

    batch_size = 2
    seq_len = 20

    # First forward pass
    x1 = torch.randn(batch_size, seq_len, 384)
    spectral_basis1 = torch.randn(batch_size, seq_len, 32)
    out1, info1 = attn(x1, spectral_basis1)

    # Capture buffer information from first pass
    state1 = info1["fhn_state"]
    state1_shape = state1.shape

    # Second forward pass with same dimensions but different data
    x2 = torch.randn(batch_size, seq_len, 384)
    spectral_basis2 = torch.randn(batch_size, seq_len, 32)
    out2, info2 = attn(x2, spectral_basis2)

    state2 = info2["fhn_state"]

    # Verify shapes are consistent (buffers can be reused)
    assert out1.shape == (batch_size, seq_len, 384)
    assert out2.shape == (batch_size, seq_len, 384)
    assert state2.shape == state1_shape

    # Verify buffers are functional across multiple passes
    assert torch.isfinite(state1).all()
    assert torch.isfinite(state2).all()

    # Third pass with different batch size to test buffer adaptation
    x3 = torch.randn(4, seq_len, 384)
    spectral_basis3 = torch.randn(4, seq_len, 32)
    out3, info3 = attn(x3, spectral_basis3)

    # Should handle different batch size
    assert out3.shape == (4, seq_len, 384)
    assert torch.isfinite(info3["fhn_state"]).all()
