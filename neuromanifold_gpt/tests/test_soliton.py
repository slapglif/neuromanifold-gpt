# neuromanifold_gpt/tests/test_soliton.py
"""Tests for Soliton Attention mechanism."""
import pytest
import torch
from neuromanifold_gpt.model.attention.soliton import SolitonAttention


def test_soliton_output_shape():
    """Output should match input dimensions."""
    attn = SolitonAttention(embed_dim=384, n_heads=8)
    x = torch.randn(2, 20, 384)
    spectral_basis = torch.randn(2, 20, 32)

    out, info = attn(x, spectral_basis)

    assert out.shape == (2, 20, 384)
    assert 'pulse_widths' in info
    assert 'soliton_state' in info


def test_soliton_pulse_width_positive():
    """Pulse widths should be positive."""
    attn = SolitonAttention(embed_dim=384, n_heads=8)
    x = torch.randn(2, 20, 384)
    spectral_basis = torch.randn(2, 20, 32)

    _, info = attn(x, spectral_basis)

    assert (info['pulse_widths'] > 0).all()


def test_soliton_threshold_behavior():
    """Sub-threshold inputs should produce weaker response."""
    attn = SolitonAttention(embed_dim=384, n_heads=8, threshold=0.5)

    # Small input
    x_small = torch.randn(1, 10, 384) * 0.1
    spectral_basis = torch.randn(1, 10, 32)

    out_small, _ = attn(x_small, spectral_basis)

    # Large input
    x_large = torch.randn(1, 10, 384) * 2.0
    out_large, _ = attn(x_large, spectral_basis)

    # Large should have stronger response (on average)
    assert out_large.abs().mean() > out_small.abs().mean()
