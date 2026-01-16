# neuromanifold_gpt/tests/test_block.py
"""Tests for NeuroManifoldBlock."""
import pytest
import torch
from neuromanifold_gpt.model.block import NeuroManifoldBlock
from neuromanifold_gpt.config.block_config import NeuroManifoldBlockConfig


def test_block_forward_shape():
    """Block should transform SDR to embeddings."""
    config = NeuroManifoldBlockConfig(
        sdr_size=2048,
        embed_dim=384,
        manifold_dim=64,
        n_eigenvectors=32,
        n_heads=8
    )
    block = NeuroManifoldBlock(config=config)
    sdr = torch.randn(2, 20, 2048)

    out, info = block(sdr)

    assert out.shape == (2, 20, 384)
    assert 'manifold_coords' in info
    assert 'spectral_basis' in info


def test_block_gradient_flow():
    """Gradients should flow through block."""
    config = NeuroManifoldBlockConfig(
        sdr_size=2048,
        embed_dim=384,
        manifold_dim=64,
        n_eigenvectors=32,
        n_heads=8
    )
    block = NeuroManifoldBlock(config=config)
    sdr = torch.randn(2, 20, 2048, requires_grad=True)

    out, _ = block(sdr)
    loss = out.sum()
    loss.backward()

    assert sdr.grad is not None
    assert not torch.isnan(sdr.grad).any()
