# neuromanifold_gpt/tests/test_block.py
"""Tests for NeuroManifoldBlock."""
import pytest
import torch
from neuromanifold_gpt.model.block import NeuroManifoldBlock


def test_block_forward_shape(block_config):
    """Block should transform SDR to embeddings."""
    block = NeuroManifoldBlock(config=block_config)
    sdr = torch.randn(2, 20, 2048)

    out, info = block(sdr)

    assert out.shape == (2, 20, 384)
    assert 'manifold_coords' in info
    assert 'spectral_basis' in info


def test_block_gradient_flow(block_config):
    """Gradients should flow through block."""
    block = NeuroManifoldBlock(config=block_config)
    sdr = torch.randn(2, 20, 2048, requires_grad=True)

    out, _ = block(sdr)
    loss = out.sum()
    loss.backward()

    assert sdr.grad is not None
    assert not torch.isnan(sdr.grad).any()
