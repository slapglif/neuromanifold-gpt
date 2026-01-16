"""
Integration tests for WaveManifoldGPT.
"""

import torch
import pytest
from neuromanifold_gpt.config.wave_manifold_config import WaveManifoldConfig
from neuromanifold_gpt.model.wave_manifold_gpt import WaveManifoldGPT

def test_wave_manifold_gpt_instantiation():
    config = WaveManifoldConfig(
        n_layer=2,
        n_head=4,
        n_embd=64,
        vocab_size=100,
        block_size=32,
        use_fno_encoder=True,
        use_mamba_backbone=True,
        use_soliton_mixing=True,
        use_topological_loss=True,
        use_continuous_head=True
    )
    
    model = WaveManifoldGPT(config)
    assert isinstance(model, WaveManifoldGPT)

def test_wave_manifold_gpt_forward_pass():
    config = WaveManifoldConfig(
        n_layer=2,
        n_head=4,
        n_embd=64,
        vocab_size=100,
        block_size=16,
        use_fno_encoder=True,
        use_continuous_head=True,
        use_mamba_backbone=False,
        use_soliton_mixing=True
    )
    
    model = WaveManifoldGPT(config)
    
    # Batch=2, Seq=16
    idx = torch.randint(0, 100, (2, 16))
    targets = torch.randint(0, 100, (2, 16))
    
    logits, loss, info = model(idx, targets)
    
    assert logits.shape == (2, 16, 100)
    assert loss is not None
    assert not torch.isnan(loss)
    assert 'loss_discrete' in info
    assert 'loss_continuous' in info

def test_legacy_compatibility_mode():
    """Test with standard embedding instead of FNO."""
    config = WaveManifoldConfig(
        n_layer=2,
        n_head=4,
        n_embd=32,
        vocab_size=100,
        use_fno_encoder=False,
        use_continuous_head=False
    )
    
    model = WaveManifoldGPT(config)
    idx = torch.randint(0, 100, (2, 16))
    logits, _, _ = model(idx)
    assert logits.shape == (2, 16, 100)
