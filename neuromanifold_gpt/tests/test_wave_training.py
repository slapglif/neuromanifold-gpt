"""
Tests for WaveManifoldGPT training via Lightning.
"""

import torch
import pytest
from neuromanifold_gpt.config.wave_manifold_config import WaveManifoldConfig
from neuromanifold_gpt.training.wave_lightning_module import WaveManifoldLightning

def test_training_step():
    config = WaveManifoldConfig(
        n_layer=2,
        n_head=4,
        n_embd=32,
        vocab_size=100,
        block_size=16,
        use_fno_encoder=True,
        use_continuous_head=True,
        use_mamba_backbone=False,
        use_soliton_mixing=True
    )
    
    module = WaveManifoldLightning(config)
    
    # Fake batch
    idx = torch.randint(0, 100, (2, 16))
    targets = torch.randint(0, 100, (2, 16))
    batch = (idx, targets)
    
    loss = module.training_step(batch, 0)
    
    assert loss is not None
    assert not torch.isnan(loss)
    
def test_configure_optimizers():
    config = WaveManifoldConfig(
        n_layer=2,
        n_head=4,
        n_embd=32,
        vocab_size=100
    )
    module = WaveManifoldLightning(config)
    optim = module.configure_optimizers()
    assert optim is not None
