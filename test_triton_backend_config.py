#!/usr/bin/env python3
"""Test Triton backend configuration integration."""

from neuromanifold_gpt.model.attention.fhn import FHNAttention
from neuromanifold_gpt.config.base import NeuroManifoldConfig

# Test 1: Config with triton backend
print("Test 1: Creating config with attention_backend='triton'")
cfg = NeuroManifoldConfig(attention_backend='triton')
print(f"✓ Config created with attention_backend='{cfg.attention_backend}'")

# Test 2: Verify backend propagation through block config
print("\nTest 2: Verifying backend propagation to FHNConfig")
from neuromanifold_gpt.config.block_config import NeuroManifoldBlockConfig
block_cfg = NeuroManifoldBlockConfig.from_model_config(cfg, layer_idx=0)
print(f"✓ Block config fhn_backend='{block_cfg.fhn.fhn_backend}'")

assert block_cfg.fhn.fhn_backend == 'triton', f"Expected 'triton', got '{block_cfg.fhn.fhn_backend}'"
print("✓ Backend correctly propagated from config to FHNConfig")

# Test 3: Direct FHNAttention instantiation with triton backend
print("\nTest 3: Creating FHNAttention with fhn_backend='triton'")
attn = FHNAttention(embed_dim=384, n_heads=8, fhn_backend='triton')
print(f"✓ FHNAttention created with backend='{attn.fhn.backend}'")

print("\n✓ All tests passed! Triton backend is configurable.")
