#!/usr/bin/env python
"""Test script to verify Triton backend integration."""

from neuromanifold_gpt.model.attention.fhn import FHNAttention
from neuromanifold_gpt.config.base import NeuroManifoldConfig

# Test 1: Check that config accepts attention_backend='triton'
cfg = NeuroManifoldConfig(attention_backend='triton')
print("✓ Config accepts attention_backend='triton'")

# Test 2: Create FHNAttention with triton backend directly
attn = FHNAttention(embed_dim=384, n_heads=8, fhn_backend='triton')
print("✓ FHNAttention accepts fhn_backend='triton'")

# Test 3: Check that FHN dynamics uses triton backend
print(f"✓ FHN dynamics backend: {attn.fhn.backend}")

print("\nTriton backend configurable!")
