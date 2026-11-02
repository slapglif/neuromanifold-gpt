#!/usr/bin/env python3
"""Test script for FHNAttention xformers integration."""

from neuromanifold_gpt.model.attention.fhn import FHNAttention
import torch

# Create FHNAttention with n_fhn_steps=0 to trigger fast path
attn = FHNAttention(384, 8, n_fhn_steps=0)

# Create test inputs
x = torch.randn(2, 10, 384)
spec = torch.randn(32, 32)

# Run forward pass
y, info = attn(x, spec)

# Print results
print(f'Backend: {info.get("backend", "unknown")}')
print(f'Output shape: {y.shape}')
print(f'✓ FHNAttention successfully uses backend: {info.get("backend")}')
