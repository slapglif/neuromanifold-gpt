#!/usr/bin/env python
"""Verification script for StandardAttention xformers integration."""

from neuromanifold_gpt.model.attention.standard import StandardAttention
import torch

# Create attention module
attn = StandardAttention(384, 8)

# Test forward pass
x = torch.randn(2, 10, 384)
y, info = attn(x)

# Print results
print(f'Backend: {info.get("backend", "unknown")}')
print(f'Output shape: {y.shape}')
print(f'Expected shape: torch.Size([2, 10, 384])')
print(f'Test: {"PASS" if y.shape == torch.Size([2, 10, 384]) else "FAIL"}')
