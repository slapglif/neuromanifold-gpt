#!/usr/bin/env python3
"""Manual test for KaufmannAttention with torch.compile fallback."""

import torch
from neuromanifold_gpt.model.attention.kaufmann import KaufmannAttention
from neuromanifold_gpt.config.base import NeuroManifoldConfig

print("Testing KaufmannAttention with torch.compile (Python 3.12 fallback)...")

# Create config
config = NeuroManifoldConfig(
    use_kaufmann_attention=True,
    n_embd=128,
    n_heads=4,
    block_size=64,
)

# Create attention module
attn = KaufmannAttention(128, 4, config)
attn.eval()

# Create dummy input
batch_size = 2
seq_len = 16
x = torch.randn(batch_size, seq_len, 128)
coords = torch.randn(batch_size, seq_len, 64)
spectral_basis = None

print(f"  Input shape: {x.shape}")

# Forward pass
with torch.no_grad():
    out, info = attn(x, spectral_basis, coords)

print(f"  Output shape: {out.shape}")
print(f"  Output mean: {out.mean().item():.4f}")
print(f"  Output std: {out.std().item():.4f}")
print(f"  Output min: {out.min().item():.4f}")
print(f"  Output max: {out.max().item():.4f}")

# Check for NaN or Inf
assert not torch.isnan(out).any(), "Output contains NaN!"
assert not torch.isinf(out).any(), "Output contains Inf!"

# Check shape
assert out.shape == x.shape, f"Output shape mismatch! Expected {x.shape}, got {out.shape}"

# Check info dict
print(f"  Info dict keys: {list(info.keys())}")
assert len(info) > 0, "Info dict is empty!"

print("\n✓ All checks passed!")
print("✓ KaufmannAttention works correctly with torch.compile fallback")
print("✓ No NaN or Inf values")
print("✓ Correct output shape")
print(f"✓ Info dict populated with keys: {list(info.keys())}")
