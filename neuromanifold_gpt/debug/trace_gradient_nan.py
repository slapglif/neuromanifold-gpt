#!/usr/bin/env python3
"""Trace exactly where NaN gradients originate."""

import torch
import torch.nn as nn

torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from neuromanifold_gpt.config.base import NeuroManifoldConfig
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT

# Track NaN
nan_sources = []

def make_hook(name):
    def hook(grad):
        if grad is not None:
            has_nan = torch.isnan(grad).any().item()
            has_inf = torch.isinf(grad).any().item()
            if has_nan or has_inf:
                nan_sources.append({
                    'name': name,
                    'nan': has_nan,
                    'inf': has_inf,
                    'shape': tuple(grad.shape),
                    'min': grad[~torch.isnan(grad)].min().item() if (~torch.isnan(grad)).any() else float('nan'),
                    'max': grad[~torch.isnan(grad)].max().item() if (~torch.isnan(grad)).any() else float('nan'),
                })
        return grad
    return hook

# Config
B, T = 4, 256
vocab_size = 65

config = NeuroManifoldConfig(
    vocab_size=vocab_size,
    block_size=T,
    n_layer=2,
    n_heads=4,
    n_embd=128,
    sdr_size=1024,
    manifold_dim=32,
    n_eigenvectors=16,
    use_sdr=True,
    use_kan=True,
    kan_type="wave",
    kan_wavelet="dog",
    fhn_threshold=0.5,
    fhn_tau=12.5,
    n_fhn_steps=2,
    use_fhn_imex=True,
    use_fhn_partitioning=True,
    dropout=0.0,
)

model = NeuroManifoldGPT(config).to(device)
model.train()

# Register hooks on all activations
hooks = []
activation_grads = {}

def register_activation_hook(module, name):
    def forward_hook(module, input, output):
        if isinstance(output, torch.Tensor) and output.requires_grad:
            output.register_hook(make_hook(f"activation:{name}"))
    hooks.append(module.register_forward_hook(forward_hook))

# Register on key modules
for name, module in model.named_modules():
    if isinstance(module, (nn.Linear, nn.LayerNorm, nn.Embedding, nn.MultiheadAttention)):
        register_activation_hook(module, name)

# Forward + backward
tokens = torch.randint(0, vocab_size, (B, T), device=device)
targets = torch.randint(0, vocab_size, (B, T), device=device)

logits, loss, info = model(tokens, targets)
print(f"Loss: {loss.item():.4f}")

loss.backward()

# Report findings
print("\n" + "="*60)
print("NaN/Inf GRADIENT SOURCES (first appearance order)")
print("="*60)

for i, src in enumerate(nan_sources[:20]):  # First 20
    print(f"{i+1}. {src['name']}")
    print(f"   Shape: {src['shape']}, NaN: {src['nan']}, Inf: {src['inf']}")
    print(f"   Valid range: [{src['min']:.2e}, {src['max']:.2e}]")

# Cleanup hooks
for h in hooks:
    h.remove()

# Also check parameter gradients in order
print("\n" + "="*60)
print("PARAMETER GRADIENTS WITH NaN (order of definition)")
print("="*60)
for name, param in model.named_parameters():
    if param.grad is not None and torch.isnan(param.grad).any():
        pct = torch.isnan(param.grad).float().mean().item() * 100
        print(f"  {name}: {pct:.1f}% NaN")
