#!/usr/bin/env python3
"""Debug FHN attention to understand why generation quality is poor."""

import torch

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from neuromanifold_gpt.config.base import NeuroManifoldConfig
from neuromanifold_gpt.model.attention.fhn import FHNAttention

# Create FHN attention module
config = NeuroManifoldConfig(
    vocab_size=65,
    block_size=256,
    n_layer=4,
    n_heads=4,
    n_embd=256,
    sdr_size=256,
    manifold_dim=32,
    n_eigenvectors=16,
    fhn_threshold=0.5,
    fhn_tau=12.5,
    n_fhn_steps=2,
    use_fhn_imex=True,
    use_fhn_partitioning=True,
    use_fhn_parallel=True,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Create FHN attention
attn = FHNAttention(
    embed_dim=256,
    n_heads=4,
    dropout=0.0,
    threshold=0.5,
    tau=12.5,
    pulse_width_base=4,
    n_fhn_steps=2,
    use_imex=True,
    use_partitioning=True,
    use_fused=False,
).to(device)

# Create test inputs
B, T, D = 2, 32, 256
x = torch.randn(B, T, D, device=device)
spectral_basis = torch.randn(B, T, 16, device=device)  # n_eigenvectors=16

print(f"Input x shape: {x.shape}")
print(f"Spectral basis shape: {spectral_basis.shape}")

# Run forward pass
attn.eval()
with torch.no_grad():
    out, info = attn(x, spectral_basis)

print(f"\nOutput shape: {out.shape}")
print(f"\nInfo keys: {info.keys()}")

# Analyze FHN state
if 'fhn_state' in info:
    fhn_state = info['fhn_state']
    print(f"\nFHN state type: {type(fhn_state)}")

    if isinstance(fhn_state, dict):
        print(f"FHN state keys: {fhn_state.keys()}")
        if 'v' in fhn_state:
            v = fhn_state['v']
            print(f"\nFHN v (activation) stats:")
            print(f"  Shape: {v.shape}")
            print(f"  Mean: {v.mean().item():.4f}")
            print(f"  Std: {v.std().item():.4f}")
            print(f"  Min: {v.min().item():.4f}")
            print(f"  Max: {v.max().item():.4f}")
            firing = (v > 0.5).float().mean().item()
            print(f"  % firing (v > 0.5): {firing*100:.1f}%")
    elif isinstance(fhn_state, tuple):
        v, w = fhn_state
        print(f"\nFHN v (activation) stats:")
        print(f"  Shape: {v.shape}")
        print(f"  Mean: {v.mean().item():.4f}")
        print(f"  Std: {v.std().item():.4f}")
        print(f"  Min: {v.min().item():.4f}")
        print(f"  Max: {v.max().item():.4f}")
        firing = (v > 0.5).float().mean().item()
        print(f"  % firing (v > 0.5): {firing*100:.1f}%")

        print(f"\nFHN w (recovery) stats:")
        print(f"  Shape: {w.shape}")
        print(f"  Mean: {w.mean().item():.4f}")
        print(f"  Std: {w.std().item():.4f}")
        print(f"  Min: {w.min().item():.4f}")
        print(f"  Max: {w.max().item():.4f}")
    elif isinstance(fhn_state, torch.Tensor):
        print(f"\nFHN state (tensor) stats:")
        print(f"  Shape: {fhn_state.shape}")
        print(f"  Mean: {fhn_state.mean().item():.4f}")
        print(f"  Std: {fhn_state.std().item():.4f}")
        print(f"  Min: {fhn_state.min().item():.4f}")
        print(f"  Max: {fhn_state.max().item():.4f}")
        firing = (fhn_state.abs() > 0.5).float().mean().item()
        print(f"  % |val| > 0.5: {firing*100:.1f}%")

# Check attention weights
if 'attn_weights' in info:
    attn_w = info['attn_weights']
    print(f"\nAttention weights shape: {attn_w.shape}")
    print(f"Attention weights stats:")
    print(f"  Mean: {attn_w.mean().item():.4f}")
    print(f"  Std: {attn_w.std().item():.4f}")
    print(f"  Sum per query (should be ~1): {attn_w.sum(dim=-1).mean().item():.4f}")

# Check if output has gradient flow
x_grad = torch.randn(B, T, D, device=device, requires_grad=True)
spectral_grad = torch.randn(B, T, 16, device=device, requires_grad=True)

attn.train()
out_grad, _ = attn(x_grad, spectral_grad)
loss = out_grad.sum()
loss.backward()

print(f"\nGradient flow check:")
print(f"  x_grad.grad exists: {x_grad.grad is not None}")
print(f"  spectral_grad.grad exists: {spectral_grad.grad is not None}")
if x_grad.grad is not None:
    print(f"  x_grad.grad mean: {x_grad.grad.mean().item():.4f}")
    print(f"  x_grad.grad std: {x_grad.grad.std().item():.4f}")
    has_nan = torch.isnan(x_grad.grad).any().item()
    print(f"  x_grad.grad has NaN: {has_nan}")

# Test: Compare output variance with input variance
print(f"\nVariance analysis:")
print(f"  Input variance: {x.var().item():.4f}")
print(f"  Output variance: {out.var().item():.4f}")
print(f"  Ratio (out/in): {out.var().item() / x.var().item():.4f}")

# Test: Causal masking - does position i only depend on positions <= i?
print(f"\nCausal masking test:")
# If causal, changing position j should NOT affect positions < j
x_test = x.clone()
x_mod = x.clone()
x_mod[:, 16, :] = 0  # Zero out position 16

with torch.no_grad():
    out_test, _ = attn(x_test, spectral_basis)
    out_mod, _ = attn(x_mod, spectral_basis)

# Positions < 16 should be identical
diff_before = (out_test[:, :16, :] - out_mod[:, :16, :]).abs().mean().item()
diff_at = (out_test[:, 16, :] - out_mod[:, 16, :]).abs().mean().item()
diff_after = (out_test[:, 17:, :] - out_mod[:, 17:, :]).abs().mean().item()

print(f"  Diff before pos 16: {diff_before:.6f} (should be ~0 if causal)")
print(f"  Diff at pos 16: {diff_at:.6f}")
print(f"  Diff after pos 16: {diff_after:.6f}")

if diff_before > 0.001:
    print("  WARNING: Not causal! Information leaking backward!")
else:
    print("  OK: Causal masking working")
