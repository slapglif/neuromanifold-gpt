#!/usr/bin/env python3
"""Diagnose NaN source and profile bottlenecks in NeuroManifoldGPT."""

import time

import torch

# Setup
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Import components
from neuromanifold_gpt.config.base import NeuroManifoldConfig
from neuromanifold_gpt.model.attention.fhn import FHNAttention, FHNDynamics
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT
from neuromanifold_gpt.model.manifold import ManifoldProjection
from neuromanifold_gpt.model.semantic_folding import SemanticFoldingEncoder
from neuromanifold_gpt.model.spectral import SpectralDecomposition


def check_tensor(name: str, t: torch.Tensor) -> dict:
    """Check tensor for NaN/Inf and return stats."""
    if t is None:
        return {"name": name, "is_none": True}
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    return {
        "name": name,
        "shape": tuple(t.shape),
        "dtype": str(t.dtype),
        "has_nan": has_nan,
        "has_inf": has_inf,
        "min": t.min().item() if not has_nan else float("nan"),
        "max": t.max().item() if not has_nan else float("nan"),
        "mean": t.mean().item() if not has_nan else float("nan"),
        "std": t.std().item() if not has_nan else float("nan"),
    }


def profile_component(name: str, fn, *args, n_runs: int = 10, **kwargs):
    """Profile a component and return timing + NaN check."""
    # Warmup
    with torch.no_grad():
        result = fn(*args, **kwargs)

    # Time it
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            result = fn(*args, **kwargs)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_runs * 1000  # ms

    return result, elapsed


print("\n" + "=" * 60)
print("COMPONENT-LEVEL DIAGNOSIS")
print("=" * 60)

# Test config
B, T = 4, 256
vocab_size = 65
sdr_size = 1024
manifold_dim = 32
n_eigenvectors = 16
n_heads = 4
embed_dim = 128

# 1. Test SemanticFoldingEncoder
print("\n1. SemanticFoldingEncoder")
encoder = SemanticFoldingEncoder(vocab_size, sdr_size, n_active=20, embed_dim=64).to(
    device
)
tokens = torch.randint(0, vocab_size, (B, T), device=device)
sdr, scores, topo_loss = encoder(tokens)
print(f"   SDR: {check_tensor('sdr', sdr)}")
print(f"   Scores: {check_tensor('scores', scores)}")
print(f"   Topo Loss: {check_tensor('topo_loss', topo_loss)}")

# 2. Test ManifoldProjection
print("\n2. ManifoldProjection")
manifold = ManifoldProjection(sdr_size, manifold_dim).to(device)
coords, metric = manifold(sdr)
print(f"   Coords: {check_tensor('coords', coords)}")
print(f"   Metric: {check_tensor('metric', metric)}")

# 3. Test SpectralDecomposition
print("\n3. SpectralDecomposition")
spectral = SpectralDecomposition(manifold_dim, n_eigenvectors).to(device)
basis, freqs, ortho_loss = spectral(coords, metric)
print(f"   Basis: {check_tensor('basis', basis)}")
print(f"   Freqs: {check_tensor('freqs', freqs)}")
print(f"   Ortho Loss: {check_tensor('ortho_loss', ortho_loss)}")

# 4. Test FHN Dynamics alone
print("\n4. FHNDynamics (isolated)")
fhn = FHNDynamics(dim=embed_dim // n_heads, tau=12.5, threshold=0.5).to(device)
test_input = torch.randn(B, n_heads, T, embed_dim // n_heads, device=device) * 0.5
response, state = fhn(test_input, n_steps=2)
print(f"   Response: {check_tensor('response', response)}")
print(f"   State: {check_tensor('state', state)}")

# 5. Test FHNAttention
print("\n5. FHNAttention")
x_embed = torch.randn(B, T, embed_dim, device=device)
attn = FHNAttention(embed_dim, n_heads, threshold=0.5, tau=12.5, n_fhn_steps=2).to(
    device
)
attn_out, attn_info = attn(x_embed, basis)
print(f"   Output: {check_tensor('attn_out', attn_out)}")
print(f"   FHN State: {check_tensor('fhn_state', attn_info.get('fhn_state'))}")

# 6. Test full forward pass
print("\n6. Full NeuroManifoldGPT Forward")
config = NeuroManifoldConfig(
    vocab_size=vocab_size,
    block_size=T,
    n_layer=2,  # Reduced for diagnosis
    n_heads=n_heads,
    n_embd=embed_dim,
    sdr_size=sdr_size,
    manifold_dim=manifold_dim,
    n_eigenvectors=n_eigenvectors,
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
targets = torch.randint(0, vocab_size, (B, T), device=device)

# Forward with gradient
model.train()
logits, loss, info = model(tokens, targets)
print(f"   Logits: {check_tensor('logits', logits)}")
print(f"   Loss: {check_tensor('loss', loss)}")

# Check if backward produces NaN
if loss is not None and not torch.isnan(loss):
    loss.backward()
    print("\n   Gradient check:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_info = check_tensor(name, param.grad)
            if grad_info["has_nan"] or grad_info["has_inf"]:
                print(
                    f"   ❌ {name}: NaN={grad_info['has_nan']}, Inf={grad_info['has_inf']}"
                )
else:
    print("   ❌ Loss is NaN, cannot backprop")

print("\n" + "=" * 60)
print("PROFILING BOTTLENECKS")
print("=" * 60)

# Profile each component
model.eval()
timings = {}

# Profile encoder
_, t = profile_component("encoder", encoder, tokens)
timings["SemanticFolding"] = t
print(f"SemanticFolding: {t:.2f}ms")

# Profile manifold
_, t = profile_component("manifold", manifold, sdr)
timings["ManifoldProjection"] = t
print(f"ManifoldProjection: {t:.2f}ms")

# Profile spectral
_, t = profile_component("spectral", spectral, coords, metric)
timings["SpectralDecomposition"] = t
print(f"SpectralDecomposition: {t:.2f}ms")

# Profile FHN attention
_, t = profile_component("attention", attn, x_embed, basis)
timings["FHNAttention"] = t
print(f"FHNAttention: {t:.2f}ms")


# Profile full forward
def full_forward():
    return model(tokens, targets)


_, t = profile_component("full", full_forward)
timings["FullForward"] = t
print(f"FullForward (2 layers): {t:.2f}ms")

# Estimate for 4 layers
est_4layer = timings["SemanticFolding"] + 4 * (
    timings["ManifoldProjection"]
    + timings["SpectralDecomposition"]
    + timings["FHNAttention"]
)
print(f"\nEstimated 4-layer forward: {est_4layer:.2f}ms")

# For 1000 iters at batch 64
est_total = est_4layer * 1000 * (64 / B) / 1000  # seconds
print(f"Estimated 1000 iters (batch 64): {est_total:.1f}s")

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)
