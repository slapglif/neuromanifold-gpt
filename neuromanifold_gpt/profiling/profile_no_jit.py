#!/usr/bin/env python3
"""
Profile without JIT-compiled FHN.
Uses plain Python FHN implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum
from rich.console import Console
from rich.table import Table
from neuromanifold_gpt.utils.profiling import profile_component, cleanup

console = Console()

BATCH_SIZE = 16
SEQ_LEN = 256
N_WARMUP = 5
N_ITERS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FHNDynamicsNoJIT(nn.Module):
    """FHN dynamics without JIT - pure Python."""

    def __init__(self, dim, tau=12.5, threshold=0.5):
        super().__init__()
        self.dim = dim
        self.tau = tau
        self.threshold = threshold
        self.a = nn.Parameter(torch.tensor(0.7))
        self.b = nn.Parameter(torch.tensor(0.8))
        self.dt = nn.Parameter(torch.tensor(1.0))

    def forward(self, stimulus, n_steps=2):
        # Normalize stimulus
        stim_scale = stimulus.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        stimulus_normed = stimulus / stim_scale

        # Soft threshold
        threshold_gate = torch.sigmoid((stimulus.abs() - self.threshold) * 10.0)
        I = stimulus_normed * (0.1 + 0.9 * threshold_gate)

        # Initialize state
        v = torch.zeros_like(stimulus)
        w = torch.zeros_like(stimulus)

        # Extract scalars
        dt_val = self.dt.item()
        a_val = self.a.item()
        b_val = self.b.item()

        alpha = dt_val / self.tau
        denom = 1.0 + alpha * b_val

        # FHN update loop
        for _ in range(n_steps):
            v3 = v * v * v
            dv = v - (v3 / 3.0) - w + I
            v_next = v + dv * dt_val
            w_next = (w + (v_next + a_val) * alpha) / denom
            v = torch.clamp(v_next, -3.0, 3.0)
            w = torch.clamp(w_next, -3.0, 3.0)

        response = v * stim_scale
        return response, v


class FHNAttentionNoJIT(nn.Module):
    """FHN Attention without JIT."""

    def __init__(self, embed_dim, n_heads=8, threshold=0.5, tau=12.5, n_fhn_steps=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.n_fhn_steps = n_fhn_steps

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.fhn = FHNDynamicsNoJIT(self.head_dim, tau, threshold)
        self.pulse_width_net = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim // 2),
            nn.SiLU(),
            nn.Linear(self.head_dim // 2, 1),
            nn.Softplus()
        )
        self.spectral_filter = nn.Parameter(torch.ones(n_heads, 32) * 0.5)

    def forward(self, x, spectral_basis):
        B, T, D = x.shape
        k = spectral_basis.shape[-1]

        qkv = self.qkv(x)
        q, key, v = rearrange(qkv, 'b t (three h d) -> three b h t d', three=3, h=self.n_heads)

        # Spectral projection
        basis = spectral_basis.unsqueeze(1)
        q_spec = einsum(q, basis, 'b h t d, b one t k -> b h k d')
        k_spec = einsum(key, basis, 'b h t d, b one t k -> b h k d')

        # Spectral attention
        attn_spec = einsum(q_spec, k_spec, 'b h k d, b h k d -> b h k')
        attn_spec = attn_spec / (self.head_dim ** 0.5)

        # Spectral filter
        filter_weights = torch.sigmoid(self.spectral_filter[:, :k])
        attn_spec = attn_spec * filter_weights.unsqueeze(0)

        # FHN dynamics
        fhn_out, fhn_state = self.fhn(
            attn_spec.unsqueeze(-1).expand(-1, -1, -1, self.head_dim),
            n_steps=self.n_fhn_steps
        )
        fhn_response = fhn_out.mean(dim=-1)

        # Apply to values
        v_spec = einsum(v, basis, 'b h t d, b one t k -> b h k d')
        out_spec = einsum(fhn_response.unsqueeze(-1), v_spec, 'b h k one, b h k d -> b h k d')

        # Project back
        out = einsum(out_spec, basis, 'b h k d, b one t k -> b h t d')
        out = rearrange(out, 'b h t d -> b t (h d)')
        out = self.out_proj(out)

        return out, {}


def main():
    console.print(f"\n[bold cyan]NeuroManifoldGPT Profiler (No JIT)[/bold cyan]")
    console.print(f"Device: {DEVICE}")
    console.print(f"Batch: {BATCH_SIZE}, SeqLen: {SEQ_LEN}\n")

    from neuromanifold_gpt.config import NeuroManifoldConfig
    from neuromanifold_gpt.model.semantic_folding import SemanticFoldingEncoder
    from neuromanifold_gpt.model.manifold import ManifoldProjection
    from neuromanifold_gpt.model.spectral import SpectralDecomposition
    from neuromanifold_gpt.model.block import SwiGLU
    from neuromanifold_gpt.model.kan.wave import WaveKANFFN
    from neuromanifold_gpt.model.kan.cheby import ChebyKANFFN

    config = NeuroManifoldConfig(
        vocab_size=65,
        block_size=SEQ_LEN,
        n_layer=6,
        n_heads=8,
        n_embd=384,
        sdr_size=2048,
        manifold_dim=64,
        n_eigenvectors=32,
    )

    results = []

    # 1. Encoder
    console.print("1. SemanticFoldingEncoder...", end=" ")
    encoder = SemanticFoldingEncoder(65, config.sdr_size, config.sdr_n_active)
    def encoder_input():
        return (torch.randint(0, 65, (BATCH_SIZE, SEQ_LEN), device=DEVICE),)
    r = profile_component("SemanticFoldingEncoder", encoder, encoder_input, n_warmup=N_WARMUP, n_iters=N_ITERS, device=DEVICE)
    results.append(r)
    del encoder
    cleanup()
    console.print(f"{r['mean_ms']:.2f} ms")

    # 2. Manifold
    console.print("2. ManifoldProjection...", end=" ")
    manifold = ManifoldProjection(config.sdr_size, config.manifold_dim)
    def manifold_input():
        return (torch.randn(BATCH_SIZE, SEQ_LEN, config.sdr_size, device=DEVICE),)
    r = profile_component("ManifoldProjection", manifold, manifold_input, n_warmup=N_WARMUP, n_iters=N_ITERS, device=DEVICE)
    results.append(r)
    del manifold
    cleanup()
    console.print(f"{r['mean_ms']:.2f} ms")

    # 3. Spectral
    console.print("3. SpectralDecomposition...", end=" ")
    spectral = SpectralDecomposition(config.manifold_dim, config.n_eigenvectors)
    def spectral_input():
        return (torch.randn(BATCH_SIZE, SEQ_LEN, config.manifold_dim, device=DEVICE), None)
    r = profile_component("SpectralDecomposition", spectral, spectral_input, n_warmup=N_WARMUP, n_iters=N_ITERS, device=DEVICE)
    results.append(r)
    del spectral
    cleanup()
    console.print(f"{r['mean_ms']:.2f} ms")

    # 4. FHN Attention (No JIT)
    console.print("4. FHNAttention (no JIT)...", end=" ")
    fhn_attn = FHNAttentionNoJIT(config.n_embd, config.n_heads, n_fhn_steps=2)
    def fhn_input():
        x = torch.randn(BATCH_SIZE, SEQ_LEN, config.n_embd, device=DEVICE)
        spectral_basis = torch.randn(BATCH_SIZE, SEQ_LEN, config.n_eigenvectors, device=DEVICE)
        return (x, spectral_basis)
    r = profile_component("FHNAttention", fhn_attn, fhn_input, n_warmup=N_WARMUP, n_iters=N_ITERS, device=DEVICE)
    results.append(r)
    del fhn_attn
    cleanup()
    console.print(f"{r['mean_ms']:.2f} ms")

    # 5. WaveKAN
    console.print("5. WaveKAN FFN...", end=" ")
    mlp_hidden = int(config.n_embd * 4.0)
    wavekan = WaveKANFFN(config.n_embd, mlp_hidden, wavelet_type="dog", use_fast_wavekan=True)
    def ffn_input():
        return (torch.randn(BATCH_SIZE, SEQ_LEN, config.n_embd, device=DEVICE),)
    r = profile_component("WaveKAN_FFN", wavekan, ffn_input, n_warmup=N_WARMUP, n_iters=N_ITERS, device=DEVICE)
    results.append(r)
    del wavekan
    cleanup()
    console.print(f"{r['mean_ms']:.2f} ms")

    # 6. SwiGLU
    console.print("6. SwiGLU FFN...", end=" ")
    swiglu = SwiGLU(config.n_embd, int(mlp_hidden * 2 / 3))
    r = profile_component("SwiGLU_FFN", swiglu, ffn_input, n_warmup=N_WARMUP, n_iters=N_ITERS, device=DEVICE)
    results.append(r)
    del swiglu
    cleanup()
    console.print(f"{r['mean_ms']:.2f} ms")

    # 7. ChebyKAN
    console.print("7. ChebyKAN FFN...", end=" ")
    chebykan = ChebyKANFFN(config.n_embd, mlp_hidden, degree=4)
    r = profile_component("ChebyKAN_FFN", chebykan, ffn_input, n_warmup=N_WARMUP, n_iters=N_ITERS, device=DEVICE)
    results.append(r)
    del chebykan
    cleanup()
    console.print(f"{r['mean_ms']:.2f} ms")

    # Results
    console.print("\n")
    table = Table(title=f"Profiling Results (B={BATCH_SIZE}, T={SEQ_LEN})")
    table.add_column("Component", style="cyan")
    table.add_column("Mean (ms)", justify="right")
    table.add_column("@ B=64 (est)", justify="right", style="yellow")

    scale = 64 / BATCH_SIZE
    for r in results:
        table.add_row(r["name"], f"{r['mean_ms']:.2f}", f"{r['mean_ms'] * scale:.2f}")

    console.print(table)

    # Analysis
    console.print("\n[bold yellow]===== ANALYSIS =====[/bold yellow]")

    bottleneck = max(results, key=lambda x: x["mean_ms"])
    console.print(f"\n[bold red]#1 BOTTLENECK: {bottleneck['name']}[/bold red]")
    console.print(f"   Time: {bottleneck['mean_ms']:.2f} ms")

    # Estimate block time
    manifold = next(r["mean_ms"] for r in results if r["name"] == "ManifoldProjection")
    spectral = next(r["mean_ms"] for r in results if r["name"] == "SpectralDecomposition")
    fhn = next(r["mean_ms"] for r in results if r["name"] == "FHNAttention")
    ffn = next(r["mean_ms"] for r in results if r["name"] == "WaveKAN_FFN")
    encoder_time = next(r["mean_ms"] for r in results if r["name"] == "SemanticFoldingEncoder")

    block_time = manifold + spectral + fhn + ffn + 2  # +2 for overhead
    console.print(f"\n[bold]Estimated Block Time:[/bold] {block_time:.2f} ms")
    console.print(f"   Manifold: {manifold:.2f} ms ({manifold/block_time*100:.1f}%)")
    console.print(f"   Spectral: {spectral:.2f} ms ({spectral/block_time*100:.1f}%)")
    console.print(f"   FHN Attn: {fhn:.2f} ms ({fhn/block_time*100:.1f}%)")
    console.print(f"   WaveKAN:  {ffn:.2f} ms ({ffn/block_time*100:.1f}%)")

    # Full model estimate
    full_fwd = encoder_time + block_time * 6
    full_fwd_bwd = full_fwd * 3  # ~3x for backward

    console.print(f"\n[bold]Full Model Estimate (6 layers):[/bold]")
    console.print(f"   Forward:     {full_fwd:.2f} ms")
    console.print(f"   Fwd+Bwd:     {full_fwd_bwd:.2f} ms")

    # Scale to B=64
    console.print(f"\n[bold]@ B=64, T=256:[/bold]")
    est_fwd_bwd = full_fwd_bwd * scale
    console.print(f"   Fwd+Bwd: {est_fwd_bwd:.2f} ms/iter")
    console.print(f"   1000 iters: {est_fwd_bwd:.1f} seconds")

    if est_fwd_bwd > 120:
        speedup = est_fwd_bwd / 120
        console.print(f"   [bold red]Need {speedup:.2f}x speedup for <120s[/bold red]")
    else:
        console.print(f"   [bold green]Under 120s target![/bold green]")

    # FFN comparison
    console.print("\n[bold]FFN Comparison:[/bold]")
    wavekan_t = next(r["mean_ms"] for r in results if r["name"] == "WaveKAN_FFN")
    swiglu_t = next(r["mean_ms"] for r in results if r["name"] == "SwiGLU_FFN")
    chebykan_t = next(r["mean_ms"] for r in results if r["name"] == "ChebyKAN_FFN")
    console.print(f"   SwiGLU:   {swiglu_t:.2f} ms (1.00x)")
    console.print(f"   WaveKAN:  {wavekan_t:.2f} ms ({wavekan_t/swiglu_t:.2f}x)")
    console.print(f"   ChebyKAN: {chebykan_t:.2f} ms ({chebykan_t/swiglu_t:.2f}x)")


if __name__ == "__main__":
    main()
