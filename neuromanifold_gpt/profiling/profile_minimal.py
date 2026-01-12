#!/usr/bin/env python3
"""
Minimal profiler - profiles each component in isolation with small inputs.
Focus on finding memory/time bottlenecks.
"""

import time
import gc
import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table

console = Console()

# Very small test parameters
BATCH_SIZE = 4
SEQ_LEN = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def profile_module(name, module, input_tensors, n_iters=5):
    """Profile a module's forward pass."""
    module = module.to(DEVICE)
    module.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(2):
            _ = module(*input_tensors)
            if DEVICE == "cuda":
                torch.cuda.synchronize()

    # Memory before
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()

    # Profile
    times = []
    with torch.no_grad():
        for _ in range(n_iters):
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = module(*input_tensors)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

    # Memory after
    if DEVICE == "cuda":
        mem_peak = torch.cuda.max_memory_allocated()
        mem_used = mem_peak - mem_before
    else:
        mem_used = 0

    mean_ms = sum(times) / len(times)

    # Cleanup
    del module
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return {
        "name": name,
        "mean_ms": mean_ms,
        "mem_mb": mem_used / 1e6,
    }


def main():
    console.print(f"\n[bold cyan]Minimal Component Profiler[/bold cyan]")
    console.print(f"Device: {DEVICE}, Batch: {BATCH_SIZE}, SeqLen: {SEQ_LEN}\n")

    results = []

    # Config params
    embed_dim = 384
    n_heads = 8
    head_dim = embed_dim // n_heads
    sdr_size = 2048
    manifold_dim = 64
    n_eigenvectors = 32
    mlp_hidden = embed_dim * 4

    # 1. SDR Projection (simple linear)
    console.print("1. SDR Projection...", end=" ")
    from neuromanifold_gpt.model.block import NeuroManifoldBlock
    sdr_proj = nn.Linear(sdr_size, embed_dim)
    x = torch.randn(BATCH_SIZE, SEQ_LEN, sdr_size, device=DEVICE)
    r = profile_module("SDR_Projection", sdr_proj, (x,))
    results.append(r)
    console.print(f"{r['mean_ms']:.2f} ms, {r['mem_mb']:.1f} MB")

    # 2. Manifold Encoder (the big 3-layer MLP)
    console.print("2. Manifold Encoder...", end=" ")
    from neuromanifold_gpt.model.manifold import ManifoldProjection
    manifold = ManifoldProjection(sdr_size, manifold_dim)
    x = torch.randn(BATCH_SIZE, SEQ_LEN, sdr_size, device=DEVICE)
    r = profile_module("ManifoldProjection", manifold, (x,))
    results.append(r)
    console.print(f"{r['mean_ms']:.2f} ms, {r['mem_mb']:.1f} MB")

    # 3. Spectral Decomposition
    console.print("3. Spectral Decomposition...", end=" ")
    from neuromanifold_gpt.model.spectral import SpectralDecomposition
    spectral = SpectralDecomposition(manifold_dim, n_eigenvectors)
    coords = torch.randn(BATCH_SIZE, SEQ_LEN, manifold_dim, device=DEVICE)
    r = profile_module("SpectralDecomp", spectral, (coords, None))
    results.append(r)
    console.print(f"{r['mean_ms']:.2f} ms, {r['mem_mb']:.1f} MB")

    # 4. FHN Dynamics Core - This is the suspected bottleneck
    console.print("4. FHN Dynamics...", end=" ")
    from neuromanifold_gpt.model.attention.fhn import FHNDynamics
    fhn = FHNDynamics(dim=head_dim, tau=12.5, threshold=0.5, use_imex=True)
    # Shape: (B, H, k, head_dim)
    stimulus = torch.randn(BATCH_SIZE, n_heads, n_eigenvectors, head_dim, device=DEVICE)
    r = profile_module("FHNDynamics", fhn, (stimulus, 2))
    results.append(r)
    console.print(f"{r['mean_ms']:.2f} ms, {r['mem_mb']:.1f} MB")

    # 5. FHN Attention Full
    console.print("5. FHN Attention...", end=" ")
    from neuromanifold_gpt.model.attention.fhn import FHNAttention
    fhn_attn = FHNAttention(embed_dim, n_heads, n_fhn_steps=2, use_imex=True, use_partitioning=True)
    x = torch.randn(BATCH_SIZE, SEQ_LEN, embed_dim, device=DEVICE)
    spectral_basis = torch.randn(BATCH_SIZE, SEQ_LEN, n_eigenvectors, device=DEVICE)
    r = profile_module("FHNAttention", fhn_attn, (x, spectral_basis))
    results.append(r)
    console.print(f"{r['mean_ms']:.2f} ms, {r['mem_mb']:.1f} MB")

    # 6. WaveKAN FFN
    console.print("6. WaveKAN FFN...", end=" ")
    from neuromanifold_gpt.model.kan.wave import WaveKANFFN
    wavekan = WaveKANFFN(embed_dim, mlp_hidden, wavelet_type="dog", use_fast_wavekan=True)
    x = torch.randn(BATCH_SIZE, SEQ_LEN, embed_dim, device=DEVICE)
    r = profile_module("WaveKAN_FFN", wavekan, (x,))
    results.append(r)
    console.print(f"{r['mean_ms']:.2f} ms, {r['mem_mb']:.1f} MB")

    # 7. SwiGLU FFN
    console.print("7. SwiGLU FFN...", end=" ")
    from neuromanifold_gpt.model.block import SwiGLU
    swiglu = SwiGLU(embed_dim, int(mlp_hidden * 2 / 3))
    x = torch.randn(BATCH_SIZE, SEQ_LEN, embed_dim, device=DEVICE)
    r = profile_module("SwiGLU_FFN", swiglu, (x,))
    results.append(r)
    console.print(f"{r['mean_ms']:.2f} ms, {r['mem_mb']:.1f} MB")

    # 8. ChebyKAN FFN
    console.print("8. ChebyKAN FFN...", end=" ")
    from neuromanifold_gpt.model.kan.cheby import ChebyKANFFN
    chebykan = ChebyKANFFN(embed_dim, mlp_hidden, degree=4)
    x = torch.randn(BATCH_SIZE, SEQ_LEN, embed_dim, device=DEVICE)
    r = profile_module("ChebyKAN_FFN", chebykan, (x,))
    results.append(r)
    console.print(f"{r['mean_ms']:.2f} ms, {r['mem_mb']:.1f} MB")

    # 9. Full Block
    console.print("9. Full Block...", end=" ")
    block = NeuroManifoldBlock(
        sdr_size=sdr_size,
        embed_dim=embed_dim,
        manifold_dim=manifold_dim,
        n_eigenvectors=n_eigenvectors,
        n_heads=n_heads,
        use_kan=True,
        kan_type="wave",
        kan_wavelet="dog",
        use_fast_wavekan=True,
    )
    sdr = torch.randn(BATCH_SIZE, SEQ_LEN, sdr_size, device=DEVICE)
    r = profile_module("NeuroManifoldBlock", block, (sdr,))
    results.append(r)
    console.print(f"{r['mean_ms']:.2f} ms, {r['mem_mb']:.1f} MB")

    # 10. SemanticFolding Encoder
    console.print("10. SemanticFoldingEncoder...", end=" ")
    from neuromanifold_gpt.model.semantic_folding import SemanticFoldingEncoder
    encoder = SemanticFoldingEncoder(vocab_size=65, sdr_size=sdr_size, n_active=40)
    tokens = torch.randint(0, 65, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    r = profile_module("SemanticFoldingEncoder", encoder, (tokens,))
    results.append(r)
    console.print(f"{r['mean_ms']:.2f} ms, {r['mem_mb']:.1f} MB")

    # Results
    console.print("\n")
    table = Table(title="Component Profiling")
    table.add_column("Component", style="cyan")
    table.add_column("Time (ms)", justify="right")
    table.add_column("Memory (MB)", justify="right")
    table.add_column("Time @ B=64", justify="right", style="yellow")

    for r in results:
        # Estimate for B=64, T=256
        scale = (64 / BATCH_SIZE) * (256 / SEQ_LEN)
        est = r["mean_ms"] * scale
        table.add_row(r["name"], f"{r['mean_ms']:.2f}", f"{r['mem_mb']:.1f}", f"{est:.2f}")

    console.print(table)

    # Analysis
    console.print("\n[bold yellow]===== ANALYSIS =====[/bold yellow]")

    # Find bottleneck
    bottleneck = max(results, key=lambda x: x["mean_ms"])
    console.print(f"\n[bold red]#1 Time Bottleneck: {bottleneck['name']}[/bold red]")
    console.print(f"   {bottleneck['mean_ms']:.2f} ms")

    mem_bottleneck = max(results, key=lambda x: x["mem_mb"])
    console.print(f"\n[bold red]#1 Memory Bottleneck: {mem_bottleneck['name']}[/bold red]")
    console.print(f"   {mem_bottleneck['mem_mb']:.1f} MB")

    # Estimate full model
    block_time = next(r["mean_ms"] for r in results if r["name"] == "NeuroManifoldBlock")
    encoder_time = next(r["mean_ms"] for r in results if r["name"] == "SemanticFoldingEncoder")

    # Scale to B=64, T=256
    scale = (64 / BATCH_SIZE) * (256 / SEQ_LEN)
    est_block = block_time * scale
    est_encoder = encoder_time * scale
    est_full_fwd = est_encoder + est_block * 6  # 6 layers
    est_full_fwd_bwd = est_full_fwd * 3  # ~3x for backward
    est_1000_iters = est_full_fwd_bwd  # ms -> seconds for 1000 iters

    console.print(f"\n[bold]Estimated performance @ B=64, T=256:[/bold]")
    console.print(f"   Per block: {est_block:.1f} ms")
    console.print(f"   Full fwd:  {est_full_fwd:.1f} ms")
    console.print(f"   Full fwd+bwd: {est_full_fwd_bwd:.1f} ms")
    console.print(f"   1000 iters: {est_1000_iters:.1f} seconds")

    if est_1000_iters > 120:
        speedup = est_1000_iters / 120
        console.print(f"   [bold red]Need {speedup:.2f}x speedup for <120s target[/bold red]")
    else:
        console.print(f"   [bold green]Under 120s target![/bold green]")


if __name__ == "__main__":
    main()
