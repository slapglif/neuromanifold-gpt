#!/usr/bin/env python3
"""
Complete NeuroManifoldGPT Profiler
Works around JIT warm-up issues by running sufficient warm-up iterations.
"""

import time
import gc
import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table

console = Console()

# Profiling parameters
BATCH_SIZE = 16
SEQ_LEN = 256
N_WARMUP = 10  # Extra warmup to handle JIT issues
N_ITERS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def cleanup():
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def profile_component(name, module, input_fn, n_warmup=N_WARMUP, n_iters=N_ITERS):
    """Profile a single component."""
    module = module.to(DEVICE)
    module.eval()

    # Extended warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            inputs = input_fn()
            _ = module(*inputs)
            if DEVICE == "cuda":
                torch.cuda.synchronize()

    # Memory before
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(n_iters):
            inputs = input_fn()
            if DEVICE == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = module(*inputs)

            if DEVICE == "cuda":
                torch.cuda.synchronize()

            times.append((time.perf_counter() - start) * 1000)

    # Memory after
    if DEVICE == "cuda":
        mem_peak = torch.cuda.max_memory_allocated()
        mem_used = (mem_peak - mem_before) / 1e6
    else:
        mem_used = 0

    # Cleanup
    del module
    cleanup()

    return {
        "name": name,
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
        "mem_mb": mem_used,
    }


def profile_fwd_bwd(name, module, input_fn, loss_fn, n_warmup=N_WARMUP, n_iters=N_ITERS):
    """Profile forward + backward pass."""
    module = module.to(DEVICE)
    module.train()

    # Extended warmup
    for _ in range(n_warmup):
        inputs = input_fn()
        output = module(*inputs)
        loss = loss_fn(output)
        loss.backward()
        module.zero_grad()
        if DEVICE == "cuda":
            torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(n_iters):
        inputs = input_fn()
        if DEVICE == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        output = module(*inputs)
        loss = loss_fn(output)
        loss.backward()
        module.zero_grad()

        if DEVICE == "cuda":
            torch.cuda.synchronize()

        times.append((time.perf_counter() - start) * 1000)

    # Cleanup
    del module
    cleanup()

    return {
        "name": name,
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
        "mem_mb": 0,
    }


def main():
    console.print(f"\n[bold cyan]NeuroManifoldGPT Complete Profiler[/bold cyan]")
    console.print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        console.print(f"GPU: {torch.cuda.get_device_name()}")
        console.print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    console.print(f"Batch Size: {BATCH_SIZE}, Seq Len: {SEQ_LEN}")
    console.print(f"Warmup: {N_WARMUP}, Iterations: {N_ITERS}\n")

    # Import components
    from neuromanifold_gpt.config import NeuroManifoldConfig
    from neuromanifold_gpt.model.gpt import NeuroManifoldGPT
    from neuromanifold_gpt.model.semantic_folding import SemanticFoldingEncoder
    from neuromanifold_gpt.model.manifold import ManifoldProjection
    from neuromanifold_gpt.model.spectral import SpectralDecomposition
    from neuromanifold_gpt.model.attention.fhn import FHNAttention
    from neuromanifold_gpt.model.block import NeuroManifoldBlock, SwiGLU
    from neuromanifold_gpt.model.kan.wave import WaveKANFFN
    from neuromanifold_gpt.model.kan.cheby import ChebyKANFFN

    # Config
    config = NeuroManifoldConfig(
        vocab_size=65,
        block_size=SEQ_LEN,
        n_layer=6,
        n_heads=8,
        n_embd=384,
        sdr_size=2048,
        manifold_dim=64,
        n_eigenvectors=32,
        use_sdr=True,
        use_kan=True,
        kan_type="wave",
        kan_wavelet="dog",
        use_fast_wavekan=True,
    )

    results = []

    # 1. SemanticFoldingEncoder
    console.print("[bold]1. SemanticFoldingEncoder...[/bold]", end=" ")
    encoder = SemanticFoldingEncoder(
        vocab_size=config.vocab_size,
        sdr_size=config.sdr_size,
        n_active=config.sdr_n_active,
        embed_dim=config.sdr_embed_dim,
        context_size=config.sdr_context_size,
    )
    def encoder_input():
        return (torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE),)
    r = profile_component("SemanticFoldingEncoder", encoder, encoder_input)
    results.append(r)
    console.print(f"{r['mean_ms']:.2f} ms")

    # 2. ManifoldProjection
    console.print("[bold]2. ManifoldProjection...[/bold]", end=" ")
    manifold = ManifoldProjection(config.sdr_size, config.manifold_dim)
    def manifold_input():
        return (torch.randn(BATCH_SIZE, SEQ_LEN, config.sdr_size, device=DEVICE),)
    r = profile_component("ManifoldProjection", manifold, manifold_input)
    results.append(r)
    console.print(f"{r['mean_ms']:.2f} ms")

    # 3. SpectralDecomposition
    console.print("[bold]3. SpectralDecomposition...[/bold]", end=" ")
    spectral = SpectralDecomposition(config.manifold_dim, config.n_eigenvectors)
    def spectral_input():
        coords = torch.randn(BATCH_SIZE, SEQ_LEN, config.manifold_dim, device=DEVICE)
        return (coords, None)
    r = profile_component("SpectralDecomposition", spectral, spectral_input)
    results.append(r)
    console.print(f"{r['mean_ms']:.2f} ms")

    # 4. FHNAttention
    console.print("[bold]4. FHNAttention...[/bold]", end=" ")
    fhn_attn = FHNAttention(
        embed_dim=config.n_embd,
        n_heads=config.n_heads,
        threshold=config.fhn_threshold,
        tau=config.fhn_tau,
        n_fhn_steps=config.n_fhn_steps,
        use_imex=config.use_fhn_imex,
        use_partitioning=config.use_fhn_partitioning,
    )
    def fhn_input():
        x = torch.randn(BATCH_SIZE, SEQ_LEN, config.n_embd, device=DEVICE)
        spectral_basis = torch.randn(BATCH_SIZE, SEQ_LEN, config.n_eigenvectors, device=DEVICE)
        return (x, spectral_basis)
    r = profile_component("FHNAttention", fhn_attn, fhn_input)
    results.append(r)
    console.print(f"{r['mean_ms']:.2f} ms")

    # 5. WaveKAN FFN
    console.print("[bold]5. WaveKAN FFN...[/bold]", end=" ")
    mlp_hidden = int(config.n_embd * 4.0)
    wavekan_ffn = WaveKANFFN(
        embed_dim=config.n_embd,
        hidden_dim=mlp_hidden,
        wavelet_type=config.kan_wavelet,
        use_fast_wavekan=config.use_fast_wavekan,
    )
    def ffn_input():
        return (torch.randn(BATCH_SIZE, SEQ_LEN, config.n_embd, device=DEVICE),)
    r = profile_component("WaveKAN_FFN", wavekan_ffn, ffn_input)
    results.append(r)
    console.print(f"{r['mean_ms']:.2f} ms")

    # 6. SwiGLU FFN
    console.print("[bold]6. SwiGLU FFN...[/bold]", end=" ")
    swiglu_ffn = SwiGLU(config.n_embd, int(mlp_hidden * 2 / 3))
    r = profile_component("SwiGLU_FFN", swiglu_ffn, ffn_input)
    results.append(r)
    console.print(f"{r['mean_ms']:.2f} ms")

    # 7. ChebyKAN FFN
    console.print("[bold]7. ChebyKAN FFN...[/bold]", end=" ")
    chebykan_ffn = ChebyKANFFN(config.n_embd, mlp_hidden, degree=config.kan_degree)
    r = profile_component("ChebyKAN_FFN", chebykan_ffn, ffn_input)
    results.append(r)
    console.print(f"{r['mean_ms']:.2f} ms")

    # 8. NeuroManifoldBlock
    console.print("[bold]8. NeuroManifoldBlock...[/bold]", end=" ")
    block = NeuroManifoldBlock(
        sdr_size=config.sdr_size,
        embed_dim=config.n_embd,
        manifold_dim=config.manifold_dim,
        n_eigenvectors=config.n_eigenvectors,
        n_heads=config.n_heads,
        fhn_threshold=config.fhn_threshold,
        fhn_tau=config.fhn_tau,
        n_fhn_steps=config.n_fhn_steps,
        use_fhn_imex=config.use_fhn_imex,
        use_fhn_partitioning=config.use_fhn_partitioning,
        use_kan=config.use_kan,
        kan_type=config.kan_type,
        kan_wavelet=config.kan_wavelet,
        use_fast_wavekan=config.use_fast_wavekan,
    )
    def block_input():
        return (torch.randn(BATCH_SIZE, SEQ_LEN, config.sdr_size, device=DEVICE),)
    r = profile_component("NeuroManifoldBlock", block, block_input)
    results.append(r)
    console.print(f"{r['mean_ms']:.2f} ms")

    # 9. Full Model Forward
    console.print("[bold]9. Full Model Forward...[/bold]", end=" ")
    model = NeuroManifoldGPT(config)
    def model_input():
        return (torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE),)
    r = profile_component("FullModel_Forward", model, model_input)
    results.append(r)
    console.print(f"{r['mean_ms']:.2f} ms")

    # 10. Full Model Forward+Backward
    console.print("[bold]10. Full Model FwdBwd...[/bold]", end=" ")
    model = NeuroManifoldGPT(config)
    def model_input_train():
        tokens = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
        targets = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
        return (tokens, targets)
    def loss_fn(output):
        logits, loss, info = output
        return loss
    r = profile_fwd_bwd("FullModel_FwdBwd", model, model_input_train, loss_fn)
    results.append(r)
    console.print(f"{r['mean_ms']:.2f} ms")

    # =========================================================================
    # Results Table
    # =========================================================================
    console.print("\n")
    table = Table(title=f"Profiling Results (B={BATCH_SIZE}, T={SEQ_LEN})")
    table.add_column("Component", style="cyan")
    table.add_column("Mean (ms)", justify="right")
    table.add_column("@ B=64 (est)", justify="right", style="yellow")
    table.add_column("% of Block", justify="right")

    block_time = next(r["mean_ms"] for r in results if r["name"] == "NeuroManifoldBlock")
    scale = 64 / BATCH_SIZE

    for r in results:
        est_64 = r["mean_ms"] * scale
        if r["name"] in ["FullModel_Forward", "FullModel_FwdBwd"]:
            pct_str = "-"
        elif r["name"] == "NeuroManifoldBlock":
            pct_str = "100%"
        else:
            pct = r["mean_ms"] / block_time * 100
            pct_str = f"{pct:.1f}%"

        table.add_row(
            r["name"],
            f"{r['mean_ms']:.2f}",
            f"{est_64:.2f}",
            pct_str,
        )

    console.print(table)

    # =========================================================================
    # Analysis
    # =========================================================================
    console.print("\n[bold yellow]===== PERFORMANCE ANALYSIS =====[/bold yellow]")

    # Bottleneck
    component_results = [r for r in results if r["name"] not in ["FullModel_Forward", "FullModel_FwdBwd", "NeuroManifoldBlock"]]
    bottleneck = max(component_results, key=lambda x: x["mean_ms"])

    console.print(f"\n[bold red]#1 BOTTLENECK: {bottleneck['name']}[/bold red]")
    console.print(f"   Time: {bottleneck['mean_ms']:.2f} ms ({bottleneck['mean_ms']/block_time*100:.1f}% of block)")

    # Time per 1000 iterations
    fwd_bwd_time = next(r["mean_ms"] for r in results if r["name"] == "FullModel_FwdBwd")
    est_fwd_bwd_64 = fwd_bwd_time * scale
    time_per_1000_iter = est_fwd_bwd_64

    console.print(f"\n[bold]Current @ B={BATCH_SIZE}:[/bold] {fwd_bwd_time:.2f} ms/iter")
    console.print(f"[bold]Estimated @ B=64:[/bold] {est_fwd_bwd_64:.2f} ms/iter")
    console.print(f"[bold]Est. time for 1000 iters:[/bold] {time_per_1000_iter:.1f} seconds")

    # Target
    target_time = 120.0
    if time_per_1000_iter > target_time:
        speedup_needed = time_per_1000_iter / target_time
        console.print(f"[bold red]Speedup needed for <120s:[/bold red] {speedup_needed:.2f}x")
    else:
        console.print(f"[bold green]Already under 120s target![/bold green]")

    # Component breakdown
    console.print("\n[bold]Block Component Breakdown:[/bold]")
    block_components = ["ManifoldProjection", "SpectralDecomposition", "FHNAttention", "WaveKAN_FFN"]
    total_accounted = 0.0
    for name in block_components:
        r = next((x for x in results if x["name"] == name), None)
        if r:
            pct = r["mean_ms"] / block_time * 100
            total_accounted += r["mean_ms"]
            bar = "#" * int(pct / 3)
            console.print(f"   {name:24s}: {r['mean_ms']:7.2f} ms ({pct:5.1f}%) {bar}")

    overhead = block_time - total_accounted
    pct = overhead / block_time * 100
    bar = "#" * int(pct / 3)
    console.print(f"   {'SDR proj/Norms/etc':24s}: {overhead:7.2f} ms ({pct:5.1f}%) {bar}")

    # FFN Comparison
    console.print("\n[bold]FFN Comparison:[/bold]")
    wavekan = next(r for r in results if r["name"] == "WaveKAN_FFN")
    swiglu = next(r for r in results if r["name"] == "SwiGLU_FFN")
    chebykan = next(r for r in results if r["name"] == "ChebyKAN_FFN")

    console.print(f"   SwiGLU:   {swiglu['mean_ms']:6.2f} ms (1.00x baseline)")
    console.print(f"   WaveKAN:  {wavekan['mean_ms']:6.2f} ms ({wavekan['mean_ms']/swiglu['mean_ms']:.2f}x)")
    console.print(f"   ChebyKAN: {chebykan['mean_ms']:6.2f} ms ({chebykan['mean_ms']/swiglu['mean_ms']:.2f}x)")

    # Full model breakdown
    console.print(f"\n[bold]Full Model Breakdown (6 layers):[/bold]")
    encoder_time = next(r["mean_ms"] for r in results if r["name"] == "SemanticFoldingEncoder")
    full_fwd = next(r["mean_ms"] for r in results if r["name"] == "FullModel_Forward")
    layers_time = block_time * 6
    other_overhead = full_fwd - encoder_time - layers_time

    console.print(f"   Encoder:      {encoder_time:.2f} ms ({encoder_time/full_fwd*100:.1f}%)")
    console.print(f"   6 Blocks:     {layers_time:.2f} ms ({layers_time/full_fwd*100:.1f}%)")
    console.print(f"   LM Head/etc:  {max(0, other_overhead):.2f} ms ({max(0, other_overhead)/full_fwd*100:.1f}%)")

    # Optimization targets
    console.print("\n[bold]OPTIMIZATION TARGETS (to achieve <120s):[/bold]")
    if time_per_1000_iter > target_time:
        speedup = time_per_1000_iter / target_time
        sorted_components = sorted(component_results, key=lambda x: x["mean_ms"], reverse=True)[:4]
        for r in sorted_components:
            target_ms = r["mean_ms"] / speedup
            console.print(f"   {r['name']:24s}: {r['mean_ms']:.2f} ms -> {target_ms:.2f} ms ({speedup:.2f}x)")

    # Memory
    if DEVICE == "cuda":
        console.print(f"\n[bold]GPU Memory:[/bold]")
        console.print(f"   Peak:      {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
