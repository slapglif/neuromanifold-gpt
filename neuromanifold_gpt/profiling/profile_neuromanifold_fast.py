#!/usr/bin/env python3
"""
NeuroManifoldGPT Performance Profiler (FAST VERSION)

Profiles each component separately with reduced iterations for quick results.
"""

import time
import torch
import torch.nn as nn
from typing import Callable
from rich.console import Console
from rich.table import Table

# Import model components
from neuromanifold_gpt.config import NeuroManifoldConfig
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT
from neuromanifold_gpt.model.semantic_folding import SemanticFoldingEncoder
from neuromanifold_gpt.model.manifold import ManifoldProjection
from neuromanifold_gpt.model.spectral import SpectralDecomposition
from neuromanifold_gpt.model.attention.fhn import FHNAttention, FHNDynamics
from neuromanifold_gpt.model.block import NeuroManifoldBlock, SwiGLU
from neuromanifold_gpt.model.kan.wave import WaveKANFFN
from neuromanifold_gpt.model.kan.cheby import ChebyKANFFN

console = Console()

# Fast profiling parameters
BATCH_SIZE = 64
SEQ_LEN = 256
N_WARMUP = 3
N_ITERS = 10  # Reduced for speed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def profile_component(
    name: str,
    module: nn.Module,
    input_fn: Callable[[], tuple],
    n_warmup: int = N_WARMUP,
    n_iters: int = N_ITERS,
) -> dict:
    """Profile a single component."""
    module = module.to(DEVICE)
    module.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            inputs = input_fn()
            _ = module(*inputs)
            if DEVICE == "cuda":
                torch.cuda.synchronize()

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(n_iters):
            inputs = input_fn()
            if DEVICE == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            output = module(*inputs)

            if DEVICE == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

    return {
        "name": name,
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
    }


def profile_forward_backward(
    name: str,
    module: nn.Module,
    input_fn: Callable[[], tuple],
    loss_fn: Callable,
    n_warmup: int = N_WARMUP,
    n_iters: int = N_ITERS,
) -> dict:
    """Profile forward + backward pass (training iteration)."""
    module = module.to(DEVICE)
    module.train()

    # Warmup
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

        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    return {
        "name": name,
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
    }


def main():
    console.print(f"\n[bold cyan]NeuroManifoldGPT Profiler (FAST)[/bold cyan]")
    console.print(f"Device: {DEVICE}")
    console.print(f"Batch Size: {BATCH_SIZE}, Seq Len: {SEQ_LEN}")
    console.print(f"Warmup: {N_WARMUP}, Iterations: {N_ITERS}\n")

    # Build config
    config = NeuroManifoldConfig(
        vocab_size=65,  # char-level
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

    # =========================================================================
    # 1. SemanticFoldingEncoder
    # =========================================================================
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

    result = profile_component("SemanticFoldingEncoder", encoder, encoder_input)
    results.append(result)
    console.print(f"{result['mean_ms']:.2f} ms")

    # =========================================================================
    # 2. ManifoldProjection
    # =========================================================================
    console.print("[bold]2. ManifoldProjection...[/bold]", end=" ")
    manifold = ManifoldProjection(
        sdr_size=config.sdr_size,
        manifold_dim=config.manifold_dim,
    )

    def manifold_input():
        sdr = torch.randn(BATCH_SIZE, SEQ_LEN, config.sdr_size, device=DEVICE)
        return (sdr,)

    result = profile_component("ManifoldProjection", manifold, manifold_input)
    results.append(result)
    console.print(f"{result['mean_ms']:.2f} ms")

    # =========================================================================
    # 3. SpectralDecomposition
    # =========================================================================
    console.print("[bold]3. SpectralDecomposition...[/bold]", end=" ")
    spectral = SpectralDecomposition(
        manifold_dim=config.manifold_dim,
        n_eigenvectors=config.n_eigenvectors,
    )

    def spectral_input():
        coords = torch.randn(BATCH_SIZE, SEQ_LEN, config.manifold_dim, device=DEVICE)
        metric = torch.eye(config.manifold_dim, device=DEVICE).unsqueeze(0).unsqueeze(0).expand(
            BATCH_SIZE, SEQ_LEN, -1, -1
        )
        return (coords, metric)

    result = profile_component("SpectralDecomposition", spectral, spectral_input)
    results.append(result)
    console.print(f"{result['mean_ms']:.2f} ms")

    # =========================================================================
    # 4. FHNDynamics (just the core dynamics) - SMALL INPUT FOR SPEED
    # =========================================================================
    console.print("[bold]4. FHNDynamics...[/bold]", end=" ")
    head_dim = config.n_embd // config.n_heads
    fhn_dynamics = FHNDynamics(
        dim=head_dim,
        tau=config.fhn_tau,
        threshold=config.fhn_threshold,
        use_imex=config.use_fhn_imex,
    )

    def fhn_input():
        # Reduced size for profiling
        stimulus = torch.randn(
            BATCH_SIZE, config.n_heads, config.n_eigenvectors, head_dim,
            device=DEVICE
        )
        return (stimulus, config.n_fhn_steps)

    result = profile_component("FHNDynamics", fhn_dynamics, fhn_input)
    results.append(result)
    console.print(f"{result['mean_ms']:.2f} ms")

    # =========================================================================
    # 5. FHNAttention (full attention)
    # =========================================================================
    console.print("[bold]5. FHNAttention...[/bold]", end=" ")
    fhn_attn = FHNAttention(
        embed_dim=config.n_embd,
        n_heads=config.n_heads,
        threshold=config.fhn_threshold,
        tau=config.fhn_tau,
        n_fhn_steps=config.n_fhn_steps,
        use_imex=config.use_fhn_imex,
        use_partitioning=config.use_fhn_partitioning,
    )

    def fhn_attn_input():
        x = torch.randn(BATCH_SIZE, SEQ_LEN, config.n_embd, device=DEVICE)
        spectral_basis = torch.randn(BATCH_SIZE, SEQ_LEN, config.n_eigenvectors, device=DEVICE)
        return (x, spectral_basis)

    result = profile_component("FHNAttention", fhn_attn, fhn_attn_input)
    results.append(result)
    console.print(f"{result['mean_ms']:.2f} ms")

    # =========================================================================
    # 6. WaveKAN FFN
    # =========================================================================
    console.print("[bold]6. WaveKAN FFN...[/bold]", end=" ")
    mlp_hidden = int(config.n_embd * 4.0)
    wavekan_ffn = WaveKANFFN(
        embed_dim=config.n_embd,
        hidden_dim=mlp_hidden,
        wavelet_type=config.kan_wavelet,
        use_fast_wavekan=config.use_fast_wavekan,
    )

    def ffn_input():
        x = torch.randn(BATCH_SIZE, SEQ_LEN, config.n_embd, device=DEVICE)
        return (x,)

    result = profile_component("WaveKAN_FFN", wavekan_ffn, ffn_input)
    results.append(result)
    console.print(f"{result['mean_ms']:.2f} ms")

    # =========================================================================
    # 7. SwiGLU FFN (baseline comparison)
    # =========================================================================
    console.print("[bold]7. SwiGLU FFN...[/bold]", end=" ")
    swiglu_hidden = int(config.n_embd * 4.0 * 2 / 3)
    swiglu_ffn = SwiGLU(
        dim=config.n_embd,
        hidden_dim=swiglu_hidden,
    )

    result = profile_component("SwiGLU_FFN", swiglu_ffn, ffn_input)
    results.append(result)
    console.print(f"{result['mean_ms']:.2f} ms")

    # =========================================================================
    # 8. ChebyKAN FFN
    # =========================================================================
    console.print("[bold]8. ChebyKAN FFN...[/bold]", end=" ")
    chebykan_ffn = ChebyKANFFN(
        embed_dim=config.n_embd,
        hidden_dim=mlp_hidden,
        degree=config.kan_degree,
    )

    result = profile_component("ChebyKAN_FFN", chebykan_ffn, ffn_input)
    results.append(result)
    console.print(f"{result['mean_ms']:.2f} ms")

    # =========================================================================
    # 9. Full NeuroManifoldBlock
    # =========================================================================
    console.print("[bold]9. NeuroManifoldBlock...[/bold]", end=" ")
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
        sdr = torch.randn(BATCH_SIZE, SEQ_LEN, config.sdr_size, device=DEVICE)
        return (sdr,)

    result = profile_component("NeuroManifoldBlock", block, block_input)
    results.append(result)
    console.print(f"{result['mean_ms']:.2f} ms")

    # =========================================================================
    # 10. Full Model Forward Pass
    # =========================================================================
    console.print("[bold]10. Full Model Forward...[/bold]", end=" ")
    model = NeuroManifoldGPT(config)

    def model_input():
        tokens = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
        return (tokens,)

    result = profile_component("FullModel_Forward", model, model_input)
    results.append(result)
    console.print(f"{result['mean_ms']:.2f} ms")

    # =========================================================================
    # 11. Full Model Forward + Backward (Training Iteration)
    # =========================================================================
    console.print("[bold]11. Full Model FwdBwd...[/bold]", end=" ")
    model = NeuroManifoldGPT(config)

    def model_input_train():
        tokens = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
        targets = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
        return (tokens, targets)

    def loss_fn_model(output):
        logits, loss, info = output
        return loss

    result = profile_forward_backward("FullModel_FwdBwd", model, model_input_train, loss_fn_model)
    results.append(result)
    console.print(f"{result['mean_ms']:.2f} ms")

    # =========================================================================
    # Results Table
    # =========================================================================
    console.print("\n")
    table = Table(title="Component Profiling Results")
    table.add_column("Component", style="cyan")
    table.add_column("Mean (ms)", justify="right")
    table.add_column("Min (ms)", justify="right")
    table.add_column("Max (ms)", justify="right")
    table.add_column("% of Block", justify="right")

    # Calculate percentage of block time
    block_time = next(r["mean_ms"] for r in results if r["name"] == "NeuroManifoldBlock")

    for r in results:
        pct = (r["mean_ms"] / block_time * 100) if r["name"] != "NeuroManifoldBlock" else 100.0
        if r["name"] in ["FullModel_Forward", "FullModel_FwdBwd"]:
            pct_str = "-"
        else:
            pct_str = f"{pct:.1f}%"

        table.add_row(
            r["name"],
            f"{r['mean_ms']:.2f}",
            f"{r['min_ms']:.2f}",
            f"{r['max_ms']:.2f}",
            pct_str,
        )

    console.print(table)

    # =========================================================================
    # Analysis
    # =========================================================================
    console.print("\n[bold yellow]===== PERFORMANCE ANALYSIS =====[/bold yellow]")

    # Find the bottleneck
    component_results = [r for r in results if r["name"] not in ["FullModel_Forward", "FullModel_FwdBwd", "NeuroManifoldBlock"]]
    bottleneck = max(component_results, key=lambda x: x["mean_ms"])

    console.print(f"\n[bold red]#1 BOTTLENECK: {bottleneck['name']}[/bold red]")
    console.print(f"   Time: {bottleneck['mean_ms']:.2f} ms ({bottleneck['mean_ms']/block_time*100:.1f}% of block)")

    # Calculate time per 1000 iterations
    fwd_bwd_time = next(r["mean_ms"] for r in results if r["name"] == "FullModel_FwdBwd")
    time_per_1000_iter = fwd_bwd_time  # ms per iteration -> seconds for 1000
    console.print(f"\n[bold]Current time per 1000 iterations:[/bold] {time_per_1000_iter:.1f} seconds")

    # Target: <120s for 1000 iters
    target_time = 120.0
    current_time = time_per_1000_iter
    if current_time > target_time:
        speedup_needed = current_time / target_time
        console.print(f"[bold]Speedup needed for <120s:[/bold] {speedup_needed:.2f}x")
    else:
        console.print(f"[bold green]Already under 120s target![/bold green]")

    # Component breakdown for block
    console.print("\n[bold]Block Component Breakdown:[/bold]")
    block_components = ["ManifoldProjection", "SpectralDecomposition", "FHNAttention", "WaveKAN_FFN"]
    total_accounted = 0.0
    for name in block_components:
        r = next((x for x in results if x["name"] == name), None)
        if r:
            pct = r["mean_ms"] / block_time * 100
            total_accounted += r["mean_ms"]
            bar = "#" * int(pct / 5)
            console.print(f"   {name:24s}: {r['mean_ms']:7.2f} ms ({pct:5.1f}%) {bar}")

    overhead = block_time - total_accounted
    pct = overhead / block_time * 100
    bar = "#" * int(pct / 5)
    console.print(f"   {'Other (SDR proj, etc)':24s}: {overhead:7.2f} ms ({pct:5.1f}%) {bar}")

    # FFN comparison
    console.print("\n[bold]FFN Comparison:[/bold]")
    wavekan = next(r for r in results if r["name"] == "WaveKAN_FFN")
    swiglu = next(r for r in results if r["name"] == "SwiGLU_FFN")
    chebykan = next(r for r in results if r["name"] == "ChebyKAN_FFN")

    console.print(f"   SwiGLU:   {swiglu['mean_ms']:6.2f} ms (1.00x baseline)")
    console.print(f"   WaveKAN:  {wavekan['mean_ms']:6.2f} ms ({wavekan['mean_ms']/swiglu['mean_ms']:.2f}x)")
    console.print(f"   ChebyKAN: {chebykan['mean_ms']:6.2f} ms ({chebykan['mean_ms']/swiglu['mean_ms']:.2f}x)")

    # Per-layer cost
    console.print(f"\n[bold]Per-Layer Analysis (6 layers):[/bold]")
    encoder_time = next(r["mean_ms"] for r in results if r["name"] == "SemanticFoldingEncoder")
    full_fwd = next(r["mean_ms"] for r in results if r["name"] == "FullModel_Forward")

    layers_time = block_time * 6
    other_model_overhead = full_fwd - encoder_time - layers_time
    console.print(f"   Encoder: {encoder_time:.2f} ms")
    console.print(f"   6 Blocks: {layers_time:.2f} ms")
    console.print(f"   LM Head/Norms: {other_model_overhead:.2f} ms")

    # Optimization targets
    console.print("\n[bold]OPTIMIZATION TARGETS:[/bold]")
    if current_time > target_time:
        speedup = current_time / target_time
        for name in ["SemanticFoldingEncoder", "ManifoldProjection", "FHNAttention", "WaveKAN_FFN"]:
            r = next((x for x in results if x["name"] == name), None)
            if r:
                target_ms = r["mean_ms"] / speedup
                console.print(f"   {name:24s}: {r['mean_ms']:.2f} ms -> {target_ms:.2f} ms ({speedup:.2f}x needed)")

    # Memory
    if DEVICE == "cuda":
        console.print(f"\n[bold]GPU Memory:[/bold]")
        console.print(f"   Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        console.print(f"   Reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
