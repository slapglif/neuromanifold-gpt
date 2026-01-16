#!/usr/bin/env python3
"""
NeuroManifoldGPT Performance Profiler v2

Uses smaller batch size initially to avoid OOM, then scales.
"""

import time
import gc
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
from neuromanifold_gpt.utils.logging import get_logger

logger = get_logger(__name__)
console = Console()  # Keep for table rendering

# Profiling parameters - REDUCED to avoid OOM
BATCH_SIZE = 16  # Reduced from 64
SEQ_LEN = 256
N_WARMUP = 2
N_ITERS = 5  # Very fast
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def cleanup():
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


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

    # Cleanup
    del module
    cleanup()

    return {
        "name": name,
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def profile_forward_backward(
    name: str,
    module: nn.Module,
    input_fn: Callable[[], tuple],
    loss_fn: Callable,
    n_warmup: int = N_WARMUP,
    n_iters: int = N_ITERS,
) -> dict:
    """Profile forward + backward pass."""
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

    # Cleanup
    del module
    cleanup()

    return {
        "name": name,
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def main():
    logger.section("NeuroManifoldGPT Profiler v2")
    logger.info(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    logger.info(f"Batch Size: {BATCH_SIZE}, Seq Len: {SEQ_LEN}")
    logger.info(f"Warmup: {N_WARMUP}, Iterations: {N_ITERS}")

    # Build config
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

    # =========================================================================
    # 1. SemanticFoldingEncoder
    # =========================================================================
    logger.info("Profiling SemanticFoldingEncoder...")
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
    logger.metric("SemanticFoldingEncoder", result['mean_ms'], unit="ms")

    # =========================================================================
    # 2. ManifoldProjection
    # =========================================================================
    logger.info("Profiling ManifoldProjection...")
    manifold = ManifoldProjection(
        sdr_size=config.sdr_size,
        manifold_dim=config.manifold_dim,
    )

    def manifold_input():
        sdr = torch.randn(BATCH_SIZE, SEQ_LEN, config.sdr_size, device=DEVICE)
        return (sdr,)

    result = profile_component("ManifoldProjection", manifold, manifold_input)
    results.append(result)
    logger.metric("ManifoldProjection", result['mean_ms'], unit="ms")

    # =========================================================================
    # 3. SpectralDecomposition
    # =========================================================================
    logger.info("Profiling SpectralDecomposition...")
    spectral = SpectralDecomposition(
        manifold_dim=config.manifold_dim,
        n_eigenvectors=config.n_eigenvectors,
    )

    def spectral_input():
        coords = torch.randn(BATCH_SIZE, SEQ_LEN, config.manifold_dim, device=DEVICE)
        metric = torch.eye(config.manifold_dim, device=DEVICE).unsqueeze(0).unsqueeze(0).expand(
            BATCH_SIZE, SEQ_LEN, -1, -1
        ).contiguous()
        return (coords, metric)

    result = profile_component("SpectralDecomposition", spectral, spectral_input)
    results.append(result)
    logger.metric("SpectralDecomposition", result['mean_ms'], unit="ms")

    # =========================================================================
    # 4. FHNDynamics - PROFILE SEPARATELY
    # =========================================================================
    logger.info("Profiling FHNDynamics...")
    head_dim = config.n_embd // config.n_heads
    fhn_dynamics = FHNDynamics(
        dim=head_dim,
        tau=config.fhn_tau,
        threshold=config.fhn_threshold,
        use_imex=config.use_fhn_imex,
    )

    # Use smaller input to avoid memory issues
    def fhn_input():
        # Shape: (B, H, k, head_dim)
        stimulus = torch.randn(
            BATCH_SIZE, config.n_heads, config.n_eigenvectors, head_dim,
            device=DEVICE
        )
        return (stimulus, config.n_fhn_steps)

    result = profile_component("FHNDynamics", fhn_dynamics, fhn_input)
    results.append(result)
    logger.metric("FHNDynamics", result['mean_ms'], unit="ms")

    # =========================================================================
    # 5. FHNAttention
    # =========================================================================
    logger.info("Profiling FHNAttention...")
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
    logger.metric("FHNAttention", result['mean_ms'], unit="ms")

    # =========================================================================
    # 6. WaveKAN FFN
    # =========================================================================
    logger.info("Profiling WaveKAN FFN...")
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
    logger.metric("WaveKAN_FFN", result['mean_ms'], unit="ms")

    # =========================================================================
    # 7. SwiGLU FFN
    # =========================================================================
    logger.info("Profiling SwiGLU FFN...")
    swiglu_hidden = int(config.n_embd * 4.0 * 2 / 3)
    swiglu_ffn = SwiGLU(
        dim=config.n_embd,
        hidden_dim=swiglu_hidden,
    )

    result = profile_component("SwiGLU_FFN", swiglu_ffn, ffn_input)
    results.append(result)
    logger.metric("SwiGLU_FFN", result['mean_ms'], unit="ms")

    # =========================================================================
    # 8. ChebyKAN FFN
    # =========================================================================
    logger.info("Profiling ChebyKAN FFN...")
    chebykan_ffn = ChebyKANFFN(
        embed_dim=config.n_embd,
        hidden_dim=mlp_hidden,
        degree=config.kan_degree,
    )

    result = profile_component("ChebyKAN_FFN", chebykan_ffn, ffn_input)
    results.append(result)
    logger.metric("ChebyKAN_FFN", result['mean_ms'], unit="ms")

    # =========================================================================
    # 9. Full NeuroManifoldBlock
    # =========================================================================
    logger.info("Profiling NeuroManifoldBlock...")
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
    logger.metric("NeuroManifoldBlock", result['mean_ms'], unit="ms")

    # =========================================================================
    # 10. Full Model Forward Pass
    # =========================================================================
    logger.info("Profiling Full Model Forward...")
    model = NeuroManifoldGPT(config)

    def model_input():
        tokens = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
        return (tokens,)

    result = profile_component("FullModel_Forward", model, model_input)
    results.append(result)
    logger.metric("FullModel_Forward", result['mean_ms'], unit="ms")

    # =========================================================================
    # 11. Full Model Forward + Backward
    # =========================================================================
    logger.info("Profiling Full Model FwdBwd...")
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
    logger.metric("FullModel_FwdBwd", result['mean_ms'], unit="ms")

    # =========================================================================
    # Results Table
    # =========================================================================
    table = Table(title=f"Profiling Results (B={BATCH_SIZE}, T={SEQ_LEN})")
    table.add_column("Component", style="cyan")
    table.add_column("Mean (ms)", justify="right")
    table.add_column("@ B=64 (est)", justify="right", style="yellow")
    table.add_column("% of Block", justify="right")

    block_time = next(r["mean_ms"] for r in results if r["name"] == "NeuroManifoldBlock")
    scale = 64 / BATCH_SIZE  # Estimate for batch=64

    for r in results:
        est_64 = r["mean_ms"] * scale
        pct = (r["mean_ms"] / block_time * 100) if "Block" not in r["name"] and "Model" not in r["name"] else 100.0
        if r["name"] in ["FullModel_Forward", "FullModel_FwdBwd"]:
            pct_str = "-"
        else:
            pct_str = f"{pct:.1f}%"

        table.add_row(
            r["name"],
            f"{r['mean_ms']:.2f}",
            f"{est_64:.2f}",
            pct_str,
        )

    logger.table(table)

    # =========================================================================
    # Analysis
    # =========================================================================
    logger.section("Performance Analysis")

    # Find the bottleneck
    component_results = [r for r in results if r["name"] not in ["FullModel_Forward", "FullModel_FwdBwd", "NeuroManifoldBlock"]]
    bottleneck = max(component_results, key=lambda x: x["mean_ms"])

    logger.warning(f"#1 BOTTLENECK: {bottleneck['name']}")
    logger.info(f"   Time: {bottleneck['mean_ms']:.2f} ms ({bottleneck['mean_ms']/block_time*100:.1f}% of block)")

    # Estimate time for batch=64
    fwd_bwd_time = next(r["mean_ms"] for r in results if r["name"] == "FullModel_FwdBwd")
    est_fwd_bwd_64 = fwd_bwd_time * scale
    time_per_1000_iter_64 = est_fwd_bwd_64  # seconds for 1000 iters at batch=64

    logger.info(f"Measured (B={BATCH_SIZE}): {fwd_bwd_time:.2f} ms/iter")
    logger.info(f"Estimated (B=64): {est_fwd_bwd_64:.2f} ms/iter")
    logger.info(f"Est. time per 1000 iters (B=64): {time_per_1000_iter_64:.1f} seconds")

    # Target: <120s for 1000 iters
    target_time = 120.0
    current_time = time_per_1000_iter_64
    if current_time > target_time:
        speedup_needed = current_time / target_time
        logger.info(f"Speedup needed for <120s: {speedup_needed:.2f}x")
    else:
        logger.success("Already under 120s target!")

    # Component breakdown
    logger.info("Block Component Breakdown:")
    block_components = ["ManifoldProjection", "SpectralDecomposition", "FHNAttention", "WaveKAN_FFN"]
    total_accounted = 0.0
    for name in block_components:
        r = next((x for x in results if x["name"] == name), None)
        if r:
            pct = r["mean_ms"] / block_time * 100
            total_accounted += r["mean_ms"]
            bar = "#" * int(pct / 3)
            logger.info(f"   {name:24s}: {r['mean_ms']:7.2f} ms ({pct:5.1f}%) {bar}")

    overhead = block_time - total_accounted
    pct = overhead / block_time * 100
    bar = "#" * int(pct / 3)
    logger.info(f"   {'SDR proj/Norms/etc':24s}: {overhead:7.2f} ms ({pct:5.1f}%) {bar}")

    # FFN comparison
    logger.info("FFN Comparison:")
    wavekan = next(r for r in results if r["name"] == "WaveKAN_FFN")
    swiglu = next(r for r in results if r["name"] == "SwiGLU_FFN")
    chebykan = next(r for r in results if r["name"] == "ChebyKAN_FFN")

    logger.info(f"   SwiGLU:   {swiglu['mean_ms']:6.2f} ms (1.00x baseline)")
    logger.info(f"   WaveKAN:  {wavekan['mean_ms']:6.2f} ms ({wavekan['mean_ms']/swiglu['mean_ms']:.2f}x)")
    logger.info(f"   ChebyKAN: {chebykan['mean_ms']:6.2f} ms ({chebykan['mean_ms']/swiglu['mean_ms']:.2f}x)")

    # Full model breakdown
    logger.info("Full Model Breakdown (6 layers):")
    encoder_time = next(r["mean_ms"] for r in results if r["name"] == "SemanticFoldingEncoder")
    full_fwd = next(r["mean_ms"] for r in results if r["name"] == "FullModel_Forward")
    layers_time = block_time * 6
    other_overhead = full_fwd - encoder_time - layers_time

    logger.info(f"   Encoder:      {encoder_time:.2f} ms ({encoder_time/full_fwd*100:.1f}%)")
    logger.info(f"   6 Blocks:     {layers_time:.2f} ms ({layers_time/full_fwd*100:.1f}%)")
    logger.info(f"   LM Head/etc:  {max(0, other_overhead):.2f} ms ({max(0, other_overhead)/full_fwd*100:.1f}%)")

    # Optimization targets
    if current_time > target_time:
        logger.info("OPTIMIZATION TARGETS (to achieve <120s):")
        speedup = current_time / target_time
        # Top 3 bottlenecks
        sorted_components = sorted(component_results, key=lambda x: x["mean_ms"], reverse=True)[:4]
        for r in sorted_components:
            target_ms = r["mean_ms"] / speedup
            logger.info(f"   {r['name']:24s}: {r['mean_ms']:.2f} ms -> {target_ms:.2f} ms (need {speedup:.2f}x)")

    # Memory
    if DEVICE == "cuda":
        logger.info("GPU Memory:")
        logger.info(f"   Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        logger.info(f"   Peak:      {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
