#!/usr/bin/env python3
"""
Complete NeuroManifoldGPT Profiler
Works around JIT warm-up issues by running sufficient warm-up iterations.
"""

import torch
from rich.console import Console
from rich.table import Table

from neuromanifold_gpt.utils.logging import get_logger
from neuromanifold_gpt.utils.profiling import (
    cleanup,
    profile_component,
    profile_forward_backward,
)

logger = get_logger(__name__)
console = Console()  # Keep for table rendering

# Profiling parameters
BATCH_SIZE = 16
SEQ_LEN = 256
N_WARMUP = 10  # Extra warmup to handle JIT issues
N_ITERS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    logger.section("NeuroManifoldGPT Complete Profiler")
    logger.info(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(
            f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    logger.info(f"Batch Size: {BATCH_SIZE}, Seq Len: {SEQ_LEN}")
    logger.info(f"Warmup: {N_WARMUP}, Iterations: {N_ITERS}")

    # Import components
    from neuromanifold_gpt.config import NeuroManifoldConfig
    from neuromanifold_gpt.config.block_config import NeuroManifoldBlockConfig
    from neuromanifold_gpt.model.attention.fhn import FHNAttention
    from neuromanifold_gpt.model.block import NeuroManifoldBlock, SwiGLU
    from neuromanifold_gpt.model.gpt import NeuroManifoldGPT
    from neuromanifold_gpt.model.kan.cheby import ChebyKANFFN
    from neuromanifold_gpt.model.kan.wave import WaveKANFFN
    from neuromanifold_gpt.model.manifold import ManifoldProjection
    from neuromanifold_gpt.model.semantic_folding import SemanticFoldingEncoder
    from neuromanifold_gpt.model.spectral import SpectralDecomposition

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
    logger.info("Profiling SemanticFoldingEncoder...")
    encoder = SemanticFoldingEncoder(
        vocab_size=config.vocab_size,
        sdr_size=config.sdr_size,
        n_active=config.sdr_n_active,
        embed_dim=config.sdr_embed_dim,
        context_size=config.sdr_context_size,
    )

    def encoder_input():
        return (
            torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE),
        )

    r = profile_component(
        "SemanticFoldingEncoder",
        encoder,
        encoder_input,
        n_warmup=N_WARMUP,
        n_iters=N_ITERS,
        device=DEVICE,
        track_memory=True,
    )
    results.append(r)
    del encoder
    cleanup()
    logger.metric("SemanticFoldingEncoder", r["mean_ms"], unit="ms")

    # 2. ManifoldProjection
    logger.info("Profiling ManifoldProjection...")
    manifold = ManifoldProjection(config.sdr_size, config.manifold_dim)

    def manifold_input():
        return (torch.randn(BATCH_SIZE, SEQ_LEN, config.sdr_size, device=DEVICE),)

    r = profile_component(
        "ManifoldProjection",
        manifold,
        manifold_input,
        n_warmup=N_WARMUP,
        n_iters=N_ITERS,
        device=DEVICE,
        track_memory=True,
    )
    results.append(r)
    del manifold
    cleanup()
    logger.metric("ManifoldProjection", r["mean_ms"], unit="ms")

    # 3. SpectralDecomposition
    logger.info("Profiling SpectralDecomposition...")
    spectral = SpectralDecomposition(config.manifold_dim, config.n_eigenvectors)

    def spectral_input():
        coords = torch.randn(BATCH_SIZE, SEQ_LEN, config.manifold_dim, device=DEVICE)
        return (coords, None)

    r = profile_component(
        "SpectralDecomposition",
        spectral,
        spectral_input,
        n_warmup=N_WARMUP,
        n_iters=N_ITERS,
        device=DEVICE,
        track_memory=True,
    )
    results.append(r)
    del spectral
    cleanup()
    logger.metric("SpectralDecomposition", r["mean_ms"], unit="ms")

    # 4. FHNAttention
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

    def fhn_input():
        x = torch.randn(BATCH_SIZE, SEQ_LEN, config.n_embd, device=DEVICE)
        spectral_basis = torch.randn(
            BATCH_SIZE, SEQ_LEN, config.n_eigenvectors, device=DEVICE
        )
        return (x, spectral_basis)

    r = profile_component(
        "FHNAttention",
        fhn_attn,
        fhn_input,
        n_warmup=N_WARMUP,
        n_iters=N_ITERS,
        device=DEVICE,
        track_memory=True,
    )
    results.append(r)
    del fhn_attn
    cleanup()
    logger.metric("FHNAttention", r["mean_ms"], unit="ms")

    # 5. WaveKAN FFN
    logger.info("Profiling WaveKAN FFN...")
    mlp_hidden = int(config.n_embd * 4.0)
    wavekan_ffn = WaveKANFFN(
        embed_dim=config.n_embd,
        hidden_dim=mlp_hidden,
        wavelet_type=config.kan_wavelet,
        use_fast_wavekan=config.use_fast_wavekan,
    )

    def ffn_input():
        return (torch.randn(BATCH_SIZE, SEQ_LEN, config.n_embd, device=DEVICE),)

    r = profile_component(
        "WaveKAN_FFN",
        wavekan_ffn,
        ffn_input,
        n_warmup=N_WARMUP,
        n_iters=N_ITERS,
        device=DEVICE,
        track_memory=True,
    )
    results.append(r)
    del wavekan_ffn
    cleanup()
    logger.metric("WaveKAN_FFN", r["mean_ms"], unit="ms")

    # 6. SwiGLU FFN
    logger.info("Profiling SwiGLU FFN...")
    swiglu_ffn = SwiGLU(config.n_embd, int(mlp_hidden * 2 / 3))
    r = profile_component(
        "SwiGLU_FFN",
        swiglu_ffn,
        ffn_input,
        n_warmup=N_WARMUP,
        n_iters=N_ITERS,
        device=DEVICE,
        track_memory=True,
    )
    results.append(r)
    del swiglu_ffn
    cleanup()
    logger.metric("SwiGLU_FFN", r["mean_ms"], unit="ms")

    # 7. ChebyKAN FFN
    logger.info("Profiling ChebyKAN FFN...")
    chebykan_ffn = ChebyKANFFN(config.n_embd, mlp_hidden, degree=config.kan_degree)
    r = profile_component(
        "ChebyKAN_FFN",
        chebykan_ffn,
        ffn_input,
        n_warmup=N_WARMUP,
        n_iters=N_ITERS,
        device=DEVICE,
        track_memory=True,
    )
    results.append(r)
    del chebykan_ffn
    cleanup()
    logger.metric("ChebyKAN_FFN", r["mean_ms"], unit="ms")

    # 8. NeuroManifoldBlock
    logger.info("Profiling NeuroManifoldBlock...")
    block_cfg = NeuroManifoldBlockConfig.from_model_config(config, layer_idx=0)
    block = NeuroManifoldBlock(config=block_cfg)

    def block_input():
        return (torch.randn(BATCH_SIZE, SEQ_LEN, config.sdr_size, device=DEVICE),)

    r = profile_component(
        "NeuroManifoldBlock",
        block,
        block_input,
        n_warmup=N_WARMUP,
        n_iters=N_ITERS,
        device=DEVICE,
        track_memory=True,
    )
    results.append(r)
    del block
    cleanup()
    logger.metric("NeuroManifoldBlock", r["mean_ms"], unit="ms")

    # 9. Full Model Forward
    logger.info("Profiling Full Model Forward...")
    model = NeuroManifoldGPT(config)

    def model_input():
        return (
            torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE),
        )

    r = profile_component(
        "FullModel_Forward",
        model,
        model_input,
        n_warmup=N_WARMUP,
        n_iters=N_ITERS,
        device=DEVICE,
        track_memory=True,
    )
    results.append(r)
    del model
    cleanup()
    logger.metric("FullModel_Forward", r["mean_ms"], unit="ms")

    # 10. Full Model Forward+Backward
    logger.info("Profiling Full Model FwdBwd...")
    model = NeuroManifoldGPT(config)

    def model_input_train():
        tokens = torch.randint(
            0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE
        )
        targets = torch.randint(
            0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE
        )
        return (tokens, targets)

    def loss_fn(output):
        logits, loss, info = output
        return loss

    r = profile_forward_backward(
        "FullModel_FwdBwd",
        model,
        model_input_train,
        loss_fn,
        n_warmup=N_WARMUP,
        n_iters=N_ITERS,
        device=DEVICE,
        track_memory=True,
    )
    results.append(r)
    del model
    cleanup()
    logger.metric("FullModel_FwdBwd", r["mean_ms"], unit="ms")

    # =========================================================================
    # Results Table
    # =========================================================================
    logger.section("Profiling Results")
    table = Table(title=f"Profiling Results (B={BATCH_SIZE}, T={SEQ_LEN})")
    table.add_column("Component", style="cyan")
    table.add_column("Mean (ms)", justify="right")
    table.add_column("@ B=64 (est)", justify="right", style="yellow")
    table.add_column("% of Block", justify="right")

    block_time = next(
        r["mean_ms"] for r in results if r["name"] == "NeuroManifoldBlock"
    )
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

    logger.table(table)

    # =========================================================================
    # Analysis
    # =========================================================================
    logger.section("PERFORMANCE ANALYSIS")

    # Bottleneck
    component_results = [
        r
        for r in results
        if r["name"]
        not in ["FullModel_Forward", "FullModel_FwdBwd", "NeuroManifoldBlock"]
    ]
    bottleneck = max(component_results, key=lambda x: x["mean_ms"])

    logger.warning(f"#1 BOTTLENECK: {bottleneck['name']}")
    logger.info(
        f"   Time: {bottleneck['mean_ms']:.2f} ms ({bottleneck['mean_ms']/block_time*100:.1f}% of block)"
    )

    # Time per 1000 iterations
    fwd_bwd_time = next(
        r["mean_ms"] for r in results if r["name"] == "FullModel_FwdBwd"
    )
    est_fwd_bwd_64 = fwd_bwd_time * scale
    time_per_1000_iter = est_fwd_bwd_64

    logger.info(f"Current @ B={BATCH_SIZE}: {fwd_bwd_time:.2f} ms/iter")
    logger.info(f"Estimated @ B=64: {est_fwd_bwd_64:.2f} ms/iter")
    logger.info(f"Est. time for 1000 iters: {time_per_1000_iter:.1f} seconds")

    # Target
    target_time = 120.0
    if time_per_1000_iter > target_time:
        speedup_needed = time_per_1000_iter / target_time
        logger.warning(f"Speedup needed for <120s: {speedup_needed:.2f}x")
    else:
        logger.info("Already under 120s target!")

    # Component breakdown
    logger.section("Block Component Breakdown")
    block_components = [
        "ManifoldProjection",
        "SpectralDecomposition",
        "FHNAttention",
        "WaveKAN_FFN",
    ]
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
    logger.info(
        f"   {'SDR proj/Norms/etc':24s}: {overhead:7.2f} ms ({pct:5.1f}%) {bar}"
    )

    # FFN Comparison
    logger.section("FFN Comparison")
    wavekan = next(r for r in results if r["name"] == "WaveKAN_FFN")
    swiglu = next(r for r in results if r["name"] == "SwiGLU_FFN")
    chebykan = next(r for r in results if r["name"] == "ChebyKAN_FFN")

    logger.info(f"   SwiGLU:   {swiglu['mean_ms']:6.2f} ms (1.00x baseline)")
    logger.info(
        f"   WaveKAN:  {wavekan['mean_ms']:6.2f} ms ({wavekan['mean_ms']/swiglu['mean_ms']:.2f}x)"
    )
    logger.info(
        f"   ChebyKAN: {chebykan['mean_ms']:6.2f} ms ({chebykan['mean_ms']/swiglu['mean_ms']:.2f}x)"
    )

    # Full model breakdown
    logger.section("Full Model Breakdown (6 layers)")
    encoder_time = next(
        r["mean_ms"] for r in results if r["name"] == "SemanticFoldingEncoder"
    )
    full_fwd = next(r["mean_ms"] for r in results if r["name"] == "FullModel_Forward")
    layers_time = block_time * 6
    other_overhead = full_fwd - encoder_time - layers_time

    logger.info(
        f"   Encoder:      {encoder_time:.2f} ms ({encoder_time/full_fwd*100:.1f}%)"
    )
    logger.info(
        f"   6 Blocks:     {layers_time:.2f} ms ({layers_time/full_fwd*100:.1f}%)"
    )
    logger.info(
        f"   LM Head/etc:  {max(0, other_overhead):.2f} ms ({max(0, other_overhead)/full_fwd*100:.1f}%)"
    )

    # Optimization targets
    logger.section("OPTIMIZATION TARGETS (to achieve <120s)")
    if time_per_1000_iter > target_time:
        speedup = time_per_1000_iter / target_time
        sorted_components = sorted(
            component_results, key=lambda x: x["mean_ms"], reverse=True
        )[:4]
        for r in sorted_components:
            target_ms = r["mean_ms"] / speedup
            logger.info(
                f"   {r['name']:24s}: {r['mean_ms']:.2f} ms -> {target_ms:.2f} ms ({speedup:.2f}x)"
            )

    # Memory
    if DEVICE == "cuda":
        logger.section("GPU Memory")
        logger.metric("Peak", torch.cuda.max_memory_allocated() / 1e9, unit="GB")


if __name__ == "__main__":
    main()
