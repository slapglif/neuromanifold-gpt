#!/usr/bin/env python3
"""
Minimal profiler - profiles each component in isolation with small inputs.
Focus on finding memory/time bottlenecks.
"""

import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table

from neuromanifold_gpt.config.block_config import NeuroManifoldBlockConfig
from neuromanifold_gpt.utils.logging import get_logger
from neuromanifold_gpt.utils.profiling import cleanup, profile_component

logger = get_logger(__name__)
console = Console()  # Keep for table rendering

# Very small test parameters
BATCH_SIZE = 4
SEQ_LEN = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    logger.section("Minimal Component Profiler")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Batch Size: {BATCH_SIZE}, Seq Length: {SEQ_LEN}")

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
    logger.info("Profiling SDR Projection...")
    from neuromanifold_gpt.model.block import NeuroManifoldBlock

    sdr_proj = nn.Linear(sdr_size, embed_dim)
    r = profile_component(
        "SDR_Projection",
        sdr_proj,
        lambda: (torch.randn(BATCH_SIZE, SEQ_LEN, sdr_size, device=DEVICE),),
        n_warmup=2,
        n_iters=5,
        device=DEVICE,
        track_memory=True,
    )
    results.append(r)
    del sdr_proj
    cleanup()
    logger.metric("SDR_Projection", r["mean_ms"], unit="ms")

    # 2. Manifold Encoder (the big 3-layer MLP)
    logger.info("Profiling ManifoldProjection...")
    from neuromanifold_gpt.model.manifold import ManifoldProjection

    manifold = ManifoldProjection(sdr_size, manifold_dim)
    r = profile_component(
        "ManifoldProjection",
        manifold,
        lambda: (torch.randn(BATCH_SIZE, SEQ_LEN, sdr_size, device=DEVICE),),
        n_warmup=2,
        n_iters=5,
        device=DEVICE,
        track_memory=True,
    )
    results.append(r)
    del manifold
    cleanup()
    logger.metric("ManifoldProjection", r["mean_ms"], unit="ms")

    # 3. Spectral Decomposition
    logger.info("Profiling SpectralDecomposition...")
    from neuromanifold_gpt.model.spectral import SpectralDecomposition

    spectral = SpectralDecomposition(manifold_dim, n_eigenvectors)
    r = profile_component(
        "SpectralDecomp",
        spectral,
        lambda: (torch.randn(BATCH_SIZE, SEQ_LEN, manifold_dim, device=DEVICE), None),
        n_warmup=2,
        n_iters=5,
        device=DEVICE,
        track_memory=True,
    )
    results.append(r)
    del spectral
    cleanup()
    logger.metric("SpectralDecomp", r["mean_ms"], unit="ms")

    # 4. FHN Dynamics Core - This is the suspected bottleneck
    logger.info("Profiling FHNDynamics...")
    from neuromanifold_gpt.model.attention.fhn import FHNDynamics

    fhn = FHNDynamics(dim=head_dim, tau=12.5, threshold=0.5, use_imex=True)
    # Shape: (B, H, k, head_dim)
    r = profile_component(
        "FHNDynamics",
        fhn,
        lambda: (
            torch.randn(BATCH_SIZE, n_heads, n_eigenvectors, head_dim, device=DEVICE),
            2,
        ),
        n_warmup=2,
        n_iters=5,
        device=DEVICE,
        track_memory=True,
    )
    results.append(r)
    del fhn
    cleanup()
    logger.metric("FHNDynamics", r["mean_ms"], unit="ms")

    # 5. FHN Attention Full
    logger.info("Profiling FHNAttention...")
    from neuromanifold_gpt.model.attention.fhn import FHNAttention

    fhn_attn = FHNAttention(
        embed_dim, n_heads, n_fhn_steps=2, use_imex=True, use_partitioning=True
    )
    r = profile_component(
        "FHNAttention",
        fhn_attn,
        lambda: (
            torch.randn(BATCH_SIZE, SEQ_LEN, embed_dim, device=DEVICE),
            torch.randn(BATCH_SIZE, SEQ_LEN, n_eigenvectors, device=DEVICE),
        ),
        n_warmup=2,
        n_iters=5,
        device=DEVICE,
        track_memory=True,
    )
    results.append(r)
    del fhn_attn
    cleanup()
    logger.metric("FHNAttention", r["mean_ms"], unit="ms")

    # 6. WaveKAN FFN
    logger.info("Profiling WaveKAN FFN...")
    from neuromanifold_gpt.model.kan.wave import WaveKANFFN

    wavekan = WaveKANFFN(
        embed_dim, mlp_hidden, wavelet_type="dog", use_fast_wavekan=True
    )
    r = profile_component(
        "WaveKAN_FFN",
        wavekan,
        lambda: (torch.randn(BATCH_SIZE, SEQ_LEN, embed_dim, device=DEVICE),),
        n_warmup=2,
        n_iters=5,
        device=DEVICE,
        track_memory=True,
    )
    results.append(r)
    del wavekan
    cleanup()
    logger.metric("WaveKAN_FFN", r["mean_ms"], unit="ms")

    # 7. SwiGLU FFN
    logger.info("Profiling SwiGLU FFN...")
    from neuromanifold_gpt.model.block import SwiGLU

    swiglu = SwiGLU(embed_dim, int(mlp_hidden * 2 / 3))
    r = profile_component(
        "SwiGLU_FFN",
        swiglu,
        lambda: (torch.randn(BATCH_SIZE, SEQ_LEN, embed_dim, device=DEVICE),),
        n_warmup=2,
        n_iters=5,
        device=DEVICE,
        track_memory=True,
    )
    results.append(r)
    del swiglu
    cleanup()
    logger.metric("SwiGLU_FFN", r["mean_ms"], unit="ms")

    # 8. ChebyKAN FFN
    logger.info("Profiling ChebyKAN FFN...")
    from neuromanifold_gpt.model.kan.cheby import ChebyKANFFN

    chebykan = ChebyKANFFN(embed_dim, mlp_hidden, degree=4)
    r = profile_component(
        "ChebyKAN_FFN",
        chebykan,
        lambda: (torch.randn(BATCH_SIZE, SEQ_LEN, embed_dim, device=DEVICE),),
        n_warmup=2,
        n_iters=5,
        device=DEVICE,
        track_memory=True,
    )
    results.append(r)
    del chebykan
    cleanup()
    logger.metric("ChebyKAN_FFN", r["mean_ms"], unit="ms")

    # 9. Full Block
    logger.info("Profiling NeuroManifoldBlock...")
    from neuromanifold_gpt.config import NeuroManifoldConfig

    temp_config = NeuroManifoldConfig(
        n_embd=embed_dim,
        n_heads=n_heads,
        sdr_size=sdr_size,
        manifold_dim=manifold_dim,
        n_eigenvectors=n_eigenvectors,
        use_kan=True,
        kan_type="wave",
        kan_wavelet="dog",
        use_fast_wavekan=True,
    )
    block_cfg = NeuroManifoldBlockConfig.from_model_config(temp_config, layer_idx=0)
    block = NeuroManifoldBlock(config=block_cfg)
    r = profile_component(
        "NeuroManifoldBlock",
        block,
        lambda: (torch.randn(BATCH_SIZE, SEQ_LEN, sdr_size, device=DEVICE),),
        n_warmup=2,
        n_iters=5,
        device=DEVICE,
        track_memory=True,
    )
    results.append(r)
    del block
    cleanup()
    logger.metric("NeuroManifoldBlock", r["mean_ms"], unit="ms")

    # 10. SemanticFolding Encoder
    logger.info("Profiling SemanticFoldingEncoder...")
    from neuromanifold_gpt.model.semantic_folding import SemanticFoldingEncoder

    encoder = SemanticFoldingEncoder(vocab_size=65, sdr_size=sdr_size, n_active=40)
    r = profile_component(
        "SemanticFoldingEncoder",
        encoder,
        lambda: (torch.randint(0, 65, (BATCH_SIZE, SEQ_LEN), device=DEVICE),),
        n_warmup=2,
        n_iters=5,
        device=DEVICE,
        track_memory=True,
    )
    results.append(r)
    del encoder
    cleanup()
    logger.metric("SemanticFoldingEncoder", r["mean_ms"], unit="ms")

    # Results
    table = Table(title="Component Profiling")
    table.add_column("Component", style="cyan")
    table.add_column("Time (ms)", justify="right")
    table.add_column("Memory (MB)", justify="right")
    table.add_column("Time @ B=64", justify="right", style="yellow")

    for r in results:
        # Estimate for B=64, T=256
        scale = (64 / BATCH_SIZE) * (256 / SEQ_LEN)
        est = r["mean_ms"] * scale
        table.add_row(
            r["name"], f"{r['mean_ms']:.2f}", f"{r['mem_mb']:.1f}", f"{est:.2f}"
        )

    logger.table(table)

    # Analysis
    logger.section("Analysis")

    # Find bottleneck
    bottleneck = max(results, key=lambda x: x["mean_ms"])
    logger.info(f"#1 Time Bottleneck: {bottleneck['name']}")
    logger.info(f"   {bottleneck['mean_ms']:.2f} ms")

    mem_bottleneck = max(results, key=lambda x: x["mem_mb"])
    logger.info(f"#1 Memory Bottleneck: {mem_bottleneck['name']}")
    logger.info(f"   {mem_bottleneck['mem_mb']:.1f} MB")

    # Estimate full model
    block_time = next(
        r["mean_ms"] for r in results if r["name"] == "NeuroManifoldBlock"
    )
    encoder_time = next(
        r["mean_ms"] for r in results if r["name"] == "SemanticFoldingEncoder"
    )

    # Scale to B=64, T=256
    scale = (64 / BATCH_SIZE) * (256 / SEQ_LEN)
    est_block = block_time * scale
    est_encoder = encoder_time * scale
    est_full_fwd = est_encoder + est_block * 6  # 6 layers
    est_full_fwd_bwd = est_full_fwd * 3  # ~3x for backward
    est_1000_iters = est_full_fwd_bwd  # ms -> seconds for 1000 iters

    logger.info("Estimated performance @ B=64, T=256:")
    logger.info(f"   Per block: {est_block:.1f} ms")
    logger.info(f"   Full fwd:  {est_full_fwd:.1f} ms")
    logger.info(f"   Full fwd+bwd: {est_full_fwd_bwd:.1f} ms")
    logger.info(f"   1000 iters: {est_1000_iters:.1f} seconds")

    if est_1000_iters > 120:
        speedup = est_1000_iters / 120
        logger.warning(f"   Need {speedup:.2f}x speedup for <120s target")
    else:
        logger.info("   Under 120s target!")


if __name__ == "__main__":
    main()
