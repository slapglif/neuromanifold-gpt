#!/usr/bin/env python3
"""
MTP (Multi-Token Prediction) Profiler - profiles MTP overhead and performance.
Compares MTP enabled vs disabled, different prediction horizons, and measures auxiliary head costs.
"""

import torch
from rich.console import Console
from rich.table import Table

from neuromanifold_gpt.utils.profiling import cleanup, profile_component

console = Console()

# Test parameters
BATCH_SIZE = 4
SEQ_LEN = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def create_model(vocab_size, embed_dim, n_layer, use_mtp, mtp_n_predict):
    """Create a minimal model for MTP profiling."""
    from neuromanifold_gpt.config import NeuroManifoldConfig
    from neuromanifold_gpt.model.gpt import NeuroManifoldGPT

    config = NeuroManifoldConfig(
        vocab_size=vocab_size,
        n_embd=embed_dim,
        n_layer=n_layer,
        n_heads=8,
        block_size=256,
        use_sdr=False,  # Use dense embeddings for speed
        skip_manifold_spectral=True,  # Skip expensive manifold ops for clean MTP test
        use_mtp=use_mtp,
        mtp_n_predict=mtp_n_predict,
        mtp_loss_weight=0.1,
    )

    return NeuroManifoldGPT(config)


def main():
    console.print("\n[bold cyan]MTP (Multi-Token Prediction) Profiler[/bold cyan]")
    console.print(f"Device: {DEVICE}, Batch: {BATCH_SIZE}, SeqLen: {SEQ_LEN}\n")

    results = []

    # Config params
    vocab_size = 1024  # Small vocab for faster testing
    embed_dim = 384
    n_layer = 2  # Small model for focused MTP testing

    # Generate input tokens
    input_tokens = torch.randint(0, vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE)

    # ========================================
    # Part 1: Compare MTP vs No-MTP
    # ========================================
    console.print("[bold yellow]Part 1: MTP vs No-MTP Comparison[/bold yellow]\n")

    # 1. Baseline: No MTP
    console.print("1. Baseline (No MTP)...", end=" ")
    model_no_mtp = create_model(
        vocab_size, embed_dim, n_layer, use_mtp=False, mtp_n_predict=1
    )
    r = profile_component(
        "No_MTP_Baseline",
        model_no_mtp,
        lambda: (input_tokens,),
        n_warmup=2,
        n_iters=5,
        device=DEVICE,
        track_memory=True,
    )
    results.append(r)
    console.print(f"{r['mean_ms']:.2f} ms, {r['mem_mb']:.1f} MB")
    baseline_time = r["mean_ms"]
    baseline_mem = r["mem_mb"]

    # Cleanup
    del model_no_mtp
    cleanup()

    # 2. MTP with n_predict=2 (predict t+1 and t+2)
    console.print("2. MTP (n_predict=2)...", end=" ")
    model_mtp_2 = create_model(
        vocab_size, embed_dim, n_layer, use_mtp=True, mtp_n_predict=2
    )
    r = profile_component(
        "MTP_n=2",
        model_mtp_2,
        lambda: (input_tokens,),
        n_warmup=2,
        n_iters=5,
        device=DEVICE,
        track_memory=True,
    )
    results.append(r)
    overhead_2 = ((r["mean_ms"] - baseline_time) / baseline_time) * 100
    console.print(
        f"{r['mean_ms']:.2f} ms, {r['mem_mb']:.1f} MB (overhead: +{overhead_2:.1f}%)"
    )

    # Cleanup
    del model_mtp_2
    cleanup()

    # 3. MTP with n_predict=4 (predict t+1, t+2, t+3, t+4)
    console.print("3. MTP (n_predict=4)...", end=" ")
    model_mtp_4 = create_model(
        vocab_size, embed_dim, n_layer, use_mtp=True, mtp_n_predict=4
    )
    r = profile_component(
        "MTP_n=4",
        model_mtp_4,
        lambda: (input_tokens,),
        n_warmup=2,
        n_iters=5,
        device=DEVICE,
        track_memory=True,
    )
    results.append(r)
    overhead_4 = ((r["mean_ms"] - baseline_time) / baseline_time) * 100
    console.print(
        f"{r['mean_ms']:.2f} ms, {r['mem_mb']:.1f} MB (overhead: +{overhead_4:.1f}%)"
    )

    # Cleanup
    del model_mtp_4
    cleanup()

    # 4. MTP with n_predict=8 (predict t+1...t+8)
    console.print("4. MTP (n_predict=8)...", end=" ")
    model_mtp_8 = create_model(
        vocab_size, embed_dim, n_layer, use_mtp=True, mtp_n_predict=8
    )
    r = profile_component(
        "MTP_n=8",
        model_mtp_8,
        lambda: (input_tokens,),
        n_warmup=2,
        n_iters=5,
        device=DEVICE,
        track_memory=True,
    )
    results.append(r)
    overhead_8 = ((r["mean_ms"] - baseline_time) / baseline_time) * 100
    console.print(
        f"{r['mean_ms']:.2f} ms, {r['mem_mb']:.1f} MB (overhead: +{overhead_8:.1f}%)"
    )

    # Cleanup
    del model_mtp_8
    cleanup()

    # ========================================
    # Part 2: Profile Individual MTP Components
    # ========================================
    console.print("\n[bold yellow]Part 2: MTP Component Breakdown[/bold yellow]\n")

    # Create a model with MTP for component testing
    model_mtp = create_model(
        vocab_size, embed_dim, n_layer, use_mtp=True, mtp_n_predict=4
    )
    model_mtp = model_mtp.to(DEVICE)
    model_mtp.eval()

    # Test input for projection heads
    x = torch.randn(BATCH_SIZE, SEQ_LEN, embed_dim, device=DEVICE)

    # Profile each auxiliary projection head
    console.print("5. MTP Projection Head (t+2)...", end=" ")
    proj_0 = model_mtp.mtp_projs[0]
    r = profile_component(
        "MTP_Proj_t+2",
        proj_0,
        lambda: (x,),
        n_warmup=2,
        n_iters=5,
        device=DEVICE,
        track_memory=True,
    )
    results.append(r)
    console.print(f"{r['mean_ms']:.2f} ms, {r['mem_mb']:.1f} MB")

    if len(model_mtp.mtp_projs) > 1:
        console.print("6. MTP Projection Head (t+3)...", end=" ")
        proj_1 = model_mtp.mtp_projs[1]
        r = profile_component(
            "MTP_Proj_t+3",
            proj_1,
            lambda: (x,),
            n_warmup=2,
            n_iters=5,
            device=DEVICE,
            track_memory=True,
        )
        results.append(r)
        console.print(f"{r['mean_ms']:.2f} ms, {r['mem_mb']:.1f} MB")

    if len(model_mtp.mtp_projs) > 2:
        console.print("7. MTP Projection Head (t+4)...", end=" ")
        proj_2 = model_mtp.mtp_projs[2]
        r = profile_component(
            "MTP_Proj_t+4",
            proj_2,
            lambda: (x,),
            n_warmup=2,
            n_iters=5,
            device=DEVICE,
            track_memory=True,
        )
        results.append(r)
        console.print(f"{r['mean_ms']:.2f} ms, {r['mem_mb']:.1f} MB")

    # Profile main lm_head
    console.print("8. Main LM Head (shared)...", end=" ")
    lm_head = model_mtp.lm_head
    r = profile_component(
        "LM_Head",
        lm_head,
        lambda: (x,),
        n_warmup=2,
        n_iters=5,
        device=DEVICE,
        track_memory=True,
    )
    results.append(r)
    console.print(f"{r['mean_ms']:.2f} ms, {r['mem_mb']:.1f} MB")

    # Cleanup
    del model_mtp
    cleanup()

    # ========================================
    # Results Table
    # ========================================
    console.print("\n")
    table = Table(title="MTP Performance Profile")
    table.add_column("Component", style="cyan")
    table.add_column("Time (ms)", justify="right")
    table.add_column("Memory (MB)", justify="right")
    table.add_column("vs Baseline", justify="right", style="yellow")

    for r in results:
        if "Baseline" in r["name"]:
            vs_baseline = "-"
        elif r["mean_ms"] < 1.0:  # Component-level profiling
            vs_baseline = "N/A"
        else:
            overhead = ((r["mean_ms"] - baseline_time) / baseline_time) * 100
            vs_baseline = f"+{overhead:.1f}%"
        table.add_row(
            r["name"], f"{r['mean_ms']:.2f}", f"{r['mem_mb']:.1f}", vs_baseline
        )

    console.print(table)

    # ========================================
    # Analysis
    # ========================================
    console.print("\n[bold yellow]===== ANALYSIS =====[/bold yellow]")

    console.print("\n[bold cyan]MTP Overhead:[/bold cyan]")
    console.print(f"  n_predict=2: +{overhead_2:.1f}% time")
    console.print(f"  n_predict=4: +{overhead_4:.1f}% time")
    console.print(f"  n_predict=8: +{overhead_8:.1f}% time")

    # Cost per auxiliary head
    cost_per_head = (overhead_4 - overhead_2) / 2  # Average cost of 2 additional heads
    console.print("\n[bold cyan]Cost per auxiliary head:[/bold cyan]")
    console.print(f"  ~{cost_per_head:.1f}% of baseline time")

    # Recommendations
    console.print("\n[bold cyan]Recommendations:[/bold cyan]")
    if overhead_4 < 10:
        console.print(
            "  [green]✓ MTP n_predict=4 has acceptable overhead (<10%)[/green]"
        )
        console.print("    Recommended for training to improve sample efficiency")
    elif overhead_4 < 20:
        console.print(
            "  [yellow]⚠ MTP n_predict=4 has moderate overhead (10-20%)[/yellow]"
        )
        console.print("    Consider using only during training, disable for inference")
    else:
        console.print("  [red]✗ MTP n_predict=4 has high overhead (>20%)[/red]")
        console.print(
            "    Consider reducing to n_predict=2 or optimizing MTP implementation"
        )

    if overhead_8 > overhead_4 * 1.8:
        console.print("  [yellow]⚠ n_predict=8 has diminishing returns[/yellow]")
        console.print("    Stick with n_predict=4 for better efficiency")

    console.print("\n[bold cyan]Memory Impact:[/bold cyan]")
    mtp_4_mem = results[2]["mem_mb"]
    mem_increase = (
        ((mtp_4_mem - baseline_mem) / baseline_mem) * 100 if baseline_mem > 0 else 0
    )
    console.print(f"  MTP n_predict=4: +{mem_increase:.1f}% memory vs baseline")

    # Estimate for full-scale model
    console.print(
        "\n[bold cyan]Estimated impact on full model (B=64, T=256):[/bold cyan]"
    )
    scale = (64 / BATCH_SIZE) * (256 / SEQ_LEN)
    est_baseline = baseline_time * scale
    est_mtp_4 = results[2]["mean_ms"] * scale
    console.print(f"  Baseline: {est_baseline:.1f} ms/iter")
    console.print(f"  MTP n=4:  {est_mtp_4:.1f} ms/iter")
    console.print(
        f"  Extra time per 1000 iters: {(est_mtp_4 - est_baseline):.1f} seconds"
    )


if __name__ == "__main__":
    main()
