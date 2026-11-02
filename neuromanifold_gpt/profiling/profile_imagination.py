#!/usr/bin/env python3
"""
Imagination Module Profiler - profiles ConsistencyImaginationModule.
Tests different configurations and counterfactual generation scenarios.
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


def main():
    console.print("\n[bold cyan]Imagination Module Profiler[/bold cyan]")
    console.print(f"Device: {DEVICE}, Batch: {BATCH_SIZE}, SeqLen: {SEQ_LEN}\n")

    results = []

    # Config params
    embed_dim = 384
    manifold_dim = 64

    from neuromanifold_gpt.model.imagination import ConsistencyImaginationModule

    # Test 1: Encoder only
    console.print("1. Imagination Encoder...", end=" ")
    imagination = ConsistencyImaginationModule(
        embed_dim, manifold_dim, n_imagination_steps=4
    )
    # Just run encoder
    imagination_encoder = imagination.encoder
    r = profile_component(
        "Imagination_Encoder",
        imagination_encoder,
        lambda: (torch.randn(BATCH_SIZE, SEQ_LEN, embed_dim, device=DEVICE),),
        n_warmup=2,
        n_iters=5,
        device=DEVICE,
        track_memory=True,
    )
    results.append(r)
    del imagination_encoder
    del imagination
    cleanup()
    console.print(f"{r['mean_ms']:.2f} ms, {r['mem_mb']:.1f} MB")

    # Test 2: Decoder only
    console.print("2. Imagination Decoder...", end=" ")
    imagination = ConsistencyImaginationModule(
        embed_dim, manifold_dim, n_imagination_steps=4
    )
    imagination_decoder = imagination.decoder
    r = profile_component(
        "Imagination_Decoder",
        imagination_decoder,
        lambda: (torch.randn(BATCH_SIZE, SEQ_LEN, manifold_dim, device=DEVICE),),
        n_warmup=2,
        n_iters=5,
        device=DEVICE,
        track_memory=True,
    )
    results.append(r)
    del imagination_decoder
    del imagination
    cleanup()
    console.print(f"{r['mean_ms']:.2f} ms, {r['mem_mb']:.1f} MB")

    # Test 3: Full module with n_alternatives=1
    console.print("3. Imagination (1 alternative, 4 steps)...", end=" ")
    imagination = ConsistencyImaginationModule(
        embed_dim, manifold_dim, n_imagination_steps=4
    )
    r = profile_component(
        "Imagination_1alt_4steps",
        imagination,
        lambda: (torch.randn(BATCH_SIZE, SEQ_LEN, embed_dim, device=DEVICE), 1),
        n_warmup=2,
        n_iters=5,
        device=DEVICE,
        track_memory=True,
    )
    results.append(r)
    del imagination
    cleanup()
    console.print(f"{r['mean_ms']:.2f} ms, {r['mem_mb']:.1f} MB")

    # Test 4: Full module with n_alternatives=4
    console.print("4. Imagination (4 alternatives, 4 steps)...", end=" ")
    imagination = ConsistencyImaginationModule(
        embed_dim, manifold_dim, n_imagination_steps=4
    )
    r = profile_component(
        "Imagination_4alt_4steps",
        imagination,
        lambda: (torch.randn(BATCH_SIZE, SEQ_LEN, embed_dim, device=DEVICE), 4),
        n_warmup=2,
        n_iters=5,
        device=DEVICE,
        track_memory=True,
    )
    results.append(r)
    del imagination
    cleanup()
    console.print(f"{r['mean_ms']:.2f} ms, {r['mem_mb']:.1f} MB")

    # Test 5: Full module with n_alternatives=8
    console.print("5. Imagination (8 alternatives, 4 steps)...", end=" ")
    imagination = ConsistencyImaginationModule(
        embed_dim, manifold_dim, n_imagination_steps=4
    )
    r = profile_component(
        "Imagination_8alt_4steps",
        imagination,
        lambda: (torch.randn(BATCH_SIZE, SEQ_LEN, embed_dim, device=DEVICE), 8),
        n_warmup=2,
        n_iters=5,
        device=DEVICE,
        track_memory=True,
    )
    results.append(r)
    del imagination
    cleanup()
    console.print(f"{r['mean_ms']:.2f} ms, {r['mem_mb']:.1f} MB")

    # Test 6: Different n_imagination_steps (2 steps)
    console.print("6. Imagination (4 alternatives, 2 steps)...", end=" ")
    imagination = ConsistencyImaginationModule(
        embed_dim, manifold_dim, n_imagination_steps=2
    )
    r = profile_component(
        "Imagination_4alt_2steps",
        imagination,
        lambda: (torch.randn(BATCH_SIZE, SEQ_LEN, embed_dim, device=DEVICE), 4),
        n_warmup=2,
        n_iters=5,
        device=DEVICE,
        track_memory=True,
    )
    results.append(r)
    del imagination
    cleanup()
    console.print(f"{r['mean_ms']:.2f} ms, {r['mem_mb']:.1f} MB")

    # Test 7: sample_single method
    console.print("7. Imagination sample_single...", end=" ")
    imagination = ConsistencyImaginationModule(
        embed_dim, manifold_dim, n_imagination_steps=4
    )
    imagination = imagination.to(DEVICE)
    imagination.eval()

    # Profile sample_single - needs manual profiling since it's a different method
    import time

    if DEVICE == "cuda":
        import torch.cuda

        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()

    times = []
    with torch.no_grad():
        for _ in range(5):
            x = torch.randn(BATCH_SIZE, SEQ_LEN, embed_dim, device=DEVICE)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = imagination.sample_single(x, temperature=1.0)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

    if DEVICE == "cuda":
        mem_peak = torch.cuda.max_memory_allocated()
        mem_used = mem_peak - mem_before
    else:
        mem_used = 0

    mean_ms = sum(times) / len(times)
    r = {
        "name": "Imagination_sample_single",
        "mean_ms": mean_ms,
        "mem_mb": mem_used / 1e6,
    }
    results.append(r)
    console.print(f"{r['mean_ms']:.2f} ms, {r['mem_mb']:.1f} MB")

    del imagination
    cleanup()

    # Test 8: Larger manifold_dim
    console.print("8. Imagination (manifold_dim=128)...", end=" ")
    imagination = ConsistencyImaginationModule(
        embed_dim, manifold_dim=128, n_imagination_steps=4
    )
    r = profile_component(
        "Imagination_manifold128",
        imagination,
        lambda: (torch.randn(BATCH_SIZE, SEQ_LEN, embed_dim, device=DEVICE), 4),
        n_warmup=2,
        n_iters=5,
        device=DEVICE,
        track_memory=True,
    )
    results.append(r)
    del imagination
    cleanup()
    console.print(f"{r['mean_ms']:.2f} ms, {r['mem_mb']:.1f} MB")

    # Results table
    console.print("\n")
    table = Table(title="Imagination Module Profiling")
    table.add_column("Configuration", style="cyan")
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

    console.print(table)

    # Analysis
    console.print("\n[bold yellow]===== ANALYSIS =====[/bold yellow]")

    # Find bottleneck
    bottleneck = max(results, key=lambda x: x["mean_ms"])
    console.print(f"\n[bold red]Slowest Configuration: {bottleneck['name']}[/bold red]")
    console.print(f"   {bottleneck['mean_ms']:.2f} ms")

    mem_bottleneck = max(results, key=lambda x: x["mem_mb"])
    console.print(f"\n[bold red]Highest Memory: {mem_bottleneck['name']}[/bold red]")
    console.print(f"   {mem_bottleneck['mem_mb']:.1f} MB")

    # Compare alternatives scaling
    alt1 = next((r for r in results if r["name"] == "Imagination_1alt_4steps"), None)
    alt4 = next((r for r in results if r["name"] == "Imagination_4alt_4steps"), None)
    alt8 = next((r for r in results if r["name"] == "Imagination_8alt_4steps"), None)

    if alt1 and alt4 and alt8:
        console.print("\n[bold]Alternatives Scaling:[/bold]")
        console.print(f"   1 alt:  {alt1['mean_ms']:.2f} ms")
        console.print(
            f"   4 alts: {alt4['mean_ms']:.2f} ms ({alt4['mean_ms']/alt1['mean_ms']:.2f}x)"
        )
        console.print(
            f"   8 alts: {alt8['mean_ms']:.2f} ms ({alt8['mean_ms']/alt1['mean_ms']:.2f}x)"
        )

        if alt4["mean_ms"] / alt1["mean_ms"] > 4.5:
            console.print("   [yellow]⚠ Slightly worse than linear scaling[/yellow]")
        else:
            console.print("   [green]✓ Good scaling efficiency[/green]")

    # Compare steps scaling
    steps2 = next((r for r in results if r["name"] == "Imagination_4alt_2steps"), None)
    steps4 = next((r for r in results if r["name"] == "Imagination_4alt_4steps"), None)

    if steps2 and steps4:
        console.print("\n[bold]Imagination Steps Scaling:[/bold]")
        console.print(f"   2 steps: {steps2['mean_ms']:.2f} ms")
        console.print(
            f"   4 steps: {steps4['mean_ms']:.2f} ms ({steps4['mean_ms']/steps2['mean_ms']:.2f}x)"
        )

    # Estimate for inference
    if alt4:
        scale = (64 / BATCH_SIZE) * (256 / SEQ_LEN)
        est_inference = alt4["mean_ms"] * scale
        console.print("\n[bold]Estimated @ B=64, T=256:[/bold]")
        console.print(f"   4 alternatives: {est_inference:.1f} ms")

        if est_inference < 100:
            console.print(
                "   [green]✓ Fast enough for inference-time reasoning[/green]"
            )
        elif est_inference < 500:
            console.print("   [yellow]⚠ May impact inference speed[/yellow]")
        else:
            console.print("   [red]⚠ Too slow for real-time inference[/red]")

    console.print(
        "\n[bold green]✓ Imagination Module Profiling Complete[/bold green]\n"
    )


if __name__ == "__main__":
    main()
