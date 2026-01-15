#!/usr/bin/env python3
"""
Benchmark Memory Retrieval: Loop vs Vectorized

Compares the old Python loop-based memory retrieval against the new
vectorized batch retrieval to demonstrate 10-50x speedup.
"""

import os
os.environ["NEUROMANIFOLD_TESTING"] = "1"

import time
import torch
from rich.console import Console
from rich.table import Table

console = Console()

# Benchmark parameters
MEMORY_CAPACITY = 1000
SDR_SIZE = 2048
N_ACTIVE = 64
CONTENT_DIM = 384
TOP_K = 5
N_WARMUP = 5
N_ITERS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]


def populate_memory(memory, n_items):
    """Fill memory with random SDR-content pairs."""
    for _ in range(n_items):
        sdr = torch.zeros(SDR_SIZE, device=DEVICE)
        active_indices = torch.randperm(SDR_SIZE, device=DEVICE)[:N_ACTIVE]
        sdr[active_indices] = 1.0
        content = torch.randn(CONTENT_DIM, device=DEVICE)
        memory.store(sdr, content)


def benchmark_loop_based(memory, query_sdrs, top_k=TOP_K, n_warmup=N_WARMUP, n_iters=N_ITERS):
    """Benchmark the old loop-based approach (calling retrieve() B times)."""
    B = query_sdrs.shape[0]

    # Warmup
    for _ in range(n_warmup):
        for b in range(B):
            _ = memory.retrieve(query_sdrs[b], top_k=top_k)
        if DEVICE == "cuda":
            torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(n_iters):
        if DEVICE == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for b in range(B):
            _ = memory.retrieve(query_sdrs[b], top_k=top_k)

        if DEVICE == "cuda":
            torch.cuda.synchronize()

        times.append((time.perf_counter() - start) * 1000)

    mean_ms = sum(times) / len(times)
    std_ms = (sum((t - mean_ms)**2 for t in times) / len(times)) ** 0.5

    return mean_ms, std_ms, min(times), max(times)


def benchmark_vectorized(memory, query_sdrs, top_k=TOP_K, n_warmup=N_WARMUP, n_iters=N_ITERS):
    """Benchmark the new vectorized approach (calling retrieve_batch() once)."""
    # Warmup
    for _ in range(n_warmup):
        _ = memory.retrieve_batch(query_sdrs, top_k=top_k)
        if DEVICE == "cuda":
            torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(n_iters):
        if DEVICE == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        _ = memory.retrieve_batch(query_sdrs, top_k=top_k)

        if DEVICE == "cuda":
            torch.cuda.synchronize()

        times.append((time.perf_counter() - start) * 1000)

    mean_ms = sum(times) / len(times)
    std_ms = (sum((t - mean_ms)**2 for t in times) / len(times)) ** 0.5

    return mean_ms, std_ms, min(times), max(times)


def verify_correctness(memory, query_sdrs, top_k=TOP_K):
    """Verify that both approaches produce equivalent results."""
    B = query_sdrs.shape[0]

    # Loop-based results
    loop_contents = []
    loop_sims = []
    for b in range(B):
        contents, sims = memory.retrieve(query_sdrs[b], top_k=top_k)
        loop_contents.append(contents)
        loop_sims.append(sims)

    # Vectorized results
    vec_contents, vec_sims = memory.retrieve_batch(query_sdrs, top_k=top_k)

    # Compare
    all_correct = True
    for b in range(B):
        # Get number of actual results (non-zero similarities)
        n_loop = loop_sims[b].shape[0]
        n_vec = (vec_sims[b] > 0).sum().item()

        if n_loop != n_vec:
            console.print(f"[red]Mismatch in batch {b}: loop returned {n_loop} results, vectorized returned {n_vec}[/red]")
            all_correct = False
            continue

        if n_loop > 0:
            # Compare similarities (should be identical)
            if not torch.allclose(loop_sims[b], vec_sims[b, :n_loop], rtol=1e-4):
                console.print(f"[red]Similarity mismatch in batch {b}[/red]")
                all_correct = False

            # Compare contents (should be identical)
            if not torch.allclose(loop_contents[b], vec_contents[b, :n_loop], rtol=1e-4):
                console.print(f"[red]Content mismatch in batch {b}[/red]")
                all_correct = False

    return all_correct


def main():
    console.print(f"\n[bold cyan]Memory Retrieval Benchmark: Loop vs Vectorized[/bold cyan]")
    console.print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        console.print(f"GPU: {torch.cuda.get_device_name()}")
    console.print(f"Memory: {MEMORY_CAPACITY} items, SDR: {SDR_SIZE} ({N_ACTIVE} active)")
    console.print(f"Content dim: {CONTENT_DIM}, Top-K: {TOP_K}")
    console.print(f"Warmup: {N_WARMUP}, Iterations: {N_ITERS}\n")

    # Import memory module
    from neuromanifold_gpt.model.memory.engram import SDREngramMemory

    # Create and populate memory
    console.print("[bold]Setting up memory...[/bold]")
    memory = SDREngramMemory(
        sdr_size=SDR_SIZE,
        capacity=MEMORY_CAPACITY,
        n_active=N_ACTIVE,
        content_dim=CONTENT_DIM,
        threshold=0.2,
    )
    memory = memory.to(DEVICE)
    populate_memory(memory, MEMORY_CAPACITY)
    console.print(f"Populated memory with {len(memory)} items\n")

    # Verify correctness first
    console.print("[bold]Verifying correctness...[/bold]")
    test_batch_size = 8
    test_queries = torch.zeros(test_batch_size, SDR_SIZE, device=DEVICE)
    for b in range(test_batch_size):
        active_indices = torch.randperm(SDR_SIZE, device=DEVICE)[:N_ACTIVE]
        test_queries[b, active_indices] = 1.0

    is_correct = verify_correctness(memory, test_queries, top_k=TOP_K)
    if is_correct:
        console.print("[green]✓ Correctness verified - both methods produce identical results[/green]\n")
    else:
        console.print("[red]✗ Correctness check failed - results differ between methods[/red]\n")
        return

    # Benchmark across different batch sizes
    console.print("[bold]Running benchmarks...[/bold]\n")
    results = []

    for batch_size in BATCH_SIZES:
        console.print(f"[cyan]Batch size {batch_size}...[/cyan]", end=" ")

        # Generate query SDRs
        query_sdrs = torch.zeros(batch_size, SDR_SIZE, device=DEVICE)
        for b in range(batch_size):
            active_indices = torch.randperm(SDR_SIZE, device=DEVICE)[:N_ACTIVE]
            query_sdrs[b, active_indices] = 1.0

        # Benchmark loop-based
        loop_mean, loop_std, loop_min, loop_max = benchmark_loop_based(memory, query_sdrs)

        # Benchmark vectorized
        vec_mean, vec_std, vec_min, vec_max = benchmark_vectorized(memory, query_sdrs)

        # Calculate speedup
        speedup = loop_mean / vec_mean

        results.append({
            "batch_size": batch_size,
            "loop_mean": loop_mean,
            "loop_std": loop_std,
            "vec_mean": vec_mean,
            "vec_std": vec_std,
            "speedup": speedup,
        })

        console.print(f"Speedup: [bold green]{speedup:.1f}x[/bold green]")

    # Results table
    console.print("\n")
    table = Table(title="Memory Retrieval Benchmark Results")
    table.add_column("Batch Size", justify="right", style="cyan")
    table.add_column("Loop (ms)", justify="right")
    table.add_column("Vectorized (ms)", justify="right")
    table.add_column("Speedup", justify="right", style="bold green")
    table.add_column("Status", justify="center")

    for r in results:
        # Color code speedup
        if r["speedup"] >= 10:
            status = "🚀"
        elif r["speedup"] >= 5:
            status = "✓"
        elif r["speedup"] >= 2:
            status = "~"
        else:
            status = "⚠"

        table.add_row(
            str(r["batch_size"]),
            f"{r['loop_mean']:.3f} ± {r['loop_std']:.3f}",
            f"{r['vec_mean']:.3f} ± {r['vec_std']:.3f}",
            f"{r['speedup']:.1f}x",
            status,
        )

    console.print(table)

    # Analysis
    console.print("\n[bold yellow]===== ANALYSIS =====[/bold yellow]")

    # Find peak speedup
    max_speedup_result = max(results, key=lambda x: x["speedup"])
    console.print(f"\n[bold]Peak Speedup:[/bold] {max_speedup_result['speedup']:.1f}x at batch size {max_speedup_result['batch_size']}")

    # Speedup trend
    console.print("\n[bold]Speedup Trend:[/bold]")
    for i in range(0, len(results) - 1):
        r_small = results[i]
        r_large = results[i + 1]
        increase = r_large["speedup"] - r_small["speedup"]
        console.print(f"   B={r_small['batch_size']:2d} -> B={r_large['batch_size']:2d}: {r_small['speedup']:.1f}x -> {r_large['speedup']:.1f}x (+{increase:.1f}x)")

    # Performance impact on model forward pass
    console.print("\n[bold]Impact on Model Training:[/bold]")
    # Assume memory retrieval happens once per forward pass
    # Typical training batch size
    training_batch = 64
    train_result = next(r for r in results if r["batch_size"] == training_batch)
    saved_per_iter = train_result["loop_mean"] - train_result["vec_mean"]
    console.print(f"   Per iteration @ B={training_batch}: saves {saved_per_iter:.2f} ms ({train_result['speedup']:.1f}x speedup)")
    console.print(f"   Per 1000 iterations: saves {saved_per_iter * 1000 / 1000:.1f} seconds")
    console.print(f"   Per epoch (5000 iters): saves {saved_per_iter * 5000 / 1000:.1f} seconds")

    # Success criteria
    console.print("\n[bold]Target Achievement:[/bold]")
    if max_speedup_result["speedup"] >= 10:
        console.print(f"   [bold green]✓ SUCCESS: Achieved {max_speedup_result['speedup']:.1f}x speedup (target: 10-50x)[/bold green]")
    else:
        console.print(f"   [bold red]✗ BELOW TARGET: Achieved {max_speedup_result['speedup']:.1f}x speedup (target: 10-50x)[/bold red]")

    # Scalability
    console.print("\n[bold]Scalability Analysis:[/bold]")
    console.print("   Loop approach: O(B × M) where B=batch size, M=memory size")
    console.print("   Vectorized approach: O(B × M) but with GPU parallelism")
    console.print("   Key advantage: Single matmul replaces B individual operations")
    console.print("   Result: Overhead elimination + GPU memory bandwidth optimization")


if __name__ == "__main__":
    main()
