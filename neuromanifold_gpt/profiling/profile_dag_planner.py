#!/usr/bin/env python3
"""
DAG Planner profiler - profiles DAG-based planning components.
Focus on hierarchical planning and task decomposition performance.
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
    console.print("\n[bold cyan]DAG Planner Profiler[/bold cyan]")
    console.print(f"Device: {DEVICE}, Batch: {BATCH_SIZE}, SeqLen: {SEQ_LEN}\n")

    results = []

    # Config params
    embed_dim = 384
    manifold_dim = 64
    max_nodes = 32
    min_nodes = 3

    # 1. ForcedDAGPlanner with default settings
    console.print("1. ForcedDAGPlanner (N=32)...", end=" ")
    from neuromanifold_gpt.model.planning.dag_planner import ForcedDAGPlanner

    planner = ForcedDAGPlanner(
        embed_dim, manifold_dim, max_nodes=max_nodes, min_nodes=min_nodes
    )
    r = profile_component(
        "DAGPlanner_N32",
        planner,
        lambda: (torch.randn(BATCH_SIZE, SEQ_LEN, embed_dim, device=DEVICE),),
        n_warmup=2,
        n_iters=5,
        device=DEVICE,
        track_memory=True,
    )
    results.append(r)
    del planner
    cleanup()
    console.print(f"{r['mean_ms']:.2f} ms, {r['mem_mb']:.1f} MB")

    # 2. ForcedDAGPlanner with smaller graph (N=16)
    console.print("2. ForcedDAGPlanner (N=16)...", end=" ")
    planner_small = ForcedDAGPlanner(embed_dim, manifold_dim, max_nodes=16, min_nodes=3)
    r = profile_component(
        "DAGPlanner_N16",
        planner_small,
        lambda: (torch.randn(BATCH_SIZE, SEQ_LEN, embed_dim, device=DEVICE),),
        n_warmup=2,
        n_iters=5,
        device=DEVICE,
        track_memory=True,
    )
    results.append(r)
    del planner_small
    cleanup()
    console.print(f"{r['mean_ms']:.2f} ms, {r['mem_mb']:.1f} MB")

    # 3. ForcedDAGPlanner with larger graph (N=64)
    console.print("3. ForcedDAGPlanner (N=64)...", end=" ")
    planner_large = ForcedDAGPlanner(embed_dim, manifold_dim, max_nodes=64, min_nodes=3)
    r = profile_component(
        "DAGPlanner_N64",
        planner_large,
        lambda: (torch.randn(BATCH_SIZE, SEQ_LEN, embed_dim, device=DEVICE),),
        n_warmup=2,
        n_iters=5,
        device=DEVICE,
        track_memory=True,
    )
    results.append(r)
    del planner_large
    cleanup()
    console.print(f"{r['mean_ms']:.2f} ms, {r['mem_mb']:.1f} MB")

    # 4. DAGExecutor
    console.print("4. DAGExecutor...", end=" ")
    from neuromanifold_gpt.model.planning.dag_planner import DAGExecutor

    executor = DAGExecutor(embed_dim, manifold_dim)
    r = profile_component(
        "DAGExecutor",
        executor,
        lambda: (
            torch.randn(BATCH_SIZE, max_nodes, manifold_dim, device=DEVICE),
            torch.randn(BATCH_SIZE, max_nodes, max_nodes, device=DEVICE),
        ),
        n_warmup=2,
        n_iters=5,
        device=DEVICE,
        track_memory=True,
    )
    results.append(r)
    del executor
    cleanup()
    console.print(f"{r['mean_ms']:.2f} ms, {r['mem_mb']:.1f} MB")

    # 5. Full pipeline: Planning + Execution
    console.print("5. Full DAG Pipeline...", end=" ")
    planner_full = ForcedDAGPlanner(
        embed_dim, manifold_dim, max_nodes=max_nodes, min_nodes=min_nodes
    )
    executor_full = DAGExecutor(embed_dim, manifold_dim)

    class DAGPipeline(torch.nn.Module):
        def __init__(self, planner, executor):
            super().__init__()
            self.planner = planner
            self.executor = executor

        def forward(self, x):
            plan = self.planner(x)
            result = self.executor(plan["node_embeddings"], plan["adj_matrix"])
            return result

    pipeline = DAGPipeline(planner_full, executor_full)
    r = profile_component(
        "DAG_Pipeline",
        pipeline,
        lambda: (torch.randn(BATCH_SIZE, SEQ_LEN, embed_dim, device=DEVICE),),
        n_warmup=2,
        n_iters=5,
        device=DEVICE,
        track_memory=True,
    )
    results.append(r)
    del pipeline
    cleanup()
    console.print(f"{r['mean_ms']:.2f} ms, {r['mem_mb']:.1f} MB")

    # Results table
    console.print("\n")
    table = Table(title="DAG Planner Component Profiling")
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

    console.print(table)

    # Analysis
    console.print("\n[bold yellow]===== ANALYSIS =====[/bold yellow]")

    # Find bottleneck
    bottleneck = max(results, key=lambda x: x["mean_ms"])
    console.print(f"\n[bold red]#1 Time Bottleneck: {bottleneck['name']}[/bold red]")
    console.print(f"   {bottleneck['mean_ms']:.2f} ms")

    mem_bottleneck = max(results, key=lambda x: x["mem_mb"])
    console.print(
        f"\n[bold red]#1 Memory Bottleneck: {mem_bottleneck['name']}[/bold red]"
    )
    console.print(f"   {mem_bottleneck['mem_mb']:.1f} MB")

    # DAG scaling analysis
    planner_16 = next(r for r in results if r["name"] == "DAGPlanner_N16")
    planner_32 = next(r for r in results if r["name"] == "DAGPlanner_N32")
    planner_64 = next(r for r in results if r["name"] == "DAGPlanner_N64")

    console.print("\n[bold]DAG Size Scaling:[/bold]")
    console.print(f"   N=16: {planner_16['mean_ms']:.2f} ms")
    console.print(f"   N=32: {planner_32['mean_ms']:.2f} ms")
    console.print(f"   N=64: {planner_64['mean_ms']:.2f} ms")

    scale_32_16 = planner_32["mean_ms"] / planner_16["mean_ms"]
    scale_64_32 = planner_64["mean_ms"] / planner_32["mean_ms"]
    console.print(f"   Scaling factor (32/16): {scale_32_16:.2f}x")
    console.print(f"   Scaling factor (64/32): {scale_64_32:.2f}x")

    # Estimate for typical workload
    pipeline_time = next(r["mean_ms"] for r in results if r["name"] == "DAG_Pipeline")
    scale = (64 / BATCH_SIZE) * (256 / SEQ_LEN)
    est_pipeline = pipeline_time * scale

    console.print("\n[bold]Estimated performance @ B=64, T=256:[/bold]")
    console.print(f"   DAG Pipeline: {est_pipeline:.1f} ms")
    console.print(f"   Per planning step: {est_pipeline:.1f} ms")

    # If used in reasoning loop (e.g., 10 planning steps per forward pass)
    planning_steps = 10
    est_reasoning = est_pipeline * planning_steps
    console.print(f"   With {planning_steps} planning steps: {est_reasoning:.1f} ms")

    if est_reasoning > 100:
        console.print(
            f"   [bold yellow]Warning: {planning_steps} planning steps may be expensive[/bold yellow]"
        )
    else:
        console.print(
            "   [bold green]Reasonable overhead for System 2 reasoning[/bold green]"
        )


if __name__ == "__main__":
    main()
