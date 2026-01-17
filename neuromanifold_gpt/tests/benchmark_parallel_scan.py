#!/usr/bin/env python3
"""
Benchmark parallel associative scan speed: sequential vs parallel implementation.

Measures:
- Forward pass time
- Backward pass time (gradient computation)
- Speedup ratio (sequential / parallel)
- Throughput (tokens/sec)
- Memory usage (with --memory flag)

Tests at different sequence lengths: 128, 256, 512, 1024, 2048
"""

import argparse
import time
from contextlib import nullcontext

import torch
import torch.nn as nn

from neuromanifold_gpt.model.ssm.selective_scan import (
    ParallelSelectiveScan,
    SelectiveScan,
)


def benchmark_scan(
    scan_module: nn.Module,
    batch_size: int,
    seq_len: int,
    embed_dim: int,
    device: torch.device,
    ctx,
    n_iters: int = 100,
    warmup: int = 10,
):
    """Benchmark a scan module's forward pass."""
    scan_module.to(device)
    scan_module.eval()  # Use eval mode for consistent timing

    # Generate random input
    x = torch.randn(batch_size, seq_len, embed_dim, device=device)

    # Warmup forward pass
    with torch.no_grad():
        for _ in range(warmup):
            with ctx:
                _ = scan_module(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark forward pass
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(n_iters):
            with ctx:
                scan_module(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        forward_time = (time.perf_counter() - start) / n_iters * 1000

    return forward_time


def benchmark_scan_backward(
    scan_module: nn.Module,
    batch_size: int,
    seq_len: int,
    embed_dim: int,
    device: torch.device,
    ctx,
    n_iters: int = 100,
    warmup: int = 10,
):
    """Benchmark a scan module's backward pass (gradient computation)."""
    scan_module.to(device)
    scan_module.train()  # Use train mode for backward pass

    # Warmup backward pass
    for _ in range(warmup):
        x = torch.randn(
            batch_size, seq_len, embed_dim, device=device, requires_grad=True
        )
        with ctx:
            output = scan_module(x)
        loss = output.mean()
        loss.backward()
        # Zero gradients
        scan_module.zero_grad()
        if x.grad is not None:
            x.grad = None

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark backward pass
    start = time.perf_counter()
    for _ in range(n_iters):
        x = torch.randn(
            batch_size, seq_len, embed_dim, device=device, requires_grad=True
        )
        with ctx:
            output = scan_module(x)
        loss = output.mean()
        loss.backward()
        # Zero gradients
        scan_module.zero_grad()
        if x.grad is not None:
            x.grad = None

    if device.type == "cuda":
        torch.cuda.synchronize()
    backward_time = (time.perf_counter() - start) / n_iters * 1000

    return backward_time


def benchmark_scan_memory(
    scan_module: nn.Module,
    batch_size: int,
    seq_len: int,
    embed_dim: int,
    device: torch.device,
    ctx,
):
    """Benchmark a scan module's memory usage.

    Returns peak memory allocated in bytes.
    """
    scan_module.to(device)
    scan_module.eval()

    # Clear memory before measurement
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Generate random input
    x = torch.randn(batch_size, seq_len, embed_dim, device=device)

    # Run forward pass and measure peak memory
    with torch.no_grad():
        with ctx:
            scan_module(x)

    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()
    else:
        # CPU memory tracking is not as straightforward, return 0 as placeholder
        peak_memory = 0

    return peak_memory


def benchmark_memory(quick_test: bool = False):
    """Run parallel scan memory benchmarks.

    Args:
        quick_test: If True, run with reduced configurations for quick validation
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type != "cuda":
        print("\n⚠ Warning: Memory profiling is only available on CUDA devices")
        print("  CPU memory tracking is not supported in this benchmark")
        return False

    print("=" * 80)

    # Clear GPU memory before starting
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Set up autocast context
    dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    device_type = "cuda"
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Test configurations
    if quick_test:
        print("\n⚠ Quick test mode - reduced sequence lengths for faster execution")
        configs = [
            {"seq_len": 128, "batch_size": 2},
            {"seq_len": 256, "batch_size": 2},
            {"seq_len": 512, "batch_size": 1},
            {"seq_len": 1024, "batch_size": 1},
        ]
    else:
        configs = [
            {"seq_len": 128, "batch_size": 8},
            {"seq_len": 256, "batch_size": 4},
            {"seq_len": 512, "batch_size": 2},
            {"seq_len": 1024, "batch_size": 1},
            {"seq_len": 2048, "batch_size": 1},
        ]

    # SSM configuration
    embed_dim = 768  # GPT-2 size
    state_dim = 16  # Standard SSM state dimension

    print("\nSSM Configuration:")
    print(f"  embed_dim: {embed_dim}")
    print(f"  state_dim: {state_dim}")
    print("=" * 80)

    # Track results for summary
    results = []

    for config in configs:
        seq_len = config["seq_len"]
        batch_size = config["batch_size"]

        print(f"\n{'=' * 80}")
        print(f"Sequence Length: {seq_len}, Batch Size: {batch_size}")
        print(f"{'=' * 80}")

        # Sequential scan (baseline)
        print("\n[1/2] Sequential Scan (baseline)...")
        sequential_scan = SelectiveScan(
            embed_dim=embed_dim,
            state_dim=state_dim,
            use_hippo_init=True,
        )

        sequential_memory = None

        try:
            sequential_memory = benchmark_scan_memory(
                sequential_scan,
                batch_size,
                seq_len,
                embed_dim,
                device,
                ctx,
            )
            sequential_memory_mb = sequential_memory / (1024**2)
            print(f"  ✓ Peak memory: {sequential_memory_mb:.2f} MB")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  ✗ OOM - skipping sequential for seq_len={seq_len}")
                sequential_memory = None
            else:
                raise

        # Parallel scan (optimized)
        print("\n[2/2] Parallel Scan (optimized)...")
        parallel_scan = ParallelSelectiveScan(
            embed_dim=embed_dim,
            state_dim=state_dim,
            use_hippo_init=True,
            gradient_checkpointing=False,
        )

        # Copy weights from sequential to parallel for fair comparison
        parallel_scan.load_state_dict(sequential_scan.state_dict(), strict=False)

        parallel_memory = None

        try:
            parallel_memory = benchmark_scan_memory(
                parallel_scan,
                batch_size,
                seq_len,
                embed_dim,
                device,
                ctx,
            )
            parallel_memory_mb = parallel_memory / (1024**2)
            print(f"  ✓ Peak memory: {parallel_memory_mb:.2f} MB")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  ✗ OOM - skipping parallel for seq_len={seq_len}")
                parallel_memory = None
            else:
                raise

        # Calculate memory ratio
        if sequential_memory is not None and parallel_memory is not None:
            memory_ratio = parallel_memory / sequential_memory

            result = {
                "seq_len": seq_len,
                "batch_size": batch_size,
                "sequential_memory": sequential_memory,
                "parallel_memory": parallel_memory,
                "memory_ratio": memory_ratio,
            }

            print(f"\n{'─' * 80}")
            print("Results:")
            print(f"  Sequential: {sequential_memory / (1024 ** 2):.2f} MB")
            print(f"  Parallel:   {parallel_memory / (1024 ** 2):.2f} MB")
            print(f"  Ratio:      {memory_ratio:.2f}x")
            print(f"{'─' * 80}")

            results.append(result)
        else:
            print(f"\n{'─' * 80}")
            print("  Results: Skipped due to OOM")
            print(f"{'─' * 80}")

        # Clean up
        del sequential_scan
        del parallel_scan
        torch.cuda.empty_cache()

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY - Memory Usage")
    print("=" * 80)
    print(
        f"{'Seq Len':<10} {'Batch':<8} {'Sequential':<15} {'Parallel':<15} {'Ratio':<10}"
    )
    print("-" * 80)

    for result in results:
        sequential_mb = result["sequential_memory"] / (1024**2)
        parallel_mb = result["parallel_memory"] / (1024**2)
        print(
            f"{result['seq_len']:<10} "
            f"{result['batch_size']:<8} "
            f"{sequential_mb:>10.2f} MB   "
            f"{parallel_mb:>10.2f} MB   "
            f"{result['memory_ratio']:>7.2f}x"
        )

    # Check acceptance criteria: <2x memory usage
    print("\n" + "=" * 80)
    print("ACCEPTANCE CRITERIA CHECK")
    print("=" * 80)

    if results:
        max_ratio = max(r["memory_ratio"] for r in results)
        avg_ratio = sum(r["memory_ratio"] for r in results) / len(results)

        print("Memory usage ratio (parallel / sequential):")
        print(f"  Maximum: {max_ratio:.2f}x")
        print(f"  Average: {avg_ratio:.2f}x")
        print("  Target:  <2.00x")

        memory_pass = max_ratio < 2.0
        if memory_pass:
            print(f"  ✓ PASS: Maximum ratio ({max_ratio:.2f}x) below 2x target")
        else:
            print(f"  ✗ FAIL: Maximum ratio ({max_ratio:.2f}x) exceeds 2x target")

        return memory_pass
    else:
        print("✗ FAIL: No memory profiling results available")
        return False


def benchmark_speed(quick_test: bool = False, backward: bool = False):
    """Run parallel scan speed benchmarks.

    Args:
        quick_test: If True, run with reduced iterations for quick validation
        backward: If True, also benchmark backward pass (gradient computation)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("=" * 80)

    # Clear GPU memory before starting
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Set up autocast context
    dtype = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    # Test configurations
    if quick_test:
        print("\n⚠ Quick test mode - reduced sequence lengths for faster execution")
        configs = [
            {"seq_len": 128, "batch_size": 2},
            {"seq_len": 256, "batch_size": 2},
            {"seq_len": 512, "batch_size": 1},
            {"seq_len": 1024, "batch_size": 1},
        ]
        n_iters = 10
        warmup = 3
    else:
        configs = [
            {"seq_len": 128, "batch_size": 8},
            {"seq_len": 256, "batch_size": 4},
            {"seq_len": 512, "batch_size": 2},
            {"seq_len": 1024, "batch_size": 1},
            {"seq_len": 2048, "batch_size": 1},
        ]
        n_iters = 100
        warmup = 10

    # SSM configuration
    embed_dim = 768  # GPT-2 size
    state_dim = 16  # Standard SSM state dimension

    print("\nSSM Configuration:")
    print(f"  embed_dim: {embed_dim}")
    print(f"  state_dim: {state_dim}")
    print(f"  iterations: {n_iters}")
    print(f"  warmup: {warmup}")
    print("=" * 80)

    # Track results for summary
    results = []

    for config in configs:
        seq_len = config["seq_len"]
        batch_size = config["batch_size"]

        print(f"\n{'=' * 80}")
        print(f"Sequence Length: {seq_len}, Batch Size: {batch_size}")
        print(f"{'=' * 80}")

        # Sequential scan (baseline)
        print("\n[1/2] Sequential Scan (baseline)...")
        sequential_scan = SelectiveScan(
            embed_dim=embed_dim,
            state_dim=state_dim,
            use_hippo_init=True,
        )

        sequential_time = None
        sequential_backward_time = None

        try:
            sequential_time = benchmark_scan(
                sequential_scan,
                batch_size,
                seq_len,
                embed_dim,
                device,
                ctx,
                n_iters,
                warmup,
            )
            print(f"  ✓ Forward pass: {sequential_time:.3f} ms")

            if backward:
                sequential_backward_time = benchmark_scan_backward(
                    sequential_scan,
                    batch_size,
                    seq_len,
                    embed_dim,
                    device,
                    ctx,
                    n_iters,
                    warmup,
                )
                print(f"  ✓ Backward pass: {sequential_backward_time:.3f} ms")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  ✗ OOM - skipping sequential for seq_len={seq_len}")
                sequential_time = None
                sequential_backward_time = None
            else:
                raise

        # Parallel scan (optimized)
        print("\n[2/2] Parallel Scan (optimized)...")
        parallel_scan = ParallelSelectiveScan(
            embed_dim=embed_dim,
            state_dim=state_dim,
            use_hippo_init=True,
            gradient_checkpointing=False,
        )

        # Copy weights from sequential to parallel for fair comparison
        parallel_scan.load_state_dict(sequential_scan.state_dict(), strict=False)

        parallel_time = None
        parallel_backward_time = None

        try:
            parallel_time = benchmark_scan(
                parallel_scan,
                batch_size,
                seq_len,
                embed_dim,
                device,
                ctx,
                n_iters,
                warmup,
            )
            print(f"  ✓ Forward pass: {parallel_time:.3f} ms")

            if backward:
                parallel_backward_time = benchmark_scan_backward(
                    parallel_scan,
                    batch_size,
                    seq_len,
                    embed_dim,
                    device,
                    ctx,
                    n_iters,
                    warmup,
                )
                print(f"  ✓ Backward pass: {parallel_backward_time:.3f} ms")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  ✗ OOM - skipping parallel for seq_len={seq_len}")
                parallel_time = None
                parallel_backward_time = None
            else:
                raise

        # Calculate speedup
        if sequential_time is not None and parallel_time is not None:
            speedup = sequential_time / parallel_time
            tokens_per_sec = (batch_size * seq_len) / (parallel_time / 1000)

            result = {
                "seq_len": seq_len,
                "batch_size": batch_size,
                "sequential_time": sequential_time,
                "parallel_time": parallel_time,
                "speedup": speedup,
                "tokens_per_sec": tokens_per_sec,
            }

            print(f"\n{'─' * 80}")
            print("Results (Forward):")
            print(f"  Sequential: {sequential_time:.3f} ms")
            print(f"  Parallel:   {parallel_time:.3f} ms")
            print(f"  Speedup:    {speedup:.2f}x")
            print(f"  Throughput: {tokens_per_sec:,.0f} tokens/sec")

            if (
                backward
                and sequential_backward_time is not None
                and parallel_backward_time is not None
            ):
                backward_speedup = sequential_backward_time / parallel_backward_time
                result["sequential_backward_time"] = sequential_backward_time
                result["parallel_backward_time"] = parallel_backward_time
                result["backward_speedup"] = backward_speedup

                print("\nResults (Backward):")
                print(f"  Sequential: {sequential_backward_time:.3f} ms")
                print(f"  Parallel:   {parallel_backward_time:.3f} ms")
                print(f"  Speedup:    {backward_speedup:.2f}x")

            print(f"{'─' * 80}")

            results.append(result)
        else:
            print(f"\n{'─' * 80}")
            print("  Results: Skipped due to OOM")
            print(f"{'─' * 80}")

        # Clean up
        del sequential_scan
        del parallel_scan
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY - Forward Pass")
    print("=" * 80)
    print(
        f"{'Seq Len':<10} {'Batch':<8} {'Sequential':<15} {'Parallel':<15} {'Speedup':<10}"
    )
    print("-" * 80)

    for result in results:
        print(
            f"{result['seq_len']:<10} "
            f"{result['batch_size']:<8} "
            f"{result['sequential_time']:>10.3f} ms   "
            f"{result['parallel_time']:>10.3f} ms   "
            f"{result['speedup']:>7.2f}x"
        )

    # Print backward summary if available
    if backward and any("backward_speedup" in r for r in results):
        print("\n" + "=" * 80)
        print("SUMMARY - Backward Pass")
        print("=" * 80)
        print(
            f"{'Seq Len':<10} {'Batch':<8} {'Sequential':<15} {'Parallel':<15} {'Speedup':<10}"
        )
        print("-" * 80)

        for result in results:
            if "backward_speedup" in result:
                print(
                    f"{result['seq_len']:<10} "
                    f"{result['batch_size']:<8} "
                    f"{result['sequential_backward_time']:>10.3f} ms   "
                    f"{result['parallel_backward_time']:>10.3f} ms   "
                    f"{result['backward_speedup']:>7.2f}x"
                )

    # Check acceptance criteria: >3x speedup for seq_len >= 1024
    print("\n" + "=" * 80)
    print("ACCEPTANCE CRITERIA CHECK")
    print("=" * 80)

    long_seq_results = [r for r in results if r["seq_len"] >= 1024]
    if long_seq_results:
        # Check forward pass
        min_speedup = min(r["speedup"] for r in long_seq_results)
        avg_speedup = sum(r["speedup"] for r in long_seq_results) / len(
            long_seq_results
        )

        print("Forward pass speedup for sequences >= 1024 tokens:")
        print(f"  Minimum: {min_speedup:.2f}x")
        print(f"  Average: {avg_speedup:.2f}x")
        print("  Target:  >3.00x")

        forward_pass = min_speedup > 3.0
        if forward_pass:
            print(f"  ✓ PASS: Minimum speedup ({min_speedup:.2f}x) exceeds 3x target")
        else:
            print(f"  ✗ FAIL: Minimum speedup ({min_speedup:.2f}x) below 3x target")

        # Check backward pass if available
        if backward:
            backward_results = [r for r in long_seq_results if "backward_speedup" in r]
            if backward_results:
                min_backward_speedup = min(
                    r["backward_speedup"] for r in backward_results
                )
                avg_backward_speedup = sum(
                    r["backward_speedup"] for r in backward_results
                ) / len(backward_results)

                print("\nBackward pass speedup for sequences >= 1024 tokens:")
                print(f"  Minimum: {min_backward_speedup:.2f}x")
                print(f"  Average: {avg_backward_speedup:.2f}x")
                print("  Target:  >3.00x")

                backward_pass = min_backward_speedup > 3.0
                if backward_pass:
                    print(
                        f"  ✓ PASS: Minimum speedup ({min_backward_speedup:.2f}x) exceeds 3x target"
                    )
                else:
                    print(
                        f"  ✗ FAIL: Minimum speedup ({min_backward_speedup:.2f}x) below 3x target"
                    )

                return forward_pass and backward_pass
            else:
                print("\n✗ FAIL: No backward pass results for sequences >= 1024 tokens")
                return False

        return forward_pass
    else:
        print("✗ FAIL: No results for sequences >= 1024 tokens")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark parallel associative scan speed",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test with reduced iterations",
    )
    parser.add_argument(
        "--backward",
        action="store_true",
        help="Also benchmark backward pass (gradient computation)",
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="Run memory profiling instead of speed benchmarks",
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("PARALLEL ASSOCIATIVE SCAN BENCHMARK")
    if args.memory:
        print("Mode: Memory Profiling")
    elif args.backward:
        print("Mode: Forward + Backward Pass")
    else:
        print("Mode: Forward Pass Only")
    print("=" * 80)

    if args.memory:
        success = benchmark_memory(quick_test=args.quick)
    else:
        success = benchmark_speed(quick_test=args.quick, backward=args.backward)

    print("\n" + "=" * 80)
    if success:
        print("✓ Benchmark completed successfully")
    else:
        print("✗ Benchmark did not meet acceptance criteria")
    print("=" * 80 + "\n")

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
