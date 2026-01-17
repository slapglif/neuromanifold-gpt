#!/usr/bin/env python3
"""
Benchmark FHN attention memory usage: chunked vs non-chunked vs flash-only.

Measures:
- Peak memory allocation during forward pass
- Peak memory allocation during backward pass
- Total peak memory usage
- Memory reduction from chunking for long sequences

Tests three configurations:
1. Flash-only: n_fhn_steps=0 (baseline, uses Flash Attention)
2. Non-chunked FHN: Short sequences (T <= chunk_size) with FHN modulation
3. Chunked FHN: Long sequences (T > chunk_size) with memory-efficient chunking

Tests at different sequence lengths: 512, 1024, 2048, 4096, 8192
"""

import argparse
from contextlib import nullcontext

import torch
import torch.nn as nn

from neuromanifold_gpt.model.attention.fhn import FHNAttention


def benchmark_fhn_memory(
    attn: nn.Module,
    batch_size: int,
    seq_len: int,
    embed_dim: int,
    device: torch.device,
    ctx,
    n_iters: int = 10,
):
    """Benchmark FHN attention memory usage.

    Args:
        attn: FHN attention module to benchmark
        batch_size: Batch size
        seq_len: Sequence length
        embed_dim: Embedding dimension
        device: Device to run on
        ctx: Autocast context
        n_iters: Number of iterations to average over

    Returns:
        Tuple of (forward_memory_mb, backward_memory_mb) averaged over n_iters
    """
    attn.to(device)
    attn.train()

    # Generate random input
    x = torch.randn(batch_size, seq_len, embed_dim, device=device, requires_grad=True)
    # Spectral basis (required by FHNAttention)
    spectral_basis = torch.randn(batch_size, seq_len, 32, device=device)

    forward_memory_total = 0.0
    backward_memory_total = 0.0

    for _ in range(n_iters):
        # Reset memory stats before forward pass
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Forward pass
        with ctx:
            out, info = attn(x, spectral_basis)
            # Sum output to get scalar loss for backward
            loss = out.sum()

        if device.type == "cuda":
            torch.cuda.synchronize()
            forward_memory = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
            forward_memory_total += forward_memory

        # Reset memory stats before backward pass
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()

        # Backward pass
        loss.backward()

        if device.type == "cuda":
            torch.cuda.synchronize()
            backward_memory = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
            backward_memory_total += backward_memory

        # Clean up
        attn.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None
        del out, info, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Return average memory usage
    forward_memory_avg = forward_memory_total / n_iters
    backward_memory_avg = backward_memory_total / n_iters

    return forward_memory_avg, backward_memory_avg


def analyze_memory_reduction(results, device):
    """Analyze memory reduction from chunking.

    Args:
        results: List of dict with keys: seq_len, flash_total, non_chunked_total, chunked_total
        device: Device used for benchmarking
    """
    if device.type != "cuda":
        print("\n" + "=" * 80)
        print("Memory reduction analysis (not available on CPU)")
        print("=" * 80)
        return

    print("\n" + "=" * 80)
    print("Memory Reduction Analysis")
    print("=" * 80)
    print(
        f"\n{'SeqLen':<10} {'Flash (MB)':<15} {'NonChunk (MB)':<15} {'Chunked (MB)':<15} {'Reduction':<12} {'vs Flash':<12}"
    )
    print("-" * 90)

    for r in results:
        seq_len = r["seq_len"]
        flash_mem = r["flash_total"]
        non_chunked_mem = r["non_chunked_total"]
        chunked_mem = r["chunked_total"]

        # Calculate reduction percentage
        if non_chunked_mem > 0:
            reduction_pct = ((non_chunked_mem - chunked_mem) / non_chunked_mem) * 100
        else:
            reduction_pct = 0.0

        # Compare to flash baseline
        if flash_mem > 0:
            vs_flash_pct = ((chunked_mem - flash_mem) / flash_mem) * 100
        else:
            vs_flash_pct = 0.0

        print(
            f"{seq_len:<10} {flash_mem:<15.1f} {non_chunked_mem:<15.1f} {chunked_mem:<15.1f} {reduction_pct:>10.1f}% {vs_flash_pct:>10.1f}%"
        )

    print("\nNotes:")
    print("  - 'Reduction': Memory saved by chunking vs non-chunked FHN")
    print("  - 'vs Flash': Memory overhead of chunked FHN vs Flash-only baseline")
    print("  - Negative 'vs Flash' means chunked FHN uses less memory than Flash")


def benchmark_chunked_memory(quick_test: bool = False):
    """Run FHN chunked memory benchmarks.

    Args:
        quick_test: If True, run with reduced iterations for quick validation
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type != "cuda":
        print(
            "WARNING: Memory benchmarking requires CUDA. Running on CPU will not measure GPU memory."
        )

    print("=" * 80)

    # Store results for analysis
    results = []

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
        seq_lens = [512, 1024]
        n_iters = 3
    else:
        seq_lens = [512, 1024, 2048, 4096, 8192]
        n_iters = 10

    # Fixed configuration
    batch_size = 1
    embed_dim = 384
    n_heads = 8
    chunk_size = 512  # Default chunk size

    for seq_len in seq_lens:
        print(f"\n{'=' * 80}")
        print(f"Sequence Length: {seq_len}")
        print(f"{'=' * 80}")

        # 1. Flash-only (baseline, no FHN modulation)
        print("\n1. Flash-only (n_fhn_steps=0)...")
        flash_attn = FHNAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_fhn_steps=0,  # Disable FHN, use Flash Attention
            chunk_size=chunk_size,
        )
        flash_fwd_mem, flash_bwd_mem = benchmark_fhn_memory(
            flash_attn, batch_size, seq_len, embed_dim, device, ctx, n_iters
        )
        flash_total_mem = flash_fwd_mem + flash_bwd_mem
        del flash_attn
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # 2. Non-chunked FHN (force short sequence behavior)
        # Create attention with large chunk_size to disable chunking
        print("\n2. Non-chunked FHN (chunk_size > seq_len)...")
        non_chunked_attn = FHNAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_fhn_steps=2,  # Enable FHN modulation
            chunk_size=seq_len + 1,  # Force non-chunked path
        )
        non_chunked_fwd_mem, non_chunked_bwd_mem = benchmark_fhn_memory(
            non_chunked_attn, batch_size, seq_len, embed_dim, device, ctx, n_iters
        )
        non_chunked_total_mem = non_chunked_fwd_mem + non_chunked_bwd_mem
        del non_chunked_attn
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # 3. Chunked FHN (memory-efficient for long sequences)
        print("\n3. Chunked FHN (chunk_size={})...".format(chunk_size))
        chunked_attn = FHNAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_fhn_steps=2,  # Enable FHN modulation
            chunk_size=chunk_size,  # Enable chunking for T > chunk_size
        )
        chunked_fwd_mem, chunked_bwd_mem = benchmark_fhn_memory(
            chunked_attn, batch_size, seq_len, embed_dim, device, ctx, n_iters
        )
        chunked_total_mem = chunked_fwd_mem + chunked_bwd_mem
        del chunked_attn
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Store results
        results.append(
            {
                "seq_len": seq_len,
                "flash_total": flash_total_mem,
                "non_chunked_total": non_chunked_total_mem,
                "chunked_total": chunked_total_mem,
            }
        )

        # Print results for this sequence length
        if device.type == "cuda":
            print(f"\n{'Results for seq_len=' + str(seq_len):}")
            print("-" * 80)

            print("\nForward Pass Peak Memory (MB):")
            print(f"  Flash-only:    {flash_fwd_mem:>8.1f}")
            print(
                f"  Non-chunked:   {non_chunked_fwd_mem:>8.1f} ({non_chunked_fwd_mem / flash_fwd_mem:.2f}x vs Flash)"
            )
            print(
                f"  Chunked:       {chunked_fwd_mem:>8.1f} ({chunked_fwd_mem / flash_fwd_mem:.2f}x vs Flash)"
            )

            print("\nBackward Pass Peak Memory (MB):")
            print(f"  Flash-only:    {flash_bwd_mem:>8.1f}")
            print(
                f"  Non-chunked:   {non_chunked_bwd_mem:>8.1f} ({non_chunked_bwd_mem / flash_bwd_mem:.2f}x vs Flash)"
            )
            print(
                f"  Chunked:       {chunked_bwd_mem:>8.1f} ({chunked_bwd_mem / flash_bwd_mem:.2f}x vs Flash)"
            )

            print("\nTotal Peak Memory (MB):")
            print(f"  Flash-only:    {flash_total_mem:>8.1f}")
            print(
                f"  Non-chunked:   {non_chunked_total_mem:>8.1f} ({non_chunked_total_mem / flash_total_mem:.2f}x vs Flash)"
            )
            print(
                f"  Chunked:       {chunked_total_mem:>8.1f} ({chunked_total_mem / flash_total_mem:.2f}x vs Flash)"
            )

            # Calculate and show reduction
            if non_chunked_total_mem > 0:
                reduction_pct = (
                    (non_chunked_total_mem - chunked_total_mem) / non_chunked_total_mem
                ) * 100
                print(
                    f"\n  Memory Reduction: {reduction_pct:.1f}% (chunked vs non-chunked)"
                )

            # Memory per token analysis
            tokens = batch_size * seq_len
            print("\nMemory per Token (KB):")
            print(f"  Flash-only:    {(flash_total_mem * 1024) / tokens:>8.2f}")
            print(f"  Non-chunked:   {(non_chunked_total_mem * 1024) / tokens:>8.2f}")
            print(f"  Chunked:       {(chunked_total_mem * 1024) / tokens:>8.2f}")
        else:
            print("\n(Memory metrics not available on CPU)")

    # Analyze memory reduction across all sequence lengths
    analyze_memory_reduction(results, device)

    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark FHN chunked vs non-chunked memory usage"
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test with reduced iterations",
    )
    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    benchmark_chunked_memory(quick_test=args.quick_test)


if __name__ == "__main__":
    main()
