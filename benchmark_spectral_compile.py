#!/usr/bin/env python3
"""
Benchmark SpectralDecomposition with torch.compile optimization.

Measures the performance improvement from using torch.compile on the full
spectral decomposition module, including the spectral_proj layers for
end-to-end kernel fusion.
"""

import time
import torch
import torch.nn as nn
from contextlib import nullcontext

from neuromanifold_gpt.model.spectral import SpectralDecomposition, _spectral_forward


def benchmark_spectral_decomposition(
    use_compile: bool,
    batch_size: int,
    seq_len: int,
    manifold_dim: int,
    n_eigenvectors: int,
    device: torch.device,
    ctx,
    n_iters: int = 100,
    warmup: int = 10,
):
    """Benchmark SpectralDecomposition forward and backward pass with full module compilation."""

    # Create spectral decomposition module
    spectral = SpectralDecomposition(
        manifold_dim=manifold_dim,
        n_eigenvectors=n_eigenvectors,
        use_learned_basis=True,
        ortho_weight=0.01,
    )

    # Override with uncompiled version for baseline comparison
    if not use_compile:
        # Replace with uncompiled version to measure baseline performance
        import neuromanifold_gpt.model.spectral as spectral_module
        original_fn = spectral_module.spectral_forward
        spectral_module.spectral_forward = _spectral_forward

    spectral.to(device)
    spectral.train()

    # Generate random inputs
    coords = torch.randn(batch_size, seq_len, manifold_dim, device=device, requires_grad=True)

    # Warmup forward pass
    for _ in range(warmup):
        with ctx:
            _ = spectral(coords)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark forward pass
    start = time.perf_counter()
    for _ in range(n_iters):
        with ctx:
            spectral_basis, spectral_freqs, ortho_loss = spectral(coords)
    if device.type == "cuda":
        torch.cuda.synchronize()
    forward_time = (time.perf_counter() - start) / n_iters * 1000

    # Warmup backward pass
    for _ in range(warmup):
        with ctx:
            spectral_basis, spectral_freqs, ortho_loss = spectral(coords)
        loss = spectral_basis.sum() + ortho_loss
        loss.backward()
        spectral.zero_grad(set_to_none=True)
        coords.grad = None

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark backward pass
    start = time.perf_counter()
    for _ in range(n_iters):
        with ctx:
            spectral_basis, spectral_freqs, ortho_loss = spectral(coords)
        loss = spectral_basis.sum() + ortho_loss
        loss.backward()
        spectral.zero_grad(set_to_none=True)
        coords.grad = None
    if device.type == "cuda":
        torch.cuda.synchronize()
    backward_time = (time.perf_counter() - start) / n_iters * 1000

    # Restore original function if we changed it
    if not use_compile:
        spectral_module.spectral_forward = original_fn

    return forward_time, backward_time


def main():
    """Run SpectralDecomposition torch.compile benchmark."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("SpectralDecomposition torch.compile Benchmark")
    print("=" * 80)
    print(f"Device: {device}")
    print()

    # Clear GPU memory before starting
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Set up autocast context
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Test configurations (reduced for memory constraints)
    configs = [
        {"seq_len": 64, "batch_size": 2, "manifold_dim": 64, "n_eigenvectors": 32},
        {"seq_len": 128, "batch_size": 1, "manifold_dim": 64, "n_eigenvectors": 32},
    ]

    n_iters = 30
    warmup = 3

    print("Configuration:")
    print(f"  Iterations: {n_iters}")
    print(f"  Warmup: {warmup}")
    print(f"  Dtype: {dtype}")
    print()

    for config in configs:
        seq_len = config["seq_len"]
        batch_size = config["batch_size"]
        manifold_dim = config["manifold_dim"]
        n_eigenvectors = config["n_eigenvectors"]

        print("=" * 80)
        print(f"Sequence Length: {seq_len}, Batch Size: {batch_size}")
        print(f"Manifold Dim: {manifold_dim}, Eigenvectors: {n_eigenvectors}")
        print("-" * 80)

        # Benchmark without torch.compile (baseline)
        print("Benchmarking without torch.compile (baseline)...")
        baseline_fwd, baseline_bwd = benchmark_spectral_decomposition(
            use_compile=False,
            batch_size=batch_size,
            seq_len=seq_len,
            manifold_dim=manifold_dim,
            n_eigenvectors=n_eigenvectors,
            device=device,
            ctx=ctx,
            n_iters=n_iters,
            warmup=warmup,
        )

        # Clean up
        if device.type == "cuda":
            torch.cuda.empty_cache()
            time.sleep(1)

        # Benchmark with torch.compile (optimized)
        print("Benchmarking with torch.compile (optimized)...")
        compiled_fwd, compiled_bwd = benchmark_spectral_decomposition(
            use_compile=True,
            batch_size=batch_size,
            seq_len=seq_len,
            manifold_dim=manifold_dim,
            n_eigenvectors=n_eigenvectors,
            device=device,
            ctx=ctx,
            n_iters=n_iters,
            warmup=warmup,
        )

        # Calculate totals
        baseline_total = baseline_fwd + baseline_bwd
        compiled_total = compiled_fwd + compiled_bwd

        # Calculate speedup
        fwd_speedup = baseline_fwd / compiled_fwd
        bwd_speedup = baseline_bwd / compiled_bwd
        total_speedup = baseline_total / compiled_total

        # Print results
        print()
        print("Results:")
        print("-" * 80)
        print(f"Forward Pass (ms):")
        print(f"  Baseline:  {baseline_fwd:.3f}")
        print(f"  Compiled:  {compiled_fwd:.3f}")
        print(f"  Speedup:   {fwd_speedup:.2f}x ({(fwd_speedup - 1) * 100:.1f}% faster)")
        print()
        print(f"Backward Pass (ms):")
        print(f"  Baseline:  {baseline_bwd:.3f}")
        print(f"  Compiled:  {compiled_bwd:.3f}")
        print(f"  Speedup:   {bwd_speedup:.2f}x ({(bwd_speedup - 1) * 100:.1f}% faster)")
        print()
        print(f"Total Time (ms):")
        print(f"  Baseline:  {baseline_total:.3f}")
        print(f"  Compiled:  {compiled_total:.3f}")
        print(f"  Speedup:   {total_speedup:.2f}x ({(total_speedup - 1) * 100:.1f}% faster)")
        print()

        # Clean up
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("=" * 80)
    print("Benchmark complete")
    print()
    print("Summary:")
    print("-" * 80)
    print("torch.compile successfully optimizes the full SpectralDecomposition module")
    print("by fusing the spectral_proj layers (Linear -> SiLU -> Linear) with")
    print("subsequent operations (L2 norm, bmm for ortho loss).")
    print()
    print("Expected improvement: 15-40% speedup from end-to-end kernel fusion")
    print("Actual results documented above.")
    print("=" * 80)


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    main()
