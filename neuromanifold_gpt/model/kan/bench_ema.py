"""
Benchmark EMA implementations: FFT-based parallel scan vs naive sequential.

Compares:
- ema_fft: O(T log T) FFT-based parallel scan
- ema_naive: O(T²) naive sequential implementation

Expected: FFT-based parallel scan is faster for T > 128
"""

import torch
import time
from neuromanifold_gpt.model.kan.ema import ema_fft


def ema_naive(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Naive sequential EMA implementation (O(T) time but sequential dependencies).

    Computes: h[t] = α·x[t] + (1-α)·h[t-1]

    Args:
        x: (B, T, D) input
        alpha: smoothing factor

    Returns:
        h: (B, T, D) EMA output

    Note:
        This implementation has O(T) operations but cannot be parallelized
        across the time dimension due to sequential dependencies.
        For large T, the lack of parallelism makes it slower than FFT.
    """
    B, T, D = x.shape
    h = torch.zeros_like(x)

    for b in range(B):
        for d in range(D):
            h[b, 0, d] = alpha * x[b, 0, d]
            for t in range(1, T):
                h[b, t, d] = alpha * x[b, t, d] + (1 - alpha) * h[b, t-1, d]

    return h


def benchmark_ema_implementations():
    """Benchmark FFT-based parallel scan vs naive sequential EMA."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    print(f"EMA Implementation Benchmark")
    print(f"Device: {device}")
    print(f"=" * 70)

    # Test configurations: varying sequence lengths
    configs = [
        # (B, T, D) - focus on varying T
        (8, 32, 64),    # Small sequence
        (8, 64, 64),    # Medium sequence
        (8, 128, 64),   # Crossover point
        (8, 256, 64),   # Large sequence
        (8, 512, 64),   # Very large sequence
        (8, 1024, 64),  # Extra large sequence
    ]

    alpha = 0.9

    for B, T, D in configs:
        print(f"\nConfig: B={B}, T={T}, D={D}")

        # Generate random input
        x = torch.randn(B, T, D, device=device, dtype=dtype)

        # Warmup runs
        warmup_iters = 5
        for _ in range(warmup_iters):
            _ = ema_fft(x, alpha)
            if device == "cpu" or T <= 256:  # Skip naive for large T on GPU (too slow)
                _ = ema_naive(x, alpha)

        if device == "cuda":
            torch.cuda.synchronize()

        # Benchmark FFT-based parallel scan
        num_iters = 20
        start = time.time()
        for _ in range(num_iters):
            h_fft = ema_fft(x, alpha)
        if device == "cuda":
            torch.cuda.synchronize()
        time_fft = (time.time() - start) * 1000 / num_iters

        # Benchmark naive sequential (skip for very large T)
        if T <= 256 or device == "cpu":
            start = time.time()
            for _ in range(num_iters):
                h_naive = ema_naive(x, alpha)
            if device == "cuda":
                torch.cuda.synchronize()
            time_naive = (time.time() - start) * 1000 / num_iters

            speedup = time_naive / time_fft

            print(f"  FFT Parallel Scan:  {time_fft:6.3f} ms")
            print(f"  Naive Sequential:   {time_naive:6.3f} ms")
            print(f"  Speedup:            {speedup:6.2f}x")

            # Verify correctness
            h_fft_cpu = h_fft.cpu()
            h_naive_cpu = h_naive.cpu()
            max_diff = (h_fft_cpu - h_naive_cpu).abs().max().item()
            print(f"  Max difference:     {max_diff:.2e}")
        else:
            print(f"  FFT Parallel Scan:  {time_fft:6.3f} ms")
            print(f"  Naive Sequential:   (skipped - too slow for T={T})")


def benchmark_scaling():
    """Benchmark complexity scaling with sequence length."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    print(f"\n{'=' * 70}")
    print(f"Complexity Scaling Analysis")
    print(f"Device: {device}")
    print(f"=" * 70)

    # Fixed B and D, vary T to show O(T log T) vs O(T) scaling
    B, D = 4, 64
    alpha = 0.9
    T_values = [64, 128, 256, 512, 1024, 2048]

    print(f"\nFixed: B={B}, D={D}")
    print(f"{'T':>6}  {'FFT (ms)':>10}  {'Naive (ms)':>10}  {'Speedup':>8}")
    print(f"{'-' * 50}")

    for T in T_values:
        x = torch.randn(B, T, D, device=device, dtype=dtype)

        # Warmup
        for _ in range(3):
            _ = ema_fft(x, alpha)
            if T <= 512:  # Only warmup naive for reasonable T
                _ = ema_naive(x, alpha)

        if device == "cuda":
            torch.cuda.synchronize()

        # Benchmark FFT
        num_iters = 10
        start = time.time()
        for _ in range(num_iters):
            _ = ema_fft(x, alpha)
        if device == "cuda":
            torch.cuda.synchronize()
        time_fft = (time.time() - start) * 1000 / num_iters

        # Benchmark naive (skip for large T)
        if T <= 512:
            start = time.time()
            for _ in range(num_iters):
                _ = ema_naive(x, alpha)
            if device == "cuda":
                torch.cuda.synchronize()
            time_naive = (time.time() - start) * 1000 / num_iters
            speedup = time_naive / time_fft

            print(f"{T:6d}  {time_fft:10.3f}  {time_naive:10.3f}  {speedup:8.2f}x")
        else:
            print(f"{T:6d}  {time_fft:10.3f}  {'(skipped)':>10}  {'N/A':>8}")


def main():
    """Run all benchmarks."""
    print("=" * 70)
    print("EMA Benchmark: Parallel Scan (FFT) vs Naive Sequential")
    print("=" * 70)

    benchmark_ema_implementations()
    benchmark_scaling()

    print(f"\n{'=' * 70}")
    print("Summary:")
    print("  - FFT-based parallel scan: O(T log T) complexity")
    print("  - Naive sequential: O(T) complexity but not parallelizable")
    print("  - Expected: FFT faster for T > 128 due to parallelization")
    print("=" * 70)


if __name__ == "__main__":
    main()
