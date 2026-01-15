#!/usr/bin/env python3
"""
Standalone Benchmark for Memory Retrieval: Loop vs Vectorized

Runs without complex imports - directly includes the SDREngramMemory code.
"""

import time
import torch
import torch.nn as nn

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


class SDREngramMemory(nn.Module):
    """Minimal SDREngramMemory for benchmarking."""

    def __init__(self, sdr_size, capacity, n_active, content_dim=384, threshold=0.3):
        super().__init__()
        self.sdr_size = sdr_size
        self.capacity = capacity
        self.n_active = n_active
        self.content_dim = content_dim
        self.threshold = threshold

        self.register_buffer("sdr_bank", torch.zeros(capacity, sdr_size))
        self.register_buffer("content_bank", torch.zeros(capacity, content_dim))
        self.register_buffer("valid_mask", torch.zeros(capacity, dtype=torch.bool))
        self.register_buffer("write_ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    def __len__(self):
        return int(min(self.count.item(), self.capacity))

    def store(self, sdr, content):
        sdr = sdr.detach()
        content = content.detach()

        ptr = self.write_ptr.view(1)
        sdr = sdr.to(self.sdr_bank.device).unsqueeze(0)
        content = content.to(self.content_bank.device).unsqueeze(0)

        self.sdr_bank.index_copy_(0, ptr, sdr)
        self.content_bank.index_copy_(0, ptr, content)
        self.valid_mask.index_fill_(0, ptr, torch.tensor(True, device=self.valid_mask.device))

        next_ptr = (self.write_ptr + 1) % self.capacity
        self.write_ptr.copy_(next_ptr)

        next_count = torch.clamp(self.count + 1, max=self.capacity)
        self.count.copy_(next_count)

    def retrieve(self, query_sdr, top_k=5):
        """OLD: Loop-based retrieval (single query)."""
        if self.count == 0:
            return (
                torch.zeros(0, self.content_dim, device=self.content_bank.device),
                torch.zeros(0, device=self.content_bank.device),
            )

        query_sdr = query_sdr.to(self.sdr_bank.device)
        overlap = (query_sdr.unsqueeze(0) * self.sdr_bank).sum(dim=-1)
        similarity = overlap / self.n_active

        similarity = torch.where(
            self.valid_mask,
            similarity,
            torch.tensor(-float("inf"), device=similarity.device),
        )

        k = min(top_k, len(self))
        top_sim, top_idx = torch.topk(similarity, k)

        mask = top_sim >= self.threshold
        top_sim = top_sim[mask]
        top_idx = top_idx[mask]

        return self.content_bank[top_idx], top_sim

    def retrieve_batch(self, query_sdrs, top_k=5):
        """NEW: Vectorized batch retrieval."""
        B = query_sdrs.shape[0]

        if self.count == 0 or B == 0:
            return (
                torch.zeros(B, top_k, self.content_dim, device=self.content_bank.device),
                torch.zeros(B, top_k, device=self.content_bank.device),
            )

        query_sdrs = query_sdrs.to(self.sdr_bank.device)

        # KEY OPTIMIZATION: Vectorized matmul instead of loop
        overlap = torch.matmul(query_sdrs, self.sdr_bank.T)
        similarity = overlap / self.n_active

        similarity = torch.where(
            self.valid_mask.unsqueeze(0),
            similarity,
            torch.tensor(-float("inf"), device=similarity.device),
        )

        k = min(top_k, len(self))
        top_sim, top_idx = torch.topk(similarity, k, dim=-1)

        mask = top_sim >= self.threshold
        top_sim = torch.where(mask, top_sim, torch.zeros_like(top_sim))

        contents = self.content_bank[top_idx]
        contents = torch.where(
            mask.unsqueeze(-1),
            contents,
            torch.zeros_like(contents),
        )

        if k < top_k:
            pad_size = top_k - k
            contents = torch.cat([
                contents,
                torch.zeros(B, pad_size, self.content_dim, device=contents.device)
            ], dim=1)
            top_sim = torch.cat([
                top_sim,
                torch.zeros(B, pad_size, device=top_sim.device)
            ], dim=1)

        return contents, top_sim


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


def main():
    print(f"\nMemory Retrieval Benchmark: Loop vs Vectorized")
    print(f"=" * 60)
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {MEMORY_CAPACITY} items, SDR: {SDR_SIZE} ({N_ACTIVE} active)")
    print(f"Content dim: {CONTENT_DIM}, Top-K: {TOP_K}")
    print(f"Warmup: {N_WARMUP}, Iterations: {N_ITERS}\n")

    # Create and populate memory
    print("Setting up memory...")
    memory = SDREngramMemory(
        sdr_size=SDR_SIZE,
        capacity=MEMORY_CAPACITY,
        n_active=N_ACTIVE,
        content_dim=CONTENT_DIM,
        threshold=0.2,
    )
    memory = memory.to(DEVICE)
    populate_memory(memory, MEMORY_CAPACITY)
    print(f"Populated memory with {len(memory)} items\n")

    # Benchmark across different batch sizes
    print("Running benchmarks...")
    print(f"{'Batch':>5} | {'Loop (ms)':>15} | {'Vec (ms)':>15} | {'Speedup':>10} | Status")
    print("-" * 70)

    results = []
    for batch_size in BATCH_SIZES:
        # Generate query SDRs
        query_sdrs = torch.zeros(batch_size, SDR_SIZE, device=DEVICE)
        for b in range(batch_size):
            active_indices = torch.randperm(SDR_SIZE, device=DEVICE)[:N_ACTIVE]
            query_sdrs[b, active_indices] = 1.0

        # Benchmark both approaches
        loop_mean, loop_std, _, _ = benchmark_loop_based(memory, query_sdrs)
        vec_mean, vec_std, _, _ = benchmark_vectorized(memory, query_sdrs)

        # Calculate speedup
        speedup = loop_mean / vec_mean

        # Status indicator
        if speedup >= 10:
            status = "🚀 FAST"
        elif speedup >= 5:
            status = "✓ Good"
        elif speedup >= 2:
            status = "~ OK"
        else:
            status = "⚠ Slow"

        print(f"{batch_size:5d} | {loop_mean:7.3f} ± {loop_std:5.3f} | {vec_mean:7.3f} ± {vec_std:5.3f} | {speedup:8.1f}x | {status}")

        results.append({
            "batch_size": batch_size,
            "loop_mean": loop_mean,
            "vec_mean": vec_mean,
            "speedup": speedup,
        })

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    # Find peak speedup
    max_speedup_result = max(results, key=lambda x: x["speedup"])
    print(f"\nPeak Speedup: {max_speedup_result['speedup']:.1f}x at batch size {max_speedup_result['batch_size']}")

    # Speedup trend
    print("\nSpeedup Trend:")
    for i in range(0, len(results) - 1):
        r_small = results[i]
        r_large = results[i + 1]
        increase = r_large["speedup"] - r_small["speedup"]
        print(f"  B={r_small['batch_size']:2d} -> B={r_large['batch_size']:2d}: {r_small['speedup']:.1f}x -> {r_large['speedup']:.1f}x (+{increase:.1f}x)")

    # Training impact
    print("\nImpact on Model Training (B=64):")
    train_result = next(r for r in results if r["batch_size"] == 64)
    saved_per_iter = train_result["loop_mean"] - train_result["vec_mean"]
    print(f"  Per iteration: saves {saved_per_iter:.2f} ms ({train_result['speedup']:.1f}x speedup)")
    print(f"  Per 1000 iterations: saves {saved_per_iter * 1000 / 1000:.1f} seconds")
    print(f"  Per epoch (5000 iters): saves {saved_per_iter * 5000 / 1000:.1f} seconds")

    # Success criteria
    print("\nTarget Achievement:")
    if max_speedup_result["speedup"] >= 10:
        print(f"  ✓ SUCCESS: Achieved {max_speedup_result['speedup']:.1f}x speedup (target: 10-50x)")
    else:
        print(f"  ✗ BELOW TARGET: Achieved {max_speedup_result['speedup']:.1f}x speedup (target: 10-50x)")

    # Key insight
    print("\nKey Optimization:")
    print("  OLD: for b in range(B): retrieve(query[b])  → O(B) sequential calls")
    print("  NEW: retrieve_batch(queries)  → Single matmul: queries @ bank.T")
    print("  Result: Eliminates Python loop overhead + GPU parallelism\n")


if __name__ == "__main__":
    main()
