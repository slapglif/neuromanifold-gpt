#!/usr/bin/env python3
"""
Benchmark attention backends: Flash Attention, xformers, Triton, and PyTorch.

Measures:
- Forward pass time
- Backward pass time
- Total time (forward + backward)
- Peak memory usage
- Tokens/sec throughput

Tests different backends at sequence lengths: 128, 256, 512, 1024
"""

import argparse
import time
from contextlib import nullcontext
from typing import Dict, Optional

import torch
import torch.nn as nn

from neuromanifold_gpt.config.base import NeuroManifoldConfig
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT
from neuromanifold_gpt.utils.gpu_detection import (
    detect_gpu_capability,
    get_optimal_attention_backend,
)


def get_memory_allocated(device: torch.device) -> float:
    """Get current GPU memory allocated in MB."""
    if device.type == "cuda":
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


def get_peak_memory(device: torch.device) -> float:
    """Get peak GPU memory allocated in MB."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0.0


def benchmark_model(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    ctx,
    n_iters: int = 100,
    warmup: int = 10,
) -> Dict[str, float]:
    """Benchmark a model's forward and backward pass.

    Returns:
        dict: Contains forward_time, backward_time, and peak_memory metrics
    """
    model.to(device)
    model.train()

    # Clear memory stats
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Generate random input tokens
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Warmup forward pass
    for _ in range(warmup):
        with ctx:
            _ = model(x, y)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark forward pass
    start = time.perf_counter()
    for _ in range(n_iters):
        with ctx:
            logits, loss, _ = model(x, y)
    if device.type == "cuda":
        torch.cuda.synchronize()
    forward_time = (time.perf_counter() - start) / n_iters * 1000

    # Warmup backward pass
    for _ in range(warmup):
        with ctx:
            logits, loss, _ = model(x, y)
        loss.backward()
        model.zero_grad(set_to_none=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Benchmark backward pass
    start = time.perf_counter()
    for _ in range(n_iters):
        with ctx:
            logits, loss, _ = model(x, y)
        loss.backward()
        model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    backward_time = (time.perf_counter() - start) / n_iters * 1000

    # Get peak memory
    peak_mem = get_peak_memory(device)

    return {
        "forward_time": forward_time,
        "backward_time": backward_time,
        "peak_memory": peak_mem,
    }


def benchmark_backends(quick_test: bool = False, backend: Optional[str] = None):
    """Run attention backend benchmarks.

    Args:
        quick_test: If True, run with reduced iterations for quick validation
        backend: If specified, only test this backend. Otherwise test all available backends.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Display GPU information
    gpu_info = detect_gpu_capability()
    print("=" * 80)
    print("GPU Detection:")
    print("-" * 80)
    if gpu_info["available"]:
        print(f"Device: {gpu_info['name']}")
        print(
            f"Compute Capability: {gpu_info['compute_capability'][0]}.{gpu_info['compute_capability'][1]}"
        )
        print(f"CUDA Version: {gpu_info['cuda_version']}")
        print("\nBackend Support:")
        print(
            f"  Flash Attention: {'✓' if gpu_info['supports_flash_attention'] else '✗'}"
        )
        print(f"  xformers:        {'✓' if gpu_info['supports_xformers'] else '✗'}")
        print(f"  Triton:          {'✓' if gpu_info['supports_triton'] else '✗'}")
        print(f"\nOptimal Backend: {get_optimal_attention_backend()}")
    else:
        print("Device: CPU (CUDA not available)")
        print("\nBackend Support:")
        print("  Flash Attention: ✗ (requires CUDA)")
        print("  xformers:        ✗ (requires CUDA)")
        print("  Triton:          ✗ (requires CUDA)")
        print("\nOptimal Backend: manual")
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
            {"seq_len": 128, "batch_size": 1},
            {"seq_len": 256, "batch_size": 1},
        ]
        n_iters = 5
        warmup = 2
    else:
        configs = [
            {"seq_len": 128, "batch_size": 8},
            {"seq_len": 256, "batch_size": 4},
            {"seq_len": 512, "batch_size": 2},
            {"seq_len": 1024, "batch_size": 1},
        ]
        n_iters = 100
        warmup = 10

    # GPT-2 124M configuration
    base_config = {
        "n_layer": 12,
        "n_embd": 768,
        "n_heads": 12,
        "vocab_size": 50304,
        "dropout": 0.0,
        "bias": False,
    }

    # Define backends to test
    if backend:
        backends_to_test = [backend]
    else:
        # Test all available backends
        backends_to_test = []
        if gpu_info["supports_flash_attention"]:
            backends_to_test.append("flash")
        if gpu_info["supports_xformers"]:
            # Check if xformers is actually installed
            try:
                import xformers  # noqa: F401

                backends_to_test.append("xformers")
            except ImportError:
                pass
        if gpu_info["supports_triton"]:
            # Check if triton is actually installed
            try:
                import triton  # noqa: F401

                backends_to_test.append("triton")
            except ImportError:
                pass
        # Always test pytorch/manual as fallback
        backends_to_test.append("pytorch")

    if not backends_to_test:
        backends_to_test = ["pytorch"]

    print(f"\nBackends to test: {', '.join(backends_to_test)}")

    # Collect results for comparison
    all_results = {}

    for config in configs:
        seq_len = config["seq_len"]
        batch_size = config["batch_size"]
        block_size = seq_len

        print(f"\n{'=' * 80}")
        print(f"Sequence Length: {seq_len}, Batch Size: {batch_size}")
        print("=" * 80)

        config_results = {}

        for backend_name in backends_to_test:
            print(f"\n{'-' * 80}")
            print(f"Testing backend: {backend_name.upper()}")
            print("-" * 80)

            # Create configuration for this backend
            model_config = NeuroManifoldConfig(
                **base_config,
                block_size=block_size,
                attention_backend=backend_name,
                use_sdr=False,
                use_multiscale_manifold=False,
                skip_manifold_spectral=True,
                use_fhn_parallel=False,
                use_mhc=False,
                use_knot_attention=False,
                use_kaufmann_attention=False,
                use_qk_norm=False,
                use_kan=False,
                use_mtp=False,
                fast_mode=True,
                skip_context_encoder=True,
                skip_semantic_retina=True,
                skip_metric_tensor=True,
            )

            try:
                # Create model
                print(f"Creating model with {backend_name} backend...")
                model = NeuroManifoldGPT(model_config)

                # Count parameters
                params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"Parameters: {params:,}")

                # Benchmark
                print(f"Benchmarking {backend_name}...")
                results = benchmark_model(
                    model,
                    batch_size,
                    seq_len,
                    base_config["vocab_size"],
                    device,
                    ctx,
                    n_iters,
                    warmup,
                )

                # Calculate totals
                total_time = results["forward_time"] + results["backward_time"]
                tokens_per_sec = (batch_size * seq_len) / (total_time / 1000)

                # Store results
                config_results[backend_name] = {
                    "forward": results["forward_time"],
                    "backward": results["backward_time"],
                    "total": total_time,
                    "throughput": tokens_per_sec,
                    "memory": results["peak_memory"],
                }

                # Print results
                print("\nResults:")
                print(f"  Forward:    {results['forward_time']:.3f} ms")
                print(f"  Backward:   {results['backward_time']:.3f} ms")
                print(f"  Total:      {total_time:.3f} ms")
                print(f"  Throughput: {tokens_per_sec:.1f} tokens/sec")
                print(f"  Peak Mem:   {results['peak_memory']:.1f} MB")

                # Clean up
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    time.sleep(1)  # Give GPU time to free memory

            except Exception as e:
                print(f"⚠ Error testing {backend_name}: {e}")
                config_results[backend_name] = None

        # Store results for this configuration
        all_results[f"seq_{seq_len}_batch_{batch_size}"] = config_results

        # Print comparison table if multiple backends were tested
        if len([r for r in config_results.values() if r is not None]) > 1:
            print(f"\n{'=' * 80}")
            print(f"Comparison for seq_len={seq_len}, batch_size={batch_size}")
            print("=" * 80)

            # Find baseline (pytorch or first successful backend)
            baseline_name = (
                "pytorch"
                if "pytorch" in config_results and config_results["pytorch"]
                else None
            )
            if not baseline_name:
                for name, result in config_results.items():
                    if result is not None:
                        baseline_name = name
                        break

            if baseline_name:
                baseline = config_results[baseline_name]

                print(
                    f"\n{'Backend':<12} {'Forward (ms)':<14} {'Backward (ms)':<15} {'Total (ms)':<13} {'Throughput':<18} {'Memory (MB)':<12}"
                )
                print("-" * 80)

                for backend_name in backends_to_test:
                    if (
                        backend_name not in config_results
                        or config_results[backend_name] is None
                    ):
                        print(
                            f"{backend_name:<12} {'N/A':<14} {'N/A':<15} {'N/A':<13} {'N/A':<18} {'N/A':<12}"
                        )
                        continue

                    result = config_results[backend_name]
                    fwd_speedup = (
                        baseline["forward"] / result["forward"]
                        if result["forward"] > 0
                        else 0
                    )
                    bwd_speedup = (
                        baseline["backward"] / result["backward"]
                        if result["backward"] > 0
                        else 0
                    )
                    total_speedup = (
                        baseline["total"] / result["total"]
                        if result["total"] > 0
                        else 0
                    )
                    throughput_speedup = (
                        result["throughput"] / baseline["throughput"]
                        if baseline["throughput"] > 0
                        else 0
                    )
                    mem_ratio = (
                        result["memory"] / baseline["memory"]
                        if baseline["memory"] > 0
                        else 1.0
                    )

                    if backend_name == baseline_name:
                        print(
                            f"{backend_name:<12} {result['forward']:>7.2f}        {result['backward']:>7.2f}         {result['total']:>7.2f}       {result['throughput']:>8.1f}          {result['memory']:>7.1f}"
                        )
                    else:
                        print(
                            f"{backend_name:<12} {result['forward']:>7.2f} ({fwd_speedup:>4.2f}x) {result['backward']:>7.2f} ({bwd_speedup:>4.2f}x)  {result['total']:>7.2f} ({total_speedup:>4.2f}x) {result['throughput']:>8.1f} ({throughput_speedup:>4.2f}x) {result['memory']:>7.1f} ({mem_ratio:>4.2f}x)"
                        )

    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark attention backends (Flash, xformers, Triton, PyTorch)"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick test with reduced iterations"
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "flash", "xformers", "triton", "pytorch"],
        default=None,
        help="Test only a specific backend",
    )
    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # Handle 'auto' backend by resolving it
    backend = args.backend
    if backend == "auto":
        backend = get_optimal_attention_backend()
        print(f"Auto-selected backend: {backend}")

    benchmark_backends(quick_test=args.quick, backend=backend)


if __name__ == "__main__":
    main()
