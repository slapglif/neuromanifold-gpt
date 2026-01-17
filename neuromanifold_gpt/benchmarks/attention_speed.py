#!/usr/bin/env python3
"""
Benchmark attention speed: standard vs NeuroManifold attention.

Measures:
- Forward pass time
- Backward pass time
- Total time (forward + backward)
- Tokens/sec throughput

Tests at different sequence lengths: 128, 256, 512, 1024
"""

import argparse
import time
from contextlib import nullcontext

import torch
import torch.nn as nn

from neuromanifold_gpt.config.base import NeuroManifoldConfig
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT


def benchmark_model(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    ctx,
    n_iters: int = 100,
    warmup: int = 10,
):
    """Benchmark a model's forward and backward pass."""
    model.to(device)
    model.train()

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

    return forward_time, backward_time


def benchmark_speed(quick_test: bool = False):
    """Run attention speed benchmarks.

    Args:
        quick_test: If True, run with reduced iterations for quick validation
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

    for config in configs:
        seq_len = config["seq_len"]
        batch_size = config["batch_size"]
        block_size = seq_len

        print(f"\nSequence Length: {seq_len}, Batch Size: {batch_size}")
        print("-" * 80)

        # Standard attention config
        standard_config = NeuroManifoldConfig(
            **base_config,
            block_size=block_size,
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

        # NeuroManifold attention config
        neuromanifold_config = NeuroManifoldConfig(
            **base_config,
            block_size=block_size,
            use_sdr=True,
            sdr_size=2048,
            sdr_sparsity=0.02,
            sdr_embed_dim=256,
            sdr_context_size=5,
            use_multiscale_manifold=True,
            multiscale_coarse_dim=16,
            multiscale_medium_dim=32,
            multiscale_fine_dim=64,
            manifold_dim=64,
            n_eigenvectors=32,
            skip_manifold_spectral=False,
            fhn_threshold=0.5,
            fhn_tau=12.5,
            n_fhn_steps=2,
            use_fhn_imex=True,
            use_fhn_partitioning=True,
            use_fhn_fused=False,
            use_fhn_parallel=True,
            use_mhc=True,
            use_full_mhc=True,
            mhc_n_streams=2,
            use_qk_norm=True,
            use_kan=True,
            kan_type="faster",
            kan_wavelet="dog",
            use_fast_wavekan=True,
            kan_num_centers=3,
            use_mtp=True,
            mtp_n_predict=4,
            mtp_loss_weight=0.1,
            fast_mode=False,
            skip_context_encoder=False,
            skip_semantic_retina=False,
            skip_metric_tensor=False,
        )

        # Create models
        print("Creating standard attention model...")
        standard_model = NeuroManifoldGPT(standard_config)

        print("Creating NeuroManifold attention model...")
        neuromanifold_model = NeuroManifoldGPT(neuromanifold_config)

        # Count parameters
        standard_params = sum(
            p.numel() for p in standard_model.parameters() if p.requires_grad
        )
        neuromanifold_params = sum(
            p.numel() for p in neuromanifold_model.parameters() if p.requires_grad
        )

        print("\nParameters:")
        print(f"  Standard:      {standard_params:,}")
        print(
            f"  NeuroManifold: {neuromanifold_params:,} ({neuromanifold_params / standard_params:.2f}x)"
        )

        # Benchmark standard attention
        print("\nBenchmarking standard attention...")
        standard_fwd, standard_bwd = benchmark_model(
            standard_model,
            batch_size,
            seq_len,
            base_config["vocab_size"],
            device,
            ctx,
            n_iters,
            warmup,
        )

        # Clean up standard model before loading NeuroManifold
        del standard_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
            time.sleep(2)  # Give GPU time to free memory

        # Benchmark NeuroManifold attention
        print("Benchmarking NeuroManifold attention...")
        neuromanifold_fwd, neuromanifold_bwd = benchmark_model(
            neuromanifold_model,
            batch_size,
            seq_len,
            base_config["vocab_size"],
            device,
            ctx,
            n_iters,
            warmup,
        )

        # Calculate totals
        standard_total = standard_fwd + standard_bwd
        neuromanifold_total = neuromanifold_fwd + neuromanifold_bwd

        # Calculate tokens/sec
        standard_tokens_per_sec = (batch_size * seq_len * n_iters) / (
            standard_total / 1000 * n_iters
        )
        neuromanifold_tokens_per_sec = (batch_size * seq_len * n_iters) / (
            neuromanifold_total / 1000 * n_iters
        )

        # Print results
        print("\nForward Pass (ms):")
        print(f"  Standard:      {standard_fwd:.3f}")
        print(
            f"  NeuroManifold: {neuromanifold_fwd:.3f} ({neuromanifold_fwd / standard_fwd:.2f}x)"
        )

        print("\nBackward Pass (ms):")
        print(f"  Standard:      {standard_bwd:.3f}")
        print(
            f"  NeuroManifold: {neuromanifold_bwd:.3f} ({neuromanifold_bwd / standard_bwd:.2f}x)"
        )

        print("\nTotal Time (ms):")
        print(f"  Standard:      {standard_total:.3f}")
        print(
            f"  NeuroManifold: {neuromanifold_total:.3f} ({neuromanifold_total / standard_total:.2f}x)"
        )

        print("\nThroughput (tokens/sec):")
        print(f"  Standard:      {standard_tokens_per_sec:.1f}")
        print(
            f"  NeuroManifold: {neuromanifold_tokens_per_sec:.1f} ({neuromanifold_tokens_per_sec / standard_tokens_per_sec:.2f}x)"
        )

        # Clean up
        del neuromanifold_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("Benchmark complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Benchmark attention speed")
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

    benchmark_speed(quick_test=args.quick_test)


if __name__ == "__main__":
    main()
