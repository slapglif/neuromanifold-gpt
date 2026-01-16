#!/usr/bin/env python3
"""
Benchmark attention memory usage: standard vs NeuroManifold attention.

Measures:
- Peak memory allocation during forward pass
- Peak memory allocation during backward pass
- Total peak memory usage
- Memory scaling with batch size and sequence length

Tests at different batch sizes: 1, 4, 8, 16
Tests at different sequence lengths: 128, 256, 512, 1024
"""

import argparse
import torch
import torch.nn as nn
from contextlib import nullcontext

from neuromanifold_gpt.model.gpt import NeuroManifoldGPT
from neuromanifold_gpt.config.base import NeuroManifoldConfig


def benchmark_model_memory(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    ctx,
    n_iters: int = 10,
):
    """Benchmark a model's memory usage.

    Args:
        model: Model to benchmark
        batch_size: Batch size
        seq_len: Sequence length
        vocab_size: Vocabulary size
        device: Device to run on
        ctx: Autocast context
        n_iters: Number of iterations to average over

    Returns:
        Tuple of (forward_memory_mb, backward_memory_mb) averaged over n_iters
    """
    model.to(device)
    model.train()

    # Generate random input tokens
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

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
            logits, loss = model(x, y)

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
        model.zero_grad(set_to_none=True)

        if device.type == "cuda":
            torch.cuda.synchronize()
            backward_memory = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
            backward_memory_total += backward_memory

        # Clean up
        del logits, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Return average memory usage
    forward_memory_avg = forward_memory_total / n_iters
    backward_memory_avg = backward_memory_total / n_iters

    return forward_memory_avg, backward_memory_avg


def analyze_memory_scaling(results, device):
    """Analyze memory scaling with batch size and sequence length.

    Args:
        results: List of dict with keys: seq_len, batch_size, standard_total, neuromanifold_total
        device: Device used for benchmarking
    """
    if device.type != "cuda":
        print("\n" + "=" * 80)
        print("Memory scaling analysis (not available on CPU)")
        print("=" * 80)
        return

    print("\n" + "=" * 80)
    print("Memory scaling analysis")
    print("=" * 80)

    # Group by sequence length for batch size scaling
    seq_lens = sorted(set(r["seq_len"] for r in results))

    for seq_len in seq_lens:
        seq_results = [r for r in results if r["seq_len"] == seq_len]
        if len(seq_results) < 2:
            continue

        seq_results.sort(key=lambda x: x["batch_size"])

        print(f"\nBatch size scaling at sequence length {seq_len}:")
        print(f"  {'Batch':<8} {'Std (MB)':<12} {'NM (MB)':<12} {'Std/tok (KB)':<15} {'NM/tok (KB)':<15}")
        print("  " + "-" * 70)

        for r in seq_results:
            batch_size = r["batch_size"]
            seq_len_val = r["seq_len"]
            tokens = batch_size * seq_len_val

            std_mem = r["standard_total"]
            nm_mem = r["neuromanifold_total"]

            std_per_tok = (std_mem * 1024) / tokens  # Convert MB to KB per token
            nm_per_tok = (nm_mem * 1024) / tokens

            print(f"  {batch_size:<8} {std_mem:<12.1f} {nm_mem:<12.1f} {std_per_tok:<15.2f} {nm_per_tok:<15.2f}")

    # Group by batch size for sequence length scaling
    batch_sizes = sorted(set(r["batch_size"] for r in results))

    for batch_size in batch_sizes:
        batch_results = [r for r in results if r["batch_size"] == batch_size]
        if len(batch_results) < 2:
            continue

        batch_results.sort(key=lambda x: x["seq_len"])

        print(f"\nSequence length scaling at batch size {batch_size}:")
        print(f"  {'SeqLen':<8} {'Std (MB)':<12} {'NM (MB)':<12} {'Std/tok (KB)':<15} {'NM/tok (KB)':<15}")
        print("  " + "-" * 70)

        for r in batch_results:
            batch_size_val = r["batch_size"]
            seq_len_val = r["seq_len"]
            tokens = batch_size_val * seq_len_val

            std_mem = r["standard_total"]
            nm_mem = r["neuromanifold_total"]

            std_per_tok = (std_mem * 1024) / tokens
            nm_per_tok = (nm_mem * 1024) / tokens

            print(f"  {seq_len_val:<8} {std_mem:<12.1f} {nm_mem:<12.1f} {std_per_tok:<15.2f} {nm_per_tok:<15.2f}")


def benchmark_memory(quick_test: bool = False):
    """Run attention memory benchmarks.

    Args:
        quick_test: If True, run with reduced iterations for quick validation
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type != "cuda":
        print("WARNING: Memory benchmarking requires CUDA. Running on CPU will not measure GPU memory.")

    print("=" * 80)

    # Store results for scaling analysis
    results = []

    # Set up autocast context
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Test configurations
    if quick_test:
        configs = [
            {"seq_len": 128, "batch_size": 1},
            {"seq_len": 128, "batch_size": 4},
        ]
        n_iters = 3
    else:
        configs = [
            {"seq_len": 128, "batch_size": 1},
            {"seq_len": 128, "batch_size": 4},
            {"seq_len": 128, "batch_size": 8},
            {"seq_len": 128, "batch_size": 16},
            {"seq_len": 256, "batch_size": 1},
            {"seq_len": 256, "batch_size": 4},
            {"seq_len": 512, "batch_size": 1},
            {"seq_len": 512, "batch_size": 2},
            {"seq_len": 1024, "batch_size": 1},
        ]
        n_iters = 10

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
        standard_params = sum(p.numel() for p in standard_model.parameters() if p.requires_grad)
        neuromanifold_params = sum(p.numel() for p in neuromanifold_model.parameters() if p.requires_grad)

        print(f"\nParameters:")
        print(f"  Standard:      {standard_params:,}")
        print(f"  NeuroManifold: {neuromanifold_params:,} ({neuromanifold_params / standard_params:.2f}x)")

        # Benchmark standard attention
        print("\nBenchmarking standard attention memory...")
        standard_fwd_mem, standard_bwd_mem = benchmark_model_memory(
            standard_model, batch_size, seq_len, base_config["vocab_size"],
            device, ctx, n_iters
        )

        # Benchmark NeuroManifold attention
        print("Benchmarking NeuroManifold attention memory...")
        neuromanifold_fwd_mem, neuromanifold_bwd_mem = benchmark_model_memory(
            neuromanifold_model, batch_size, seq_len, base_config["vocab_size"],
            device, ctx, n_iters
        )

        # Calculate totals
        standard_total_mem = standard_fwd_mem + standard_bwd_mem
        neuromanifold_total_mem = neuromanifold_fwd_mem + neuromanifold_bwd_mem

        # Store results for scaling analysis
        results.append({
            "seq_len": seq_len,
            "batch_size": batch_size,
            "standard_total": standard_total_mem,
            "neuromanifold_total": neuromanifold_total_mem,
        })

        # Print results
        if device.type == "cuda":
            print(f"\nForward Pass Peak Memory (MB):")
            print(f"  Standard:      {standard_fwd_mem:.1f}")
            print(f"  NeuroManifold: {neuromanifold_fwd_mem:.1f} ({neuromanifold_fwd_mem / standard_fwd_mem:.2f}x)")

            print(f"\nBackward Pass Peak Memory (MB):")
            print(f"  Standard:      {standard_bwd_mem:.1f}")
            print(f"  NeuroManifold: {neuromanifold_bwd_mem:.1f} ({neuromanifold_bwd_mem / standard_bwd_mem:.2f}x)")

            print(f"\nTotal Peak Memory (MB):")
            print(f"  Standard:      {standard_total_mem:.1f}")
            print(f"  NeuroManifold: {neuromanifold_total_mem:.1f} ({neuromanifold_total_mem / standard_total_mem:.2f}x)")
        else:
            print("\n(Memory metrics not available on CPU)")

        # Clean up
        del standard_model, neuromanifold_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Analyze memory scaling
    analyze_memory_scaling(results, device)

    print("\n" + "=" * 80)
    print("Benchmark complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Benchmark attention memory usage")
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test with reduced iterations"
    )
    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    benchmark_memory(quick_test=args.quick_test)


if __name__ == "__main__":
    main()
