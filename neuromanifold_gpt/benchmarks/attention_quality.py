"""Quality benchmarking for NeuroManifold attention mechanisms.

Compares standard transformer attention vs NeuroManifold attention (FHN/SDR)
on perplexity, convergence speed, and sample quality metrics.

Usage:
    from neuromanifold_gpt.benchmarks.attention_quality import benchmark_quality

    results = benchmark_quality(
        dataset='openwebtext',
        eval_iters=200,
        device='cuda'
    )
    print(f"Standard perplexity: {results['standard']['perplexity']:.2f}")
    print(f"NeuroManifold perplexity: {results['neuromanifold']['perplexity']:.2f}")

Command line:
    python -m neuromanifold_gpt.benchmarks.attention_quality --dataset openwebtext --eval_iters 200
"""
import argparse
import math
import os
import pickle
import time
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch

from neuromanifold_gpt.config.base import NeuroManifoldConfig
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT


def load_model(config_path: str, device: str = "cuda", dtype: str = "bfloat16") -> tuple[NeuroManifoldGPT, dict]:
    """Load a model from configuration file.

    Args:
        config_path: Path to config module (e.g., 'neuromanifold_gpt.config.benchmarks.standard_attention')
        device: Device to load model on ('cuda' or 'cpu')
        dtype: Data type ('bfloat16', 'float16', or 'float32')

    Returns:
        Tuple of (model, config_dict)
    """
    # Import config module
    import importlib
    config_module = importlib.import_module(config_path)

    # Get config instance
    if hasattr(config_module, 'config'):
        model_config = config_module.config
    elif hasattr(config_module, 'out'):
        # Create config from dict
        model_config = NeuroManifoldConfig(**config_module.out)
    else:
        raise ValueError(f"Config module {config_path} must have 'config' or 'out' attribute")

    # Initialize model
    model = NeuroManifoldGPT(model_config)
    model.to(device)

    # Get config dict for reference
    config_dict = {k: v for k, v in vars(model_config).items() if not k.startswith('_')}

    return model, config_dict


def get_batch(
    split: str,
    data_dir: str,
    batch_size: int,
    block_size: int,
    device: str,
    device_type: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a batch of data.

    Uses memmap to avoid memory leak (pattern from train_nanogpt.py).

    Args:
        split: 'train' or 'val'
        data_dir: Path to data directory
        batch_size: Number of sequences in batch
        block_size: Sequence length
        device: Device to load tensors on
        device_type: 'cuda' or 'cpu'

    Returns:
        Tuple of (input_ids, target_ids)
    """
    if split == "train":
        data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    else:
        data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])

    if device_type == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y


@torch.no_grad()
def estimate_loss(
    model: NeuroManifoldGPT,
    data_dir: str,
    eval_iters: int,
    batch_size: int,
    block_size: int,
    device: str,
    ctx: Any
) -> dict[str, float]:
    """Estimate loss on train and val splits.

    Pattern from train_nanogpt.py estimate_loss().

    Args:
        model: Model to evaluate
        data_dir: Path to data directory
        eval_iters: Number of iterations to average
        batch_size: Batch size
        block_size: Sequence length
        device: Device to run on
        ctx: Autocast context manager

    Returns:
        Dict with 'train' and 'val' loss
    """
    out = {}
    model.eval()
    device_type = "cuda" if "cuda" in device else "cpu"

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, data_dir, batch_size, block_size, device, device_type)
            with ctx:
                logits, loss, info = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()

    model.train()
    return out


def benchmark_quality(
    dataset: str = "openwebtext",
    eval_iters: int = 200,
    batch_size: int = 12,
    block_size: int = 1024,
    device: str = "cuda",
    dtype: str = "bfloat16",
    verbose: bool = True
) -> dict[str, dict[str, float]]:
    """Benchmark quality metrics for standard vs NeuroManifold attention.

    Measures:
    - Validation loss
    - Perplexity (exp(loss))
    - Train/val gap (overfitting indicator)

    Args:
        dataset: Dataset name (directory under data/)
        eval_iters: Number of batches to average for loss estimation
        batch_size: Batch size
        block_size: Sequence length
        device: 'cuda' or 'cpu'
        dtype: 'bfloat16', 'float16', or 'float32'
        verbose: Print progress

    Returns:
        Dict with 'standard' and 'neuromanifold' results, each containing:
        - val_loss: Validation loss
        - train_loss: Training loss
        - perplexity: exp(val_loss)
        - train_val_gap: train_loss - val_loss
    """
    if verbose:
        print("=" * 80)
        print("NeuroManifold Attention Quality Benchmark")
        print("=" * 80)
        print(f"Dataset: {dataset}")
        print(f"Eval iterations: {eval_iters}")
        print(f"Batch size: {batch_size}")
        print(f"Sequence length: {block_size}")
        print(f"Device: {device}")
        print()

    # Setup
    data_dir = os.path.join("data", dataset)
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory not found: {data_dir}")

    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16
    }[dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    results = {}

    # Benchmark standard attention
    if verbose:
        print("Benchmarking standard attention...")
        start = time.time()

    model_standard, config_standard = load_model(
        "neuromanifold_gpt.config.benchmarks.standard_attention",
        device=device,
        dtype=dtype
    )

    losses_standard = estimate_loss(
        model_standard,
        data_dir,
        eval_iters,
        batch_size,
        block_size,
        device,
        ctx
    )

    results["standard"] = {
        "val_loss": losses_standard["val"],
        "train_loss": losses_standard["train"],
        "perplexity": math.exp(losses_standard["val"]),
        "train_val_gap": losses_standard["train"] - losses_standard["val"]
    }

    if verbose:
        elapsed = time.time() - start
        print(f"  Val loss: {results['standard']['val_loss']:.4f}")
        print(f"  Perplexity: {results['standard']['perplexity']:.2f}")
        print(f"  Train/val gap: {results['standard']['train_val_gap']:.4f}")
        print(f"  Time: {elapsed:.2f}s")
        print()

    # Clean up
    del model_standard
    if device_type == "cuda":
        torch.cuda.empty_cache()

    # Benchmark NeuroManifold attention
    if verbose:
        print("Benchmarking NeuroManifold attention (FHN/SDR)...")
        start = time.time()

    model_neuromanifold, config_neuromanifold = load_model(
        "neuromanifold_gpt.config.benchmarks.neuromanifold_attention",
        device=device,
        dtype=dtype
    )

    losses_neuromanifold = estimate_loss(
        model_neuromanifold,
        data_dir,
        eval_iters,
        batch_size,
        block_size,
        device,
        ctx
    )

    results["neuromanifold"] = {
        "val_loss": losses_neuromanifold["val"],
        "train_loss": losses_neuromanifold["train"],
        "perplexity": math.exp(losses_neuromanifold["val"]),
        "train_val_gap": losses_neuromanifold["train"] - losses_neuromanifold["val"]
    }

    if verbose:
        elapsed = time.time() - start
        print(f"  Val loss: {results['neuromanifold']['val_loss']:.4f}")
        print(f"  Perplexity: {results['neuromanifold']['perplexity']:.2f}")
        print(f"  Train/val gap: {results['neuromanifold']['train_val_gap']:.4f}")
        print(f"  Time: {elapsed:.2f}s")
        print()

    # Clean up
    del model_neuromanifold
    if device_type == "cuda":
        torch.cuda.empty_cache()

    # Summary
    if verbose:
        print("=" * 80)
        print("Summary")
        print("=" * 80)
        perplexity_improvement = (
            (results["standard"]["perplexity"] - results["neuromanifold"]["perplexity"])
            / results["standard"]["perplexity"] * 100
        )
        print(f"Standard perplexity:      {results['standard']['perplexity']:>10.2f}")
        print(f"NeuroManifold perplexity: {results['neuromanifold']['perplexity']:>10.2f}")
        print(f"Improvement:              {perplexity_improvement:>9.1f}%")
        print()

    return results


def main():
    """Command-line interface for quality benchmarking."""
    parser = argparse.ArgumentParser(
        description="Benchmark NeuroManifold attention quality metrics"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="openwebtext",
        help="Dataset name (default: openwebtext)"
    )
    parser.add_argument(
        "--eval_iters",
        type=int,
        default=200,
        help="Number of evaluation iterations (default: 200)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=12,
        help="Batch size (default: 12)"
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=1024,
        help="Sequence length (default: 1024)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16",
        choices=["float32", "bfloat16", "float16"],
        help="Data type (default: bfloat16 if supported)"
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Quick test mode (reduced iterations)"
    )

    args = parser.parse_args()

    # Override for quick test
    if args.quick_test:
        args.eval_iters = 10
        args.batch_size = 4
        args.block_size = 256

    # Run benchmark
    results = benchmark_quality(
        dataset=args.dataset,
        eval_iters=args.eval_iters,
        batch_size=args.batch_size,
        block_size=args.block_size,
        device=args.device,
        dtype=args.dtype,
        verbose=True
    )

    return 0


if __name__ == "__main__":
    exit(main())
