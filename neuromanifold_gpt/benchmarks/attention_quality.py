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
from typing import Any, Callable

import numpy as np
import torch
import tiktoken

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


def setup_encoding(dataset: str) -> tuple[Callable, Callable]:
    """Setup encoding/decoding functions for a dataset.

    Pattern from sample.py - tries to load meta.pkl, falls back to GPT-2 encoding.

    Args:
        dataset: Dataset name (directory under data/)

    Returns:
        Tuple of (encode_fn, decode_fn)
    """
    meta_path = os.path.join("data", dataset, "meta.pkl")
    load_meta = os.path.exists(meta_path)

    if load_meta:
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        stoi, itos = meta["stoi"], meta["itos"]
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: "".join([itos[i] for i in l])
    else:
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

    return encode, decode


def measure_sample_diversity(samples: list[str]) -> dict[str, float]:
    """Measure diversity metrics for generated samples.

    Args:
        samples: List of generated text samples

    Returns:
        Dict with diversity metrics:
        - unique_unigrams: Ratio of unique words to total words
        - unique_bigrams: Ratio of unique bigrams to total bigrams
        - unique_trigrams: Ratio of unique trigrams to total trigrams
        - avg_length: Average sample length in characters
    """
    all_words = []
    all_bigrams = []
    all_trigrams = []

    for sample in samples:
        words = sample.split()
        all_words.extend(words)
        if len(words) >= 2:
            all_bigrams.extend([tuple(words[i:i+2]) for i in range(len(words)-1)])
        if len(words) >= 3:
            all_trigrams.extend([tuple(words[i:i+3]) for i in range(len(words)-2)])

    metrics = {
        "unique_unigrams": len(set(all_words)) / max(len(all_words), 1),
        "unique_bigrams": len(set(all_bigrams)) / max(len(all_bigrams), 1),
        "unique_trigrams": len(set(all_trigrams)) / max(len(all_trigrams), 1),
        "avg_length": sum(len(s) for s in samples) / max(len(samples), 1)
    }

    return metrics


@torch.no_grad()
def benchmark_sample_quality(
    dataset: str = "openwebtext",
    num_samples: int = 10,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 200,
    device: str = "cuda",
    dtype: str = "bfloat16",
    seed: int = 1337,
    verbose: bool = True
) -> dict[str, dict[str, float]]:
    """Benchmark sample generation quality for standard vs NeuroManifold attention.

    Generates samples from both models and measures:
    - Diversity (unique n-grams)
    - Average length
    - Relative quality metrics

    Args:
        dataset: Dataset name (directory under data/)
        num_samples: Number of samples to generate per model
        max_new_tokens: Tokens to generate per sample
        temperature: Sampling temperature (higher = more random)
        top_k: Keep only top-k tokens for sampling
        device: 'cuda' or 'cpu'
        dtype: 'bfloat16', 'float16', or 'float32'
        seed: Random seed for reproducibility
        verbose: Print progress

    Returns:
        Dict with 'standard' and 'neuromanifold' results, each containing diversity metrics
    """
    if verbose:
        print("=" * 80)
        print("NeuroManifold Sample Quality Benchmark")
        print("=" * 80)
        print(f"Dataset: {dataset}")
        print(f"Samples: {num_samples}")
        print(f"Max tokens: {max_new_tokens}")
        print(f"Temperature: {temperature}")
        print(f"Top-k: {top_k}")
        print(f"Device: {device}")
        print()

    # Setup
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16
    }[dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Setup encoding
    encode, decode = setup_encoding(dataset)

    # Start prompt
    start = "\n"
    start_ids = encode(start)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    results = {}

    # Generate samples from standard attention model
    if verbose:
        print("Generating samples from standard attention model...")
        start_time = time.time()

    model_standard, _ = load_model(
        "neuromanifold_gpt.config.benchmarks.standard_attention",
        device=device,
        dtype=dtype
    )
    model_standard.eval()

    samples_standard = []
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                y = model_standard.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                sample_text = decode(y[0].tolist())
                samples_standard.append(sample_text)

    diversity_standard = measure_sample_diversity(samples_standard)
    results["standard"] = diversity_standard

    if verbose:
        elapsed = time.time() - start_time
        print(f"  Unique unigrams: {diversity_standard['unique_unigrams']:.4f}")
        print(f"  Unique bigrams:  {diversity_standard['unique_bigrams']:.4f}")
        print(f"  Unique trigrams: {diversity_standard['unique_trigrams']:.4f}")
        print(f"  Avg length:      {diversity_standard['avg_length']:.1f} chars")
        print(f"  Time: {elapsed:.2f}s")
        print()

    # Clean up
    del model_standard
    if device_type == "cuda":
        torch.cuda.empty_cache()

    # Generate samples from NeuroManifold attention model
    if verbose:
        print("Generating samples from NeuroManifold attention model...")
        start_time = time.time()

    model_neuromanifold, _ = load_model(
        "neuromanifold_gpt.config.benchmarks.neuromanifold_attention",
        device=device,
        dtype=dtype
    )
    model_neuromanifold.eval()

    samples_neuromanifold = []
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                y = model_neuromanifold.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                sample_text = decode(y[0].tolist())
                samples_neuromanifold.append(sample_text)

    diversity_neuromanifold = measure_sample_diversity(samples_neuromanifold)
    results["neuromanifold"] = diversity_neuromanifold

    if verbose:
        elapsed = time.time() - start_time
        print(f"  Unique unigrams: {diversity_neuromanifold['unique_unigrams']:.4f}")
        print(f"  Unique bigrams:  {diversity_neuromanifold['unique_bigrams']:.4f}")
        print(f"  Unique trigrams: {diversity_neuromanifold['unique_trigrams']:.4f}")
        print(f"  Avg length:      {diversity_neuromanifold['avg_length']:.1f} chars")
        print(f"  Time: {elapsed:.2f}s")
        print()

    # Clean up
    del model_neuromanifold
    if device_type == "cuda":
        torch.cuda.empty_cache()

    # Summary
    if verbose:
        print("=" * 80)
        print("Sample quality Summary")
        print("=" * 80)
        print(f"Standard diversity (unigrams):      {results['standard']['unique_unigrams']:>10.4f}")
        print(f"NeuroManifold diversity (unigrams): {results['neuromanifold']['unique_unigrams']:>10.4f}")
        diversity_change = (
            (results["neuromanifold"]["unique_unigrams"] - results["standard"]["unique_unigrams"])
            / results["standard"]["unique_unigrams"] * 100
        )
        print(f"Diversity change:                   {diversity_change:>9.1f}%")
        print()

    return results


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

    # Run perplexity benchmark
    results = benchmark_quality(
        dataset=args.dataset,
        eval_iters=args.eval_iters,
        batch_size=args.batch_size,
        block_size=args.block_size,
        device=args.device,
        dtype=args.dtype,
        verbose=True
    )

    # Run sample quality benchmark
    sample_results = benchmark_sample_quality(
        dataset=args.dataset,
        num_samples=5 if args.quick_test else 10,
        max_new_tokens=50 if args.quick_test else 100,
        temperature=0.8,
        top_k=200,
        device=args.device,
        dtype=args.dtype,
        verbose=True
    )

    return 0


if __name__ == "__main__":
    exit(main())
