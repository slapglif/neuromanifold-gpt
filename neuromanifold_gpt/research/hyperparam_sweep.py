#!/usr/bin/env python3
"""Fast hyperparameter sweep using perplexity scoring.

Quick 200-iteration runs to find optimal architecture settings.
"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger

# Configure loguru
logger.remove()
logger.add(sys.stderr, format="{time:HH:mm:ss} | {level} | {message}")

sys.path.insert(0, str(Path(__file__).parent))

from neuromanifold_gpt.config import NeuroManifoldConfig
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT


@dataclass
class SweepResult:
    """Result from a single hyperparameter configuration."""

    config_name: str
    final_loss: float
    perplexity: float
    params: int
    time_per_iter_ms: float


def load_shakespeare():
    """Load Shakespeare dataset."""
    data_path = Path("neuromanifold_gpt/data/shakespeare.txt")
    if not data_path.exists():
        data_path = Path("data/shakespeare_char/input.txt")

    with open(data_path, "r") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return data, stoi, itos, len(chars)


def get_batch(data, block_size, batch_size, device):
    """Get random batch."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def evaluate_config(
    config_overrides: dict,
    config_name: str,
    data: torch.Tensor,
    vocab_size: int,
    device: str,
    n_iters: int = 200,
    batch_size: int = 32,
) -> SweepResult:
    """Evaluate a single configuration."""

    # Base config
    config = NeuroManifoldConfig(
        vocab_size=vocab_size,
        block_size=256,
        n_layer=4,
        n_heads=4,
        n_embd=256,
        manifold_dim=32,
        n_eigenvectors=16,
        dropout=0.0,
        use_sdr=False,
        use_mhc=True,
        mhc_n_streams=2,
        n_fhn_steps=2,
        use_fhn_imex=True,
        use_fhn_partitioning=True,
        fhn_threshold=0.5,
        fhn_tau=12.5,
    )

    # Apply overrides
    for k, v in config_overrides.items():
        setattr(config, k, v)

    # Create model
    model = NeuroManifoldGPT(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # Optimizer
    optimizer = model.configure_optimizers(
        weight_decay=0.1, learning_rate=3e-4, device_type=device
    )

    # Training
    import time

    start_time = time.time()

    model.train()
    final_loss = 0.0

    for it in range(n_iters):
        x, y = get_batch(data, config.block_size, batch_size, device)

        logits, loss, _ = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if it == n_iters - 1:
            final_loss = loss.item()

    elapsed = time.time() - start_time
    time_per_iter = elapsed / n_iters * 1000

    perplexity = math.exp(min(final_loss, 10))  # Cap to avoid overflow

    # Cleanup
    del model
    del optimizer
    torch.cuda.empty_cache()

    return SweepResult(
        config_name=config_name,
        final_loss=final_loss,
        perplexity=perplexity,
        params=n_params,
        time_per_iter_ms=time_per_iter,
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Load data
    data, stoi, itos, vocab_size = load_shakespeare()
    logger.info(f"Vocab size: {vocab_size}")

    # Define sweep configurations
    sweep_configs = [
        # mHC variants
        ({"use_mhc": False, "use_full_mhc": False}, "no_mhc"),
        ({"use_mhc": True, "use_full_mhc": False}, "simplified_mhc"),
        ({"use_mhc": True, "use_full_mhc": True, "mhc_n_streams": 2}, "full_mhc_2"),
        # KAN variants
        ({"use_kan": False}, "swiglu_mlp"),
        ({"use_kan": True, "kan_type": "wave", "kan_wavelet": "dog"}, "wavekan_dog"),
        (
            {"use_kan": True, "kan_type": "wave", "kan_wavelet": "mexican_hat"},
            "wavekan_mhat",
        ),
        ({"use_kan": True, "kan_type": "cheby", "kan_degree": 4}, "chebykan_d4"),
        # FHN steps
        ({"n_fhn_steps": 1}, "fhn_1step"),
        ({"n_fhn_steps": 2}, "fhn_2step"),
        ({"n_fhn_steps": 3}, "fhn_3step"),
        # FHN threshold
        ({"fhn_threshold": 0.3}, "fhn_thresh_0.3"),
        ({"fhn_threshold": 0.5}, "fhn_thresh_0.5"),
        ({"fhn_threshold": 0.7}, "fhn_thresh_0.7"),
        # Combined best (hypothesis)
        (
            {
                "use_mhc": True,
                "use_full_mhc": True,
                "mhc_n_streams": 2,
                "use_kan": True,
                "kan_type": "wave",
                "kan_wavelet": "dog",
                "n_fhn_steps": 2,
                "fhn_threshold": 0.5,
            },
            "full_combo",
        ),
    ]

    results = []

    for config_override, config_name in sweep_configs:
        logger.info(f"\n=== Testing: {config_name} ===")
        try:
            result = evaluate_config(
                config_override,
                config_name,
                data,
                vocab_size,
                device,
                n_iters=200,
                batch_size=32,
            )
            results.append(result)
            logger.info(
                f"{config_name}: loss={result.final_loss:.4f}, ppl={result.perplexity:.2f}, "
                f"params={result.params:,}, time={result.time_per_iter_ms:.1f}ms/iter"
            )
        except Exception as e:
            logger.error(f"Failed {config_name}: {e}")
            continue

    # Sort by perplexity
    results.sort(key=lambda x: x.perplexity)

    logger.info("\n\n=== RESULTS (sorted by perplexity) ===")
    for i, r in enumerate(results):
        logger.info(
            f"{i+1}. {r.config_name}: ppl={r.perplexity:.2f}, loss={r.final_loss:.4f}, "
            f"params={r.params:,}, speed={r.time_per_iter_ms:.1f}ms"
        )

    # Best config
    if results:
        best = results[0]
        logger.info(f"\n=== BEST CONFIG: {best.config_name} ===")
        logger.info(f"Perplexity: {best.perplexity:.2f}")
        logger.info(f"Final Loss: {best.final_loss:.4f}")
        logger.info(f"Parameters: {best.params:,}")


if __name__ == "__main__":
    main()
