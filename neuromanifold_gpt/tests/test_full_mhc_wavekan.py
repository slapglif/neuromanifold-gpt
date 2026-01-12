#!/usr/bin/env python3
"""Test full mHC + WaveKAN + FHN combination (no SDR)."""

import torch
import torch.nn.functional as F
from loguru import logger
import sys
from pathlib import Path

# Configure loguru
logger.remove()
logger.add(sys.stderr, format="{time:HH:mm:ss} | {level} | {message}")

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neuromanifold_gpt.config import NeuroManifoldConfig
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT


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
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=0.8, top_k=40):
    """Generate text."""
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        logits, _, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, idx_next], dim=1)
    return idx


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Load data
    data, stoi, itos, vocab_size = load_shakespeare()
    logger.info(f"Vocab size: {vocab_size}, Data size: {len(data)}")

    # Config: Full mHC + WaveKAN + FHN, NO SDR
    # Reduced size to fit in 6GB VRAM with full mHC
    config = NeuroManifoldConfig(
        vocab_size=vocab_size,
        block_size=256,
        n_layer=4,  # Reduced
        n_heads=4,  # Reduced
        n_embd=256,  # Reduced
        manifold_dim=32,  # Reduced
        n_eigenvectors=16,  # Reduced
        dropout=0.0,
        # SDR DISABLED
        use_sdr=False,
        # Full mHC enabled
        use_mhc=True,
        use_full_mhc=True,  # Multi-stream Sinkhorn-Knopp
        mhc_n_streams=2,  # Reduced for memory efficiency
        # WaveKAN enabled
        use_kan=True,
        kan_type="wave",
        kan_wavelet="dog",  # Fast DOG wavelet
        use_fast_wavekan=True,
        # FHN with IMEX
        n_fhn_steps=2,
        use_fhn_imex=True,
        use_fhn_partitioning=True,
        fhn_threshold=0.5,
        fhn_tau=12.5,
    )

    logger.info(f"Config: mHC={config.use_mhc}, full_mhc={config.use_full_mhc}, streams={config.mhc_n_streams}")
    logger.info(f"KAN: type={config.kan_type}, wavelet={config.kan_wavelet}")
    logger.info(f"SDR: {config.use_sdr}")

    # Create model
    model = NeuroManifoldGPT(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {n_params:,}")

    # Optimizer
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=3e-4,
        device_type=device
    )

    # Training (smaller batch for memory)
    batch_size = 32
    n_iters = 2000
    log_interval = 200

    logger.info(f"Training for {n_iters} iterations...")
    model.train()

    import time
    start_time = time.time()

    for it in range(n_iters):
        x, y = get_batch(data, config.block_size, batch_size, device)

        logits, loss, info = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if it % log_interval == 0 or it == n_iters - 1:
            # Generate sample
            model.eval()
            start_tokens = torch.tensor([[stoi['\n']]], dtype=torch.long, device=device)
            generated = generate(model, start_tokens, 50, temperature=0.8, top_k=40)
            sample = ''.join([itos[i.item()] for i in generated[0]])
            model.train()

            # Count unique chars in sample
            unique_chars = len(set(sample.strip()))

            ortho_loss = info.get('ortho_loss', torch.tensor(0.0))
            logger.info(f"iter {it}: loss={loss.item():.4f}, ortho={ortho_loss.item():.4f}, unique_chars={unique_chars}")
            logger.info(f"  Sample: {repr(sample[:60])}")

    elapsed = time.time() - start_time
    logger.info(f"\nTotal time: {elapsed:.1f}s ({elapsed/n_iters*1000:.1f}ms/iter)")

    # Final generation test
    logger.info("\n=== Final Generation ===")
    model.eval()
    prompts = ["\nKING:", "\nTo be", "\nO "]
    for prompt in prompts:
        start_tokens = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)
        generated = generate(model, start_tokens, 100, temperature=0.8, top_k=40)
        sample = ''.join([itos[i.item()] for i in generated[0]])
        logger.info(f"Prompt: {repr(prompt)}")
        logger.info(f"Output: {repr(sample)}\n")


if __name__ == "__main__":
    main()
