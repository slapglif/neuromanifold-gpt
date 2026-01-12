#!/usr/bin/env python3
"""Quick A/B comparison: Full mHC vs Simplified mHC."""

import torch
import torch.nn.functional as F
from loguru import logger
import sys
from pathlib import Path
import math

logger.remove()
logger.add(sys.stderr, format="{time:HH:mm:ss} | {level} | {message}")

sys.path.insert(0, str(Path(__file__).parent))

from neuromanifold_gpt.config import NeuroManifoldConfig
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT


def load_shakespeare():
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
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=0.8, top_k=40):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        logits, _, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, idx_next], dim=1)
    return idx


def train_and_eval(config, name, data, stoi, itos, device, n_iters=500):
    """Train for n_iters and return final loss and sample."""
    import time

    model = NeuroManifoldGPT(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"{name}: {n_params:,} params")

    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=3e-4, device_type=device)

    start = time.time()
    model.train()

    losses = []
    for it in range(n_iters):
        x, y = get_batch(data, config.block_size, 32, device)
        logits, loss, _ = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

        if it % 100 == 0:
            logger.info(f"  {name} iter {it}: loss={loss.item():.4f}")

    elapsed = time.time() - start

    # Generate sample
    model.eval()
    start_tokens = torch.tensor([[stoi['\n'], stoi['K'], stoi['I'], stoi['N'], stoi['G'], stoi[':']]], device=device)
    generated = generate(model, start_tokens, 60, temperature=0.8)
    sample = ''.join([itos[i.item()] for i in generated[0]])

    final_loss = sum(losses[-10:]) / 10  # Average last 10
    perplexity = math.exp(min(final_loss, 10))

    del model
    del optimizer
    torch.cuda.empty_cache()

    return {
        'name': name,
        'params': n_params,
        'final_loss': final_loss,
        'perplexity': perplexity,
        'time': elapsed,
        'sample': sample
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    data, stoi, itos, vocab_size = load_shakespeare()

    # Base config
    base = dict(
        vocab_size=vocab_size,
        block_size=256,
        n_layer=4,
        n_heads=4,
        n_embd=256,
        manifold_dim=32,
        n_eigenvectors=16,
        dropout=0.0,
        use_sdr=False,
        use_kan=True,
        kan_type="wave",
        kan_wavelet="dog",
        use_fast_wavekan=True,
        n_fhn_steps=2,
        use_fhn_imex=True,
        use_fhn_partitioning=True,
    )

    configs = [
        (NeuroManifoldConfig(**{**base, 'use_mhc': False}), "no_mhc"),
        (NeuroManifoldConfig(**{**base, 'use_mhc': True, 'use_full_mhc': False}), "simplified_mhc"),
        (NeuroManifoldConfig(**{**base, 'use_mhc': True, 'use_full_mhc': True, 'mhc_n_streams': 2}), "full_mhc_2"),
    ]

    results = []
    for config, name in configs:
        logger.info(f"\n=== Testing: {name} ===")
        result = train_and_eval(config, name, data, stoi, itos, device, n_iters=500)
        results.append(result)
        logger.info(f"Result: loss={result['final_loss']:.4f}, ppl={result['perplexity']:.2f}")
        logger.info(f"Sample: {repr(result['sample'][:80])}")

    # Summary
    logger.info("\n\n=== SUMMARY ===")
    for r in sorted(results, key=lambda x: x['perplexity']):
        logger.info(f"{r['name']}: ppl={r['perplexity']:.2f}, loss={r['final_loss']:.4f}, params={r['params']:,}")

    best = min(results, key=lambda x: x['perplexity'])
    logger.info(f"\n=== BEST: {best['name']} (ppl={best['perplexity']:.2f}) ===")


if __name__ == "__main__":
    main()
