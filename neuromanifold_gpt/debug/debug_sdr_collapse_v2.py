#!/usr/bin/env python3
"""Test SDR with different sparsity levels to find sweet spot."""

from collections import Counter

import torch
import torch.nn.functional as F
from loguru import logger

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from neuromanifold_gpt.config.base import NeuroManifoldConfig
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT


def load_shakespeare():
    data_path = "neuromanifold_gpt/data/input.txt"
    with open(data_path, "r") as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]

    def decode(l):
        return "".join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)
    return data, decode, encode


def test_sparsity(sparsity: float, n_iters: int = 1000):
    """Test a specific sparsity level."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = NeuroManifoldConfig(
        vocab_size=65,
        block_size=256,
        n_layer=4,
        n_heads=4,
        n_embd=256,
        sdr_size=1024,
        sdr_sparsity=sparsity,  # Test different sparsity
        sdr_embed_dim=128,
        manifold_dim=32,
        n_eigenvectors=16,
        use_sdr=True,
        use_kan=True,
        kan_type="wave",
        kan_wavelet="dog",
        use_fast_wavekan=True,
        fhn_threshold=0.5,
        fhn_tau=12.5,
        n_fhn_steps=2,
        use_fhn_imex=True,
        use_fhn_partitioning=True,
        use_fhn_parallel=True,
        dropout=0.0,
        learning_rate=6e-4,
    )

    # Calculate n_active from sparsity
    n_active = int(config.sdr_size * sparsity)
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing sparsity={sparsity:.1%} ({n_active} active bits)")
    logger.info(f"{'='*60}")

    data, decode, encode = load_shakespeare()

    model = NeuroManifoldGPT(config).to(device)
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=6e-4,
        betas=(0.9, 0.95),
        device_type=str(device),
    )

    # Training
    model.train()
    batch_size = 16
    block_size = config.block_size

    for i in range(n_iters):
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[j : j + block_size] for j in ix]).to(device)
        y = torch.stack([data[j + 1 : j + block_size + 1] for j in ix]).to(device)

        logits, loss, info = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if i % 200 == 0:
            logger.info(f"iter {i}: loss={loss.item():.4f}")

    logger.info(f"Final loss: {loss.item():.4f}")

    # Generation test
    model.eval()
    context = torch.tensor([[encode("ROMEO:")[0]]], dtype=torch.long, device=device)
    with torch.no_grad():
        generated = model.generate(
            context, max_new_tokens=100, temperature=0.8, top_k=40
        )

    gen_text = decode(generated[0].tolist())

    # Analyze diversity
    char_counts = Counter(generated[0].tolist())
    top_char = char_counts.most_common(1)[0]
    diversity = len(char_counts) / 65  # Fraction of unique chars used

    logger.info(f"Generated: {gen_text[:80]}")
    logger.info(f"Top char: '{decode([top_char[0]])}' ({top_char[1]}/100)")
    logger.info(f"Char diversity: {diversity:.1%} ({len(char_counts)} unique chars)")

    # Also check entropy
    with torch.no_grad():
        ix = torch.randint(len(data) - 256, (16,))
        tokens = torch.stack([data[i : i + 256] for i in ix]).to(device)
        logits, _, _ = model(tokens)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * probs.log().clamp(min=-100)).sum(dim=-1).mean()
        logger.info(f"Mean entropy: {entropy:.4f}")

    return {
        "sparsity": sparsity,
        "n_active": n_active,
        "final_loss": loss.item(),
        "top_char_count": top_char[1],
        "diversity": diversity,
        "entropy": entropy.item(),
        "sample": gen_text[:80],
    }


def main():
    # Test different sparsity levels
    # 2% = 20 bits, 5% = 51 bits, 10% = 102 bits, 20% = 204 bits
    sparsities = [0.02, 0.05, 0.10, 0.20]

    results = []
    for sparsity in sparsities:
        result = test_sparsity(sparsity, n_iters=1000)
        results.append(result)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for r in results:
        logger.info(
            f"Sparsity {r['sparsity']:.0%} ({r['n_active']} bits): "
            f"loss={r['final_loss']:.3f}, "
            f"top_char={r['top_char_count']}/100, "
            f"diversity={r['diversity']:.0%}, "
            f"entropy={r['entropy']:.2f}"
        )


if __name__ == "__main__":
    main()
