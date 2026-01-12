#!/usr/bin/env python3
"""Test SDR with entropy regularization to prevent prediction collapse."""

import torch
import torch.nn.functional as F
from collections import Counter
from loguru import logger

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from neuromanifold_gpt.config.base import NeuroManifoldConfig
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT


def load_shakespeare():
    data_path = "neuromanifold_gpt/data/input.txt"
    with open(data_path, 'r') as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    data = torch.tensor(encode(text), dtype=torch.long)
    return data, decode, encode


def compute_entropy_reg(logits: torch.Tensor, min_entropy: float = 1.5) -> torch.Tensor:
    """Penalize predictions with entropy below min_entropy.

    This prevents the model from collapsing to always predicting the same tokens.
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)  # (B, T)

    # Penalize if entropy is below minimum (encourages diversity)
    low_entropy_penalty = F.relu(min_entropy - entropy).mean()
    return low_entropy_penalty


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")

    config = NeuroManifoldConfig(
        vocab_size=65,
        block_size=256,
        n_layer=4,
        n_heads=4,
        n_embd=256,
        sdr_size=1024,
        sdr_sparsity=0.05,  # 5% = 51 active bits (more capacity)
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

    logger.info(f"SDR config: size={config.sdr_size}, n_active={config.sdr_n_active}")

    data, decode, encode = load_shakespeare()
    model = NeuroManifoldGPT(config).to(device)
    logger.info(f"Parameters: {model.num_parameters():,}")

    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=config.learning_rate,
        betas=(0.9, 0.95),
        device_type=str(device),
    )

    # Training with entropy regularization
    model.train()
    batch_size = 16
    block_size = config.block_size
    n_iters = 2000

    # Entropy reg parameters
    entropy_weight = 0.1  # Weight for entropy regularization
    min_entropy = 2.0  # Minimum expected entropy (log2(65) ≈ 6, so 2 is reasonable)

    logger.info(f"Training for {n_iters} iterations with entropy regularization...")
    logger.info(f"Entropy weight={entropy_weight}, min_entropy={min_entropy}")

    for i in range(n_iters):
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[j:j+block_size] for j in ix]).to(device)
        y = torch.stack([data[j+1:j+block_size+1] for j in ix]).to(device)

        logits, loss, info = model(x, y)

        # Add entropy regularization
        entropy_reg = compute_entropy_reg(logits, min_entropy)
        total_loss = loss + entropy_weight * entropy_reg

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if i % 200 == 0:
            # Test generation
            model.eval()
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            with torch.no_grad():
                generated = model.generate(context, max_new_tokens=50, temperature=0.8, top_k=40)
            gen_text = decode(generated[0].tolist())
            char_counts = Counter(generated[0].tolist())
            diversity = len(char_counts)

            # Compute current entropy
            with torch.no_grad():
                probs = F.softmax(logits, dim=-1)
                log_probs = F.log_softmax(logits, dim=-1)
                mean_entropy = -(probs * log_probs).sum(dim=-1).mean().item()

            logger.info(f"iter {i}: loss={loss.item():.3f}, ent_reg={entropy_reg.item():.3f}, "
                       f"entropy={mean_entropy:.2f}, div={diversity}")
            logger.info(f"  sample='{gen_text[:40]}'")
            model.train()

    logger.info(f"\nFinal loss: {loss.item():.4f}")

    # Final generation test
    model.eval()
    prompts = ["ROMEO:", "To be or ", "The ", "First"]

    for prompt in prompts:
        prompt_ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
        with torch.no_grad():
            generated = model.generate(prompt_ids.clone(), max_new_tokens=100, temperature=0.8, top_k=40)
        output = decode(generated[0].tolist())
        logger.info(f"\n'{prompt}': {output[:100]}")


if __name__ == "__main__":
    main()
