#!/usr/bin/env python3
"""Test SDR with discrimination loss to prevent mode collapse."""

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
        sdr_sparsity=0.02,  # Back to 2% for testing
        sdr_embed_dim=128,
        manifold_dim=32,
        n_eigenvectors=16,
        use_sdr=True,  # Test with SDR enabled
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
    logger.info(f"Data: {len(data)} chars")

    model = NeuroManifoldGPT(config).to(device)
    logger.info(f"Parameters: {model.num_parameters():,}")

    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=config.learning_rate,
        betas=(0.9, 0.95),
        device_type=str(device),
    )

    # Training
    model.train()
    batch_size = 16
    block_size = config.block_size
    n_iters = 2000

    logger.info(f"Training for {n_iters} iterations with discrimination loss...")
    for i in range(n_iters):
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[j:j+block_size] for j in ix]).to(device)
        y = torch.stack([data[j+1:j+block_size+1] for j in ix]).to(device)

        logits, loss, info = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if i % 200 == 0:
            discrim = info.get('discrimination_loss', torch.tensor(0.0)).item()
            # Test generation
            model.eval()
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            with torch.no_grad():
                generated = model.generate(context, max_new_tokens=50, temperature=0.8, top_k=40)
            gen_text = decode(generated[0].tolist())
            char_counts = Counter(generated[0].tolist())
            diversity = len(char_counts)
            logger.info(f"iter {i}: loss={loss.item():.4f}, discrim={discrim:.4f}, "
                       f"div={diversity}, sample='{gen_text[:35]}'")
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
        logger.info(f"\nPrompt '{prompt}': {output[:100]}")


if __name__ == "__main__":
    main()
