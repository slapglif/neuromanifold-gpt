#!/usr/bin/env python3
"""Test generation to debug quality issues."""

import os

import torch
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from neuromanifold_gpt.config.base import NeuroManifoldConfig
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT

# Same config as training
config = NeuroManifoldConfig(
    vocab_size=65,
    block_size=256,
    n_layer=4,
    n_heads=4,
    n_embd=256,
    sdr_size=1024,
    sdr_sparsity=0.02,
    sdr_embed_dim=128,
    manifold_dim=32,
    n_eigenvectors=16,
    use_sdr=True,  # Full pipeline with SDR
    use_kan=True,  # With KAN
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
)


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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data, decode, encode = load_shakespeare()
    print(f"Data: {len(data)} chars")

    # Create and train a fresh model for a bit
    model = NeuroManifoldGPT(config).to(device)
    print(f"Parameters: {model.num_parameters():,}")

    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=6e-4,
        betas=(0.9, 0.95),
        device_type=device,
    )

    # Train for 2000 iterations
    model.train()
    batch_size = 16
    block_size = config.block_size

    print("Training for 2000 iterations...")
    for i in range(2000):
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[j : j + block_size] for j in ix]).to(device)
        y = torch.stack([data[j + 1 : j + block_size + 1] for j in ix]).to(device)

        logits, loss, _ = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if i % 200 == 0:
            print(f"iter {i}: loss={loss.item():.4f}")

    print(f"\nFinal loss: {loss.item():.4f}")

    # Test generation
    model.eval()

    # Test with different prompts and temperatures
    prompts = [
        "ROMEO:",
        "To be or ",
        "The ",
        "First",
    ]

    temperatures = [0.5, 0.8, 1.0]

    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: '{prompt}'")

        # Encode prompt
        prompt_ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

        for temp in temperatures:
            print(f"\nTemp={temp}:")
            with torch.no_grad():
                generated = model.generate(
                    prompt_ids.clone(), max_new_tokens=100, temperature=temp, top_k=40
                )
            output = decode(generated[0].tolist())
            print(f"  {output[:150]}")

    # Also test just logits to see if model is working
    print(f"\n{'='*60}")
    print("Raw logits test:")
    test_input = torch.tensor([encode("ROMEO:")], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _, _ = model(test_input)
        last_logits = logits[0, -1, :]
        probs = F.softmax(last_logits, dim=-1)
        top_probs, top_ids = torch.topk(probs, 10)

        print("Top 10 next char predictions:")
        for prob, idx in zip(top_probs.tolist(), top_ids.tolist()):
            char = decode([idx])
            print(f"  '{char}': {prob:.4f}")


if __name__ == "__main__":
    main()
