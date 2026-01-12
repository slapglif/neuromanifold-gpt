#!/usr/bin/env python3
"""Diagnose SDR mode collapse - why does SDR cause degenerate generation?"""

import torch
import torch.nn.functional as F
from collections import Counter
from loguru import logger

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from neuromanifold_gpt.config.base import NeuroManifoldConfig
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT


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
    use_sdr=True,  # Enable SDR to analyze collapse
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


def analyze_sdr_diversity(model, data, device, n_samples=1000):
    """Check if different tokens produce different SDRs."""
    logger.info("Analyzing SDR diversity...")

    model.eval()
    with torch.no_grad():
        # Get random sample of tokens
        idx = torch.randint(len(data) - 32, (n_samples,))
        tokens = torch.stack([data[i:i+32] for i in idx]).to(device)

        # Get SDRs from encoder
        sdr, scores, _ = model.encoder(tokens)

        # Analyze SDR patterns
        logger.info(f"SDR shape: {sdr.shape}")  # (n_samples, 32, sdr_size)

        # Check active bits per SDR
        active_per_sdr = sdr.sum(dim=-1)  # (n_samples, 32)
        logger.info(f"Active bits per SDR: mean={active_per_sdr.mean():.1f}, min={active_per_sdr.min():.0f}, max={active_per_sdr.max():.0f}")

        # Check bit usage across all SDRs
        bit_usage = sdr.sum(dim=(0, 1))  # (sdr_size,)
        total_sdrs = n_samples * 32
        bit_freq = bit_usage / total_sdrs
        logger.info(f"Bit usage: {(bit_usage > 0).sum().item()} / {sdr.size(-1)} bits used")
        logger.info(f"Bit freq: mean={bit_freq.mean():.4f}, max={bit_freq.max():.4f}, min non-zero={bit_freq[bit_freq > 0].min():.4f}")

        # Check temperature
        temp = model.encoder.temperature.item()
        logger.info(f"Temperature: {temp:.6f}")

        # Check duty cycle
        duty = model.encoder.bit_duty_cycle
        logger.info(f"Duty cycle: mean={duty.mean():.6f}, std={duty.std():.6f}")

        # Check SDR uniqueness - how many unique SDR patterns?
        sdr_flat = sdr.view(-1, sdr.size(-1))  # (n_samples*32, sdr_size)
        # Convert to hashable tuples for counting
        sdr_hashes = [tuple(s.nonzero().squeeze(-1).tolist()) for s in sdr_flat]
        unique_count = len(set(sdr_hashes))
        logger.info(f"Unique SDR patterns: {unique_count} / {len(sdr_hashes)} ({100*unique_count/len(sdr_hashes):.1f}%)")

        # Check if different tokens get different SDRs
        # Group by token ID
        token_flat = tokens.view(-1)
        token_sdr_map = {}
        for tok, sdr_hash in zip(token_flat.tolist(), sdr_hashes):
            if tok not in token_sdr_map:
                token_sdr_map[tok] = []
            token_sdr_map[tok].append(sdr_hash)

        # For each token, how many unique SDRs does it produce?
        token_diversity = {}
        for tok, sdrs in token_sdr_map.items():
            unique = len(set(sdrs))
            token_diversity[tok] = (unique, len(sdrs))

        # Report
        logger.info("Token-SDR diversity (should be many unique SDRs per token if context-aware):")
        for tok in sorted(token_diversity.keys())[:10]:
            unique, total = token_diversity[tok]
            logger.info(f"  Token {tok}: {unique}/{total} unique SDRs ({100*unique/total:.1f}%)")

        return sdr


def analyze_after_training(model, data, device, n_iters=500):
    """Train briefly and analyze SDR collapse."""
    logger.info(f"\nTraining for {n_iters} iterations to observe collapse...")

    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=6e-4,
        betas=(0.9, 0.95),
        device_type=str(device),
    )

    model.train()
    batch_size = 16
    block_size = config.block_size

    for i in range(n_iters):
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[j:j+block_size] for j in ix]).to(device)
        y = torch.stack([data[j+1:j+block_size+1] for j in ix]).to(device)

        logits, loss, info = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if i % 100 == 0:
            logger.info(f"iter {i}: loss={loss.item():.4f}")

    logger.info(f"Final loss: {loss.item():.4f}")

    # Analyze SDRs after training
    logger.info("\n--- After Training ---")
    analyze_sdr_diversity(model, data, device)

    # Check generation
    logger.info("\nGeneration test:")
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=100, temperature=0.8, top_k=40)

    # Analyze what was generated
    gen_ids = generated[0].tolist()
    char_counts = Counter(gen_ids)
    logger.info(f"Character distribution in generated text:")
    for tok, count in char_counts.most_common(5):
        logger.info(f"  Token {tok}: {count} occurrences")

    return model


def analyze_logits_entropy(model, data, device, decode):
    """Check if logits have collapsed to always predicting same token."""
    logger.info("\nAnalyzing logit entropy...")

    model.eval()
    with torch.no_grad():
        # Get batch
        ix = torch.randint(len(data) - 256, (16,))
        tokens = torch.stack([data[i:i+256] for i in ix]).to(device)

        logits, _, info = model(tokens)

        # Check entropy of predictions
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * probs.log().clamp(min=-100)).sum(dim=-1)  # (B, T)

        logger.info(f"Prediction entropy: mean={entropy.mean():.4f}, min={entropy.min():.4f}, max={entropy.max():.4f}")

        # Check most predicted tokens
        top_probs, top_ids = torch.topk(probs, 5, dim=-1)

        # Average top prediction across all positions
        avg_top_prob = top_probs[:, :, 0].mean()
        logger.info(f"Average top-1 probability: {avg_top_prob:.4f}")

        # What's the most common prediction?
        argmax_preds = logits.argmax(dim=-1).view(-1)
        pred_counts = Counter(argmax_preds.tolist())
        logger.info("Most common predictions (argmax):")
        for tok, count in pred_counts.most_common(5):
            char = decode([tok])
            logger.info(f"  '{char}' (tok {tok}): {count} times ({100*count/argmax_preds.numel():.1f}%)")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")

    data, decode, encode = load_shakespeare()
    logger.info(f"Data: {len(data)} chars")

    model = NeuroManifoldGPT(config).to(device)
    logger.info(f"Parameters: {model.num_parameters():,}")

    logger.info("\n=== Before Training ===")
    analyze_sdr_diversity(model, data, device)

    logger.info("\n=== Training ===")
    model = analyze_after_training(model, data, device, n_iters=500)

    logger.info("\n=== Logit Analysis ===")
    analyze_logits_entropy(model, data, device, decode)

    # Final temperature
    logger.info(f"\nFinal temperature: {model.encoder.temperature.item():.6f}")

    # Final duty cycle stats
    duty = model.encoder.bit_duty_cycle
    logger.info(f"Final duty cycle: mean={duty.mean():.6f}, std={duty.std():.6f}, max={duty.max():.6f}")


if __name__ == "__main__":
    main()
