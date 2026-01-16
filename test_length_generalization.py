"""
Test length generalization of position embeddings.

This script trains models with different position embedding types on seq_len=64,
then tests them on seq_len=128 (2x training length) to measure extrapolation capability.

Expected results:
- RoPE and ALiBi should maintain reasonable perplexity at 2x length
- Learned embeddings should show degradation (poor extrapolation)

Usage:
    python test_length_generalization.py
"""
import os
import pickle
import time
from contextlib import nullcontext
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from neuromanifold_gpt.config import NeuroManifoldConfigNano
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT


# Configuration
TRAIN_SEQ_LEN = 64
TEST_SEQ_LEN = 128
BATCH_SIZE = 32
MAX_ITERS = 300  # Reduced for faster testing
EVAL_ITERS = 50
LEARNING_RATE = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
DATA_DIR = 'data/shakespeare_char'
SEED = 1337

# Position embedding types to test
POS_EMB_TYPES = ['learned', 'rotary', 'alibi']


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def prepare_data() -> Tuple[np.ndarray, np.ndarray, int]:
    """Load Shakespeare dataset and prepare train/val splits."""
    # Check if data exists
    if not os.path.exists(DATA_DIR):
        logger.info("Shakespeare data not found. Preparing dataset...")
        from neuromanifold_gpt.data.prepare_shakespeare import main as prepare_shakespeare
        prepare_shakespeare()

    # Load data
    train_data = np.memmap(os.path.join(DATA_DIR, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(DATA_DIR, 'val.bin'), dtype=np.uint16, mode='r')

    # Load vocab size
    meta_path = os.path.join(DATA_DIR, 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    vocab_size = meta['vocab_size']

    logger.info(f"Dataset loaded: train={len(train_data):,} tokens, val={len(val_data):,} tokens, vocab={vocab_size}")

    return train_data, val_data, vocab_size


def get_batch(
    data: np.ndarray,
    seq_len: int,
    batch_size: int,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get a batch of data with specified sequence length."""
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+seq_len].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+seq_len].astype(np.int64)) for i in ix])

    x = x.to(device)
    y = y.to(device)

    return x, y


@torch.no_grad()
def estimate_loss(
    model: NeuroManifoldGPT,
    train_data: np.ndarray,
    val_data: np.ndarray,
    seq_len: int,
    batch_size: int,
    eval_iters: int,
    device: str,
    ctx
) -> Dict[str, float]:
    """Estimate loss on train and validation sets."""
    model.eval()
    out = {}

    for split, data in [('train', train_data), ('val', val_data)]:
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(data, seq_len, batch_size, device)
            with ctx:
                _, loss, _ = model(x, y)
            losses.append(loss.item())
        out[split] = np.mean(losses)

    model.train()
    return out


def train_model(
    pos_emb_type: str,
    train_data: np.ndarray,
    val_data: np.ndarray,
    vocab_size: int,
    device: str,
    ctx
) -> NeuroManifoldGPT:
    """Train a model with specified position embedding type on TRAIN_SEQ_LEN."""
    logger.info(f"\n{'='*70}")
    logger.info(f"Training {pos_emb_type} model on seq_len={TRAIN_SEQ_LEN}")
    logger.info(f"{'='*70}")

    # Create model - use Nano preset which is known to work
    config = NeuroManifoldConfigNano(
        vocab_size=vocab_size,
        block_size=TEST_SEQ_LEN,  # Allow testing at 2x length
        pos_emb_type=pos_emb_type,
        skip_manifold_spectral=True,  # Skip spectral to avoid torch.compile/nvcc issues
    )

    model = NeuroManifoldGPT(config)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    logger.info(f"Starting training for {MAX_ITERS} iterations...")
    start_time = time.time()

    for iter_num in range(MAX_ITERS):
        # Get batch at training length
        x, y = get_batch(train_data, TRAIN_SEQ_LEN, BATCH_SIZE, device)

        # Forward pass
        with ctx:
            _, loss, _ = model(x, y)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Logging
        if iter_num % 100 == 0:
            losses = estimate_loss(
                model, train_data, val_data,
                TRAIN_SEQ_LEN, BATCH_SIZE, EVAL_ITERS,
                device, ctx
            )
            train_losses.append(losses['train'])
            val_losses.append(losses['val'])

            elapsed = time.time() - start_time
            logger.info(
                f"iter {iter_num:4d} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | "
                f"time {elapsed:.2f}s"
            )

            best_val_loss = min(best_val_loss, losses['val'])

    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.2f}s")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")

    return model


def evaluate_extrapolation(
    model: NeuroManifoldGPT,
    pos_emb_type: str,
    val_data: np.ndarray,
    device: str,
    ctx
) -> Dict[str, float]:
    """Evaluate model at both training and test sequence lengths."""
    logger.info(f"\nEvaluating {pos_emb_type} extrapolation capability...")

    model.eval()
    results = {}

    # Evaluate at training length
    logger.info(f"  Testing at training length (seq_len={TRAIN_SEQ_LEN})...")
    losses_train_len = []
    for _ in range(EVAL_ITERS):
        x, y = get_batch(val_data, TRAIN_SEQ_LEN, BATCH_SIZE, device)
        with ctx:
            _, loss, _ = model(x, y)
        losses_train_len.append(loss.item())

    train_len_loss = np.mean(losses_train_len)
    train_len_ppl = np.exp(train_len_loss)
    results['train_len_loss'] = train_len_loss
    results['train_len_ppl'] = train_len_ppl

    logger.info(f"    Loss: {train_len_loss:.4f}, Perplexity: {train_len_ppl:.4f}")

    # Evaluate at 2x length
    logger.info(f"  Testing at 2x length (seq_len={TEST_SEQ_LEN})...")
    losses_test_len = []
    for _ in range(EVAL_ITERS):
        x, y = get_batch(val_data, TEST_SEQ_LEN, BATCH_SIZE, device)
        with ctx:
            _, loss, _ = model(x, y)
        losses_test_len.append(loss.item())

    test_len_loss = np.mean(losses_test_len)
    test_len_ppl = np.exp(test_len_loss)
    results['test_len_loss'] = test_len_loss
    results['test_len_ppl'] = test_len_ppl

    logger.info(f"    Loss: {test_len_loss:.4f}, Perplexity: {test_len_ppl:.4f}")

    # Calculate degradation
    degradation = ((test_len_ppl - train_len_ppl) / train_len_ppl) * 100
    results['degradation_pct'] = degradation

    logger.info(f"  Perplexity degradation: {degradation:+.2f}%")

    return results


def save_results(all_results: Dict[str, Dict[str, float]]) -> None:
    """Save results to markdown file."""
    output_file = 'LENGTH_GENERALIZATION_RESULTS.md'

    with open(output_file, 'w') as f:
        f.write("# Length Generalization Test Results\n\n")
        f.write("## Overview\n\n")
        f.write(f"- **Training sequence length**: {TRAIN_SEQ_LEN}\n")
        f.write(f"- **Test sequence length**: {TEST_SEQ_LEN} (2x training)\n")
        f.write(f"- **Dataset**: Shakespeare (character-level)\n")
        f.write(f"- **Training iterations**: {MAX_ITERS}\n")
        f.write(f"- **Model size**: 4 layers, 4 heads, 256 embedding dim\n")
        f.write(f"- **Device**: {DEVICE}\n\n")

        f.write("## Results Summary\n\n")
        f.write("| Position Embedding | Train Length PPL | Test Length PPL (2x) | Degradation | Status |\n")
        f.write("|-------------------|------------------|---------------------|-------------|--------|\n")

        for pos_emb_type in POS_EMB_TYPES:
            results = all_results[pos_emb_type]
            train_ppl = results['train_len_ppl']
            test_ppl = results['test_len_ppl']
            degradation = results['degradation_pct']

            # RoPE and ALiBi should have lower degradation than learned
            if pos_emb_type in ['rotary', 'alibi']:
                status = "✅ Good extrapolation" if degradation < 50 else "⚠️ High degradation"
            else:
                status = "⚠️ Expected degradation"

            f.write(f"| {pos_emb_type.capitalize():17s} | {train_ppl:16.2f} | {test_ppl:19.2f} | {degradation:+10.1f}% | {status} |\n")

        f.write("\n## Detailed Results\n\n")
        for pos_emb_type in POS_EMB_TYPES:
            results = all_results[pos_emb_type]
            f.write(f"### {pos_emb_type.capitalize()}\n\n")
            f.write(f"**At training length ({TRAIN_SEQ_LEN}):**\n")
            f.write(f"- Loss: {results['train_len_loss']:.4f}\n")
            f.write(f"- Perplexity: {results['train_len_ppl']:.4f}\n\n")
            f.write(f"**At test length ({TEST_SEQ_LEN}):**\n")
            f.write(f"- Loss: {results['test_len_loss']:.4f}\n")
            f.write(f"- Perplexity: {results['test_len_ppl']:.4f}\n\n")
            f.write(f"**Degradation**: {results['degradation_pct']:+.2f}%\n\n")

        f.write("## Analysis\n\n")
        f.write("### Expected Behavior\n\n")
        f.write("1. **RoPE (Rotary Position Embeddings)**:\n")
        f.write("   - Should extrapolate well due to relative position encoding\n")
        f.write("   - Rotation-based mechanism works at any sequence length\n")
        f.write("   - Expected: Low degradation (< 50%)\n\n")

        f.write("2. **ALiBi (Attention with Linear Biases)**:\n")
        f.write("   - Should extrapolate well due to linear bias design\n")
        f.write("   - Bias slopes are position-agnostic\n")
        f.write("   - Expected: Low degradation (< 50%)\n\n")

        f.write("3. **Learned Position Embeddings**:\n")
        f.write("   - Limited to training sequence length\n")
        f.write("   - No learned representation for positions > 64\n")
        f.write("   - Expected: Higher degradation (baseline for comparison)\n\n")

        # Comparative analysis
        learned_degradation = all_results['learned']['degradation_pct']
        rotary_degradation = all_results['rotary']['degradation_pct']
        alibi_degradation = all_results['alibi']['degradation_pct']

        f.write("### Comparative Analysis\n\n")

        rotary_improvement = learned_degradation - rotary_degradation
        alibi_improvement = learned_degradation - alibi_degradation

        f.write(f"- **RoPE vs Learned**: {rotary_improvement:+.1f} percentage points ")
        f.write("better " if rotary_improvement > 0 else "worse ")
        f.write("extrapolation\n")

        f.write(f"- **ALiBi vs Learned**: {alibi_improvement:+.1f} percentage points ")
        f.write("better " if alibi_improvement > 0 else "worse ")
        f.write("extrapolation\n\n")

        f.write("### Conclusion\n\n")
        if rotary_degradation < learned_degradation and alibi_degradation < learned_degradation:
            f.write("✅ **SUCCESS**: Both RoPE and ALiBi demonstrate better length generalization ")
            f.write("than learned position embeddings, confirming their extrapolation capabilities.\n")
        elif rotary_degradation < learned_degradation or alibi_degradation < learned_degradation:
            f.write("⚠️ **PARTIAL SUCCESS**: At least one of RoPE/ALiBi shows better extrapolation ")
            f.write("than learned embeddings.\n")
        else:
            f.write("⚠️ **UNEXPECTED**: RoPE and ALiBi did not show expected extrapolation benefits. ")
            f.write("This may be due to limited training, model capacity, or dataset characteristics.\n")

    logger.info(f"\nResults saved to {output_file}")


def main():
    """Main execution function."""
    logger.info("="*70)
    logger.info("Position Embedding Length Generalization Test")
    logger.info("="*70)
    logger.info(f"Training length: {TRAIN_SEQ_LEN}")
    logger.info(f"Test length: {TEST_SEQ_LEN} (2x)")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Dtype: {DTYPE}")

    # Set seed
    set_seed(SEED)

    # Prepare data
    train_data, val_data, vocab_size = prepare_data()

    # Setup mixed precision context
    device_type = 'cuda' if 'cuda' in DEVICE else 'cpu'
    ptdtype = {
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        'float16': torch.float16
    }[DTYPE]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
        device_type=device_type, dtype=ptdtype
    )

    # Train and evaluate each position embedding type
    all_results = {}

    for pos_emb_type in POS_EMB_TYPES:
        # Train model
        model = train_model(pos_emb_type, train_data, val_data, vocab_size, DEVICE, ctx)

        # Evaluate extrapolation
        results = evaluate_extrapolation(model, pos_emb_type, val_data, DEVICE, ctx)
        all_results[pos_emb_type] = results

        # Clean up
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save results
    save_results(all_results)

    logger.info("\n" + "="*70)
    logger.info("Length Generalization Test Complete!")
    logger.info("="*70)

    # Print summary
    logger.info("\nSummary:")
    for pos_emb_type in POS_EMB_TYPES:
        results = all_results[pos_emb_type]
        logger.info(f"  {pos_emb_type.capitalize():8s}: "
                   f"train_ppl={results['train_len_ppl']:.2f}, "
                   f"test_ppl={results['test_len_ppl']:.2f}, "
                   f"degradation={results['degradation_pct']:+.1f}%")


if __name__ == '__main__':
    main()
