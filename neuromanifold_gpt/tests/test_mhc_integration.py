#!/usr/bin/env python3
"""Test mHC integration with NeuroManifold architecture."""

from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from neuromanifold_gpt.config.base import NeuroManifoldConfig
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT
from neuromanifold_gpt.model.mhc import SimplifiedMHC
from neuromanifold_gpt.model.sinkhorn import sinkhorn_log


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


class MHCNeuroManifoldGPT(nn.Module):
    """NeuroManifoldGPT with mHC residual connections."""

    def __init__(self, base_model: NeuroManifoldGPT):
        super().__init__()
        self.base = base_model
        self.config = base_model.config

        # Add mHC to each block's residual connections
        self.block_mhc = nn.ModuleList(
            [
                SimplifiedMHC(self.config.n_embd, init_residual_weight=0.9)
                for _ in range(self.config.n_layer)
            ]
        )

    def forward(self, tokens, targets=None):
        """Forward with mHC residual connections."""
        B, T = tokens.shape
        device = tokens.device

        # Use base model's embedding
        if self.base.use_sdr:
            sdr, sdr_scores, topo_loss, discrim_loss = self.base.encoder(tokens)
            x = None
        else:
            tok_emb = self.base.token_embedding(tokens)
            pos_emb = self.base.position_embedding(
                torch.arange(T, device=device).unsqueeze(0)
            )
            x = tok_emb + pos_emb
            sdr = torch.zeros(B, T, self.config.sdr_size, device=device)
            torch.tensor(0.0, device=device)
            discrim_loss = torch.tensor(0.0, device=device)

        # Through blocks with mHC
        total_ortho_loss = torch.tensor(0.0, device=device)

        for i, (block, mhc) in enumerate(zip(self.base.blocks, self.block_mhc)):
            if self.base.use_sdr:
                if i == 0:
                    x, info = block(sdr)
                    pos_emb = self.base.ramanujan_pos(torch.zeros(1, T, device=device))
                    x = x + pos_emb
                else:
                    x_as_sdr = self.base.embed_to_sdr(x)
                    block_out, info = block(x_as_sdr)
                    # Apply mHC residual instead of simple addition
                    x = mhc(x, block_out)
            else:
                block_out, info = block(x)
                # Apply mHC residual
                x = mhc(x, block_out)

            if "ortho_loss" in info:
                total_ortho_loss = total_ortho_loss + info["ortho_loss"]

        # Final norm and head
        x = self.base.ln_f(x)
        logits = self.base.lm_head(x)

        # Compute loss
        loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
            loss = ce_loss + total_ortho_loss + 0.5 * discrim_loss

        info = {
            "sdr": sdr,
            "ortho_loss": total_ortho_loss,
            "discrimination_loss": discrim_loss,
        }

        return logits, loss, info

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Use base model's generate with our forward."""
        for _ in range(max_new_tokens):
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        return idx

    def configure_optimizers(self, **kwargs):
        return self.base.configure_optimizers(**kwargs)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # First test without SDR to verify mHC works
    logger.info("Testing mHC with standard embedding (no SDR)...")

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
        use_sdr=False,  # Start without SDR
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

    data, decode, encode = load_shakespeare()

    base_model = NeuroManifoldGPT(config).to(device)
    model = MHCNeuroManifoldGPT(base_model).to(device)
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

    logger.info(f"Training for {n_iters} iterations with mHC (no SDR)...")
    for i in range(n_iters):
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[j : j + block_size] for j in ix]).to(device)
        y = torch.stack([data[j + 1 : j + block_size + 1] for j in ix]).to(device)

        logits, loss, info = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if i % 500 == 0:
            model.eval()
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            with torch.no_grad():
                generated = model.generate(
                    context, max_new_tokens=50, temperature=0.8, top_k=40
                )
            gen_text = decode(generated[0].tolist())
            char_counts = Counter(generated[0].tolist())
            diversity = len(char_counts)
            logger.info(f"iter {i}: loss={loss.item():.4f}, div={diversity}")
            logger.info(f"  sample='{gen_text[:50]}'")
            model.train()

    logger.info(f"\nFinal loss (no SDR, with mHC): {loss.item():.4f}")

    # Final generation
    model.eval()
    for prompt in ["ROMEO:", "To be or "]:
        prompt_ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
        with torch.no_grad():
            generated = model.generate(
                prompt_ids.clone(), max_new_tokens=80, temperature=0.8, top_k=40
            )
        output = decode(generated[0].tolist())
        logger.info(f"\n'{prompt}': {output[:100]}")


if __name__ == "__main__":
    main()
