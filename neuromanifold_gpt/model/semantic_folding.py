"""Complete Semantic Folding Encoder.

Full pipeline:
1. Token -> Initial embedding
2. Context encoding (word meaning in context)
3. Project to semantic grid (2D topographic map)
4. Apply retina smoothing (local activation spread)
5. Fold into SDR (flatten + sparsify)

The "folding" refers to collapsing rich semantic space into
compressed SDR while preserving similarity structure.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from neuromanifold_gpt.model.sdr_ops import SDROperations
from neuromanifold_gpt.model.semantic_retina import SemanticRetina
from neuromanifold_gpt.model.context_encoder import ContextEncoder


class SemanticFoldingEncoder(nn.Module):
    """Encode tokens as context-aware Sparse Distributed Representations.

    The encoder pipeline:
    1. Embed tokens (vocab_size -> embed_dim)
    2. Apply context encoding (local attention + gating)
    3. Project to 2D semantic grid
    4. Smooth with semantic retina (Gaussian convolution)
    5. Fold to SDR space and sparsify (top-k selection)

    Args:
        vocab_size: Number of tokens in vocabulary
        sdr_size: Size of output SDR (default 2048)
        n_active: Number of active bits in SDR (default 40, ~2% sparsity)
        embed_dim: Internal embedding dimension (default 256)
        context_size: Half-width of context window (default 5)
        grid_size: Size of semantic retina grid (default 64)
    """

    def __init__(
        self,
        vocab_size: int,
        sdr_size: int = 2048,
        n_active: int = 40,
        embed_dim: int = 256,
        context_size: int = 5,
        grid_size: int = 64,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.sdr_size = sdr_size
        self.n_active = n_active
        self.embed_dim = embed_dim

        # Compute grid dimensions for semantic retina
        self.grid_h = int(math.sqrt(sdr_size))
        self.grid_w = sdr_size // self.grid_h

        # 1. Token embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.token_embed.weight, std=0.02)

        # 2. Context encoder - modulates embeddings based on local context
        self.context_encoder = ContextEncoder(
            embed_dim=embed_dim,
            context_size=context_size,
            n_heads=4,
            dropout=0.0,
            use_layer_norm=True,
        )

        # 3. Semantic Retina - topographic smoothing
        self.retina = SemanticRetina(
            grid_size=grid_size,
            n_features=self.grid_h * self.grid_w,
            kernel_size=5,
            sigma=1.0,
        )

        # 4. Project to semantic grid
        self.to_grid = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, self.grid_h * self.grid_w),
        )

        # 5. Fold to SDR space
        self.fold_proj = nn.Linear(self.grid_h * self.grid_w, sdr_size, bias=False)

        # Training helpers
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("bit_duty_cycle", torch.ones(sdr_size) / sdr_size)
        self.boost_strength = 0.1

    def forward(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode tokens to SDRs.

        Args:
            tokens: (B, T) token indices

        Returns:
            sdr: (B, T, sdr_size) binary SDR
            scores: (B, T, sdr_size) raw scores for gradients
        """
        B, T = tokens.shape

        # 1. Embed tokens
        embeds = self.token_embed(tokens)  # (B, T, embed_dim)

        # 2. Context modulation
        context_embeds = self.context_encoder(embeds)  # (B, T, embed_dim)

        # 3. Project to semantic grid
        grid_flat = self.to_grid(context_embeds)  # (B, T, grid_h * grid_w)
        semantic_grid = grid_flat.view(B, T, self.grid_h, self.grid_w)

        # 4. Retina smoothing (topographic Gaussian convolution)
        smoothed = self.retina(semantic_grid)  # (B, T, grid_h, grid_w)

        # 5. Fold to SDR space
        scores = self.fold_proj(smoothed.view(B, T, -1))  # (B, T, sdr_size)

        # Apply boosting for uniform bit usage during training
        if self.training:
            boost = (1.0 / (self.bit_duty_cycle + 1e-6)) ** self.boost_strength
            scores = scores * boost

        # Sparsify to create SDR
        if self.training:
            sdr = SDROperations.soft_topk(
                scores, self.n_active, self.temperature.abs()
            )
            # Update duty cycle exponential moving average
            with torch.no_grad():
                self.bit_duty_cycle = (
                    0.99 * self.bit_duty_cycle + 0.01 * sdr.mean(dim=(0, 1))
                )
        else:
            sdr = SDROperations.hard_topk(scores, self.n_active)

        return sdr, scores

    def semantic_similarity(
        self, sdr_a: torch.Tensor, sdr_b: torch.Tensor
    ) -> torch.Tensor:
        """Compute semantic similarity via normalized overlap.

        Args:
            sdr_a: First SDR tensor
            sdr_b: Second SDR tensor (same shape as sdr_a)

        Returns:
            Similarity score in [0, 1] where 1.0 = identical
        """
        return SDROperations.semantic_similarity(sdr_a, sdr_b, self.n_active)

    def encode_phrase(self, token_sdrs: torch.Tensor) -> torch.Tensor:
        """Encode a sequence of token SDRs into a phrase SDR.

        Unions all token SDRs, then re-sparsifies to n_active bits.

        Args:
            token_sdrs: (B, T, sdr_size) tensor of token SDRs

        Returns:
            (B, sdr_size) phrase SDR with exactly n_active bits
        """
        # Sum along sequence dimension (union-like operation)
        combined = token_sdrs.sum(dim=1)  # (B, sdr_size)
        # Re-sparsify to exactly n_active bits
        return SDROperations.hard_topk(combined, self.n_active)

    def extra_repr(self) -> str:
        """String representation for print(module)."""
        return (
            f"vocab_size={self.vocab_size}, "
            f"sdr_size={self.sdr_size}, "
            f"n_active={self.n_active}, "
            f"embed_dim={self.embed_dim}"
        )
