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
import torch.nn.functional as F

from neuromanifold_gpt.model.context_encoder import ContextEncoder
from neuromanifold_gpt.model.sdr_ops import SDROperations
from neuromanifold_gpt.model.semantic_retina import SemanticRetina


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

    def token_discrimination_loss(
        self, tokens: torch.Tensor, sdr: torch.Tensor, n_samples: int = 64
    ) -> torch.Tensor:
        """Ensure different tokens produce sufficiently different SDRs.

        This prevents mode collapse where all tokens map to similar SDRs.

        Args:
            tokens: (B, T) token indices
            sdr: (B, T, sdr_size) SDR representations
            n_samples: Number of pairs to sample for efficiency

        Returns:
            Scalar loss penalizing SDR collapse
        """
        B, T = tokens.shape
        device = tokens.device

        # Flatten for sampling
        tokens_flat = tokens.view(-1)  # (B*T,)
        sdr_flat = sdr.view(-1, sdr.size(-1))  # (B*T, sdr_size)
        N = tokens_flat.size(0)

        if N < 2:
            return torch.tensor(0.0, device=device)

        # Sample random pairs
        n_pairs = min(n_samples, N * (N - 1) // 2)
        idx1 = torch.randint(N, (n_pairs,), device=device)
        idx2 = torch.randint(N, (n_pairs,), device=device)

        # Get tokens and SDRs for pairs
        tok1 = tokens_flat[idx1]
        tok2 = tokens_flat[idx2]
        sdr1 = sdr_flat[idx1]
        sdr2 = sdr_flat[idx2]

        # Compute SDR overlap (similarity)
        overlap = (sdr1 * sdr2).sum(dim=-1) / self.n_active  # [0, 1]

        # Different tokens should have low overlap
        # Same tokens can have high overlap (context variation)
        different_tokens = (tok1 != tok2).float()

        # Loss: penalize high overlap for different tokens
        # Target overlap for different tokens: ~0.1 (some overlap is OK for semantically similar)
        target_overlap = 0.1
        loss = different_tokens * F.relu(overlap - target_overlap)

        return loss.mean()

    def topographic_loss(
        self, embeds: torch.Tensor, grid_activations: torch.Tensor
    ) -> torch.Tensor:
        """Compute O(N) topographic loss using temporal neighbors.

        Instead of all-to-all O(N^2), we enforce that:
        Similarity(t, t+1) in embedding space ~= Similarity(t, t+1) in grid space.
        This ensures the 'path' of thought is continuous on the manifold.
        """
        B, T = embeds.shape[:2]
        if T < 2:
            return torch.tensor(0.0, device=embeds.device)

        # 1. Compute grid centroids per token (Same as before)
        # grid_activations: (B, T, sdr_size)
        H, W = self.grid_h, self.grid_w
        used_size = H * W
        scores_grid = grid_activations[..., :used_size].view(B, T, H, W)

        # Centroids
        y_coords = torch.arange(H, device=embeds.device).float()
        x_coords = torch.arange(W, device=embeds.device).float()

        scores_flat = scores_grid.flatten(start_dim=-2)
        weights_flat = F.softmax(scores_flat, dim=-1)
        weights = weights_flat.view(B, T, H, W)

        cy = (weights.sum(dim=-1) * y_coords.view(1, 1, -1)).sum(dim=-1)  # (B, T)
        cx = (weights.sum(dim=-2) * x_coords.view(1, 1, -1)).sum(dim=-1)  # (B, T)

        # Normalize
        cy = cy / (H - 1)
        cx = cx / (W - 1)

        # 2. Compute Temporal Differences (t vs t+1)
        # Embeddings
        embeds_norm = F.normalize(embeds, p=2, dim=-1)  # (B, T, D)
        # Cosine sim between t and t+1
        # (B, T-1, D) * (B, T-1, D) -> sum -> (B, T-1)
        sim_sem = (embeds_norm[:, :-1] * embeds_norm[:, 1:]).sum(dim=-1)
        sim_sem = (sim_sem + 1.0) / 2.0  # [0, 1]

        # Spatial Distance between t and t+1
        dy = cy[:, 1:] - cy[:, :-1]
        dx = cx[:, 1:] - cx[:, :-1]
        dist_spatial = (
            dy**2 + dx**2 + 1e-8
        ).sqrt()  # (B, T-1) - eps prevents NaN grad at sqrt(0)

        # RBF Similarity
        sigma = 0.2
        sim_spatial = torch.exp(-(dist_spatial**2) / (2 * sigma**2))

        # Loss: Match transition probabilities
        loss = (sim_sem - sim_spatial) ** 2

        return loss.mean()

    def forward(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode tokens to SDRs.

        Args:
            tokens: (B, T) token indices

        Returns:
            sdr: (B, T, sdr_size) binary SDR
            scores: (B, T, sdr_size) raw scores for gradients
            topographic_loss: Scalar loss for topographic organization
            discrimination_loss: Scalar loss preventing SDR collapse
            contrastive_loss: Scalar loss for contrastive learning (placeholder)
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
            sdr = SDROperations.soft_topk(scores, self.n_active, self.temperature.abs())
            # Update duty cycle exponential moving average
            with torch.no_grad():
                self.bit_duty_cycle = 0.99 * self.bit_duty_cycle + 0.01 * sdr.mean(
                    dim=(0, 1)
                )
        else:
            sdr = SDROperations.hard_topk(scores, self.n_active)

        # Compute losses during training
        topo_loss = torch.tensor(0.0, device=tokens.device)
        discrim_loss = torch.tensor(0.0, device=tokens.device)
        contrastive_loss = torch.tensor(
            0.0, device=tokens.device
        )  # Placeholder for future implementation
        if self.training:
            topo_loss = self.topographic_loss(context_embeds, scores)
            discrim_loss = self.token_discrimination_loss(tokens, sdr)

        return sdr, scores, topo_loss, discrim_loss, contrastive_loss

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
