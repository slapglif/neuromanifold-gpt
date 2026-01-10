# neuromanifold_gpt/model/block.py
"""
NeuroManifold Transformer Block.

Combines: SDR -> Manifold -> Spectral -> Soliton Attention -> MLP

This block wires together all components:
1. SDR projection to embedding space
2. ManifoldProjection (SDR -> manifold coordinates + metric)
3. SpectralDecomposition (coords -> eigenvectors)
4. SolitonAttention (attention via wave dynamics)
5. MLP with GELU and residual connections
6. Pre-norm architecture (layer norms before attention/MLP)
"""
import torch
import torch.nn as nn

from .manifold import ManifoldProjection
from .spectral import SpectralDecomposition
from .attention.soliton import SolitonAttention


class NeuroManifoldBlock(nn.Module):
    """Single transformer block with manifold-spectral-soliton attention."""

    def __init__(
        self,
        sdr_size: int,
        embed_dim: int,
        manifold_dim: int = 64,
        n_eigenvectors: int = 32,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()

        # SDR to embedding
        self.sdr_proj = nn.Linear(sdr_size, embed_dim)

        # Manifold + Spectral
        self.manifold = ManifoldProjection(sdr_size, manifold_dim)
        self.spectral = SpectralDecomposition(manifold_dim, n_eigenvectors)

        # Soliton attention
        self.attention = SolitonAttention(embed_dim, n_heads, dropout=dropout)

        # MLP
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, sdr: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Forward pass.

        Args:
            sdr: (B, T, sdr_size)

        Returns:
            out: (B, T, embed_dim)
            info: diagnostic dict
        """
        # Project SDR
        x = self.sdr_proj(sdr)

        # Manifold projection
        coords, metric = self.manifold(sdr)

        # Spectral decomposition
        spectral_basis, spectral_freqs, laplacian = self.spectral(coords, metric)

        # Soliton attention + residual
        attn_out, attn_info = self.attention(self.norm1(x), spectral_basis)
        x = x + attn_out

        # MLP + residual
        x = x + self.mlp(self.norm2(x))

        info = {
            'manifold_coords': coords,
            'metric': metric,
            'spectral_basis': spectral_basis,
            'spectral_freqs': spectral_freqs,
            **attn_info
        }

        return x, info
