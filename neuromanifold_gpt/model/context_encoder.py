"""Context Encoder for Semantic Folding.

This module modulates embeddings based on local context window,
enabling context-aware understanding. For example, "bank" near "money"
should produce different representations than "bank" near "river".

The context encoder uses local attention with a gating mechanism:
1. Compute local attention within context window
2. Gate between original and context-modulated embeddings
3. Output is a blend controlled by learned gate values

This follows the NeuroManifoldGPT architecture where context-aware
embeddings feed into the manifold projection layer.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from neuromanifold_gpt.errors import ConfigurationError

if TYPE_CHECKING:
    from neuromanifold_gpt.config.base import NeuroManifoldConfig


class ContextEncoder(nn.Module):
    """Context-aware embedding encoder using local attention.
    
    Uses a sliding window attention mechanism to incorporate local
    context into token representations. A learned gate controls how
    much context information to blend with the original embeddings.
    
    Args:
        embed_dim: Dimension of input embeddings
        context_size: Half-width of attention window (attends to +-context_size)
        n_heads: Number of attention heads (must divide embed_dim)
        dropout: Dropout probability for attention and gate
        use_layer_norm: Whether to apply layer normalization before attention
    
    Example:
        >>> encoder = ContextEncoder(embed_dim=256, context_size=5)
        >>> x = torch.randn(2, 20, 256)  # batch, seq, embed
        >>> out = encoder(x)  # same shape as input
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        context_size: int = 5, 
        n_heads: int = 4,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        
        if embed_dim % n_heads != 0:
            raise ConfigurationError(
                problem="embed_dim must be divisible by n_heads",
                cause=f"embed_dim={embed_dim} is not divisible by n_heads={n_heads}",
                recovery=f"Set embed_dim to a multiple of {n_heads} (e.g., embed_dim={n_heads * (embed_dim // n_heads + 1)})"
            )
        
        self.context_size = context_size
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm
        
        # Optional layer normalization for training stability
        if use_layer_norm:
            self.ln = nn.LayerNorm(embed_dim)
        
        # Project queries for local attention
        self.context_proj = nn.Linear(embed_dim, embed_dim)
        
        # Multi-head attention for local context
        self.context_attn = nn.MultiheadAttention(
            embed_dim, 
            n_heads, 
            batch_first=True, 
            dropout=dropout,
        )
        
        # Gating mechanism: controls blend between original and context-modulated
        # Input is concatenation of original and context embeddings
        self.context_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Sigmoid(),
        )
        
        # Cache for mask to avoid recomputation
        self._cached_mask: torch.Tensor | None = None
        self._cached_seq_len: int = 0
    
    @classmethod
    def from_config(cls, config: NeuroManifoldConfig) -> ContextEncoder:
        """Create ContextEncoder from NeuroManifoldConfig.
        
        Args:
            config: Configuration object with sdr_embed_dim, sdr_context_size, etc.
            
        Returns:
            Configured ContextEncoder instance
        """
        return cls(
            embed_dim=config.sdr_embed_dim,
            context_size=config.sdr_context_size,
            n_heads=config.n_heads,
            dropout=config.dropout,
            use_layer_norm=True,
        )
    
    def _make_local_mask(
        self, 
        seq_len: int, 
        device: torch.device,
    ) -> torch.Tensor:
        """Create local attention mask (vectorized for efficiency).
        
        For each position i, allows attention to positions in
        [i - context_size, i + context_size] inclusive.
        
        Uses caching to avoid recomputation for same sequence length.
        
        Args:
            seq_len: Sequence length
            device: Device for the mask tensor
            
        Returns:
            Boolean mask of shape (seq_len, seq_len) where True means BLOCKED
        """
        # Disable caching for now to avoid CUDA assertion failures during dynamic generation
        # The caching logic with .to("cpu") seems to cause issues when moving back to device
        # during generation step where T changes every step.
        # Just creating fresh mask is fast enough for small T.
        
        # Create position indices
        positions = torch.arange(seq_len, device=device)
        
        # Compute absolute distance matrix
        # dist[i, j] = |i - j|
        dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
        
        # Positions within context window are NOT masked (False)
        # Positions outside context window ARE masked (True)
        mask = dist > self.context_size
        
        return mask
    
    def forward(self, token_embeds: torch.Tensor) -> torch.Tensor:
        """Apply context-aware encoding.
        
        Args:
            token_embeds: Input embeddings of shape (batch, seq_len, embed_dim)
            
        Returns:
            Context-modulated embeddings of same shape as input
        """
        B, T, D = token_embeds.shape
        
        # Optional layer norm
        if self.use_layer_norm:
            x_normed = self.ln(token_embeds)
        else:
            x_normed = token_embeds
        
        # Create local attention mask
        mask = self._make_local_mask(T, token_embeds.device)
        
        # Project queries
        context_q = self.context_proj(x_normed)
        
        # Apply local attention
        # Keys and values are the (optionally normalized) embeddings
        context_out, _ = self.context_attn(
            context_q,      # queries
            x_normed,       # keys  
            x_normed,       # values
            attn_mask=mask  # local attention mask
        )
        
        # Compute gate: how much to blend context vs original
        gate_input = torch.cat([token_embeds, context_out], dim=-1)
        gate = self.context_gate(gate_input)
        
        # Blend: output = (1 - gate) * original + gate * context
        output = token_embeds * (1 - gate) + context_out * gate
        
        return output
    
    def extra_repr(self) -> str:
        """String representation for print(module)."""
        return (
            f"embed_dim={self.embed_dim}, "
            f"context_size={self.context_size}, "
            f"n_heads={self.n_heads}, "
            f"dropout={self.dropout}, "
            f"use_layer_norm={self.use_layer_norm}"
        )


def _test_context_encoder() -> None:
    """Quick sanity check for the context encoder."""
    from loguru import logger
    
    encoder = ContextEncoder(embed_dim=256, context_size=5)
    x = torch.randn(2, 20, 256)
    out = encoder(x)
    
    logger.info(f"Input shape: {x.shape}")
    logger.info(f"Output shape: {out.shape}")
    
    assert out.shape == x.shape, "Shape mismatch"
    logger.success("Context encoder basic test passed")


if __name__ == "__main__":
    _test_context_encoder()
