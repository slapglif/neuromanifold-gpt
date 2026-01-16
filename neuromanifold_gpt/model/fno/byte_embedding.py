# neuromanifold_gpt/model/fno/byte_embedding.py
"""
Byte-level Embedding Module.

Provides byte-level embedding functionality with optional positional encoding
for sequence modeling. This module is extracted from the multimodal encoder
to support domain-specific organization.

Key features:
- Discrete byte token embedding (vocab_size=256)
- Optional learned position embeddings
- Optional sinusoidal position encoding
- Layer normalization and dropout for stability

This enables byte-level language modeling as the foundation for
multimodal processing in the Neuromanifold GPT architecture.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


def sinusoidal_position_encoding(
    seq_len: int,
    embed_dim: int,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generate sinusoidal position encodings.

    Uses sine/cosine functions at different frequencies to encode position:
    PE(pos, 2i) = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    Args:
        seq_len: Sequence length
        embed_dim: Embedding dimension
        device: Device for output tensor
        dtype: Data type for output tensor

    Returns:
        Position encoding of shape (seq_len, embed_dim)
    """
    if device is None:
        device = torch.device('cpu')

    position = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2, device=device, dtype=dtype)
        * (-math.log(10000.0) / embed_dim)
    )

    pe = torch.zeros(seq_len, embed_dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[:embed_dim // 2])  # Handle odd embed_dim

    return pe


class ByteEmbedding(nn.Module):
    """
    Byte-level embedding with optional positional encoding.

    Embeds discrete byte tokens (0-255) into continuous vectors,
    adding positional information for sequence modeling.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        embed_dim: int = 384,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        use_learned_pos: bool = True,
        use_sinusoidal_pos: bool = True,
    ):
        """
        Initialize byte embedding layer.

        Args:
            vocab_size: Size of vocabulary (256 for bytes)
            embed_dim: Output embedding dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            use_learned_pos: Add learned position embeddings
            use_sinusoidal_pos: Add sinusoidal position encoding
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.use_learned_pos = use_learned_pos
        self.use_sinusoidal_pos = use_sinusoidal_pos

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, embed_dim)

        # Position embeddings
        if use_learned_pos:
            self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        else:
            self.register_parameter('pos_embed', None)

        # Sinusoidal encoding (precomputed buffer)
        if use_sinusoidal_pos:
            self.register_buffer(
                'sinusoidal_pe',
                sinusoidal_position_encoding(max_seq_len, embed_dim),
            )
        else:
            self.register_buffer('sinusoidal_pe', None)

        # Scaling factor for embedding
        self.scale = math.sqrt(embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Layer norm for output stabilization
        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        if self.pos_embed is not None:
            nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Embed byte sequence.

        Args:
            x: Input tensor of shape (B, T) with byte values [0, vocab_size)
            positions: Optional position indices (B, T). If None, uses 0, 1, ..., T-1

        Returns:
            Embedded tensor of shape (B, T, embed_dim)
        """
        B, T = x.shape

        # Token embedding: (B, T) -> (B, T, D)
        embedded = self.token_embed(x) * self.scale

        # Add positional information
        if positions is None:
            positions = torch.arange(T, device=x.device)

        if self.use_learned_pos and self.pos_embed is not None:
            # Clamp positions to valid range
            pos_clamped = positions.clamp(0, self.max_seq_len - 1)
            if positions.dim() == 1:
                pos_clamped = pos_clamped.unsqueeze(0).expand(B, -1)
            embedded = embedded + self.pos_embed(pos_clamped)

        if self.use_sinusoidal_pos and self.sinusoidal_pe is not None:
            # Add sinusoidal encoding
            seq_len = min(T, self.max_seq_len)
            embedded[:, :seq_len] = embedded[:, :seq_len] + self.sinusoidal_pe[:seq_len]

        # Normalize and dropout
        embedded = self.norm(embedded)
        embedded = self.dropout(embedded)

        return embedded

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.vocab_size}, "
            f"embed_dim={self.embed_dim}, "
            f"max_seq_len={self.max_seq_len}"
        )
