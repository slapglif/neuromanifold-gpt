# neuromanifold_gpt/model/attention/memory_efficient.py
"""
xformers Memory-Efficient Attention.

Wrapper for xformers memory_efficient_attention that provides reduced memory
footprint during attention computation while maintaining quality.
"""

from typing import Optional

import torch
import torch.nn as nn

try:
    from xformers.ops import memory_efficient_attention

    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False
    memory_efficient_attention = None


def xformers_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_bias: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Memory-efficient attention using xformers.

    Args:
        q: Query tensor of shape (B, n_heads, T, head_dim)
        k: Key tensor of shape (B, n_heads, T, head_dim)
        v: Value tensor of shape (B, n_heads, T, head_dim)
        attn_bias: Optional attention bias tensor
        dropout_p: Dropout probability (default: 0.0)
        is_causal: Whether to apply causal masking (default: False)

    Returns:
        Output tensor of shape (B, n_heads, T, head_dim)

    Raises:
        ImportError: If xformers is not installed
    """
    if not XFORMERS_AVAILABLE:
        raise ImportError(
            "xformers is not installed. Install it with: pip install xformers"
        )

    # xformers expects (B, T, n_heads, head_dim) format
    # Convert from (B, n_heads, T, head_dim) to (B, T, n_heads, head_dim)
    B, n_heads, T, head_dim = q.shape
    q = q.transpose(1, 2)  # (B, T, n_heads, head_dim)
    k = k.transpose(1, 2)  # (B, T, n_heads, head_dim)
    v = v.transpose(1, 2)  # (B, T, n_heads, head_dim)

    # Handle causal masking
    if is_causal:
        # Create causal attention bias if needed
        from xformers.ops import LowerTriangularMask

        attn_bias = LowerTriangularMask()

    # Apply memory-efficient attention
    out = memory_efficient_attention(
        q,
        k,
        v,
        attn_bias=attn_bias,
        p=dropout_p,
    )

    # Convert back to (B, n_heads, T, head_dim) format
    out = out.transpose(1, 2)

    return out


class MemoryEfficientAttention(nn.Module):
    """
    Memory-efficient causal self-attention using xformers.

    Uses xformers memory_efficient_attention for reduced memory footprint
    during attention computation while maintaining quality.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = True,
        block_size: int = 1024,
    ):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        if not XFORMERS_AVAILABLE:
            raise ImportError(
                "xformers is not installed. Install it with: pip install xformers"
            )

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.dropout = dropout
        self.block_size = block_size

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        # output projection
        self.c_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # regularization
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, spectral_basis: torch.Tensor = None
    ) -> tuple[torch.Tensor, dict]:
        """
        Forward pass with causal masking using memory-efficient attention.

        Args:
            x: Input tensor of shape (B, T, C) where
               B = batch size, T = sequence length, C = embed_dim
            spectral_basis: Optional spectral basis (unused, for API compatibility)

        Returns:
            Tuple of (output, info) where:
            - output: Attention output of shape (B, T, C)
            - info: Dictionary with attention statistics
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.embed_dim, dim=2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # efficient attention using xformers memory_efficient_attention
        y = xformers_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        # Return info dict matching API
        info = {
            "attention_type": "memory_efficient",
            "attn_probs": None,  # xformers doesn't return attention probs
        }

        return y, info
