# neuromanifold_gpt/model/attention/standard.py
"""
Standard Causal Self-Attention from nanoGPT.

Reference implementation of scaled dot-product attention with causal masking.
Adapted from Andrej Karpathy's nanoGPT.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class StandardAttention(nn.Module):
    """
    Standard causal self-attention mechanism.

    Implements scaled dot-product attention with causal masking:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Uses Flash Attention (PyTorch >= 2.0) when available for efficiency.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = True,
        block_size: int = 1024,
        pos_emb_type: str = "learned",  # Position embedding type
        max_seq_len: int = 1024,  # For RoPE/ALiBi initialization
    ):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.dropout = dropout
        self.block_size = block_size
        self.pos_emb_type = pos_emb_type

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        # output projection
        self.c_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # (always create this since we may need manual attention even with flash, e.g., for ALiBi)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )

        # RoPE and ALiBi position embeddings (lazy initialization)
        self.rope = None
        self.alibi = None
        self.max_seq_len = max_seq_len  # Store for lazy init

    def forward(self, x: torch.Tensor, spectral_basis: torch.Tensor = None) -> tuple[torch.Tensor, dict]:
        """
        Forward pass with causal masking.

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

        # Lazy initialization of position embeddings (only when first used)
        if self.pos_emb_type == 'rotary' and self.rope is None:
            from neuromanifold_gpt.model.embeddings import RotaryPositionalEmbedding
            self.rope = RotaryPositionalEmbedding(
                embed_dim=self.embed_dim,
                head_dim=self.head_dim,
                max_seq_len=self.max_seq_len
            )
        elif self.pos_emb_type == 'alibi' and self.alibi is None:
            from neuromanifold_gpt.model.embeddings import ALiBiPositionalBias
            self.alibi = ALiBiPositionalBias(
                n_heads=self.n_heads,
                embed_dim=self.embed_dim,
                max_seq_len=self.max_seq_len
            )

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.embed_dim, dim=2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)  # (B, nh, T, hs)

        # Apply RoPE if enabled
        if self.rope is not None:
            q, k = self.rope(q, k)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        attn_probs = None
        # Flash attention cannot be used with ALiBi (custom bias not supported)
        if self.flash and self.alibi is None:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

            # Add ALiBi bias if enabled (before causal mask)
            if self.alibi is not None:
                alibi_bias = self.alibi(T)  # Shape: (1, n_heads, T, T)
                # Expand to batch size and add to attention scores
                att = att + alibi_bias.squeeze(0)  # Now shape: (B, nh, T, T)

            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            attn_probs = att
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        # Return info dict matching FHNAttention API
        info = {
            "attention_type": "standard",
            "attn_probs": attn_probs,  # None when using flash attention
        }

        return y, info
