"""
Multi-Head Latent Attention (MLA) components.

Includes RMSNorm and other MLA-related utilities.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    More efficient than LayerNorm as it doesn't require mean centering.
    Used in many modern architectures (LLaMA, etc.) for ~15% speedup.

    Reference: https://arxiv.org/abs/1910.07467
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """Compute RMS normalization."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """Apply RMS normalization and learned scaling."""
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) - Stub implementation.

    DeepSeek-style KV cache compression using latent projections.

    TODO: Full implementation of latent attention mechanism.
    This is a minimal stub to satisfy imports.
    """
    def __init__(self, embed_dim, n_heads, latent_dim=None, rope_dim=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.latent_dim = latent_dim or (embed_dim // 8)
        self.rope_dim = rope_dim or 8
        self.head_dim = embed_dim // n_heads

        # Minimal projection layers for now
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, spectral_basis=None):
        """
        Stub forward pass - implements standard attention for now.

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
            spectral_basis: Optional spectral basis (unused in stub)

        Returns:
            Output tensor of shape (batch, seq_len, embed_dim)
        """
        B, T, C = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.embed_dim, dim=-1)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Standard scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(self.head_dim, dtype=x.dtype)))
        att = F.softmax(att, dim=-1)
        y = att @ v

        # Reshape back
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        return self.out_proj(y)
