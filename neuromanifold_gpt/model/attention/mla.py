"""
Multi-Head Latent Attention (MLA) components.

Includes RMSNorm and other MLA-related utilities.
"""
import torch
import torch.nn as nn


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
    Multi-Head Latent Attention (MLA) - DeepSeek-style KV cache compression.

    Placeholder implementation for module structure.
    Full implementation will project K/V into a compressed latent space
    before attention, reducing KV cache memory by ~4x.

    Reference: DeepSeek-V2 Technical Report
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 8,
        kv_latent_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.kv_latent_dim = kv_latent_dim

        # Standard attention projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm(embed_dim)

    def forward(self, x: torch.Tensor, spectral_basis: torch.Tensor = None):
        """
        Forward pass with standard attention (latent compression not yet implemented).

        Args:
            x: Input tensor of shape (B, T, embed_dim)
            spectral_basis: Optional spectral basis (unused, for API compatibility)

        Returns:
            Tuple of (output, info) matching other attention module APIs
        """
        B, T, D = x.shape

        # Standard Q, K, V projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Use Flash Attention when available
        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True,
        )

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)

        info = {
            "attention_type": "mla",
            "attn_probs": None,  # Not computed with Flash Attention
        }

        return out, info
