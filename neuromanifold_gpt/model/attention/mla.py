# neuromanifold_gpt/model/attention/mla.py
"""
Multi-Head Latent Attention (MLA) from DeepSeek-V2/V3.

KV cache compression via low-dimensional latent space projection.
Key properties:
- 8x memory reduction for KV cache
- Decoupled RoPE for position encoding
- Maintains attention quality while reducing memory

Reference: https://arxiv.org/abs/2405.04434 (DeepSeek-V2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum
import math


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Simpler than LayerNorm (no mean centering, no bias).
    Used in LLaMA, Mistral, DeepSeek models for efficiency.

    Formula: x_norm = x / sqrt(mean(x^2) + eps) * weight
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize using RMS and scale with learned weight.

        Args:
            x: Input tensor of shape (..., dim)

        Returns:
            Normalized tensor of same shape
        """
        # Compute RMS: sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        # Normalize and scale
        x_norm = x / rms
        return self.weight * x_norm


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention with KV compression.

    Architecture:
    1. Q: Standard projection (embed_dim -> embed_dim)
    2. KV: Compressed projection (embed_dim -> latent_dim)
    3. Attention in latent space (8x memory reduction)
    4. Output expansion (latent_dim -> embed_dim)

    Benefits:
    - Reduced KV cache size (critical for long context)
    - Lower memory bandwidth requirements
    - Faster inference with large batch sizes
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 8,
        latent_dim: int = 64,
        rope_dim: int = 32,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.latent_dim = latent_dim
        self.rope_dim = rope_dim

        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        # Validate rope_dim constraint
        assert rope_dim <= self.head_dim, (
            f"rope_dim ({rope_dim}) must be <= head_dim ({self.head_dim}). "
            f"For embed_dim={embed_dim} with n_heads={n_heads}, head_dim={self.head_dim}. "
            f"Try rope_dim={self.head_dim // 2}."
        )

        # Q projection (standard multi-head)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # KV compression to latent space
        # Instead of separate K, V projections of size embed_dim each,
        # we project to a shared latent space of size latent_dim
        self.kv_compress = nn.Linear(embed_dim, latent_dim, bias=bias)

        # Separate K and V extraction from latent (with RoPE dimension)
        # K gets rope_dim for rotary position encoding
        self.k_proj = nn.Linear(latent_dim, n_heads * rope_dim, bias=bias)
        self.v_proj = nn.Linear(latent_dim, n_heads * (self.head_dim - rope_dim), bias=bias)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # RMSNorm for Q and K (prevents attention logit explosion)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(rope_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        use_causal_mask: bool = True,
    ) -> tuple[torch.Tensor, dict]:
        """
        Apply multi-head latent attention.

        Args:
            x: Input tensor of shape (B, T, D)
            use_causal_mask: Whether to apply causal masking for autoregressive generation

        Returns:
            output: Attention output of shape (B, T, D)
            info: Dictionary with attention statistics
        """
        B, T, D = x.shape

        # Q projection: (B, T, D) -> (B, H, T, d_head)
        q = self.q_proj(x)
        q = rearrange(q, "b t (h d) -> b h t d", h=self.n_heads)

        # KV compression: (B, T, D) -> (B, T, latent_dim)
        kv_latent = self.kv_compress(x)

        # Extract K and V from compressed latent
        # K: (B, T, H * rope_dim) -> (B, H, T, rope_dim)
        k_rope = self.k_proj(kv_latent)
        k_rope = rearrange(k_rope, "b t (h d) -> b h t d", h=self.n_heads)

        # V: (B, T, H * (d_head - rope_dim)) -> (B, H, T, d_head - rope_dim)
        v_compressed = self.v_proj(kv_latent)
        v_compressed = rearrange(v_compressed, "b t (h d) -> b h t d", h=self.n_heads)

        # Pad V to full head_dim (fill with zeros for non-RoPE dimensions)
        v = F.pad(v_compressed, (0, self.rope_dim), value=0.0)  # (B, H, T, d_head)

        # Apply RMSNorm to Q and K for stability
        q = self.q_norm(q)

        # For K, only normalize the RoPE dimension part
        k = k_rope  # We'll use k_rope directly after normalization
        k = self.k_norm(k)

        # Pad K to full head_dim for attention computation
        k = F.pad(k, (0, self.head_dim - self.rope_dim), value=0.0)

        # Scaled dot-product attention
        # (B, H, T, d_head) @ (B, H, d_head, T) -> (B, H, T, T)
        attn_weights = einsum(q, k, "b h t d, b h s d -> b h t s")
        attn_weights = attn_weights / math.sqrt(self.head_dim)

        # Apply causal mask if requested
        if use_causal_mask:
            causal_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
            attn_weights = attn_weights.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

        # Softmax and dropout
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Apply attention to values
        # (B, H, T, T) @ (B, H, T, d_head) -> (B, H, T, d_head)
        out = einsum(attn_probs, v, "b h t s, b h s d -> b h t d")

        # Combine heads and project
        out = rearrange(out, "b h t d -> b t (h d)")
        out = self.out_proj(out)
        out = self.dropout(out)

        # Collect info for debugging/analysis
        info = {
            "attn_probs": attn_probs.detach(),
            "attn_entropy": -(attn_probs * torch.log(attn_probs + 1e-9)).sum(dim=-1).mean().item(),
            "kv_compression_ratio": self.embed_dim / self.latent_dim,
        }

        return out, info
