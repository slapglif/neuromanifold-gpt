import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) from "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    https://arxiv.org/abs/2104.09864

    RoPE encodes position information by rotating query and key vectors in alternating dimensions.
    Each consecutive pair of dimensions is treated as a 2D plane and rotated by an angle that
    depends on the position and a frequency that decreases with dimension index.
    """

    def __init__(self, embed_dim, head_dim, max_seq_len=8192, base=10000):
        """
        Args:
            embed_dim: Total embedding dimension (not used in forward, kept for API compatibility)
            head_dim: Dimension per attention head (must be even)
            max_seq_len: Maximum sequence length to precompute
            base: Base for computing rotation frequencies (default 10000 from paper)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")

        # Compute rotation frequencies for each dimension pair
        # θ_i = base^(-2i/head_dim) for i in [0, head_dim/2)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos and sin for positions [0, max_seq_len)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        """Precompute cos and sin values for positions [0, seq_len)"""
        # positions: [seq_len]
        positions = torch.arange(seq_len, dtype=self.inv_freq.dtype)

        # freqs: [seq_len, head_dim/2]
        freqs = torch.outer(positions, self.inv_freq)

        # emb: [seq_len, head_dim] where each pair (2i, 2i+1) has (cos, sin) or (sin, cos)
        # CORRECT: Each dimension pair must share the same frequency for proper 2D rotation
        # This creates pattern [freq0, freq0, freq1, freq1, ...] instead of [freq0, freq1, ...]
        emb = torch.repeat_interleave(freqs, 2, dim=-1)

        # Register cos and sin caches
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.cached_seq_len = seq_len

    def _rotate_half(self, x):
        """
        Rotate half the hidden dims of the input.

        For input [..., d0, d1, d2, d3, ...], returns [..., -d1, d0, -d3, d2, ...]
        This implements the rotation by swapping and negating pairs.
        """
        x1 = x[..., ::2]  # Even indices: 0, 2, 4, ...
        x2 = x[..., 1::2]  # Odd indices: 1, 3, 5, ...
        # Stack and reshape to interleave: -x2, x1, -x2, x1, ...
        return torch.stack([-x2, x1], dim=-1).flatten(-2)

    def forward(self, q, k):
        """
        Apply rotary position embeddings to query and key tensors.

        Args:
            q: Query tensor of shape (batch, n_heads, seq_len, head_dim)
            k: Key tensor of shape (batch, n_heads, seq_len, head_dim)

        Returns:
            q_rot: Rotated query tensor of same shape as q
            k_rot: Rotated key tensor of same shape as k
        """
        seq_len = q.shape[2]

        # Extend cache if needed
        if seq_len > self.cached_seq_len:
            self._build_cache(seq_len)

        # Get cos and sin for current sequence length
        # Shape: [seq_len, head_dim]
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]

        # Reshape for broadcasting: [1, 1, seq_len, head_dim]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Apply rotation: x_rot = x * cos + rotate_half(x) * sin
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin

        return q_rot, k_rot
