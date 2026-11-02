import math

import torch
import torch.nn as nn


class ALiBiPositionalBias(nn.Module):
    """
    Attention with Linear Biases (ALiBi) from "Train Short, Test Long:
    Attention with Linear Biases Enables Input Length Extrapolation"
    https://arxiv.org/abs/2108.12409

    ALiBi adds position information by biasing attention scores with a linear penalty
    that increases with distance between tokens. Each attention head uses a different slope,
    allowing the model to learn different notions of locality per head.

    Key properties:
    - No learned position embeddings required
    - Enables better extrapolation to longer sequences than seen during training
    - Bias is proportional to distance: bias[i,j] = -slope * |i - j|
    """

    def __init__(self, n_heads, embed_dim, max_seq_len=8192):
        """
        Args:
            n_heads: Number of attention heads
            embed_dim: Total embedding dimension (kept for API compatibility)
            max_seq_len: Maximum sequence length to precompute biases for
        """
        super().__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Compute slopes for each head: m_h = 2^(-8h/n_heads) for h in [1, n_heads]
        slopes = self._get_slopes(n_heads)
        self.register_buffer("slopes", slopes)

        # Precompute bias matrix for max_seq_len
        self._build_cache(max_seq_len)

    def _get_slopes(self, n_heads):
        """
        Compute the slopes for each attention head.

        Following the ALiBi paper, slopes are geometric: m_h = 2^(-8h/n)
        For non-power-of-2 heads, we interpolate between adjacent powers of 2.
        """

        def get_slopes_power_of_2(n):
            # For n heads (power of 2): slopes are 2^(-8/n), 2^(-16/n), ..., 2^(-8)
            start = 2 ** (-8 / n)
            ratio = start
            return torch.tensor([start * (ratio**i) for i in range(n)])

        # Check if n_heads is a power of 2
        if math.log2(n_heads).is_integer():
            return get_slopes_power_of_2(n_heads)
        else:
            # For non-power-of-2, interpolate using closest power of 2
            closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
            # Get slopes for closest power of 2 and double
            slopes_a = get_slopes_power_of_2(closest_power_of_2)
            slopes_b = get_slopes_power_of_2(2 * closest_power_of_2)[::2]
            # Concatenate to get n_heads slopes
            remaining = n_heads - closest_power_of_2
            all_slopes = torch.cat([slopes_a, slopes_b[:remaining]])
            return torch.sort(all_slopes, descending=True).values

    def _build_cache(self, seq_len):
        """
        Precompute bias matrix for positions [0, seq_len).

        Bias matrix has shape [n_heads, 1, seq_len, seq_len] where
        bias[h, 0, i, j] = -slopes[h] * |i - j|
        """
        # Create position indices: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, dtype=self.slopes.dtype)

        # Compute pairwise distance matrix: |i - j| for all position pairs
        # positions[:, None] broadcasts to [seq_len, 1]
        # positions[None, :] broadcasts to [1, seq_len]
        # Result shape: [seq_len, seq_len]
        distances = torch.abs(positions[:, None] - positions[None, :])

        # Apply per-head slopes to distance matrix
        # slopes: [n_heads] -> [n_heads, 1, 1]
        # distances: [seq_len, seq_len] -> [1, seq_len, seq_len]
        # bias: [n_heads, seq_len, seq_len]
        bias = -self.slopes[:, None, None] * distances[None, :, :]

        # Add batch dimension for broadcasting with attention scores
        # Final shape: [1, n_heads, seq_len, seq_len]
        self.register_buffer("bias_cached", bias.unsqueeze(0), persistent=False)
        self.cached_seq_len = seq_len

    def forward(self, seq_len):
        """
        Get ALiBi bias for a given sequence length.

        Args:
            seq_len: Sequence length (int)

        Returns:
            bias: Bias tensor of shape (n_heads, 1, seq_len, seq_len)
                  to be added to attention scores before softmax
        """
        # Extend cache if sequence is longer than precomputed
        if seq_len > self.cached_seq_len:
            self._build_cache(seq_len)

        # Return cached bias for current sequence length
        # Slice to get exactly seq_len x seq_len
        return self.bias_cached[:, :, :seq_len, :seq_len]
