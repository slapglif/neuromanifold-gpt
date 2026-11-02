import math
from functools import lru_cache

import torch
import torch.nn as nn


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


@lru_cache(maxsize=None)
def ramanujan_sum(q, n):
    """
    Compute Ramanujan sum c_q(n).
    c_q(n) = sum_{k=1, gcd(k,q)=1}^q cos(2*pi*k*n/q)
    """
    if q == 0:
        return 0
    val = 0.0
    for k in range(1, q + 1):
        if gcd(k, q) == 1:
            val += math.cos(2.0 * math.pi * k * n / q)
    return val


class RamanujanPositionalEmbedding(nn.Module):
    def __init__(self, block_size, embed_dim):
        super().__init__()
        self.block_size = block_size
        self.embed_dim = embed_dim

        # We compute periods q for each dimension
        qs = torch.arange(1, embed_dim + 1)
        self.register_buffer("qs", qs)

        # Precompute positional embeddings for block_size
        self._build_cache(block_size)

    def _build_cache(self, seq_len):
        """Precompute Ramanujan positional embeddings for positions [0, seq_len)"""
        pe = torch.zeros(seq_len, self.embed_dim)
        for i in range(self.embed_dim):
            q = int(self.qs[i])
            for n in range(seq_len):
                pe[n, i] = ramanujan_sum(q, n)

        # Normalize: c_q(n) can be large.
        # phi(q) is the max value. Normalize by sqrt(q) roughly.
        pe = pe / (self.qs.float().unsqueeze(0) ** 0.5)

        self.register_buffer("pe", pe, persistent=False)
        self.cached_seq_len = seq_len

    def forward(self, idx):
        # idx: (B, T)
        T = idx.shape[1]

        # Extend cache if needed
        if T > self.cached_seq_len:
            self._build_cache(T)

        return self.pe[:T, :].unsqueeze(0)
