# neuromanifold_gpt/model/fno/position_encoding.py
"""
Position Encoding Utilities for Sequence Modeling.

Provides position encoding functions for adding positional information
to sequence representations in neural networks.

Key features:
- Sinusoidal position encoding (Vaswani et al. 2017)
- Device and dtype flexibility for mixed-precision training
- Handles arbitrary sequence lengths and embedding dimensions

Reference:
- Vaswani et al. "Attention Is All You Need" (2017)
"""

import math
import torch


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
