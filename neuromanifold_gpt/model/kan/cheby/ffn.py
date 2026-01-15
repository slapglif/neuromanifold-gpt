# neuromanifold_gpt/model/kan/cheby/ffn.py
"""
Chebyshev KAN Feed-Forward Network.

FFN layer using ChebyKAN instead of standard linear layers.
"""

import torch
import torch.nn as nn

from .linear import ChebyKANLinear


class ChebyKANFFN(nn.Module):
    """
    Feed-Forward Network using ChebyKAN layers.

    Replaces SwiGLU FFN with KAN-based architecture.
    Structure: ChebyKAN(in, hidden) -> SiLU -> ChebyKAN(hidden, out)
    """

    def __init__(
        self, embed_dim: int, hidden_dim: int, degree: int = 4, dropout: float = 0.0
    ):
        super().__init__()
        self.layer1 = ChebyKANLinear(embed_dim, hidden_dim, degree=degree)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.layer2 = ChebyKANLinear(hidden_dim, embed_dim, degree=degree)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x
