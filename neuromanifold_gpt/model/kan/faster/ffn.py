# neuromanifold_gpt/model/kan/faster/ffn.py
"""
FasterKAN-based Feed Forward Network.

Replaces standard MLP or SwiGLU with FasterKAN layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer import FasterKANLayer


class FasterKANFFN(nn.Module):
    """FasterKAN-based Feed Forward Network.

    Replaces standard MLP or SwiGLU with FasterKAN layers.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_centers: int = 8,
        h: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.layer1 = FasterKANLayer(dim, hidden_dim, num_centers=num_centers, h=h)
        self.layer2 = FasterKANLayer(hidden_dim, dim, num_centers=num_centers, h=h)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = F.gelu(x)  # Activation between layers
        x = self.dropout(x)
        x = self.layer2(x)
        return x
