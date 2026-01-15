# neuromanifold_gpt/model/kan/faster/linear.py
"""
FasterKAN linear layer - drop-in replacement for nn.Linear.

Simplified version with fewer centers for efficiency.
"""

import torch
import torch.nn as nn

from .layer import FasterKANLayer


class FasterKANLinear(nn.Module):
    """Drop-in replacement for nn.Linear using FasterKAN.

    Simplified version with fewer centers for efficiency.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        num_centers: int = 4,  # Fewer centers for speed
    ):
        super().__init__()
        self.kan = FasterKANLayer(
            in_features, out_features,
            num_centers=num_centers,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kan(x)
