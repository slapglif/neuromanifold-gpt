# neuromanifold_gpt/model/kan/faster/layer.py
"""
FasterKAN layer implementation.

Implements: y = W_base @ x + W_spline @ basis(x)

Where basis(x) uses RSWAF functions centered on a grid.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .basis import RSWAFBasis


class FasterKANLayer(nn.Module):
    """FasterKAN layer using RSWAF basis.

    Implements: y = W_base @ x + W_spline @ basis(x)

    Where basis(x) uses RSWAF functions centered on a grid.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_centers: int = 8,
        h: float = 1.0,
        use_base: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_centers = num_centers
        self.use_base = use_base

        # RSWAF basis
        self.basis = RSWAFBasis(num_centers=num_centers, h=h)

        # Spline weights: (out, in * num_centers)
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features * num_centers) * 0.02
        )

        # Base linear (optional but recommended)
        if use_base:
            self.base_weight = nn.Parameter(
                torch.randn(out_features, in_features) * 0.02
            )
        else:
            self.register_parameter('base_weight', None)

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # LayerNorm for input scaling (key to FasterKAN)
        self.layer_norm = nn.LayerNorm(in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (..., in_features)

        Returns:
            (..., out_features)
        """
        # Normalize input to grid range
        x_norm = self.layer_norm(x)

        # Apply RSWAF basis
        basis_out = self.basis(x_norm)  # (..., in_features, num_centers)

        # Flatten basis output
        basis_flat = rearrange(basis_out, '... i c -> ... (i c)')  # (..., in * num_centers)

        # Spline transformation
        spline_out = F.linear(basis_flat, self.spline_weight)  # (..., out_features)

        # Add base linear (if enabled)
        if self.use_base:
            base_out = F.linear(x, self.base_weight)
            out = spline_out + base_out
        else:
            out = spline_out

        # Add bias
        if self.bias is not None:
            out = out + self.bias

        return out
