# neuromanifold_gpt/model/kan/faster/basis.py
"""
RSWAF (Reflectional Switch Activation Function) basis functions.

RSWAF: b_i(u) = 1 - tanh((u - u_i) / h)^2

This approximates 3rd-order B-splines while being computationally efficient.
Used by FasterKAN for ~1.5x speedup over FastKAN.
"""

import torch
import torch.nn as nn


class RSWAFBasis(nn.Module):
    """RSWAF basis function layer.

    RSWAF: b_i(u) = 1 - tanh((u - u_i) / h)^2

    Custom gradient optimization: derivative shares computation with forward.
    d/du[1 - tanh(x)^2] = -2 * tanh(x) * sech(x)^2 = -2 * tanh(x) * (1 - tanh(x)^2)
    """

    def __init__(
        self,
        num_centers: int = 8,
        h: float = 1.0,
        grid_range: tuple = (-1.0, 1.0),
        learnable_h: bool = False,
        learnable_grid: bool = False,
    ):
        super().__init__()
        self.num_centers = num_centers
        self.grid_range = grid_range

        # Initialize grid centers uniformly
        grid = torch.linspace(grid_range[0], grid_range[1], num_centers)
        if learnable_grid:
            self.grid = nn.Parameter(grid)
        else:
            self.register_buffer('grid', grid)

        # Bandwidth parameter
        if learnable_h:
            self.h = nn.Parameter(torch.tensor(h))
        else:
            self.register_buffer('h', torch.tensor(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RSWAF basis functions.

        Args:
            x: (..., in_features) input tensor

        Returns:
            (..., in_features, num_centers) basis function outputs
        """
        # x: (..., in_features)
        # Expand for broadcasting: (..., in_features, 1)
        x_expanded = x.unsqueeze(-1)

        # Compute (u - u_i) / h for each center
        # grid: (num_centers,) -> (1, num_centers)
        diff = (x_expanded - self.grid) / self.h  # (..., in_features, num_centers)

        # RSWAF: 1 - tanh(diff)^2
        tanh_diff = torch.tanh(diff)
        basis = 1.0 - tanh_diff ** 2

        return basis
