# neuromanifold_gpt/model/kan/faster.py
"""
FasterKAN implementation using RSWAF basis functions.

Based on: https://github.com/AthanasiosDelis/faster-kan

RSWAF (Reflectional Switch Activation Function):
    b_i(u) = 1 - tanh((u - u_i) / h)^2

This approximates 3rd-order B-splines while being computationally efficient.
~1.5x faster than FastKAN, ~2x slower than MLP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


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


def replace_linear_with_fasterkan(
    module: nn.Module,
    num_centers: int = 4,
    skip_names: set[str] | None = None,
    skip_module_types: tuple | None = None,
    _prefix: str = "",
) -> nn.Module:
    """Recursively replace nn.Linear with FasterKANLinear.

    Args:
        module: PyTorch module to modify
        num_centers: Number of RSWAF centers
        skip_names: Set of full path names to skip (e.g., {"lm_head", "token_embedding"})
        skip_module_types: Tuple of module types to skip recursing into (e.g., nn.MultiheadAttention)
        _prefix: Internal prefix for tracking full path

    Returns:
        Modified module with FasterKAN layers
    """
    if skip_names is None:
        skip_names = set()
    if skip_module_types is None:
        # Skip nn.MultiheadAttention - it directly accesses out_proj.weight
        skip_module_types = (nn.MultiheadAttention,)

    for name, child in module.named_children():
        full_name = f"{_prefix}.{name}" if _prefix else name

        # Skip certain module types entirely (they access Linear internals)
        if isinstance(child, skip_module_types):
            continue

        if isinstance(child, nn.Linear):
            # Skip certain layers (output head, embeddings)
            if name in skip_names or full_name in skip_names:
                continue
            # Replace with FasterKAN
            kan_linear = FasterKANLinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                num_centers=num_centers
            )
            setattr(module, name, kan_linear)
        else:
            # Recurse
            replace_linear_with_fasterkan(
                child,
                num_centers=num_centers,
                skip_names=skip_names,
                skip_module_types=skip_module_types,
                _prefix=full_name,
            )

    return module
