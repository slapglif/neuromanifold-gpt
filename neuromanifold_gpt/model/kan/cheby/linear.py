# neuromanifold_gpt/model/kan/cheby/linear.py
"""
Chebyshev KAN Linear Layer.

Linear layer with Chebyshev polynomial basis functions.
"""

import torch
import torch.nn as nn


class ChebyKANLinear(nn.Module):
    """
    Linear layer with Chebyshev polynomial basis functions.

    Instead of y = xW + b, we compute:
    y = sum_{i=0}^{degree} T_i(x) W_i + b

    Where T_i(x) are Chebyshev polynomials of the first kind.
    """

    def __init__(
        self, in_features: int, out_features: int, degree: int = 4, bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree

        # Polynomial coefficients: (degree + 1, in_features, out_features)
        # This allows each input-output pair to have its own polynomial activation
        # Init: std=1/(input_dim * (degree+1)) from production ChebyKAN repos
        self.coeffs = nn.Parameter(torch.empty(degree + 1, in_features, out_features))
        nn.init.normal_(self.coeffs, mean=0.0, std=1.0 / (in_features * (degree + 1)))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Normalization to keep inputs in [-1, 1] range for Chebyshev stability
        # CRITICAL: Use LayerNorm over features (not InstanceNorm over time) to preserve causality
        # InstanceNorm1d normalizes across T, violating autoregressive causality
        self.layer_norm = nn.LayerNorm(in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, in_features)

        Returns:
            out: (B, T, out_features)
        """
        # Normalize inputs to [-1, 1] for Chebyshev stability
        # LayerNorm over features preserves causality (each position independent)
        x_norm = torch.tanh(self.layer_norm(x))

        # Compute Chebyshev polynomials recursively
        # T_0(x) = 1
        # T_1(x) = x
        # T_{n+1}(x) = 2xT_n(x) - T_{n-1}(x)

        cheby_polys = []
        # T_0
        cheby_polys.append(torch.ones_like(x_norm))
        # T_1
        cheby_polys.append(x_norm)

        for i in range(2, self.degree + 1):
            t_next = 2 * x_norm * cheby_polys[-1] - cheby_polys[-2]
            cheby_polys.append(t_next)

        # Stack polynomials: (degree+1, B, T, D)
        poly_stack = torch.stack(cheby_polys, dim=0)

        # Compute output: sum_i T_i(x) W_i
        # poly: (deg+1, B, T, in)
        # coeffs: (deg+1, in, out)
        # contract over deg and in dimensions

        # Use einsum for efficient contraction
        # d: degree, b: batch, t: time, i: in_features, o: out_features
        y = torch.einsum("dbti,dio->bto", poly_stack, self.coeffs)

        if self.bias is not None:
            y = y + self.bias

        return y
