"""
Multi-Head Latent Attention (MLA) components.

Includes RMSNorm and other MLA-related utilities.
"""
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    More efficient than LayerNorm as it doesn't require mean centering.
    Used in many modern architectures (LLaMA, etc.) for ~15% speedup.

    Reference: https://arxiv.org/abs/1910.07467
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """Compute RMS normalization."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """Apply RMS normalization and learned scaling."""
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
