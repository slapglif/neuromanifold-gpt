# neuromanifold_gpt/model/kan/wave/ffn.py
"""
WaveKAN-based Feed Forward Network.

Replaces standard MLP or SwiGLU with WaveKAN layers.
"""

import torch
import torch.nn as nn

from .linear import WaveKANLinear


class WaveKANFFN(nn.Module):
    """WaveKAN-based Feed Forward Network.

    Replaces standard MLP or SwiGLU with WaveKAN layers using wavelet activations.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        wavelet_type: str = "mexican_hat",
        dropout: float = 0.0,
        use_base_linear: bool = True,
        use_fast_wavekan: bool = True,  # Default to Fast mode as Full is OOM/slow
    ):
        super().__init__()
        self.layer1 = WaveKANLinear(
            embed_dim,
            hidden_dim,
            wavelet_type=wavelet_type,
            use_base_linear=use_base_linear,
            use_fast_wavekan=use_fast_wavekan,
        )
        self.dropout = nn.Dropout(dropout)
        self.layer2 = WaveKANLinear(
            hidden_dim,
            embed_dim,
            wavelet_type=wavelet_type,
            use_base_linear=use_base_linear,
            use_fast_wavekan=use_fast_wavekan,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x
