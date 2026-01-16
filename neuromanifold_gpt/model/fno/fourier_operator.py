# neuromanifold_gpt/model/fno/fourier_operator.py
"""
Fourier Neural Operator Blocks for NeuroManifoldGPT.

Implements complete FNO layers that combine spectral convolution with local
operations. The FNO architecture learns resolution-invariant mappings between
function spaces via operations in the frequency domain.

FNOBlock architecture (per layer):
    x -> SpectralConv(x) + LocalLinear(x) -> Activation -> LayerNorm -> Output

The spectral path captures global patterns via FFT, while the local path
handles fine-grained features. This combination achieves both global receptive
field and local expressivity.

Key properties:
- Resolution invariance: Same weights work at any sequence length
- Global receptive field: O(N log N) vs O(N^2) for attention
- Physics-aligned: Natural for modeling wave-like phenomena
- Efficient: FFT-based computation is highly optimized

References:
- "Fourier Neural Operator for Parametric PDEs" - Li et al. 2020
- "Neural Operator: Learning Maps Between Function Spaces" - Kovachki et al. 2021
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuromanifold_gpt.model.fno.spectral_conv import SpectralConv1d


@dataclass
class FNOConfig:
    """
    Configuration for Fourier Neural Operator modules.

    Provides default hyperparameters for FNO blocks and encoders.
    """

    embed_dim: int = 384  # Embedding dimension (matches transformer models)
    modes: int = 32  # Number of Fourier modes to retain
    n_layers: int = 4  # Number of FNO blocks in encoder
    activation: str = "gelu"  # Activation function: "gelu", "relu", "silu"
    dropout: float = 0.1  # Dropout probability
    layer_norm_eps: float = 1e-6  # LayerNorm epsilon
    residual: bool = True  # Use residual connections
    prenorm: bool = True  # Pre-norm (True) vs post-norm (False)
    local_expansion: int = 4  # Expansion factor for local MLP
    bias: bool = True  # Include bias in linear layers


def get_activation(name: str) -> nn.Module:
    """
    Get activation function by name.

    Args:
        name: Activation name ("gelu", "relu", "silu", "tanh")

    Returns:
        PyTorch activation module
    """
    activations = {
        "gelu": nn.GELU(),
        "relu": nn.ReLU(),
        "silu": nn.SiLU(),
        "tanh": nn.Tanh(),
    }
    if name.lower() not in activations:
        raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
    return activations[name.lower()]


class FNOBlock(nn.Module):
    """
    Single Fourier Neural Operator Block.

    Combines spectral convolution (global) with local linear projection,
    followed by activation and normalization. This is the core building
    block for FNO-based architectures.

    Architecture:
        Input x (batch, seq_len, embed_dim)
            |
            +-> SpectralConv1d (global, frequency domain)
            |
            +-> Linear (local, pointwise)
            |
            v
        Add + Activation + LayerNorm
            |
            v
        Output (batch, seq_len, embed_dim)

    The spectral path provides global receptive field in O(N log N) time,
    while the local path handles position-specific transformations.

    Example:
        >>> fno = FNOBlock(embed_dim=384, modes=32)
        >>> x = torch.randn(2, 32, 384)  # (batch, seq_len, embed_dim)
        >>> y = fno(x)  # Same shape, globally mixed via FFT
        >>> assert y.shape == x.shape
    """

    def __init__(
        self,
        embed_dim: int,
        modes: int = 32,
        dropout: float = 0.1,
        activation: str = "gelu",
        residual: bool = True,
        prenorm: bool = True,
        layer_norm_eps: float = 1e-6,
        bias: bool = True,
    ):
        """
        Initialize FNO block.

        Args:
            embed_dim: Embedding dimension (input and output)
            modes: Number of Fourier modes to retain (lower = more regularization)
            dropout: Dropout probability for regularization
            activation: Activation function name ("gelu", "relu", "silu")
            residual: Whether to use residual connection
            prenorm: Use pre-normalization (True) vs post-normalization (False)
            layer_norm_eps: Epsilon for layer normalization
            bias: Whether to include bias in linear layers
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.modes = modes
        self.residual = residual
        self.prenorm = prenorm

        # Pre-normalization (if enabled)
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

        # Spectral convolution path (global)
        # Note: SpectralConv1d expects (batch, channels, length)
        # We transpose (batch, seq, dim) -> (batch, dim, seq) before calling
        self.spectral_conv = SpectralConv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            modes=modes,
            bias=False,  # Bias handled separately
            dropout=0.0,  # Dropout applied after combination
        )

        # Local linear path (pointwise)
        self.local_linear = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Output projection (optional, for flexibility)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Activation and regularization
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout)

        # Post-normalization (if not prenorm)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps) if not prenorm else None

    def forward(
        self,
        x: torch.Tensor,
        return_spectral: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through FNO block.

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
            return_spectral: If True, also return spectral component separately

        Returns:
            Output tensor of same shape as input.
            If return_spectral=True, returns tuple (output, spectral_component)
        """
        # Store residual
        residual = x

        # Pre-normalization
        if self.prenorm:
            x = self.norm1(x)

        # Spectral path: transpose for SpectralConv1d (batch, seq, dim) -> (batch, dim, seq)
        x_transposed = x.transpose(-1, -2)  # (batch, embed_dim, seq_len)
        x_spectral = self.spectral_conv(x_transposed)
        x_spectral = x_spectral.transpose(-1, -2)  # Back to (batch, seq_len, embed_dim)

        # Local path: pointwise linear transformation
        x_local = self.local_linear(x)

        # Combine spectral and local paths
        x_combined = x_spectral + x_local

        # Activation
        x_combined = self.activation(x_combined)

        # Output projection
        x_out = self.out_proj(x_combined)

        # Dropout
        x_out = self.dropout(x_out)

        # Residual connection
        if self.residual:
            x_out = x_out + residual

        # Post-normalization (if not prenorm)
        if self.norm2 is not None:
            x_out = self.norm2(x_out)

        if return_spectral:
            return x_out, x_spectral
        return x_out

    def extra_repr(self) -> str:
        """Extra representation for module printing."""
        return (
            f"embed_dim={self.embed_dim}, "
            f"modes={self.modes}, "
            f"residual={self.residual}, "
            f"prenorm={self.prenorm}"
        )


class FNOEncoder(nn.Module):
    """
    Stack of FNO Blocks for Feature Extraction.

    Chains multiple FNOBlock layers to build deep representations.
    Each layer applies spectral + local transformations with residual
    connections for stable gradient flow.

    Architecture:
        Input -> [FNOBlock_1 -> FNOBlock_2 -> ... -> FNOBlock_n] -> Output

    This encoder processes sequences in O(L * N log N) time where L is
    the number of layers and N is sequence length, compared to O(L * N^2)
    for attention-based encoders.

    Use cases:
    - Encoding continuous signals (audio, time series)
    - Processing resolution-varying inputs
    - Learning operators between function spaces
    - Pre-processing for transformer integration

    Example:
        >>> encoder = FNOEncoder(embed_dim=384, n_layers=4, modes=32)
        >>> x = torch.randn(2, 128, 384)  # (batch, seq_len, embed_dim)
        >>> y = encoder(x)  # Same shape, deeply transformed
        >>> assert y.shape == x.shape
    """

    def __init__(
        self,
        embed_dim: int,
        n_layers: int = 4,
        modes: int = 32,
        dropout: float = 0.1,
        activation: str = "gelu",
        residual: bool = True,
        prenorm: bool = True,
        layer_norm_eps: float = 1e-6,
        bias: bool = True,
    ):
        """
        Initialize FNO encoder.

        Args:
            embed_dim: Embedding dimension (preserved throughout)
            n_layers: Number of FNO blocks
            modes: Number of Fourier modes per block
            dropout: Dropout probability
            activation: Activation function name
            residual: Use residual connections in blocks
            prenorm: Use pre-normalization in blocks
            layer_norm_eps: LayerNorm epsilon
            bias: Include bias in linear layers
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.modes = modes

        # Stack of FNO blocks
        self.blocks = nn.ModuleList([
            FNOBlock(
                embed_dim=embed_dim,
                modes=modes,
                dropout=dropout,
                activation=activation,
                residual=residual,
                prenorm=prenorm,
                layer_norm_eps=layer_norm_eps,
                bias=bias,
            )
            for _ in range(n_layers)
        ])

        # Final layer normalization
        self.final_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        return_all_layers: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass through FNO encoder.

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
            return_all_layers: If True, return outputs from all layers

        Returns:
            Output tensor of same shape as input.
            If return_all_layers=True, returns tuple (output, [layer_outputs])
        """
        layer_outputs = [] if return_all_layers else None

        for block in self.blocks:
            x = block(x)
            if return_all_layers:
                layer_outputs.append(x)

        # Final normalization
        x = self.final_norm(x)

        if return_all_layers:
            return x, layer_outputs
        return x

    def extra_repr(self) -> str:
        """Extra representation for module printing."""
        return (
            f"embed_dim={self.embed_dim}, "
            f"n_layers={self.n_layers}, "
            f"modes={self.modes}"
        )


class FNOWithMLP(nn.Module):
    """
    FNO Block with Feed-Forward MLP (Full Transformer-style Block).

    Extends FNOBlock with an additional MLP sub-layer, similar to
    transformer blocks. This provides more expressive power for
    complex feature transformations.

    Architecture:
        Input x
            |
            v
        FNOBlock (spectral + local)
            |
            v
        MLP (expand -> activation -> contract)
            |
            v
        Output

    Example:
        >>> block = FNOWithMLP(embed_dim=384, modes=32, mlp_ratio=4)
        >>> x = torch.randn(2, 32, 384)
        >>> y = block(x)
        >>> assert y.shape == x.shape
    """

    def __init__(
        self,
        embed_dim: int,
        modes: int = 32,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-6,
        bias: bool = True,
    ):
        """
        Initialize FNO block with MLP.

        Args:
            embed_dim: Embedding dimension
            modes: Number of Fourier modes
            mlp_ratio: MLP hidden dimension = embed_dim * mlp_ratio
            dropout: Dropout probability
            activation: Activation function name
            layer_norm_eps: LayerNorm epsilon
            bias: Include bias in linear layers
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.modes = modes
        self.mlp_ratio = mlp_ratio

        # FNO sub-layer
        self.fno_block = FNOBlock(
            embed_dim=embed_dim,
            modes=modes,
            dropout=dropout,
            activation=activation,
            residual=True,
            prenorm=True,
            layer_norm_eps=layer_norm_eps,
            bias=bias,
        )

        # MLP sub-layer (transformer-style FFN)
        mlp_hidden = embed_dim * mlp_ratio
        self.mlp_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden, bias=bias),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)

        Returns:
            Output tensor of same shape
        """
        # FNO sub-layer (with residual inside)
        x = self.fno_block(x)

        # MLP sub-layer with residual
        x = x + self.mlp(self.mlp_norm(x))

        return x

    def extra_repr(self) -> str:
        """Extra representation for module printing."""
        return (
            f"embed_dim={self.embed_dim}, "
            f"modes={self.modes}, "
            f"mlp_ratio={self.mlp_ratio}"
        )
