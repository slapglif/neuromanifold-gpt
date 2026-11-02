# neuromanifold_gpt/model/fno/spectral_conv.py
"""
Spectral Convolution Layers for Fourier Neural Operators.

Implements the core FNO operation: pointwise multiplication in the frequency domain.
This enables global convolutions in O(N log N) time via the FFT.

The spectral convolution operation:
    y = F^(-1)(W * F(x))
where:
    - F is the Fourier transform (FFT)
    - W are learnable complex weights for each frequency mode
    - F^(-1) is the inverse Fourier transform (IFFT)

Key concepts:
- Mode truncation: Keep only k lowest frequency modes for efficiency and regularization
- Complex weights: Separate real and imaginary parts for stable learning
- Resolution invariance: Same weights work at any discretization level

References:
- "Fourier Neural Operator for Parametric PDEs" - Li et al. 2020
- "Neural Operator: Learning Maps Between Function Spaces" - Kovachki et al. 2021
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SpectralConvConfig:
    """
    Configuration for spectral convolution layers.

    Provides default hyperparameters for FNO spectral convolution.
    """

    in_channels: int = 64
    out_channels: int = 64
    modes: int = 16  # Number of Fourier modes to keep (frequency truncation)
    modes2: int = 16  # Second dimension modes for 2D convolution
    scale: float = 1.0  # Initialization scale factor
    bias: bool = True  # Whether to include bias term
    dropout: float = 0.0  # Dropout probability


class SpectralConv1d(nn.Module):
    """
    1D Spectral Convolution Layer (Core FNO Operation).

    Performs pointwise multiplication of Fourier coefficients by learnable complex
    weights. This is equivalent to a global convolution in physical space but
    computed efficiently via FFT in O(N log N) time.

    The operation:
        1. Transform input to frequency domain: x_hat = FFT(x)
        2. Truncate to first `modes` frequencies (low-pass filtering)
        3. Multiply by learnable complex weights: y_hat = W * x_hat
        4. Transform back to physical domain: y = IFFT(y_hat)

    Key properties:
    - Resolution invariant: Same weights work at any discretization
    - Global receptive field: Each output depends on all inputs
    - Efficient: O(N log N) vs O(N^2) for dense linear layers

    Example:
        >>> sc = SpectralConv1d(in_channels=64, out_channels=64, modes=16)
        >>> x = torch.randn(2, 64, 128)  # (batch, channels, length)
        >>> y = sc(x)  # Same shape, globally mixed via FFT
        >>> assert y.shape == x.shape
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int = 16,
        scale: float = 1.0,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        """
        Initialize 1D spectral convolution layer.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            modes: Number of Fourier modes to retain (lower = more regularization)
            scale: Initialization scale factor (default 1.0)
            bias: Whether to include bias term in output
            dropout: Dropout probability applied after convolution
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale_init = scale

        # Initialize complex weights as separate real/imag for stability
        # Weight shape: (in_channels, out_channels, modes)
        # Using 1/sqrt(in_channels * modes) initialization
        init_scale = scale / (in_channels * modes) ** 0.5

        # Real and imaginary parts of complex weight
        self.weight_real = nn.Parameter(
            init_scale * torch.randn(in_channels, out_channels, modes)
        )
        self.weight_imag = nn.Parameter(
            init_scale * torch.randn(in_channels, out_channels, modes)
        )

        # Optional bias (applied in physical space)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def complex_multiply(
        self,
        x_hat: torch.Tensor,
        weight_real: torch.Tensor,
        weight_imag: torch.Tensor,
    ) -> torch.Tensor:
        """
        Complex multiplication of input Fourier coefficients by complex weights.

        Computes: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i

        Args:
            x_hat: Complex input tensor (batch, in_channels, modes)
            weight_real: Real part of weights (in_channels, out_channels, modes)
            weight_imag: Imaginary part of weights (in_channels, out_channels, modes)

        Returns:
            Complex output tensor (batch, out_channels, modes)
        """
        # Extract real and imaginary parts of input
        x_real = x_hat.real
        x_imag = x_hat.imag

        # Complex multiplication via einsum
        # (batch, in_channels, modes) x (in_channels, out_channels, modes)
        # -> (batch, out_channels, modes)
        out_real = torch.einsum("bim,iom->bom", x_real, weight_real) - torch.einsum(
            "bim,iom->bom", x_imag, weight_imag
        )
        out_imag = torch.einsum("bim,iom->bom", x_real, weight_imag) + torch.einsum(
            "bim,iom->bom", x_imag, weight_real
        )

        # Handle bfloat16 for torch.complex (requires float/double)
        if out_real.dtype == torch.bfloat16:
            return torch.complex(out_real.float(), out_imag.float())

        return torch.complex(out_real, out_imag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: spectral convolution via FFT.

        Args:
            x: Input tensor of shape (batch, channels, length)

        Returns:
            Output tensor of same shape as input
        """
        batch_size = x.shape[0]
        seq_len = x.shape[-1]

        # Ensure we don't exceed Nyquist frequency
        modes = min(self.modes, seq_len // 2 + 1)

        # Forward FFT along last dimension
        x_hat = torch.fft.rfft(x, dim=-1)

        # Truncate to first `modes` frequencies
        x_hat_truncated = x_hat[..., :modes]

        # Complex multiplication with learnable weights
        out_hat = self.complex_multiply(
            x_hat_truncated,
            self.weight_real[..., :modes],
            self.weight_imag[..., :modes],
        )

        # Zero-pad to original frequency count for inverse FFT
        out_hat_padded = torch.zeros(
            batch_size,
            self.out_channels,
            seq_len // 2 + 1,
            dtype=out_hat.dtype,
            device=out_hat.device,
        )
        out_hat_padded[..., :modes] = out_hat

        # Inverse FFT back to physical space
        out = torch.fft.irfft(out_hat_padded, n=seq_len, dim=-1)

        # Apply bias if present
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1)

        # Apply dropout
        out = self.dropout(out)

        return out

    def extra_repr(self) -> str:
        """Extra representation for module printing."""
        return (
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"modes={self.modes}, "
            f"bias={self.bias is not None}"
        )


class SpectralConv2d(nn.Module):
    """
    2D Spectral Convolution Layer for Image/Grid Data.

    Extends SpectralConv1d to 2D spatial domains. Useful for processing:
    - Images (raw pixels or feature maps)
    - 2D physical fields (temperature, pressure, etc.)
    - Grid-structured data

    The operation applies spectral convolution independently along both
    spatial dimensions, mixing information globally across the entire grid.

    Example:
        >>> sc = SpectralConv2d(in_channels=3, out_channels=64, modes1=12, modes2=12)
        >>> x = torch.randn(2, 3, 64, 64)  # (batch, channels, height, width)
        >>> y = sc(x)  # (batch, 64, 64, 64)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int = 12,
        modes2: int = 12,
        scale: float = 1.0,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        """
        Initialize 2D spectral convolution layer.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            modes1: Number of Fourier modes in first spatial dimension
            modes2: Number of Fourier modes in second spatial dimension
            scale: Initialization scale factor
            bias: Whether to include bias term
            dropout: Dropout probability
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale_init = scale

        # Initialize complex weights
        init_scale = scale / (in_channels * modes1 * modes2) ** 0.5

        # Weights for positive frequency quadrant
        self.weight1_real = nn.Parameter(
            init_scale * torch.randn(in_channels, out_channels, modes1, modes2)
        )
        self.weight1_imag = nn.Parameter(
            init_scale * torch.randn(in_channels, out_channels, modes1, modes2)
        )

        # Weights for negative frequency quadrant (first dim negative, second positive)
        self.weight2_real = nn.Parameter(
            init_scale * torch.randn(in_channels, out_channels, modes1, modes2)
        )
        self.weight2_imag = nn.Parameter(
            init_scale * torch.randn(in_channels, out_channels, modes1, modes2)
        )

        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def complex_multiply_2d(
        self,
        x_hat: torch.Tensor,
        weight_real: torch.Tensor,
        weight_imag: torch.Tensor,
    ) -> torch.Tensor:
        """
        2D complex multiplication of Fourier coefficients.

        Args:
            x_hat: Complex input (batch, in_channels, modes1, modes2)
            weight_real: Real weights (in_channels, out_channels, modes1, modes2)
            weight_imag: Imaginary weights (in_channels, out_channels, modes1, modes2)

        Returns:
            Complex output (batch, out_channels, modes1, modes2)
        """
        x_real = x_hat.real
        x_imag = x_hat.imag

        # Complex multiplication via einsum
        out_real = torch.einsum("bijk,iojk->bojk", x_real, weight_real) - torch.einsum(
            "bijk,iojk->bojk", x_imag, weight_imag
        )
        out_imag = torch.einsum("bijk,iojk->bojk", x_real, weight_imag) + torch.einsum(
            "bijk,iojk->bojk", x_imag, weight_real
        )

        return torch.complex(out_real, out_imag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: 2D spectral convolution via FFT2.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Output tensor of shape (batch, out_channels, height, width)
        """
        batch_size = x.shape[0]
        height, width = x.shape[-2], x.shape[-1]

        # Ensure modes don't exceed Nyquist
        modes1 = min(self.modes1, height // 2 + 1)
        modes2 = min(self.modes2, width // 2 + 1)

        # 2D real FFT
        x_hat = torch.fft.rfft2(x, dim=(-2, -1))

        # Get frequency dimensions
        freq_height = x_hat.shape[-2]
        freq_width = x_hat.shape[-1]

        # Initialize output in frequency domain
        out_hat = torch.zeros(
            batch_size,
            self.out_channels,
            freq_height,
            freq_width,
            dtype=x_hat.dtype,
            device=x_hat.device,
        )

        # Process positive frequencies in first dimension (top-left quadrant)
        out_hat[:, :, :modes1, :modes2] = self.complex_multiply_2d(
            x_hat[:, :, :modes1, :modes2],
            self.weight1_real[:, :, :modes1, :modes2],
            self.weight1_imag[:, :, :modes1, :modes2],
        )

        # Process negative frequencies in first dimension (bottom-left quadrant)
        # Note: For rfft2, only the second dimension is half-sized
        if freq_height > modes1:
            out_hat[:, :, -modes1:, :modes2] = self.complex_multiply_2d(
                x_hat[:, :, -modes1:, :modes2],
                self.weight2_real[:, :, :modes1, :modes2],
                self.weight2_imag[:, :, :modes1, :modes2],
            )

        # Inverse FFT2 back to physical space
        out = torch.fft.irfft2(out_hat, s=(height, width), dim=(-2, -1))

        # Apply bias
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)

        # Apply dropout
        out = self.dropout(out)

        return out

    def extra_repr(self) -> str:
        """Extra representation for module printing."""
        return (
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"modes=({self.modes1}, {self.modes2}), "
            f"bias={self.bias is not None}"
        )


class SpectralConvNd(nn.Module):
    """
    N-dimensional Spectral Convolution (Generalized).

    Supports arbitrary spatial dimensions through recursive FFT operations.
    Primarily used as a reference implementation; for production use,
    prefer SpectralConv1d or SpectralConv2d for better performance.

    Note: This implementation is more general but less optimized than
    the specialized 1D and 2D versions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: tuple[int, ...],
        scale: float = 1.0,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        """
        Initialize N-dimensional spectral convolution.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            modes: Tuple of modes for each spatial dimension
            scale: Initialization scale
            bias: Whether to include bias
            dropout: Dropout probability
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.ndim = len(modes)

        # Initialize complex weights
        init_scale = scale / (in_channels * torch.tensor(modes).prod().item()) ** 0.5
        weight_shape = (in_channels, out_channels) + modes

        self.weight_real = nn.Parameter(init_scale * torch.randn(*weight_shape))
        self.weight_imag = nn.Parameter(init_scale * torch.randn(*weight_shape))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for N-dimensional spectral convolution.

        Args:
            x: Input tensor (batch, channels, *spatial_dims)

        Returns:
            Output tensor of same spatial shape
        """
        x.shape[0]
        spatial_shape = x.shape[2:]

        # N-dimensional real FFT
        x_hat = torch.fft.rfftn(x, dim=tuple(range(2, 2 + self.ndim)))

        # Truncate to modes
        slices = [slice(None), slice(None)]  # batch, channels
        for i, m in enumerate(self.modes):
            max_mode = min(
                m, spatial_shape[i] // 2 + 1 if i == self.ndim - 1 else spatial_shape[i]
            )
            slices.append(slice(0, max_mode))

        x_truncated = x_hat[tuple(slices)]

        # Complex multiplication
        x_real = x_truncated.real
        x_imag = x_truncated.imag

        # Adjust weight slices for truncated modes
        weight_slices = [slice(None), slice(None)]
        for i in range(self.ndim):
            weight_slices.append(slice(0, x_truncated.shape[2 + i]))

        w_real = self.weight_real[tuple(weight_slices)]
        w_imag = self.weight_imag[tuple(weight_slices)]

        # Build einsum string dynamically
        in_dims = "".join([chr(ord("j") + i) for i in range(self.ndim)])
        out_dims = in_dims
        ein_str = f"bi{in_dims},io{in_dims}->bo{out_dims}"

        out_real = torch.einsum(ein_str, x_real, w_real) - torch.einsum(
            ein_str, x_imag, w_imag
        )
        out_imag = torch.einsum(ein_str, x_real, w_imag) + torch.einsum(
            ein_str, x_imag, w_real
        )

        out_hat_truncated = torch.complex(out_real, out_imag)

        # Pad back to full frequency size
        freq_shape = list(x_hat.shape)
        freq_shape[1] = self.out_channels
        out_hat = torch.zeros(*freq_shape, dtype=x_hat.dtype, device=x_hat.device)

        out_slices = [slice(None), slice(None)]
        for i in range(self.ndim):
            out_slices.append(slice(0, out_hat_truncated.shape[2 + i]))
        out_hat[tuple(out_slices)] = out_hat_truncated

        # Inverse FFT
        out = torch.fft.irfftn(
            out_hat, s=spatial_shape, dim=tuple(range(2, 2 + self.ndim))
        )

        # Bias and dropout
        if self.bias is not None:
            bias_shape = [1, -1] + [1] * self.ndim
            out = out + self.bias.view(*bias_shape)

        out = self.dropout(out)

        return out


# Convenience aliases matching common FNO paper naming conventions
SpectralConv = SpectralConv1d
