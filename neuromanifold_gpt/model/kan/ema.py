# neuromanifold_gpt/model/kan/ema.py
"""
Efficient Exponential Moving Average (EMA) implementations using parallel scan.

Uses FFT-based convolution for O(T log T) complexity instead of O(T²) matrix operations.
Based on the insight that EMA is a discrete convolution: h[t] = Σ_k x[k]·α·(1-α)^(t-k)

Classes:
    - DampedEMA: Base single-channel EMA using parallel scan
    - MultiHeadDampedEMA: Multi-head variant with vectorized head processing
    - CEMA: Causal EMA wrapper with explicit dimension parameter

Example:
    >>> ema = DampedEMA(alpha=0.9)
    >>> x = torch.randn(2, 10, 8)
    >>> h = ema(x)
    >>> h.shape
    torch.Size([2, 10, 8])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union


def ema_fft(
    x: torch.Tensor,
    alpha: Union[float, torch.Tensor],
    eps: float = 1e-6
) -> torch.Tensor:
    """
    FFT-based parallel EMA computation.

    Computes: h[t] = α·x[t] + (1-α)·h[t-1] using FFT convolution.

    Algorithm:
        1. Construct exponential decay kernel: w[τ] = α·(1-α)^τ
        2. Compute h = x ⊗ w using FFT convolution
        3. FFT(x) * FFT(w) → IFFT → h

    Args:
        x: Input tensor - shape (..., T, ...) where T is time dimension (dim=1)
           Commonly (B, T, D) or (B, T, H, D)
        alpha: Smoothing factor
            - float: Scalar damping (same for all channels)
            - Tensor: Per-channel damping (shape must broadcast with x)
        eps: Numerical stability epsilon

    Returns:
        h: EMA output (same shape as x)

    Complexity:
        Time: O(T log T) per channel
        Space: O(T) per channel

    Example:
        >>> x = torch.randn(2, 10, 8)
        >>> h = ema_fft(x, alpha=0.9)
        >>> h.shape
        torch.Size([2, 10, 8])
    """
    # Get dimensions (assume time is dim=1)
    shape = x.shape
    T = shape[1]
    device = x.device
    dtype = x.dtype

    # Convert alpha to tensor if needed
    if isinstance(alpha, float):
        alpha = torch.tensor(alpha, device=device, dtype=dtype)

    # Edge case: alpha = 1.0 (no smoothing, identity)
    if (alpha >= 1.0 - eps).all():
        return x

    # Edge case: alpha near 0 (full smoothing, degenerate)
    if (alpha <= eps).any():
        raise ValueError(f"alpha must be > {eps}, got {alpha}")

    # Edge case: single timestep
    if T == 1:
        return alpha * x

    # Step 1: Construct exponential decay kernel
    # w[τ] = α·(1-α)^τ for τ = 0, 1, 2, ..., T-1
    decay = 1.0 - alpha
    powers = torch.arange(T, device=device, dtype=dtype)

    # Handle per-channel alpha (broadcast powers to match alpha shape)
    if alpha.numel() > 1:
        # alpha: (D,) or (H, D) -> powers: (T,) -> need to broadcast
        # kernel: (T, ...) matching trailing dims of alpha
        powers_expanded = powers.view(-1, *([1] * (alpha.ndim)))  # (T, 1, 1, ...)
        kernel = alpha * (decay ** powers_expanded)  # Broadcasting
    else:
        # Scalar alpha
        kernel = alpha * (decay ** powers)  # (T,)

    # Step 2: Pad to 2T for linear (non-circular) convolution
    n_fft = 2 * T

    # Step 3: Forward FFT of input (use rfft for real signals)
    # FFT operates on dim=1 (time dimension)
    x_f = torch.fft.rfft(x, n=n_fft, dim=1)  # (..., T_freq, ...)

    # Step 4: Forward FFT of kernel
    if kernel.ndim == 1:
        # Scalar alpha: kernel is (T,)
        k_f = torch.fft.rfft(kernel, n=n_fft, dim=0)  # (T_freq,)
    else:
        # Per-channel alpha: kernel is (T, ...) matching trailing dims
        k_f = torch.fft.rfft(kernel, n=n_fft, dim=0)  # (T_freq, ...)

    # Step 5: Pointwise multiplication in frequency domain
    # Broadcasting: x_f shape (..., T_freq, ...) * k_f shape (T_freq, ...) or (T_freq,)
    # Need to align k_f with x_f dimensions
    if k_f.ndim == 1:
        # Scalar alpha: k_f is (T_freq,) -> reshape to (1, T_freq, 1, ...)
        k_f_expanded = k_f.view(*([1] * (len(shape) - 2)), -1, *([1] * (len(shape) - 2)))
        # For (B, T, D): k_f becomes (1, T_freq, 1)
        h_f = x_f * k_f_expanded
    else:
        # Per-channel alpha: broadcast naturally
        h_f = x_f * k_f

    # Step 6: Inverse FFT
    h = torch.fft.irfft(h_f, n=n_fft, dim=1)  # (..., 2T, ...)

    # Step 7: Crop to valid length T (discard padding)
    h = h[:, :T, ...]  # (..., T, ...)

    # Sanity check for NaN/Inf
    if torch.isnan(h).any() or torch.isinf(h).any():
        raise RuntimeError("NaN or Inf detected in EMA output - numerical instability")

    return h


class DampedEMA(nn.Module):
    """
    Damped Exponential Moving Average using parallel FFT convolution.

    Computes: h[t] = α·x[t] + (1-α)·h[t-1] in O(T log T) time.

    This replaces naive O(T²) matrix operations with efficient FFT-based
    parallel scan. All channels are processed simultaneously (no Python loops).

    Args:
        alpha: Smoothing factor (0 < alpha <= 1). Can be:
            - float: Fixed damping (stored as buffer)
            - 'learnable': Trainable parameter initialized to 0.9
            - Tensor: Per-channel damping (shape: (D,))
        eps: Numerical stability epsilon (default: 1e-6)

    Input:
        x: (B, T, D) - Batch, Time, Channels

    Output:
        h: (B, T, D) - EMA smoothed sequence

    Example:
        >>> ema = DampedEMA(alpha=0.9)
        >>> x = torch.randn(2, 10, 8)
        >>> h = ema(x)
        >>> h.shape
        torch.Size([2, 10, 8])

        >>> # Learnable alpha
        >>> ema = DampedEMA(alpha='learnable')
        >>> h = ema(x)
        >>> # ema.alpha_logit is a trainable parameter
    """

    def __init__(
        self,
        alpha: Union[float, str, torch.Tensor] = 0.9,
        eps: float = 1e-6
    ):
        super().__init__()
        self.eps = eps

        if isinstance(alpha, str) and alpha == 'learnable':
            # Learnable parameter (logit space for stability)
            # sigmoid(2.197) ≈ 0.9
            self.alpha_logit = nn.Parameter(torch.tensor(2.197))
            self.alpha_mode = 'learnable'
        elif isinstance(alpha, float):
            # Fixed scalar
            self.register_buffer('alpha', torch.tensor(alpha))
            self.alpha_mode = 'fixed'
        elif isinstance(alpha, torch.Tensor):
            # Fixed per-channel
            self.register_buffer('alpha', alpha)
            self.alpha_mode = 'per_channel'
        else:
            raise ValueError(
                f"alpha must be float, 'learnable', or Tensor, got {type(alpha)}"
            )

    def get_alpha(self) -> torch.Tensor:
        """
        Get current alpha value.

        Returns:
            alpha: Smoothing factor (applies sigmoid if learnable)
        """
        if self.alpha_mode == 'learnable':
            return torch.sigmoid(self.alpha_logit).clamp(self.eps, 1 - self.eps)
        else:
            return self.alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply damped EMA.

        Args:
            x: (B, T, D) input sequence

        Returns:
            h: (B, T, D) smoothed sequence
        """
        alpha = self.get_alpha()
        return ema_fft(x, alpha, eps=self.eps)

    def extra_repr(self) -> str:
        """String representation for print(module)."""
        if self.alpha_mode == 'learnable':
            alpha_val = self.get_alpha().item()
            return f'alpha=learnable (current={alpha_val:.4f}), eps={self.eps}'
        else:
            return f'alpha={self.get_alpha().item():.4f}, eps={self.eps}'


class MultiHeadDampedEMA(nn.Module):
    """
    Multi-Head Damped EMA using parallel FFT convolution.

    Splits input into multiple heads and applies independent EMA to each head.
    All heads are processed in parallel (no Python loops) using vectorized FFT.

    This is the multi-head attention equivalent of DampedEMA:
        - Input: (B, T, D) where D = num_heads * head_dim
        - Reshape: (B, T, num_heads, head_dim)
        - Apply EMA per head (vectorized across all heads)
        - Reshape back: (B, T, D)

    Args:
        num_heads: Number of attention heads
        head_dim: Dimension per head (D = num_heads * head_dim)
        alpha: Smoothing factor (0 < alpha <= 1). Can be:
            - float: Fixed damping (same for all heads/channels)
            - 'learnable': Trainable parameter initialized to 0.9
            - Tensor: Per-head or per-channel damping
                - Shape (num_heads,): per-head damping
                - Shape (num_heads, head_dim): per-channel damping
        eps: Numerical stability epsilon (default: 1e-6)

    Input:
        x: (B, T, D) where D = num_heads * head_dim

    Output:
        h: (B, T, D) - EMA smoothed sequence

    Example:
        >>> # 4 heads, 16 dims each = 64 total
        >>> ema = MultiHeadDampedEMA(num_heads=4, head_dim=16, alpha=0.9)
        >>> x = torch.randn(2, 10, 64)
        >>> h = ema(x)
        >>> h.shape
        torch.Size([2, 10, 64])

        >>> # Learnable per-head alpha
        >>> ema = MultiHeadDampedEMA(num_heads=4, head_dim=16, alpha='learnable')
        >>> h = ema(x)
        >>> # ema.alpha_logit is a trainable parameter

    Complexity:
        Time: O(T log T * num_heads * head_dim)
        Space: O(T * num_heads * head_dim)

    Note:
        All heads are processed simultaneously via broadcasting - no sequential loops.
        This is much more efficient than iterating over heads in Python.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        alpha: Union[float, str, torch.Tensor] = 0.9,
        eps: float = 1e-6
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = num_heads * head_dim
        self.eps = eps

        if isinstance(alpha, str) and alpha == 'learnable':
            # Learnable parameter (logit space for stability)
            # sigmoid(2.197) ≈ 0.9
            self.alpha_logit = nn.Parameter(torch.tensor(2.197))
            self.alpha_mode = 'learnable'
        elif isinstance(alpha, float):
            # Fixed scalar
            self.register_buffer('alpha', torch.tensor(alpha))
            self.alpha_mode = 'fixed'
        elif isinstance(alpha, torch.Tensor):
            # Fixed per-head or per-channel
            # Validate shape
            if alpha.shape == (num_heads,):
                # Per-head: broadcast to (num_heads, 1) for compatibility
                alpha = alpha.unsqueeze(-1)
            elif alpha.shape != (num_heads, head_dim):
                raise ValueError(
                    f"alpha tensor must have shape ({num_heads},) or "
                    f"({num_heads}, {head_dim}), got {alpha.shape}"
                )
            self.register_buffer('alpha', alpha)
            self.alpha_mode = 'per_head'
        else:
            raise ValueError(
                f"alpha must be float, 'learnable', or Tensor, got {type(alpha)}"
            )

    def get_alpha(self) -> torch.Tensor:
        """
        Get current alpha value.

        Returns:
            alpha: Smoothing factor (applies sigmoid if learnable)
        """
        if self.alpha_mode == 'learnable':
            return torch.sigmoid(self.alpha_logit).clamp(self.eps, 1 - self.eps)
        else:
            return self.alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-head damped EMA.

        Args:
            x: (B, T, D) input sequence where D = num_heads * head_dim

        Returns:
            h: (B, T, D) smoothed sequence

        Raises:
            ValueError: If input dimension doesn't match num_heads * head_dim
        """
        B, T, D = x.shape

        # Validate input dimension
        if D != self.dim:
            raise ValueError(
                f"Input dimension {D} doesn't match num_heads * head_dim = "
                f"{self.num_heads} * {self.head_dim} = {self.dim}"
            )

        # Reshape to multi-head format: (B, T, num_heads, head_dim)
        x_heads = x.view(B, T, self.num_heads, self.head_dim)

        # Apply EMA (vectorized across all heads)
        # ema_fft handles multi-dimensional inputs with time at dim=1
        alpha = self.get_alpha()
        h_heads = ema_fft(x_heads, alpha, eps=self.eps)  # (B, T, num_heads, head_dim)

        # Reshape back to flat format: (B, T, D)
        h = h_heads.view(B, T, D)

        return h

    def extra_repr(self) -> str:
        """String representation for print(module)."""
        if self.alpha_mode == 'learnable':
            alpha_val = self.get_alpha().item()
            alpha_str = f'learnable (current={alpha_val:.4f})'
        else:
            alpha_val = self.get_alpha()
            if alpha_val.numel() == 1:
                alpha_str = f'{alpha_val.item():.4f}'
            else:
                alpha_str = f'per_head (shape={tuple(alpha_val.shape)})'

        return (
            f'num_heads={self.num_heads}, head_dim={self.head_dim}, '
            f'alpha={alpha_str}, eps={self.eps}'
        )


class CEMA(nn.Module):
    """
    Causal Exponential Moving Average (CEMA) using parallel scan.

    A simplified wrapper around DampedEMA with explicit dimension parameter.
    Ensures causality: each timestep only depends on current and previous inputs.

    Uses FFT-based parallel scan for O(T log T) complexity instead of O(T²).

    Args:
        dim: Feature dimension (D)
        alpha: Smoothing factor (0 < alpha <= 1). Can be:
            - float: Fixed damping (stored as buffer)
            - 'learnable': Trainable parameter initialized to 0.9
        eps: Numerical stability epsilon (default: 1e-6)

    Input:
        x: (B, T, D) - Batch, Time, Channels

    Output:
        h: (B, T, D) - Causal EMA smoothed sequence

    Example:
        >>> cema = CEMA(dim=64, alpha=0.9)
        >>> x = torch.randn(2, 10, 64)
        >>> h = cema(x)
        >>> h.shape
        torch.Size([2, 10, 64])

        >>> # Learnable alpha
        >>> cema = CEMA(dim=64, alpha='learnable')
        >>> h = cema(x)
        >>> # cema.ema.alpha_logit is a trainable parameter

    Complexity:
        Time: O(T log T * D)
        Space: O(T * D)

    Note:
        The underlying ema_fft implementation is inherently causal - the
        exponential kernel only looks backward in time (τ ≥ 0).
    """

    def __init__(
        self,
        dim: int,
        alpha: Union[float, str] = 0.9,
        eps: float = 1e-6
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps

        # Delegate to DampedEMA for the actual computation
        self.ema = DampedEMA(alpha=alpha, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply causal EMA.

        Args:
            x: (B, T, D) input sequence

        Returns:
            h: (B, T, D) causally smoothed sequence

        Raises:
            ValueError: If input dimension doesn't match expected dim
        """
        B, T, D = x.shape

        # Validate input dimension
        if D != self.dim:
            raise ValueError(
                f"Input dimension {D} doesn't match expected dim {self.dim}"
            )

        # Apply causal EMA
        h = self.ema(x)

        return h

    def extra_repr(self) -> str:
        """String representation for print(module)."""
        return f'dim={self.dim}, alpha={self.ema.extra_repr()}'
