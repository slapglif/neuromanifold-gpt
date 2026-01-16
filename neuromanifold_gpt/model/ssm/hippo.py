# neuromanifold_gpt/model/ssm/hippo.py
"""
HiPPO (High-order Polynomial Projection Operators) matrix initialization.

HiPPO provides optimal memory for state space models by projecting continuous
signals onto polynomial bases. Different basis choices yield different memory
decay characteristics:

- LegS (Legendre Shifted): Uniform memory over sliding window
- LegT (Legendre Translated): Memory decays uniformly over time
- LagT (Laguerre Translated): Exponential memory decay

Key insight: The A matrix encodes how past information is compressed into
the state. HiPPO matrices are specifically designed to approximate history
optimally under different measures.

Reference: Gu et al., "HiPPO: Recurrent Memory with Optimal Polynomial Projections"
           https://arxiv.org/abs/2008.07669
"""

import math
from typing import Literal, Optional

import torch
import torch.nn as nn


def make_hippo_legs(N: int, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Construct HiPPO-LegS (Legendre Shifted) matrices.

    LegS provides uniform memory over a sliding window of fixed length.
    This is the most commonly used variant for sequence modeling.

    The measure μ(t) is uniform over [t-1, t], meaning all history
    within the window is remembered equally.

    Args:
        N: State dimension (number of Legendre polynomial coefficients)
        dtype: Data type for output tensors

    Returns:
        Tuple of (A, B) matrices where:
        - A: State transition matrix of shape (N, N)
        - B: Input matrix of shape (N, 1)
    """
    A = torch.zeros(N, N, dtype=dtype)
    B = torch.zeros(N, 1, dtype=dtype)

    for n in range(N):
        for k in range(N):
            if n > k:
                # Lower triangular structure
                A[n, k] = (2 * n + 1) ** 0.5 * (2 * k + 1) ** 0.5
            elif n == k:
                A[n, k] = n + 1

        # Input vector scaled by sqrt(2n+1)
        B[n, 0] = (2 * n + 1) ** 0.5

    # Negate A for stability (eigenvalues should be negative for stable dynamics)
    A = -A

    return A, B


def make_hippo_legt(N: int, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Construct HiPPO-LegT (Legendre Translated) matrices.

    LegT provides memory that decays uniformly over all of history.
    The measure μ(t) is uniform over [0, t], meaning older information
    gets compressed as new information arrives.

    Args:
        N: State dimension (number of Legendre polynomial coefficients)
        dtype: Data type for output tensors

    Returns:
        Tuple of (A, B) matrices where:
        - A: State transition matrix of shape (N, N)
        - B: Input matrix of shape (N, 1)
    """
    A = torch.zeros(N, N, dtype=dtype)
    B = torch.zeros(N, 1, dtype=dtype)

    for n in range(N):
        for k in range(N):
            if n > k:
                A[n, k] = (2 * n + 1) ** 0.5 * (2 * k + 1) ** 0.5
            elif n == k:
                A[n, k] = n + 1
            else:  # n < k
                A[n, k] = 0

        B[n, 0] = (2 * n + 1) ** 0.5

    # Scale by 1/t factor (approximated for continuous-time formulation)
    # LegT dynamics include 1/t scaling which we absorb into initialization
    A = -A

    return A, B


def make_hippo_lagt(N: int, alpha: float = 0.5, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Construct HiPPO-LagT (Laguerre Translated) matrices.

    LagT provides exponential memory decay with timescale controlled by alpha.
    The measure μ(t) = exp(-alpha * (t - s)) ds gives exponential forgetting.

    This is useful when recent information is more important than distant past.

    Args:
        N: State dimension (number of Laguerre polynomial coefficients)
        alpha: Memory decay rate (higher = faster forgetting)
        dtype: Data type for output tensors

    Returns:
        Tuple of (A, B) matrices where:
        - A: State transition matrix of shape (N, N)
        - B: Input matrix of shape (N, 1)
    """
    A = torch.zeros(N, N, dtype=dtype)
    B = torch.zeros(N, 1, dtype=dtype)

    # Laguerre recurrence structure
    for n in range(N):
        for k in range(N):
            if k <= n:
                A[n, k] = -1.0  # Laguerre matrices have constant -1 entries
            if k == n:
                A[n, k] = -0.5  # Diagonal adjustment for stability

        B[n, 0] = 1.0

    # Apply alpha scaling for memory decay rate
    A = alpha * A

    return A, B


def make_hippo_foud(N: int, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Construct HiPPO-FouD (Fourier Diagonal) matrices.

    FouD uses Fourier basis with diagonal A matrix for efficient computation.
    This variant is particularly fast as it avoids dense matrix operations.

    Args:
        N: State dimension (must be even for Fourier pairs)
        dtype: Data type for output tensors

    Returns:
        Tuple of (A, B) matrices where:
        - A: Diagonal state transition matrix of shape (N, N)
        - B: Input matrix of shape (N, 1)
    """
    if N % 2 != 0:
        N = N + 1  # Ensure even dimension for sin/cos pairs

    A = torch.zeros(N, N, dtype=dtype)
    B = torch.zeros(N, 1, dtype=dtype)

    # Fourier frequencies (pairs of sin/cos)
    for k in range(N // 2):
        freq = 2 * math.pi * (k + 1)

        # Real part (cosine)
        A[2 * k, 2 * k] = 0
        A[2 * k, 2 * k + 1] = freq
        # Imaginary part (sine)
        A[2 * k + 1, 2 * k] = -freq
        A[2 * k + 1, 2 * k + 1] = 0

        # Input coupling
        B[2 * k, 0] = 1.0
        B[2 * k + 1, 0] = 0.0

    return A, B


class HiPPO(nn.Module):
    """
    HiPPO matrix initialization for optimal memory in SSMs.

    Provides different polynomial bases for compressing continuous history:
    - 'legs': Legendre Shifted (uniform sliding window memory)
    - 'legt': Legendre Translated (uniform decay over all history)
    - 'lagt': Laguerre Translated (exponential memory decay)
    - 'foud': Fourier Diagonal (oscillatory, efficient diagonal)

    The matrices can be registered as parameters (learnable=True) or
    kept as fixed buffers (learnable=False).

    Example:
        >>> hippo = HiPPO(state_dim=64, hippo_type='legs')
        >>> A, B = hippo.get_matrices()
        >>> assert A.shape == (64, 64)
        >>> assert B.shape == (64, 1)

    Reference: Gu et al., "HiPPO: Recurrent Memory with Optimal Polynomial Projections"
    """

    HIPPO_TYPES = Literal["legs", "legt", "lagt", "foud"]

    def __init__(
        self,
        state_dim: int,
        hippo_type: str = "legs",
        learnable: bool = False,
        alpha: float = 0.5,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize HiPPO matrices.

        Args:
            state_dim: Dimension of the state space (N)
            hippo_type: Type of HiPPO basis ('legs', 'legt', 'lagt', 'foud')
            learnable: If True, register matrices as parameters for fine-tuning
            alpha: Memory decay rate for 'lagt' type
            dtype: Data type for matrices
        """
        super().__init__()
        self.state_dim = state_dim
        self.hippo_type = hippo_type.lower()
        self.learnable = learnable
        self.alpha = alpha
        self.dtype = dtype

        # Generate HiPPO matrices based on type
        A, B = self._make_matrices()

        # Register as parameters or buffers
        if learnable:
            self.A = nn.Parameter(A)
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("A", A)
            self.register_buffer("B", B)

    def _make_matrices(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate HiPPO matrices based on configured type."""
        if self.hippo_type == "legs":
            return make_hippo_legs(self.state_dim, self.dtype)
        elif self.hippo_type == "legt":
            return make_hippo_legt(self.state_dim, self.dtype)
        elif self.hippo_type == "lagt":
            return make_hippo_lagt(self.state_dim, self.alpha, self.dtype)
        elif self.hippo_type == "foud":
            return make_hippo_foud(self.state_dim, self.dtype)
        else:
            raise ValueError(
                f"Unknown HiPPO type: {self.hippo_type}. "
                f"Valid options: 'legs', 'legt', 'lagt', 'foud'"
            )

    def get_matrices(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the HiPPO A and B matrices.

        Returns:
            Tuple of (A, B) where:
            - A: State transition matrix of shape (state_dim, state_dim)
            - B: Input matrix of shape (state_dim, 1)
        """
        return self.A, self.B

    def get_diagonal(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get diagonal approximation of HiPPO matrices.

        For efficient computation, SSMs often use diagonal A matrices.
        This extracts the diagonal and returns a compatible B.

        Returns:
            Tuple of (A_diag, B_flat) where:
            - A_diag: Diagonal of A matrix, shape (state_dim,)
            - B_flat: Flattened B matrix, shape (state_dim,)
        """
        A_diag = torch.diag(self.A)
        B_flat = self.B.squeeze(-1)
        return A_diag, B_flat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply HiPPO state update to input.

        Computes: x_new = A @ x + B @ u

        This is primarily for testing/debugging. In practice, SSM layers
        use discretized versions of these matrices.

        Args:
            x: Input of shape (batch, state_dim) or (batch, seq_len, state_dim)

        Returns:
            State update of shape matching input
        """
        if x.dim() == 2:
            # (batch, state_dim) -> single step update
            return torch.matmul(x, self.A.T)
        elif x.dim() == 3:
            # (batch, seq_len, state_dim) -> batch matmul
            return torch.matmul(x, self.A.T)
        else:
            raise ValueError(f"Expected 2D or 3D input, got shape {x.shape}")

    def extra_repr(self) -> str:
        """Extra representation for module printing."""
        return (
            f"state_dim={self.state_dim}, "
            f"hippo_type='{self.hippo_type}', "
            f"learnable={self.learnable}"
        )


class DiagonalHiPPO(nn.Module):
    """
    Diagonal approximation of HiPPO for efficient SSM computation.

    Modern SSMs like S4D and Mamba use diagonal state matrices for
    O(N) complexity per step instead of O(N^2) for dense matrices.

    The diagonal approximation preserves the key memory properties
    while enabling much faster parallel computation.

    Example:
        >>> dhippo = DiagonalHiPPO(state_dim=64)
        >>> A_diag, B = dhippo.get_matrices()
        >>> assert A_diag.shape == (64,)
    """

    def __init__(
        self,
        state_dim: int,
        hippo_type: str = "legs",
        learnable: bool = True,
        init_std: float = 1.0,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize diagonal HiPPO.

        Args:
            state_dim: Dimension of the state space
            hippo_type: Base HiPPO type for initialization
            learnable: If True, parameters can be learned
            init_std: Standard deviation for initialization noise
            dtype: Data type for parameters
        """
        super().__init__()
        self.state_dim = state_dim
        self.hippo_type = hippo_type.lower()
        self.learnable = learnable

        # Initialize from full HiPPO, then take diagonal
        full_hippo = HiPPO(state_dim, hippo_type, learnable=False, dtype=dtype)
        A_full, B_full = full_hippo.get_matrices()

        # Extract diagonal with optional noise
        A_diag = torch.diag(A_full)
        if init_std > 0:
            A_diag = A_diag + torch.randn_like(A_diag) * init_std * 0.01
        B_flat = B_full.squeeze(-1)

        if learnable:
            # For S4D/Mamba: use log parametrization for A (ensures negative real part)
            # A = -exp(log_A_real) + i * A_imag
            self.log_A = nn.Parameter(torch.log(-A_diag.clamp(max=-1e-8).abs() + 1e-8))
            self.B = nn.Parameter(B_flat)
        else:
            self.register_buffer("log_A", torch.log(-A_diag.clamp(max=-1e-8).abs() + 1e-8))
            self.register_buffer("B", B_flat)

    def get_matrices(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get diagonal A and B matrices.

        Returns:
            Tuple of (A_diag, B) where:
            - A_diag: Diagonal elements of A, shape (state_dim,)
            - B: Input matrix, shape (state_dim,)
        """
        # Convert from log parametrization
        A_diag = -torch.exp(self.log_A)
        return A_diag, self.B

    def extra_repr(self) -> str:
        """Extra representation for module printing."""
        return (
            f"state_dim={self.state_dim}, "
            f"hippo_type='{self.hippo_type}', "
            f"learnable={self.learnable}"
        )
