# neuromanifold_gpt/model/ssm/base.py
"""
Base classes for State Space Models (SSM).

Implements Mamba-style selective state space models for continuous dynamics.
SSMs provide an alternative to attention by modeling sequence relationships
as continuous-time dynamical systems with learnable state transitions.

Key concepts:
- State: Hidden representation that evolves over time
- Input-to-state (B): How inputs affect state
- State-to-state (A): How state evolves (typically HiPPO-initialized)
- State-to-output (C): How state produces outputs
- Skip connection (D): Direct input-to-output path

Reference: Gu et al., "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class SSMBase(nn.Module, ABC):
    """
    Abstract base class for State Space Models.

    Provides the interface for SSM layers that can replace attention mechanisms.
    Subclasses implement specific SSM variants (e.g., S4, Mamba, selective scan).

    The continuous-time SSM is defined by:
        dx/dt = Ax + Bu
        y = Cx + Du

    where:
        x: state vector (state_dim,)
        u: input (embed_dim,)
        y: output (embed_dim,)
        A: state transition matrix (state_dim, state_dim)
        B: input projection (state_dim, embed_dim)
        C: output projection (embed_dim, state_dim)
        D: skip connection (embed_dim,)
    """

    def __init__(
        self,
        embed_dim: int,
        state_dim: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dropout: float = 0.0,
    ):
        """
        Initialize SSM base class.

        Args:
            embed_dim: Input/output embedding dimension
            state_dim: Hidden state dimension (N in Mamba paper)
            dt_min: Minimum discretization timestep
            dt_max: Maximum discretization timestep
            dropout: Dropout probability
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.state_dim = state_dim
        self.dt_min = dt_min
        self.dt_max = dt_max

        # Skip connection (D matrix) - direct input-to-output path
        self.D = nn.Parameter(torch.ones(embed_dim))

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Forward pass through the SSM.

        Args:
            x: Input tensor of shape (B, T, embed_dim)
            state: Optional initial state of shape (B, state_dim)
                   If None, starts from zero state.

        Returns:
            Tuple of (output, info) where:
            - output: SSM output of shape (B, T, embed_dim)
            - info: Dictionary with SSM statistics (e.g., final state)
        """
        pass

    @abstractmethod
    def step(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single step of SSM for autoregressive generation.

        Args:
            x: Input tensor of shape (B, embed_dim)
            state: Current state of shape (B, state_dim)

        Returns:
            Tuple of (output, new_state) where:
            - output: Step output of shape (B, embed_dim)
            - new_state: Updated state of shape (B, state_dim)
        """
        pass

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Initialize hidden state to zeros.

        Args:
            batch_size: Batch size
            device: Device to create state on
            dtype: Data type for state tensor

        Returns:
            Zero-initialized state of shape (B, state_dim)
        """
        return torch.zeros(
            batch_size,
            self.state_dim,
            device=device,
            dtype=dtype,
        )

    def discretize(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        dt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Discretize continuous-time SSM parameters using zero-order hold (ZOH).

        Converts continuous-time (A, B) to discrete-time (A_bar, B_bar):
            A_bar = exp(A * dt)
            B_bar = (A^-1)(A_bar - I) * B

        For diagonal A (common in practice), this simplifies to:
            A_bar = exp(A * dt)
            B_bar = (exp(A * dt) - 1) / A * B

        Args:
            A: State transition matrix of shape (..., state_dim)
               Assumed diagonal for efficiency
            B: Input matrix of shape (..., state_dim, embed_dim) or (..., state_dim)
            dt: Discretization timestep of shape (...,) or scalar

        Returns:
            Tuple of (A_bar, B_bar) discretized matrices
        """
        # Ensure dt has correct shape for broadcasting
        if dt.dim() == 0:
            dt = dt.unsqueeze(0)

        # A is assumed diagonal, so A * dt is element-wise
        # dt shape: (...,), A shape: (..., state_dim)
        dt_A = dt.unsqueeze(-1) * A  # (..., state_dim)

        # Discretized A: exp(A * dt)
        A_bar = torch.exp(dt_A)

        # Discretized B: (exp(A * dt) - 1) / A * B
        # Use stable computation to avoid division by zero
        # When A -> 0: (exp(dt*A) - 1) / A -> dt
        dB = (A_bar - 1.0) / (A + 1e-8)

        if B.dim() == A.dim():
            # B is (..., state_dim), same as A
            B_bar = dB * B
        else:
            # B is (..., state_dim, embed_dim)
            B_bar = dB.unsqueeze(-1) * B

        return A_bar, B_bar

    def extra_repr(self) -> str:
        """Extra representation for module printing."""
        return (
            f"embed_dim={self.embed_dim}, "
            f"state_dim={self.state_dim}, "
            f"dt_range=[{self.dt_min:.4f}, {self.dt_max:.4f}]"
        )


class SSMConfig:
    """
    Configuration for SSM modules.

    Provides default hyperparameters for SSM variants.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        state_dim: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dropout: float = 0.0,
        use_hippo: bool = True,
        hippo_type: str = "legs",
        expand_factor: int = 2,
        conv_kernel_size: int = 4,
        use_selective_scan: bool = True,
    ):
        """
        Initialize SSM configuration.

        Args:
            embed_dim: Model embedding dimension
            state_dim: SSM hidden state dimension
            dt_min: Minimum discretization timestep
            dt_max: Maximum discretization timestep
            dropout: Dropout probability
            use_hippo: Whether to use HiPPO initialization for A matrix
            hippo_type: Type of HiPPO matrix ('legs', 'legt', 'lagt')
            expand_factor: Expansion factor for inner dimension
            conv_kernel_size: Kernel size for local convolution
            use_selective_scan: Whether to use input-dependent selection
        """
        self.embed_dim = embed_dim
        self.state_dim = state_dim
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dropout = dropout
        self.use_hippo = use_hippo
        self.hippo_type = hippo_type
        self.expand_factor = expand_factor
        self.conv_kernel_size = conv_kernel_size
        self.use_selective_scan = use_selective_scan
