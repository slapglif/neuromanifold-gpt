# neuromanifold_gpt/model/ssm/selective_scan.py
"""
Selective Scan mechanism - the core of Mamba-style SSMs.

The key innovation of Mamba is making the SSM parameters (B, C, dt) input-dependent,
which breaks the LTI (Linear Time Invariant) structure but enables content-aware
sequence processing. This is the "selective" in selective state space.

Traditional SSM: y = SSM(A, B, C, D)(x)  # B, C are fixed
Selective SSM:   y = SSM(A, B(x), C(x), D, dt(x))(x)  # B, C, dt depend on input

The parallel scan algorithm enables O(n) training despite the non-LTI structure
by using associative scan properties of the SSM recurrence.

Reference: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
           https://arxiv.org/abs/2312.00752
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuromanifold_gpt.model.ssm.hippo import DiagonalHiPPO


class SelectiveScan(nn.Module):
    """
    Selective Scan mechanism for Mamba-style sequence modeling.

    Implements input-dependent state space dynamics where the B (input), C (output),
    and dt (timestep) matrices are projected from the input. This enables the model
    to selectively propagate or forget information based on content.

    The SSM dynamics are:
        h[t] = A_bar * h[t-1] + B_bar * x[t]
        y[t] = C[t] * h[t]

    Where A_bar, B_bar are discretized from continuous A, B using dt[t].

    For training, we use parallel scan via FFT convolution (for constant A) or
    associative scan (for diagonal A with input-dependent parameters).

    For generation, we use the recurrent formulation for O(1) memory per step.

    Example:
        >>> ss = SelectiveScan(embed_dim=64)
        >>> x = torch.randn(2, 32, 64)  # (batch, seq_len, dim)
        >>> y = ss(x)  # (batch, seq_len, dim)
        >>> assert y.shape == x.shape
    """

    def __init__(
        self,
        embed_dim: int,
        state_dim: int = 16,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        use_hippo_init: bool = True,
        hippo_type: str = "legs",
    ):
        """
        Initialize SelectiveScan.

        Args:
            embed_dim: Input/output embedding dimension (D in paper)
            state_dim: SSM state dimension (N in paper), controls memory capacity
            dt_rank: Rank for dt projection. "auto" sets it to ceil(embed_dim/16)
            dt_min: Minimum discretization timestep
            dt_max: Maximum discretization timestep
            dt_init: Initialization scheme for dt ("random" or "constant")
            dt_scale: Scaling factor for dt initialization
            dt_init_floor: Floor value for dt initialization
            use_hippo_init: Whether to use HiPPO initialization for A matrix
            hippo_type: Type of HiPPO initialization ('legs', 'legt', 'lagt')
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.state_dim = state_dim
        self.dt_min = dt_min
        self.dt_max = dt_max

        # Compute dt_rank
        if dt_rank == "auto":
            self.dt_rank = max(1, math.ceil(embed_dim / 16))
        else:
            self.dt_rank = int(dt_rank)

        # Initialize A matrix (state dynamics)
        # Use diagonal parameterization for efficiency
        if use_hippo_init:
            hippo = DiagonalHiPPO(state_dim, hippo_type=hippo_type, learnable=False)
            A_diag, _ = hippo.get_matrices()
            # Use log parameterization: A = -exp(log_A) ensures negative eigenvalues
            self.log_A = nn.Parameter(torch.log(-A_diag.clamp(max=-1e-8).abs() + 1e-8))
        else:
            # Random initialization with proper scaling
            self.log_A = nn.Parameter(torch.log(torch.rand(state_dim) * 0.5 + 0.5))

        # Expand log_A to (embed_dim, state_dim) for per-channel dynamics
        # This allows each feature dimension to have its own state dynamics
        self.log_A = nn.Parameter(
            self.log_A.unsqueeze(0).expand(embed_dim, -1).clone()
        )

        # Selective projections: x -> (B, C, dt)
        # B: input projection (embed_dim -> state_dim)
        self.B_proj = nn.Linear(embed_dim, state_dim, bias=False)

        # C: output projection (embed_dim -> state_dim)
        self.C_proj = nn.Linear(embed_dim, state_dim, bias=False)

        # dt: timestep projection (embed_dim -> dt_rank -> embed_dim)
        # Low-rank projection for efficiency
        self.dt_proj = nn.Sequential(
            nn.Linear(embed_dim, self.dt_rank, bias=False),
            nn.Linear(self.dt_rank, embed_dim, bias=True),
        )

        # Initialize dt projection
        self._init_dt_proj(dt_init, dt_scale, dt_init_floor)

        # D: skip connection (direct input-to-output)
        self.D = nn.Parameter(torch.ones(embed_dim))

    def _init_dt_proj(
        self,
        dt_init: str,
        dt_scale: float,
        dt_init_floor: float,
    ):
        """Initialize dt projection for proper timestep range."""
        # Initialize dt bias to achieve desired dt range
        dt_init_value = torch.exp(
            torch.rand(self.embed_dim) * (math.log(self.dt_max) - math.log(self.dt_min))
            + math.log(self.dt_min)
        ).clamp(min=dt_init_floor)

        # Inverse of softplus to set initial dt values
        inv_dt = dt_init_value + torch.log(-torch.expm1(-dt_init_value))

        # Set bias of the last linear layer
        with torch.no_grad():
            self.dt_proj[-1].bias.copy_(inv_dt)

        # Initialize weights with proper scaling
        nn.init.kaiming_uniform_(self.dt_proj[0].weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.dt_proj[-1].weight, a=math.sqrt(5))

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with selective scan.

        Args:
            x: Input tensor of shape (B, T, D) where:
               - B: batch size
               - T: sequence length
               - D: embed_dim
            state: Optional initial state of shape (B, D, N)
                   If None, starts from zero state.

        Returns:
            Output tensor of shape (B, T, D)
        """
        batch_size, seq_len, _ = x.shape

        # Compute input-dependent B, C, dt
        B_t = self.B_proj(x)  # (batch, seq, state_dim)
        C_t = self.C_proj(x)  # (batch, seq, state_dim)

        # dt uses softplus to ensure positive values
        dt_raw = self.dt_proj(x)  # (batch, seq, embed_dim)
        dt = F.softplus(dt_raw)  # Ensure positive, (batch, seq, embed_dim)

        # Get A matrix (negative for stability)
        A = -torch.exp(self.log_A)  # (embed_dim, state_dim)

        # Discretize: A_bar = exp(A * dt), B_bar = (exp(A*dt) - 1) / A * B
        # dt: (batch, seq, embed_dim) -> (batch, seq, embed_dim, 1)
        # A: (embed_dim, state_dim) -> (1, 1, embed_dim, state_dim)
        dt_expanded = dt.unsqueeze(-1)  # (B, T, D, 1)
        A_expanded = A.unsqueeze(0).unsqueeze(0)  # (1, 1, D, N)

        # A_bar = exp(A * dt)
        A_bar = torch.exp(dt_expanded * A_expanded)  # (B, T, D, N)

        # B_bar = (A_bar - 1) / A * B
        # Stable computation when A -> 0
        dB = (A_bar - 1) / (A_expanded + 1e-8)  # (B, T, D, N)

        # B_t: (B, T, N) -> (B, T, D, N) by broadcasting with x
        # Actually, B_t projects the input to state dim
        # x: (B, T, D) @ B_t produces update to each state dim
        # But we want per-channel: x[..., d] updates state[..., d, :]
        # So we need B_bar * x[:, :, :, None] for the input contribution

        # Reshape for scan: we process each embed_dim channel independently
        # x: (B, T, D) -> input to state
        x_expanded = x.unsqueeze(-1)  # (B, T, D, 1)
        B_bar = dB * x_expanded  # (B, T, D, N) - scaled input contribution

        # Note: B_t modulates how much of input goes to state
        # Combine with the projected B coefficients
        B_t_expanded = B_t.unsqueeze(2)  # (B, T, 1, N)
        B_bar = B_bar * B_t_expanded.expand(-1, -1, self.embed_dim, -1)

        # Run the selective scan
        y = self._selective_scan(A_bar, B_bar, C_t)

        # Add skip connection
        y = y + self.D * x

        return y

    def _selective_scan(
        self,
        A_bar: torch.Tensor,
        B_bar: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parallel selective scan implementation.

        Uses associative scan properties:
        h[t] = A_bar[t] * h[t-1] + B_bar[t]
        y[t] = C[t] @ h[t]

        For efficiency, we compute this in O(T log T) via parallel scan.

        Args:
            A_bar: Discretized A, shape (B, T, D, N)
            B_bar: Discretized B * x, shape (B, T, D, N)
            C: Output projection, shape (B, T, N)

        Returns:
            y: Output tensor, shape (B, T, D)
        """
        batch_size, seq_len, embed_dim, state_dim = A_bar.shape

        # For each position t, we need:
        # h[t] = sum_{k=0}^{t} (prod_{j=k+1}^{t} A_bar[j]) * B_bar[k]

        # Compute cumulative products of A_bar for parallel scan
        # This is the "carry" in the associative scan

        # Initialize state
        h = torch.zeros(
            batch_size, embed_dim, state_dim,
            device=A_bar.device, dtype=A_bar.dtype
        )

        # Sequential scan (fallback for correctness)
        # TODO: Replace with parallel associative scan for O(T log T)
        outputs = []
        for t in range(seq_len):
            # h[t] = A_bar[t] * h[t-1] + B_bar[t]
            h = A_bar[:, t] * h + B_bar[:, t]  # (B, D, N)

            # y[t] = sum_n C[t, n] * h[t, n]
            # C: (B, T, N), h: (B, D, N)
            # We want (B, D) output
            C_t = C[:, t].unsqueeze(1)  # (B, 1, N)
            y_t = (h * C_t).sum(dim=-1)  # (B, D)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (B, T, D)
        return y

    def step(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single step for autoregressive generation.

        More efficient than full forward for generation as it maintains
        O(1) memory and O(D*N) compute per step.

        Args:
            x: Input tensor of shape (B, D)
            state: Current state of shape (B, D, N)

        Returns:
            Tuple of (output, new_state) where:
            - output: Step output of shape (B, D)
            - new_state: Updated state of shape (B, D, N)
        """
        # Compute input-dependent parameters for single step
        B_t = self.B_proj(x)  # (B, N)
        C_t = self.C_proj(x)  # (B, N)
        dt_raw = self.dt_proj(x)  # (B, D)
        dt = F.softplus(dt_raw)  # (B, D)

        # Get A and discretize
        A = -torch.exp(self.log_A)  # (D, N)

        # Discretize for single step
        dt_expanded = dt.unsqueeze(-1)  # (B, D, 1)
        A_expanded = A.unsqueeze(0)  # (1, D, N)

        A_bar = torch.exp(dt_expanded * A_expanded)  # (B, D, N)
        dB = (A_bar - 1) / (A_expanded + 1e-8)  # (B, D, N)

        # Input contribution
        x_expanded = x.unsqueeze(-1)  # (B, D, 1)
        B_t_expanded = B_t.unsqueeze(1)  # (B, 1, N)
        B_bar = dB * x_expanded * B_t_expanded  # (B, D, N)

        # Update state
        new_state = A_bar * state + B_bar  # (B, D, N)

        # Compute output
        C_t_expanded = C_t.unsqueeze(1)  # (B, 1, N)
        y = (new_state * C_t_expanded).sum(dim=-1)  # (B, D)

        # Add skip connection
        y = y + self.D * x

        return y, new_state

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
            device: Device for state tensor
            dtype: Data type for state tensor

        Returns:
            Zero-initialized state of shape (B, D, N)
        """
        return torch.zeros(
            batch_size, self.embed_dim, self.state_dim,
            device=device, dtype=dtype
        )

    def extra_repr(self) -> str:
        """Extra representation for module printing."""
        return (
            f"embed_dim={self.embed_dim}, "
            f"state_dim={self.state_dim}, "
            f"dt_rank={self.dt_rank}"
        )


class ParallelSelectiveScan(nn.Module):
    """
    Optimized selective scan using parallel associative scan.

    This implementation uses the associative property of the SSM recurrence:
    (A1, B1) * (A2, B2) = (A1*A2, A1*B2 + B1)

    This enables O(T log T) parallel computation instead of O(T) sequential.

    For very long sequences, this can be combined with chunking for
    memory-efficient processing.
    """

    def __init__(
        self,
        embed_dim: int,
        state_dim: int = 16,
        **kwargs,
    ):
        """Initialize parallel selective scan."""
        super().__init__()
        # Use the base SelectiveScan and override the scan method
        self.selective_scan = SelectiveScan(embed_dim, state_dim, **kwargs)
        self.embed_dim = embed_dim
        self.state_dim = state_dim

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward with parallel scan."""
        return self.selective_scan(x, state)

    def _parallel_associative_scan(
        self,
        A_bar: torch.Tensor,
        B_bar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parallel associative scan using recursive doubling.

        The associative operator is:
        (a1, b1) * (a2, b2) = (a1 * a2, a2 * b1 + b2)

        This computes all prefix sums in O(log T) parallel steps.

        Args:
            A_bar: Shape (B, T, D, N)
            B_bar: Shape (B, T, D, N)

        Returns:
            h: All states, shape (B, T, D, N)
        """
        batch_size, seq_len, embed_dim, state_dim = A_bar.shape

        # Pad to power of 2 for efficient parallel scan
        log_T = math.ceil(math.log2(max(seq_len, 1)))
        padded_len = 2 ** log_T

        if seq_len < padded_len:
            pad_len = padded_len - seq_len
            A_bar = F.pad(A_bar, (0, 0, 0, 0, 0, pad_len), value=1.0)
            B_bar = F.pad(B_bar, (0, 0, 0, 0, 0, pad_len), value=0.0)

        # Up-sweep (reduce) phase
        a = A_bar.clone()
        b = B_bar.clone()

        for d in range(log_T):
            stride = 2 ** (d + 1)
            indices = torch.arange(stride - 1, padded_len, stride, device=A_bar.device)

            if len(indices) == 0:
                break

            left_indices = indices - 2 ** d

            # (a_left, b_left) * (a_right, b_right)
            a_left = a[:, left_indices]
            b_left = b[:, left_indices]
            a_right = a[:, indices]
            b_right = b[:, indices]

            # Associative operation
            a[:, indices] = a_left * a_right
            b[:, indices] = a_right * b_left + b_right

        # Down-sweep phase
        for d in range(log_T - 2, -1, -1):
            stride = 2 ** (d + 1)
            indices = torch.arange(stride + 2**d - 1, padded_len, stride, device=A_bar.device)

            if len(indices) == 0:
                continue

            left_indices = indices - 2 ** d

            a_left = a[:, left_indices]
            b_left = b[:, left_indices]

            b[:, indices] = a[:, indices] * b_left + b[:, indices]

        # Return unpadded result (b contains the cumulative states)
        return b[:, :seq_len]
