# neuromanifold_gpt/model/soliton/kdv.py
"""
Korteweg-de Vries (KdV) Equation Solver for Dispersive Wave Dynamics.

The KdV equation:
    u_t + 6u·u_x + u_xxx = 0

Supports soliton solutions that:
- Maintain shape during propagation
- Survive collisions unchanged (phase shift only)
- Balance nonlinear steepening with dispersion

Key properties for language modeling:
- Solitons represent coherent semantic units
- Stable propagation maintains meaning over distance
- Collision invariance allows compositionality

Analytic soliton solution:
    u(x,t) = (c/2) · sech²(√c/2 · (x - ct))
    where c is the soliton speed (taller = faster)

Reference:
- "Solitons and the Inverse Scattering Transform" - Ablowitz & Segur
- "Solitons in Mathematics and Physics" - Newell
"""

from typing import Optional

import torch
import torch.nn as nn

from .base import PDESolver


@torch.jit.script
def kdv_rk4_step(
    u: torch.Tensor,
    dt: torch.Tensor,
    nonlin_coeff: torch.Tensor,
    disp_coeff: torch.Tensor,
    u_x: torch.Tensor,
    u_xxx: torch.Tensor,
) -> torch.Tensor:
    """
    JIT-compiled RK4 step for KdV equation.

    Optimized inner loop for time stepping.
    Note: u_x and u_xxx must be computed externally (spectral/FD not JIT-compatible).

    Args:
        u: Wave field u(x,t)
        dt: Time step (scalar tensor)
        nonlin_coeff: Nonlinearity coefficient (default 6.0)
        disp_coeff: Dispersion coefficient (default 1.0)
        u_x: Pre-computed first spatial derivative
        u_xxx: Pre-computed third spatial derivative

    Returns:
        u_new: Updated wave field
    """
    # KdV: u_t = -nonlin_coeff * u * u_x - disp_coeff * u_xxx

    # k1 (at current point)
    k1 = -nonlin_coeff * u * u_x - disp_coeff * u_xxx

    # k2 (midpoint using k1)
    u_mid1 = u + 0.5 * dt * k1
    k2 = -nonlin_coeff * u_mid1 * u_x - disp_coeff * u_xxx  # u_x approx same

    # k3 (midpoint using k2)
    u_mid2 = u + 0.5 * dt * k2
    k3 = -nonlin_coeff * u_mid2 * u_x - disp_coeff * u_xxx

    # k4 (endpoint using k3)
    u_end = u + dt * k3
    k4 = -nonlin_coeff * u_end * u_x - disp_coeff * u_xxx

    # Combine
    u_new = u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return u_new


class KdVSolver(PDESolver):
    """
    Solver for the Korteweg-de Vries (KdV) equation.

    The KdV equation: u_t + α·u·u_x + β·u_xxx = 0

    This PDE admits soliton solutions that:
    - Propagate with constant velocity proportional to amplitude
    - Pass through each other during collisions (phase shift only)
    - Balance nonlinear steepening against dispersive spreading

    For language modeling, we use solitons to represent:
    - Semantic units that maintain coherence over long distances
    - Information packets that survive attention mixing
    - Wave dynamics for token interactions

    Example:
        >>> solver = KdVSolver(dim=64)
        >>> u = torch.randn(2, 32, 64)  # (batch, seq, dim)
        >>> u_evolved, info = solver(u, n_steps=5)
        >>> print(info['mass'])  # Should be approximately conserved
    """

    def __init__(
        self,
        dim: int,
        dt: float = 0.01,
        dx: float = 1.0,
        n_steps: int = 5,
        nonlin_coeff: float = 6.0,
        disp_coeff: float = 1.0,
        use_spectral: bool = True,
        use_rk4: bool = True,
        clamp_min: float = -10.0,
        clamp_max: float = 10.0,
        dropout: float = 0.0,
        damping: float = 0.0,
        causal: bool = False,
    ):
        """
        Initialize KdV solver.

        Args:
            dim: Feature dimension for the wave field
            dt: Time step for numerical integration (smaller = more stable)
            dx: Spatial step for finite differences
            n_steps: Default number of integration steps per forward pass
            nonlin_coeff: Coefficient for nonlinear term u·u_x (default 6.0)
            disp_coeff: Coefficient for dispersion term u_xxx (default 1.0)
            use_spectral: Use FFT-based spectral derivatives (more accurate)
            use_rk4: Use RK4 time stepping (more accurate than Euler)
            clamp_min: Minimum value for clamping (numerical stability)
            clamp_max: Maximum value for clamping (numerical stability)
            dropout: Dropout probability for regularization
            damping: Optional damping coefficient (adds -γ·u term)
            causal: Force causal derivatives (backward differences only)
        """
        super().__init__(
            dim=dim,
            dt=dt,
            dx=dx,
            n_steps=n_steps,
            use_spectral=use_spectral,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            dropout=dropout,
            causal=causal,
        )

        self.use_rk4 = use_rk4
        self.damping = damping

        # Learnable nonlinearity coefficient (standard KdV has 6)
        self.nonlin_coeff = nn.Parameter(torch.tensor(nonlin_coeff))

        # Learnable dispersion coefficient
        self.disp_coeff = nn.Parameter(torch.tensor(disp_coeff))

    def compute_rhs(
        self,
        u: torch.Tensor,
        u_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute right-hand side: u_t = -α·u·u_x - β·u_xxx - γ·u

        Args:
            u: Wave field u of shape (B, T, D) or (B, H, T, D)
            u_t: Time derivative (unused for first-order KdV)

        Returns:
            Time derivative ∂u/∂t
        """
        # First spatial derivative: u_x
        u_x = self.spatial_derivative(u, order=1)

        # Third spatial derivative: u_xxx
        u_xxx = self.spatial_derivative(u, order=3)

        # RHS = -α·u·u_x - β·u_xxx (nonlinear advection + dispersion)
        rhs = -self.nonlin_coeff * u * u_x - self.disp_coeff * u_xxx

        # Optional damping: -γ·u
        if self.damping > 0:
            rhs = rhs - self.damping * u

        return rhs

    def compute_potential_energy(self, u: torch.Tensor) -> torch.Tensor:
        """
        Compute KdV "potential energy": proportional to u³/3

        For KdV, the conserved Hamiltonian is:
        H = ∫ [½·u_x² - u³] dx

        Args:
            u: Wave field u

        Returns:
            Potential energy per batch element
        """
        # KdV potential is cubic: -u³/3
        # We compute positive version for energy interpretation
        potential = (u ** 3 / 3).sum(dim=(-2, -1))
        return potential.abs()

    def compute_mass(self, u: torch.Tensor) -> torch.Tensor:
        """
        Compute conserved mass: M = ∫ u dx

        Mass is the first conserved quantity of KdV.

        Args:
            u: Wave field u of shape (B, T, D) or (B, H, T, D)

        Returns:
            Mass per batch element, shape (B,) or (B, H)
        """
        # Sum over spatial dimension
        mass = u.sum(dim=-2)

        # Average over feature dimension
        return mass.mean(dim=-1)

    def compute_momentum(self, u: torch.Tensor) -> torch.Tensor:
        """
        Compute conserved momentum: P = ∫ u² dx

        Momentum is the second conserved quantity of KdV.

        Args:
            u: Wave field u of shape (B, T, D) or (B, H, T, D)

        Returns:
            Momentum per batch element, shape (B,) or (B, H)
        """
        # Sum of u² over spatial dimension
        momentum = (u ** 2).sum(dim=-2)

        # Average over feature dimension
        return momentum.mean(dim=-1)

    def forward(
        self,
        u: torch.Tensor,
        n_steps: Optional[int] = None,
        u_t: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """
        Evolve wave field according to KdV equation.

        Args:
            u: Initial wave field u(x,0) of shape (B, T, D) or (B, H, T, D)
            n_steps: Number of time steps (overrides self.n_steps)
            u_t: Unused (KdV is first-order), kept for API compatibility
            return_trajectory: If True, return full trajectory (expensive)

        Returns:
            Tuple of (u_final, info) where:
            - u_final: Evolved wave field u(x, T)
            - info: Dictionary containing:
                - 'mass': Final mass (conserved quantity)
                - 'mass_initial': Initial mass (for conservation check)
                - 'momentum': Final momentum (conserved quantity)
                - 'momentum_initial': Initial momentum
                - 'energy': Final total energy
                - 'trajectory': Full trajectory if return_trajectory=True
        """
        steps = n_steps if n_steps is not None else self.n_steps
        dt = self.dt.abs()  # Ensure positive

        # Compute initial conserved quantities for monitoring
        mass_initial = self.compute_mass(u)
        momentum_initial = self.compute_momentum(u)
        energy_initial = self.compute_energy(u, None)

        # Optional trajectory storage
        trajectory = [u.clone()] if return_trajectory else None

        # Time stepping loop
        for _ in range(steps):
            if self.use_rk4:
                # Pre-compute spatial derivatives (not JIT-compatible)
                u_x = self.spatial_derivative(u, order=1)
                u_xxx = self.spatial_derivative(u, order=3)

                # Use JIT-compiled RK4 step
                u = kdv_rk4_step(
                    u, dt, self.nonlin_coeff, self.disp_coeff, u_x, u_xxx
                )

                # Apply damping manually if needed
                if self.damping > 0:
                    u = u - dt * self.damping * u
            else:
                # Fallback to base class Euler step (first-order, no u_t)
                u, _ = self.euler_step(u, None)

            # Clamp for numerical stability
            u = self.clamp_field(u)

            if return_trajectory:
                trajectory.append(u.clone())

        # Apply dropout to output (regularization)
        u = self.dropout(u)

        # Compute final diagnostics
        mass_final = self.compute_mass(u)
        momentum_final = self.compute_momentum(u)
        energy_final = self.compute_energy(u, None)

        info = {
            'mass': mass_final.mean().item(),
            'mass_initial': mass_initial.mean().item(),
            'mass_conservation': (
                (mass_final - mass_initial).abs() /
                (mass_initial.abs() + 1e-8)
            ).mean().item(),
            'momentum': momentum_final.mean().item(),
            'momentum_initial': momentum_initial.mean().item(),
            'momentum_conservation': (
                (momentum_final - momentum_initial).abs() /
                (momentum_initial.abs() + 1e-8)
            ).mean().item(),
            'energy': energy_final.mean().item(),
            'energy_initial': energy_initial.mean().item(),
        }

        if return_trajectory:
            info['trajectory'] = torch.stack(trajectory, dim=0)

        return u, info

    def create_soliton(
        self,
        shape: tuple,
        device: torch.device,
        position: float = 0.5,
        amplitude: float = 1.0,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Create an analytical KdV soliton solution.

        The soliton solution: u(x,t) = (c/2)·sech²(√(c/2)·(x - x₀ - ct))
        where c is the soliton speed (proportional to amplitude).

        At t=0: u(x) = A·sech²(√(A/2)·(x - x₀))
        where A = c/2 is the amplitude.

        Useful for:
        - Testing the solver against known solutions
        - Initializing with physically meaningful states
        - Studying soliton dynamics

        Args:
            shape: Shape of wave field (B, T, D) or (B, H, T, D)
            device: Device for tensors
            position: Normalized soliton position in [0, 1]
            amplitude: Soliton amplitude (determines speed)
            dtype: Data type for tensors

        Returns:
            u: Wave field representing a soliton
        """
        # Get sequence length (spatial dimension)
        seq_len = shape[-2]

        # Create spatial coordinate normalized to [-0.5, 0.5]
        x = torch.linspace(-0.5, 0.5, seq_len, device=device, dtype=dtype)
        x = x - (position - 0.5)  # Shift center to position

        # Soliton width parameter: √(A/2) scaled by sequence length
        # Ensure amplitude is positive
        A = max(abs(amplitude), 0.1)
        width_param = torch.sqrt(torch.tensor(A / 2, device=device, dtype=dtype))

        # Scale spatial coordinate for the soliton profile
        scale = width_param * seq_len * self.dx

        # Soliton profile: u = A · sech²(scale · x)
        # sech(x) = 2 / (exp(x) + exp(-x)) = 1 / cosh(x)
        sech_val = 1.0 / torch.cosh(scale * x)
        u_profile = A * sech_val ** 2

        # Expand to full shape
        # u_profile is (T,), need to broadcast to shape
        u = u_profile.view(*([1] * (len(shape) - 2)), seq_len, 1).expand(shape)

        return u.clone()

    def create_two_soliton(
        self,
        shape: tuple,
        device: torch.device,
        positions: tuple[float, float] = (0.3, 0.7),
        amplitudes: tuple[float, float] = (2.0, 1.0),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Create a two-soliton initial condition for collision experiments.

        Creates two solitons at different positions with different amplitudes.
        The taller soliton travels faster and will overtake the shorter one.

        Args:
            shape: Shape of wave field (B, T, D) or (B, H, T, D)
            device: Device for tensors
            positions: Tuple of (pos1, pos2) normalized positions
            amplitudes: Tuple of (amp1, amp2) amplitudes
            dtype: Data type for tensors

        Returns:
            u: Wave field representing two solitons
        """
        u1 = self.create_soliton(shape, device, positions[0], amplitudes[0], dtype)
        u2 = self.create_soliton(shape, device, positions[1], amplitudes[1], dtype)

        # Linear superposition (not exact for KdV, but good approximation)
        return u1 + u2

    def extra_repr(self) -> str:
        """Extra representation for module printing."""
        base = super().extra_repr()
        return (
            f"{base}, "
            f"nonlin={self.nonlin_coeff.item():.3f}, "
            f"disp={self.disp_coeff.item():.3f}, "
            f"rk4={self.use_rk4}, "
            f"damping={self.damping:.3f}"
        )
