# neuromanifold_gpt/model/soliton/sine_gordon.py
"""
Sine-Gordon Equation Solver for Soliton Propagation.

The Sine-Gordon equation:
    phi_tt - phi_xx + sin(phi) = 0

Supports topological soliton solutions (kinks/antikinks) that:
- Maintain shape during propagation
- Survive collisions unchanged (phase shift only)
- Have quantized topological charge

Key properties for language modeling:
- Kinks represent semantic boundaries/transitions
- Stable propagation maintains coherent meaning
- Collision invariance allows compositionality

Analytic soliton solution (kink):
    phi(x,t) = 4 * arctan(exp(gamma(x - vt)))
    where gamma = 1/sqrt(1-v^2) is the Lorentz factor

Reference:
- "Solitons and the Inverse Scattering Transform" - Ablowitz & Segur
- "Solitons in Mathematics and Physics" - Newell
"""

from typing import Optional

import torch
import torch.nn as nn

from .base import PDESolver


@torch.jit.script
def sine_gordon_rk4_step(
    u: torch.Tensor,
    u_t: torch.Tensor,
    dt: torch.Tensor,
    c_sq: torch.Tensor,
    u_xx_func_result: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    JIT-compiled RK4 step for Sine-Gordon equation.

    Optimized inner loop for time stepping.
    Note: u_xx must be computed externally (spectral/FD not JIT-compatible).

    Args:
        u: Wave field phi
        u_t: Time derivative dphi/dt
        dt: Time step (scalar tensor)
        c_sq: Wave speed squared (learnable parameter)
        u_xx_func_result: Pre-computed second spatial derivative

    Returns:
        Tuple of (u_new, u_t_new)
    """
    # RK4 for second-order ODE: u_tt = c^2 * u_xx - sin(u)
    # Convert to first-order system: [u, v] where v = u_t
    # du/dt = v
    # dv/dt = c^2 * u_xx - sin(u)

    # k1
    k1_u = u_t
    k1_v = c_sq * u_xx_func_result - torch.sin(u)

    # k2 (midpoint using k1)
    u_mid1 = u + 0.5 * dt * k1_u
    v_mid1 = u_t + 0.5 * dt * k1_v
    k2_u = v_mid1
    k2_v = c_sq * u_xx_func_result - torch.sin(u_mid1)  # u_xx approx same

    # k3 (midpoint using k2)
    u_mid2 = u + 0.5 * dt * k2_u
    v_mid2 = u_t + 0.5 * dt * k2_v
    k3_u = v_mid2
    k3_v = c_sq * u_xx_func_result - torch.sin(u_mid2)

    # k4 (endpoint using k3)
    u_end = u + dt * k3_u
    v_end = u_t + dt * k3_v
    k4_u = v_end
    k4_v = c_sq * u_xx_func_result - torch.sin(u_end)

    # Combine
    u_new = u + (dt / 6.0) * (k1_u + 2.0 * k2_u + 2.0 * k3_u + k4_u)
    u_t_new = u_t + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

    return u_new, u_t_new


class SineGordonSolver(PDESolver):
    """
    Solver for the Sine-Gordon equation.

    The Sine-Gordon equation: phi_tt - c^2*phi_xx + sin(phi) = 0

    This PDE admits topological soliton solutions (kinks) that:
    - Propagate with constant velocity without changing shape
    - Pass through each other during collisions (phase shift only)
    - Carry quantized topological charge Q = (phi(+inf) - phi(-inf)) / 2*pi

    For language modeling, we use solitons to represent:
    - Semantic units that maintain coherence over long distances
    - Information packets that survive attention mixing
    - Boundary markers between semantic regions

    Example:
        >>> solver = SineGordonSolver(dim=64, wave_speed=1.0)
        >>> u = torch.randn(2, 32, 64)  # (batch, seq, dim)
        >>> u_evolved, info = solver(u, n_steps=5)
        >>> print(info['energy'])  # Should be approximately conserved
    """

    def __init__(
        self,
        dim: int,
        dt: float = 0.05,
        dx: float = 1.0,
        n_steps: int = 5,
        wave_speed: float = 1.0,
        use_spectral: bool = True,
        use_rk4: bool = True,
        clamp_min: float = -10.0,
        clamp_max: float = 10.0,
        dropout: float = 0.0,
        damping: float = 0.0,
        causal: bool = False,
    ):
        """
        Initialize Sine-Gordon solver.

        Args:
            dim: Feature dimension for the wave field
            dt: Time step for numerical integration (smaller = more stable)
            dx: Spatial step for finite differences
            n_steps: Default number of integration steps per forward pass
            wave_speed: Wave propagation speed c (appears as c^2 in the equation)
            use_spectral: Use FFT-based spectral derivatives (more accurate)
            use_rk4: Use RK4 time stepping (more accurate than Euler)
            clamp_min: Minimum value for clamping (numerical stability)
            clamp_max: Maximum value for clamping (numerical stability)
            dropout: Dropout probability for regularization
            damping: Optional damping coefficient (adds -gamma*phi_t term)
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

        # Learnable wave speed squared (c^2)
        self.c_squared = nn.Parameter(torch.tensor(wave_speed**2))

        # Optional: learnable nonlinearity strength
        # Standard Sine-Gordon has sin(phi), but we can learn a scaling
        self.nonlin_scale = nn.Parameter(torch.tensor(1.0))

    def compute_rhs(
        self,
        u: torch.Tensor,
        u_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute right-hand side: phi_tt = c^2*phi_xx - sin(phi) - gamma*phi_t

        Args:
            u: Wave field phi of shape (B, T, D) or (B, H, T, D)
            u_t: Time derivative dphi/dt (required for second-order)

        Returns:
            Acceleration d^2phi/dt^2
        """
        # Spatial second derivative: phi_xx
        u_xx = self.spatial_derivative(u, order=2)

        # Nonlinear term: sin(phi) scaled by learnable parameter
        sin_term = torch.sin(self.nonlin_scale * u)

        # RHS = c^2*phi_xx - sin(phi)
        rhs = self.c_squared * u_xx - sin_term

        # Optional damping: -gamma*phi_t
        if self.damping > 0 and u_t is not None:
            rhs = rhs - self.damping * u_t

        return rhs

    def compute_potential_energy(self, u: torch.Tensor) -> torch.Tensor:
        """
        Compute Sine-Gordon potential energy: V(phi) = 1 - cos(phi)

        The full Hamiltonian is:
        H = integral [1/2*phi_t^2 + 1/2*c^2*phi_x^2 + (1 - cos(phi))] dx

        Args:
            u: Wave field phi

        Returns:
            Potential energy per batch element
        """
        # V(phi) = 1 - cos(phi), integrated over space
        potential = (1 - torch.cos(self.nonlin_scale * u)).sum(dim=(-2, -1))
        return potential

    def compute_topological_charge(self, u: torch.Tensor) -> torch.Tensor:
        """
        Compute topological charge Q = (1/2*pi) * integral phi_x dx = [phi(+inf) - phi(-inf)] / 2*pi

        For kink: Q = +1
        For antikink: Q = -1
        Charge is conserved during evolution.

        Args:
            u: Wave field phi of shape (B, T, D) or (B, H, T, D)

        Returns:
            Topological charge per batch element, shape (B,) or (B, H)
        """
        # Boundary difference (assuming periodic -> 0, or use endpoints)
        # For finite domain: Q approx (u[:, -1, :] - u[:, 0, :]) / (2*pi)
        phi_diff = u[..., -1, :] - u[..., 0, :]
        charge = phi_diff / (2 * torch.pi)

        # Average over feature dimension
        return charge.mean(dim=-1)

    def forward(
        self,
        u: torch.Tensor,
        n_steps: Optional[int] = None,
        u_t: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """
        Evolve wave field according to Sine-Gordon equation.

        Args:
            u: Initial wave field phi(x,0) of shape (B, T, D) or (B, H, T, D)
            n_steps: Number of time steps (overrides self.n_steps)
            u_t: Initial velocity dphi/dt|_{t=0} (optional, defaults to zeros)
            return_trajectory: If True, return full trajectory (expensive)

        Returns:
            Tuple of (u_final, info) where:
            - u_final: Evolved wave field phi(x, T)
            - info: Dictionary containing:
                - 'energy': Final total energy
                - 'energy_initial': Initial energy (for conservation check)
                - 'topological_charge': Topological charge Q
                - 'trajectory': Full trajectory if return_trajectory=True
        """
        steps = n_steps if n_steps is not None else self.n_steps
        dt = self.dt.abs()  # Ensure positive

        # Initialize velocity if not provided
        if u_t is None:
            u_t = torch.zeros_like(u)

        # Compute initial energy for conservation monitoring
        energy_initial = self.compute_energy(u, u_t)

        # Optional trajectory storage
        trajectory = [u.clone()] if return_trajectory else None

        # Time stepping loop
        for _ in range(steps):
            if self.use_rk4:
                # Pre-compute spatial derivative (not JIT-compatible)
                u_xx = self.spatial_derivative(u, order=2)

                # Use JIT-compiled RK4 step
                u, u_t = sine_gordon_rk4_step(u, u_t, dt, self.c_squared, u_xx)

                # Apply damping manually after RK4 if needed
                if self.damping > 0:
                    u_t = u_t - dt * self.damping * u_t
            else:
                # Fallback to base class Euler step
                u, u_t = self.euler_step(u, u_t)

            # Clamp for numerical stability
            u = self.clamp_field(u)
            u_t = self.clamp_field(u_t)

            if return_trajectory:
                trajectory.append(u.clone())

        # Apply dropout to output (regularization)
        u = self.dropout(u)

        # Compute final diagnostics
        energy_final = self.compute_energy(u, u_t)
        charge = self.compute_topological_charge(u)

        info = {
            "energy": energy_final.mean().item(),
            "energy_initial": energy_initial.mean().item(),
            "energy_conservation": (
                (energy_final - energy_initial).abs() / (energy_initial.abs() + 1e-8)
            )
            .mean()
            .item(),
            "topological_charge": charge.mean().item(),
            "u_t": u_t,  # Return velocity for chaining
        }

        if return_trajectory:
            info["trajectory"] = torch.stack(trajectory, dim=0)

        return u, info

    def create_kink_soliton(
        self,
        shape: tuple,
        device: torch.device,
        position: float = 0.5,
        velocity: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create an analytical kink soliton solution.

        The kink solution: phi(x,t) = 4*arctan(exp(gamma*(x - x0 - v*t)))
        where gamma = 1/sqrt(1-v^2) is the Lorentz factor.

        Useful for:
        - Testing the solver against known solutions
        - Initializing with physically meaningful states
        - Studying soliton dynamics

        Args:
            shape: Shape of wave field (B, T, D) or (B, H, T, D)
            device: Device for tensors
            position: Normalized kink position in [0, 1]
            velocity: Kink velocity (|v| < 1 for stability)
            dtype: Data type for tensors

        Returns:
            Tuple of (u, u_t) representing kink state
        """
        # Get sequence length (spatial dimension)
        seq_len = shape[-2]

        # Create spatial coordinate
        x = torch.linspace(0, 1, seq_len, device=device, dtype=dtype)
        x = x - position  # Center at kink position

        # Lorentz factor (cap velocity for numerical stability)
        v = min(max(velocity, -0.99), 0.99)
        gamma = 1.0 / torch.sqrt(torch.tensor(1.0 - v**2, device=device, dtype=dtype))

        # Kink profile: phi = 4*arctan(exp(gamma*(x - v*t)))
        # At t=0: phi = 4*arctan(exp(gamma*x))
        # Scale spatial coordinate by wave speed
        c = torch.sqrt(self.c_squared.abs())
        scale = (gamma * c * seq_len).clamp(min=0.1)

        phi = 4 * torch.atan(torch.exp(scale * x))

        # Time derivative: phi_t = -4*gamma*v*sech(gamma(x-vt))*c / sqrt(1+(...)^2)
        # At t=0, simplify using chain rule
        exp_term = torch.exp(scale * x)
        sech_sq = 4 * exp_term / (1 + exp_term**2) ** 2
        phi_t = -v * c * scale * sech_sq

        # Expand to full shape
        # phi is (T,), need to broadcast to shape
        u = phi.view(*([1] * (len(shape) - 2)), seq_len, 1).expand(shape)
        u_t = phi_t.view(*([1] * (len(shape) - 2)), seq_len, 1).expand(shape)

        return u, u_t

    def extra_repr(self) -> str:
        """Extra representation for module printing."""
        base = super().extra_repr()
        return (
            f"{base}, "
            f"wave_speed={torch.sqrt(self.c_squared.abs()).item():.3f}, "
            f"rk4={self.use_rk4}, "
            f"damping={self.damping:.3f}"
        )
