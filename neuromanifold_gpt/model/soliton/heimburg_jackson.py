# neuromanifold_gpt/model/soliton/heimburg_jackson.py
"""
Heimburg-Jackson Thermodynamic Soliton Model for Neural Membranes.

The Heimburg-Jackson equation:
    ρ_tt = ∂/∂x[(c₀² + p·ρ + q·ρ²)·ρ_x] - h·ρ_xxxx

This model describes action potentials as acoustic solitons in lipid membranes.
Unlike Hodgkin-Huxley (electrical), this captures the mechanical/thermodynamic
nature of nerve impulse propagation.

Key physics:
- Acoustic compression wave in lipid bilayer
- State-dependent wave speed (nonlinear elasticity)
- Fourth-order dispersion balances nonlinear steepening
- Adiabatic: no heat exchange during propagation

Key properties for language modeling:
- Solitons represent coherent semantic units
- Thermodynamic stability ensures robustness
- State-dependent speed creates amplitude-velocity coupling
- Fourth-order dispersion enables sharper localization

Analytic soliton solution (approximate):
    ρ(x,t) ≈ A · sech²(β(x - v·t))
    where v depends on amplitude A through state equation

Reference:
- Heimburg & Jackson (2005) "On soliton propagation in biomembranes"
- Heimburg (2007) "Thermal Biophysics of Membranes"
- Kaufmann (1989) "Action Potentials and Electrochemical Coupling"
"""

from typing import Optional

import torch
import torch.nn as nn

from .base import PDESolver


@torch.jit.script
def _compute_hj_rhs(
    r: torch.Tensor,
    r_x: torch.Tensor,
    r_xx: torch.Tensor,
    r_xxxx: torch.Tensor,
    c0_sq: torch.Tensor,
    p_coeff: torch.Tensor,
    q_coeff: torch.Tensor,
    h_disp: torch.Tensor,
) -> torch.Tensor:
    """
    Compute RHS of Heimburg-Jackson equation (JIT helper).

    RHS = c²·ρ_xx + dc²/dρ·ρ_x² - h·ρ_xxxx
    where c² = c₀² + p·ρ + q·ρ²
    """
    c2 = c0_sq + p_coeff * r + q_coeff * r * r
    dc2 = p_coeff + 2.0 * q_coeff * r
    return c2 * r_xx + dc2 * r_x * r_x - h_disp * r_xxxx


@torch.jit.script
def heimburg_jackson_rk4_step(
    rho: torch.Tensor,
    rho_t: torch.Tensor,
    dt: torch.Tensor,
    c0_sq: torch.Tensor,
    p_coeff: torch.Tensor,
    q_coeff: torch.Tensor,
    h_disp: torch.Tensor,
    rho_x: torch.Tensor,
    rho_xx: torch.Tensor,
    rho_xxxx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    JIT-compiled RK4 step for Heimburg-Jackson equation.

    The equation: ρ_tt = ∂/∂x[(c₀² + p·ρ + q·ρ²)·ρ_x] - h·ρ_xxxx

    Expanding the spatial derivative:
    ρ_tt = (c₀² + p·ρ + q·ρ²)·ρ_xx + (p + 2q·ρ)·ρ_x² - h·ρ_xxxx

    Args:
        rho: Density field ρ(x,t)
        rho_t: Time derivative ∂ρ/∂t
        dt: Time step (scalar tensor)
        c0_sq: Base wave speed squared c₀²
        p_coeff: Linear nonlinearity coefficient p
        q_coeff: Quadratic nonlinearity coefficient q
        h_disp: Dispersion coefficient h
        rho_x: Pre-computed first spatial derivative
        rho_xx: Pre-computed second spatial derivative
        rho_xxxx: Pre-computed fourth spatial derivative

    Returns:
        Tuple of (rho_new, rho_t_new)
    """
    # k1
    k1_rho = rho_t
    k1_v = _compute_hj_rhs(rho, rho_x, rho_xx, rho_xxxx, c0_sq, p_coeff, q_coeff, h_disp)

    # k2 (midpoint using k1)
    rho_mid1 = rho + 0.5 * dt * k1_rho
    v_mid1 = rho_t + 0.5 * dt * k1_v
    k2_rho = v_mid1
    # Derivatives approx same at midpoint
    k2_v = _compute_hj_rhs(rho_mid1, rho_x, rho_xx, rho_xxxx, c0_sq, p_coeff, q_coeff, h_disp)

    # k3 (midpoint using k2)
    rho_mid2 = rho + 0.5 * dt * k2_rho
    v_mid2 = rho_t + 0.5 * dt * k2_v
    k3_rho = v_mid2
    k3_v = _compute_hj_rhs(rho_mid2, rho_x, rho_xx, rho_xxxx, c0_sq, p_coeff, q_coeff, h_disp)

    # k4 (endpoint using k3)
    rho_end = rho + dt * k3_rho
    v_end = rho_t + dt * k3_v
    k4_rho = v_end
    k4_v = _compute_hj_rhs(rho_end, rho_x, rho_xx, rho_xxxx, c0_sq, p_coeff, q_coeff, h_disp)

    # Combine (RK4 formula)
    rho_new = rho + (dt / 6.0) * (k1_rho + 2.0 * k2_rho + 2.0 * k3_rho + k4_rho)
    rho_t_new = rho_t + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

    return rho_new, rho_t_new


class HeimburgJacksonSolver(PDESolver):
    """
    Solver for the Heimburg-Jackson thermodynamic soliton equation.

    The Heimburg-Jackson equation:
        ρ_tt = ∂/∂x[(c₀² + p·ρ + q·ρ²)·ρ_x] - h·ρ_xxxx

    This PDE describes acoustic solitons in lipid membranes:
    - State-dependent wave speed creates nonlinear focusing
    - Fourth-order dispersion prevents wave breaking
    - Balance creates stable soliton solutions

    For language modeling, we use this thermodynamic soliton model to:
    - Represent semantic units as compression waves
    - Model attention as wave interference
    - Maintain coherence through stable propagation

    Example:
        >>> solver = HeimburgJacksonSolver(dim=64)
        >>> rho = torch.randn(2, 32, 64)  # (batch, seq, dim)
        >>> rho_evolved, info = solver(rho, n_steps=5)
        >>> print(info['enthalpy'])  # Adiabatic invariant
    """

    def __init__(
        self,
        dim: int,
        dt: float = 0.01,
        dx: float = 1.0,
        n_steps: int = 5,
        c0_squared: float = 1.0,
        p_coeff: float = -16.6,
        q_coeff: float = 79.5,
        h_disp: float = 0.1,
        use_spectral: bool = True,
        use_rk4: bool = True,
        clamp_min: float = -5.0,
        clamp_max: float = 5.0,
        dropout: float = 0.0,
        damping: float = 0.0,
    ):
        """
        Initialize Heimburg-Jackson solver.

        The default coefficients are from Heimburg & Jackson (2005) for lipid
        membranes near phase transition. The values are:
        - c₀² ≈ 176.6 m²/s² (adjusted to 1.0 for normalization)
        - p ≈ -16.6 c₀²/ρ₀ (negative: speed decreases with compression)
        - q ≈ +79.5 c₀²/ρ₀² (positive: eventually speed increases)
        - h ≈ dispersion length scale

        Args:
            dim: Feature dimension for the density field
            dt: Time step for numerical integration (smaller = more stable)
            dx: Spatial step for finite differences
            n_steps: Default number of integration steps per forward pass
            c0_squared: Low-amplitude sound speed squared c₀²
            p_coeff: Linear nonlinearity coefficient p (typically negative)
            q_coeff: Quadratic nonlinearity coefficient q (typically positive)
            h_disp: Dispersion coefficient h (controls soliton width)
            use_spectral: Use FFT-based spectral derivatives (more accurate)
            use_rk4: Use RK4 time stepping (more accurate than Euler)
            clamp_min: Minimum value for clamping (numerical stability)
            clamp_max: Maximum value for clamping (numerical stability)
            dropout: Dropout probability for regularization
            damping: Optional damping coefficient (adds -γ·ρ_t term)
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
        )

        self.use_rk4 = use_rk4
        self.damping = damping

        # Learnable thermodynamic parameters
        self.c0_squared = nn.Parameter(torch.tensor(c0_squared))
        self.p_coeff = nn.Parameter(torch.tensor(p_coeff))
        self.q_coeff = nn.Parameter(torch.tensor(q_coeff))
        self.h_disp = nn.Parameter(torch.tensor(h_disp))

    def compute_state_dependent_speed(self, rho: torch.Tensor) -> torch.Tensor:
        """
        Compute state-dependent wave speed: c(ρ) = √(c₀² + p·ρ + q·ρ²)

        This is the key nonlinearity in the Heimburg-Jackson model.
        The wave speed depends on the local density (compression level).

        Args:
            rho: Density field

        Returns:
            Local wave speed c(ρ)
        """
        c_sq = self.c0_squared + self.p_coeff * rho + self.q_coeff * rho ** 2
        # Ensure positive for stability
        c_sq = c_sq.clamp(min=1e-6)
        return torch.sqrt(c_sq)

    def compute_rhs(
        self,
        rho: torch.Tensor,
        rho_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute right-hand side: ρ_tt = ∂/∂x[(c₀² + p·ρ + q·ρ²)·ρ_x] - h·ρ_xxxx - γ·ρ_t

        Args:
            rho: Density field ρ of shape (B, T, D) or (B, H, T, D)
            rho_t: Time derivative ∂ρ/∂t (required for second-order)

        Returns:
            Acceleration ∂²ρ/∂t²
        """
        # Spatial derivatives
        rho_x = self.spatial_derivative(rho, order=1)
        rho_xx = self.spatial_derivative(rho, order=2)

        # Fourth-order derivative for dispersion
        rho_xxxx = self.spatial_derivative(self.spatial_derivative(rho, order=2), order=2)

        # State-dependent sound speed squared
        c_sq = self.c0_squared + self.p_coeff * rho + self.q_coeff * rho ** 2

        # Derivative of c² with respect to ρ
        dc_sq_drho = self.p_coeff + 2.0 * self.q_coeff * rho

        # RHS = ∂/∂x[c²·ρ_x] - h·ρ_xxxx
        #     = c²·ρ_xx + dc²/dρ·ρ_x² - h·ρ_xxxx
        rhs = c_sq * rho_xx + dc_sq_drho * rho_x ** 2 - self.h_disp * rho_xxxx

        # Optional damping: -γ·ρ_t
        if self.damping > 0 and rho_t is not None:
            rhs = rhs - self.damping * rho_t

        return rhs

    def compute_potential_energy(self, rho: torch.Tensor) -> torch.Tensor:
        """
        Compute thermodynamic potential energy.

        For the Heimburg-Jackson model, the potential energy density is:
        U(ρ) = ½c₀²·ρ² + (p/3)·ρ³ + (q/4)·ρ⁴

        This is derived from integrating the state equation.

        Args:
            rho: Density field ρ

        Returns:
            Potential energy per batch element
        """
        # Integrate c²(ρ) = c₀² + p·ρ + q·ρ² to get potential
        # U = ½c₀²·ρ² + (p/3)·ρ³ + (q/4)·ρ⁴
        potential = (
            0.5 * self.c0_squared * rho ** 2
            + (self.p_coeff / 3) * rho ** 3
            + (self.q_coeff / 4) * rho ** 4
        )
        return potential.sum(dim=(-2, -1))

    def compute_enthalpy(
        self,
        rho: torch.Tensor,
        rho_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute adiabatic enthalpy (conserved for undamped system).

        The enthalpy H is the sum of:
        - Kinetic energy: ½∫ρ_t² dx
        - Elastic potential: ∫U(ρ) dx
        - Gradient energy: ½∫(∂ρ/∂x)² dx
        - Dispersion energy: ½h∫(∂²ρ/∂x²)² dx

        Args:
            rho: Density field
            rho_t: Time derivative (optional)

        Returns:
            Total enthalpy per batch element
        """
        # Kinetic energy
        if rho_t is not None:
            kinetic = 0.5 * (rho_t ** 2).sum(dim=(-2, -1))
        else:
            kinetic = torch.zeros(rho.shape[:-2], device=rho.device, dtype=rho.dtype)
            if kinetic.numel() == 0:
                kinetic = torch.zeros(rho.shape[0], device=rho.device, dtype=rho.dtype)

        # Potential energy from state equation
        potential = self.compute_potential_energy(rho)

        # Gradient energy (elastic)
        rho_x = self.spatial_derivative(rho, order=1)
        gradient = 0.5 * (rho_x ** 2).sum(dim=(-2, -1))

        # Dispersion energy (fourth-order term)
        rho_xx = self.spatial_derivative(rho, order=2)
        dispersion = 0.5 * self.h_disp.abs() * (rho_xx ** 2).sum(dim=(-2, -1))

        return kinetic + potential + gradient + dispersion

    def compute_compression(self, rho: torch.Tensor) -> torch.Tensor:
        """
        Compute peak compression (max |ρ|) as a measure of soliton amplitude.

        In the Heimburg-Jackson model, compression correlates with:
        - Wave speed (via state equation)
        - Pulse width (via dispersion balance)
        - Energy content

        Args:
            rho: Density field

        Returns:
            Peak compression per batch element
        """
        # Max absolute value over spatial and feature dimensions
        return rho.abs().amax(dim=(-2, -1))

    def forward(
        self,
        rho: torch.Tensor,
        n_steps: Optional[int] = None,
        rho_t: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """
        Evolve density field according to Heimburg-Jackson equation.

        Args:
            rho: Initial density field ρ(x,0) of shape (B, T, D) or (B, H, T, D)
            n_steps: Number of time steps (overrides self.n_steps)
            rho_t: Initial velocity ∂ρ/∂t|_{t=0} (optional, defaults to zeros)
            return_trajectory: If True, return full trajectory (expensive)

        Returns:
            Tuple of (rho_final, info) where:
            - rho_final: Evolved density field ρ(x, T)
            - info: Dictionary containing:
                - 'enthalpy': Final adiabatic enthalpy
                - 'enthalpy_initial': Initial enthalpy (for conservation check)
                - 'compression': Peak compression level
                - 'wave_speed': Average wave speed
                - 'rho_t': Final velocity (for chaining)
                - 'trajectory': Full trajectory if return_trajectory=True
        """
        steps = n_steps if n_steps is not None else self.n_steps
        dt = self.dt.abs()  # Ensure positive

        # Initialize velocity if not provided
        if rho_t is None:
            rho_t = torch.zeros_like(rho)

        # Compute initial conserved quantity for monitoring
        enthalpy_initial = self.compute_enthalpy(rho, rho_t)

        # Optional trajectory storage
        trajectory = [rho.clone()] if return_trajectory else None

        # Time stepping loop
        for _ in range(steps):
            if self.use_rk4:
                # Pre-compute spatial derivatives (not JIT-compatible)
                rho_x = self.spatial_derivative(rho, order=1)
                rho_xx = self.spatial_derivative(rho, order=2)
                rho_xxxx = self.spatial_derivative(rho_xx, order=2)

                # Use JIT-compiled RK4 step
                rho, rho_t = heimburg_jackson_rk4_step(
                    rho,
                    rho_t,
                    dt,
                    self.c0_squared,
                    self.p_coeff,
                    self.q_coeff,
                    self.h_disp,
                    rho_x,
                    rho_xx,
                    rho_xxxx,
                )

                # Apply damping manually after RK4 if needed
                if self.damping > 0:
                    rho_t = rho_t - dt * self.damping * rho_t
            else:
                # Fallback to base class Euler step
                rho, rho_t = self.euler_step(rho, rho_t)

            # Clamp for numerical stability
            rho = self.clamp_field(rho)
            rho_t = self.clamp_field(rho_t)

            if return_trajectory:
                trajectory.append(rho.clone())

        # Apply dropout to output (regularization)
        rho = self.dropout(rho)

        # Compute final diagnostics
        enthalpy_final = self.compute_enthalpy(rho, rho_t)
        compression = self.compute_compression(rho)
        wave_speed = self.compute_state_dependent_speed(rho).mean()

        info = {
            'enthalpy': enthalpy_final.mean().item(),
            'enthalpy_initial': enthalpy_initial.mean().item(),
            'enthalpy_conservation': (
                (enthalpy_final - enthalpy_initial).abs() /
                (enthalpy_initial.abs() + 1e-8)
            ).mean().item(),
            'compression': compression.mean().item(),
            'wave_speed': wave_speed.item(),
            'rho_t': rho_t,  # Return velocity for chaining
        }

        if return_trajectory:
            info['trajectory'] = torch.stack(trajectory, dim=0)

        return rho, info

    def create_soliton(
        self,
        shape: tuple,
        device: torch.device,
        position: float = 0.5,
        amplitude: float = 0.1,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create an approximate soliton solution for the Heimburg-Jackson equation.

        The soliton profile: ρ(x) ≈ A · sech²(β(x - x₀))
        where β depends on the thermodynamic parameters.

        The soliton width and velocity are coupled through the state equation.

        Args:
            shape: Shape of density field (B, T, D) or (B, H, T, D)
            device: Device for tensors
            position: Normalized soliton position in [0, 1]
            amplitude: Soliton amplitude A
            dtype: Data type for tensors

        Returns:
            Tuple of (rho, rho_t) representing soliton state
        """
        # Get sequence length (spatial dimension)
        seq_len = shape[-2]

        # Create spatial coordinate normalized to [-0.5, 0.5]
        x = torch.linspace(-0.5, 0.5, seq_len, device=device, dtype=dtype)
        x = x - (position - 0.5)  # Shift center to position

        # Soliton width parameter from dispersion balance
        # β ≈ √(|p|·A / (12·h)) for small q contribution
        h_val = self.h_disp.abs().item() + 1e-6
        p_val = self.p_coeff.abs().item() + 1e-6
        beta = torch.sqrt(torch.tensor(p_val * abs(amplitude) / (12 * h_val), device=device, dtype=dtype))

        # Scale for sequence length
        beta_scaled = beta * seq_len * self.dx

        # Soliton profile: ρ = A · sech²(β·x)
        sech_val = 1.0 / torch.cosh(beta_scaled * x)
        rho_profile = amplitude * sech_val ** 2

        # Initial velocity: moving soliton has ρ_t = -v · ρ_x
        # For stationary soliton, start with zero velocity
        rho_t_profile = torch.zeros_like(rho_profile)

        # Expand to full shape
        rho = rho_profile.view(*([1] * (len(shape) - 2)), seq_len, 1).expand(shape).clone()
        rho_t = rho_t_profile.view(*([1] * (len(shape) - 2)), seq_len, 1).expand(shape).clone()

        return rho, rho_t

    def create_moving_soliton(
        self,
        shape: tuple,
        device: torch.device,
        position: float = 0.5,
        amplitude: float = 0.1,
        velocity: float = 0.5,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create a moving soliton with initial velocity.

        For a traveling soliton, the velocity is related to amplitude through
        the state equation: v² ≈ c₀² + p·A + q·A²

        Args:
            shape: Shape of density field
            device: Device for tensors
            position: Normalized soliton position in [0, 1]
            amplitude: Soliton amplitude A
            velocity: Normalized velocity (fraction of c₀)
            dtype: Data type for tensors

        Returns:
            Tuple of (rho, rho_t) representing moving soliton
        """
        rho, _ = self.create_soliton(shape, device, position, amplitude, dtype)

        # Compute spatial derivative for velocity initialization
        rho_x = self.spatial_derivative(rho, order=1)

        # For traveling wave: ρ_t = -v · ρ_x
        # Scale velocity by base wave speed
        c0 = torch.sqrt(self.c0_squared.abs()).item()
        v_scaled = velocity * c0

        rho_t = -v_scaled * rho_x

        return rho, rho_t

    def create_collision_state(
        self,
        shape: tuple,
        device: torch.device,
        positions: tuple[float, float] = (0.25, 0.75),
        amplitudes: tuple[float, float] = (0.15, 0.1),
        velocities: tuple[float, float] = (0.5, -0.5),
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create two counter-propagating solitons for collision experiments.

        Soliton collisions in the Heimburg-Jackson model exhibit:
        - Phase shift (delay/advance) after collision
        - Shape preservation (no energy loss)
        - Amplitude-dependent interaction strength

        Args:
            shape: Shape of density field
            device: Device for tensors
            positions: Tuple of (pos1, pos2) normalized positions
            amplitudes: Tuple of (amp1, amp2) amplitudes
            velocities: Tuple of (v1, v2) normalized velocities
            dtype: Data type for tensors

        Returns:
            Tuple of (rho, rho_t) representing two-soliton collision state
        """
        rho1, rho_t1 = self.create_moving_soliton(
            shape, device, positions[0], amplitudes[0], velocities[0], dtype
        )
        rho2, rho_t2 = self.create_moving_soliton(
            shape, device, positions[1], amplitudes[1], velocities[1], dtype
        )

        # Linear superposition (good approximation when solitons are separated)
        return rho1 + rho2, rho_t1 + rho_t2

    def extra_repr(self) -> str:
        """Extra representation for module printing."""
        base = super().extra_repr()
        c0 = torch.sqrt(self.c0_squared.abs()).item()
        return (
            f"{base}, "
            f"c0={c0:.3f}, "
            f"p={self.p_coeff.item():.3f}, "
            f"q={self.q_coeff.item():.3f}, "
            f"h={self.h_disp.item():.3f}, "
            f"rk4={self.use_rk4}, "
            f"damping={self.damping:.3f}"
        )
