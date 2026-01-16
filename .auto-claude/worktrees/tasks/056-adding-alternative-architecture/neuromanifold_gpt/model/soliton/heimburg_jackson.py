# neuromanifold_gpt/model/soliton/heimburg_jackson.py
"""
Heimburg-Jackson Thermodynamic Soliton Model.

The Heimburg-Jackson equation models soliton propagation in lipid membranes:
    rho_tt = (c^2(rho) * rho_x)_x - h * rho_xxxx

Where:
    rho: Density deviation from equilibrium
    c^2(rho) = c0^2 + p*rho + q*rho^2: State-dependent wave speed
    h: Dispersion coefficient (fourth-order term)

Key properties:
- State-dependent wave speed enables nonlinear focusing
- Fourth-order dispersion balances steepening
- Stable soliton solutions (adiabatic pulses)
- Models action potential propagation in nerves

For language modeling:
- Thermodynamic solitons model stable semantic coherence
- State-dependent speed allows context-aware propagation
- Fourth-order dynamics provide richer structure than KdV

Reference:
- Heimburg & Jackson (2005), "On soliton propagation in biomembranes
  and nerves", PNAS
- Kaufmann, "Action Potentials and Electrochemical Coupling" (1989)
"""

from typing import Optional

import torch
import torch.nn as nn

from .base import PDESolver


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
    damping: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    JIT-compiled RK4 step for Heimburg-Jackson equation.

    The equation: rho_tt = d/dx[c^2(rho) * rho_x] - h * rho_xxxx
    where c^2(rho) = c0^2 + p*rho + q*rho^2

    Args:
        rho: Density field
        rho_t: Time derivative drho/dt
        dt: Time step
        c0_sq: Base wave speed squared
        p_coeff: Linear coefficient for c^2(rho)
        q_coeff: Quadratic coefficient for c^2(rho)
        h_disp: Dispersion coefficient
        rho_x: First spatial derivative
        rho_xx: Second spatial derivative
        rho_xxxx: Fourth spatial derivative
        damping: Damping coefficient

    Returns:
        Tuple of (rho_new, rho_t_new)
    """
    # c^2(rho) = c0^2 + p*rho + q*rho^2
    c_sq = c0_sq + p_coeff * rho + q_coeff * rho * rho

    # dc^2/drho = p + 2*q*rho
    dc_sq_drho = p_coeff + 2.0 * q_coeff * rho

    # rhs = d/dx[c^2(rho) * rho_x] - h * rho_xxxx
    #     = dc^2/drho * rho_x^2 + c^2(rho) * rho_xx - h * rho_xxxx
    rhs_base = dc_sq_drho * rho_x * rho_x + c_sq * rho_xx - h_disp * rho_xxxx

    # RK4 for second-order ODE (inlined compute_rhs: rhs_base - damping * v)
    # k1
    k1_rho = rho_t
    k1_v = rhs_base - damping * rho_t

    # k2
    rho_mid1 = rho + 0.5 * dt * k1_rho
    v_mid1 = rho_t + 0.5 * dt * k1_v
    k2_rho = v_mid1
    k2_v = rhs_base - damping * v_mid1

    # k3
    rho_mid2 = rho + 0.5 * dt * k2_rho
    v_mid2 = rho_t + 0.5 * dt * k2_v
    k3_rho = v_mid2
    k3_v = rhs_base - damping * v_mid2

    # k4
    rho_end = rho + dt * k3_rho
    v_end = rho_t + dt * k3_v
    k4_rho = v_end
    k4_v = rhs_base - damping * v_end

    # Combine
    rho_new = rho + (dt / 6.0) * (k1_rho + 2.0 * k2_rho + 2.0 * k3_rho + k4_rho)
    rho_t_new = rho_t + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

    return rho_new, rho_t_new


class HeimburgJacksonSolver(PDESolver):
    """
    Solver for the Heimburg-Jackson thermodynamic soliton equation.

    The equation: rho_tt = d/dx[c^2(rho) * rho_x] - h * rho_xxxx

    Where the state-dependent wave speed is:
        c^2(rho) = c0^2 + p*rho + q*rho^2

    This PDE models:
    - Thermodynamic solitons in lipid membranes
    - Action potential propagation in nerves
    - Adiabatic pulses that conserve enthalpy

    For language modeling:
    - State-dependent dynamics enable context-aware processing
    - Fourth-order dispersion provides richer structure
    - Thermodynamic constraints ensure stability

    Example:
        >>> solver = HeimburgJacksonSolver(dim=64)
        >>> rho = torch.randn(2, 32, 64)  # (batch, seq, dim)
        >>> rho_evolved, info = solver(rho, n_steps=5)
        >>> print(info['enthalpy'])  # Thermodynamic diagnostic
    """

    def __init__(
        self,
        dim: int,
        dt: float = 0.01,
        dx: float = 1.0,
        n_steps: int = 5,
        c0_squared: float = 1.0,
        p_coeff: float = -1.0,
        q_coeff: float = 1.0,
        h_disp: float = 0.1,
        use_spectral: bool = True,
        use_rk4: bool = True,
        clamp_min: float = -10.0,
        clamp_max: float = 10.0,
        dropout: float = 0.0,
        damping: float = 0.0,
    ):
        """
        Initialize Heimburg-Jackson solver.

        Args:
            dim: Feature dimension for the wave field
            dt: Time step for numerical integration
            dx: Spatial step for finite differences
            n_steps: Default number of integration steps
            c0_squared: Base wave speed squared (c0^2)
            p_coeff: Linear coefficient for c^2(rho) (p)
            q_coeff: Quadratic coefficient for c^2(rho) (q)
            h_disp: Dispersion coefficient (h)
            use_spectral: Use FFT-based spectral derivatives
            use_rk4: Use RK4 time stepping
            clamp_min: Minimum value for clamping
            clamp_max: Maximum value for clamping
            dropout: Dropout probability
            damping: Optional damping coefficient
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

    def compute_wave_speed_squared(self, rho: torch.Tensor) -> torch.Tensor:
        """
        Compute state-dependent wave speed squared.

        c^2(rho) = c0^2 + p*rho + q*rho^2

        Args:
            rho: Density field

        Returns:
            Wave speed squared at each point
        """
        return self.c0_squared + self.p_coeff * rho + self.q_coeff * rho * rho

    def compute_rhs(
        self,
        u: torch.Tensor,
        u_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute right-hand side: rho_tt = d/dx[c^2(rho)*rho_x] - h*rho_xxxx - gamma*rho_t

        Args:
            u: Density field rho of shape (B, T, D)
            u_t: Time derivative drho/dt (required for second-order)

        Returns:
            Acceleration d^2rho/dt^2
        """
        rho = u

        # Spatial derivatives
        rho_x = self.spatial_derivative(rho, order=1)
        rho_xx = self.spatial_derivative(rho, order=2)

        # State-dependent wave speed
        c_sq = self.compute_wave_speed_squared(rho)

        # dc^2/drho = p + 2*q*rho
        dc_sq_drho = self.p_coeff + 2.0 * self.q_coeff * rho

        # d/dx[c^2(rho)*rho_x] = dc^2/drho * rho_x^2 + c^2(rho) * rho_xx
        advection = dc_sq_drho * rho_x * rho_x + c_sq * rho_xx

        # Fourth-order dispersion term: -h * rho_xxxx
        # Compute as second derivative of rho_xx
        rho_xxxx = self.spatial_derivative(rho_xx, order=2)
        dispersion = -self.h_disp * rho_xxxx

        # RHS = advection + dispersion
        rhs = advection + dispersion

        # Optional damping
        if self.damping > 0 and u_t is not None:
            rhs = rhs - self.damping * u_t

        return rhs

    def compute_potential_energy(self, u: torch.Tensor) -> torch.Tensor:
        """
        Compute thermodynamic potential energy.

        For Heimburg-Jackson, the potential involves:
        - Quadratic term from c0^2
        - Cubic term from p coefficient
        - Quartic term from q coefficient

        Args:
            u: Density field rho

        Returns:
            Potential energy per batch element
        """
        rho = u
        # Integrate the potential: V(rho) such that dV/drho relates to c^2(rho)
        # Simplified: V = c0^2/2 * rho^2 + p/3 * rho^3 + q/4 * rho^4
        potential = (
            self.c0_squared / 2 * (rho**2).sum(dim=(-2, -1)) +
            self.p_coeff / 3 * (rho**3).sum(dim=(-2, -1)) +
            self.q_coeff / 4 * (rho**4).sum(dim=(-2, -1))
        )
        return potential.abs()

    def compute_enthalpy(self, rho: torch.Tensor, rho_t: torch.Tensor) -> torch.Tensor:
        """
        Compute adiabatic enthalpy (conserved for undamped system).

        H = integral [1/2 * rho_t^2 + 1/2 * c^2(rho) * rho_x^2 + h/2 * rho_xx^2] dx

        Args:
            rho: Density field
            rho_t: Time derivative

        Returns:
            Enthalpy per batch element
        """
        # Kinetic energy
        kinetic = 0.5 * (rho_t**2).sum(dim=(-2, -1))

        # Elastic energy (wave speed term)
        rho_x = self.spatial_derivative(rho, order=1)
        c_sq = self.compute_wave_speed_squared(rho)
        elastic = 0.5 * (c_sq * rho_x**2).sum(dim=(-2, -1))

        # Dispersion energy
        rho_xx = self.spatial_derivative(rho, order=2)
        dispersion = 0.5 * self.h_disp * (rho_xx**2).sum(dim=(-2, -1))

        return kinetic + elastic + dispersion

    def forward(
        self,
        u: torch.Tensor,
        n_steps: Optional[int] = None,
        u_t: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """
        Evolve density field according to Heimburg-Jackson equation.

        Args:
            u: Initial density field rho(x,0) of shape (B, T, D)
            n_steps: Number of time steps
            u_t: Initial velocity drho/dt|_{t=0} (defaults to zeros)
            return_trajectory: If True, return full trajectory

        Returns:
            Tuple of (rho_final, info)
        """
        steps = n_steps if n_steps is not None else self.n_steps
        dt = self.dt.abs()

        rho = u
        rho_t = u_t if u_t is not None else torch.zeros_like(u)

        # Compute initial diagnostics
        enthalpy_initial = self.compute_enthalpy(rho, rho_t)
        energy_initial = self.compute_energy(rho, rho_t)

        # Optional trajectory storage
        trajectory = [rho.clone()] if return_trajectory else None

        # Time stepping loop
        for _ in range(steps):
            if self.use_rk4:
                # Pre-compute derivatives
                rho_x = self.spatial_derivative(rho, order=1)
                rho_xx = self.spatial_derivative(rho, order=2)
                rho_xxxx = self.spatial_derivative(rho_xx, order=2)

                # JIT-compiled RK4 step
                rho, rho_t = heimburg_jackson_rk4_step(
                    rho, rho_t, dt,
                    self.c0_squared, self.p_coeff, self.q_coeff, self.h_disp,
                    rho_x, rho_xx, rho_xxxx,
                    self.damping,
                )
            else:
                # Euler step
                rho, rho_t = self.euler_step(rho, rho_t)

            # Clamp for stability
            rho = self.clamp_field(rho)
            rho_t = self.clamp_field(rho_t)

            if return_trajectory:
                trajectory.append(rho.clone())

        # Apply dropout
        rho = self.dropout(rho)

        # Compute final diagnostics
        enthalpy_final = self.compute_enthalpy(rho, rho_t)
        energy_final = self.compute_energy(rho, rho_t)
        wave_speed = torch.sqrt(self.compute_wave_speed_squared(rho).mean().clamp(min=1e-6))

        info = {
            'enthalpy': enthalpy_final.mean().item(),
            'enthalpy_initial': enthalpy_initial.mean().item(),
            'enthalpy_conservation': (
                (enthalpy_final - enthalpy_initial).abs() /
                (enthalpy_initial.abs() + 1e-8)
            ).mean().item(),
            'energy': energy_final.mean().item(),
            'energy_initial': energy_initial.mean().item(),
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
        amplitude: float = 0.5,
        position: float = 0.5,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create an approximate stationary soliton solution.

        For Heimburg-Jackson, the exact soliton profile depends on
        the specific parameters. We use a Gaussian approximation.

        Args:
            shape: Shape of density field (B, T, D)
            device: Device for tensors
            amplitude: Soliton amplitude
            position: Normalized position in [0, 1]
            dtype: Data type

        Returns:
            Tuple of (rho, rho_t) for soliton state
        """
        seq_len = shape[-2]

        # Spatial coordinate
        x = torch.linspace(-0.5, 0.5, seq_len, device=device, dtype=dtype)
        x = x - (position - 0.5)

        # Approximate soliton with sech^2 profile
        width = 0.1  # Normalized width
        scale = 1.0 / (width * seq_len)
        sech_sq = 1.0 / torch.cosh(scale * x)**2
        profile = amplitude * sech_sq

        # Expand to full shape
        rho = profile.view(*([1] * (len(shape) - 2)), seq_len, 1).expand(shape).clone()
        rho_t = torch.zeros_like(rho)  # Stationary soliton

        return rho, rho_t

    def create_moving_soliton(
        self,
        shape: tuple,
        device: torch.device,
        amplitude: float = 0.5,
        position: float = 0.3,
        velocity: float = 0.5,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create a moving soliton initial condition.

        Args:
            shape: Shape of density field
            device: Device for tensors
            amplitude: Soliton amplitude
            position: Initial position
            velocity: Soliton velocity
            dtype: Data type

        Returns:
            Tuple of (rho, rho_t)
        """
        rho, _ = self.create_soliton(shape, device, amplitude, position, dtype)

        # For moving soliton, rho_t is related to rho_x
        # rho_t = -v * rho_x for right-moving wave
        rho_x = self.spatial_derivative(rho, order=1)
        rho_t = -velocity * rho_x

        return rho, rho_t

    def create_collision_state(
        self,
        shape: tuple,
        device: torch.device,
        amplitudes: tuple[float, float] = (0.5, 0.3),
        positions: tuple[float, float] = (0.3, 0.7),
        velocities: tuple[float, float] = (0.5, -0.5),
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create a two-soliton collision initial condition.

        Args:
            shape: Shape of density field
            device: Device for tensors
            amplitudes: (amp1, amp2) amplitudes
            positions: (pos1, pos2) positions
            velocities: (vel1, vel2) velocities
            dtype: Data type

        Returns:
            Tuple of (rho, rho_t)
        """
        rho1, rho_t1 = self.create_moving_soliton(
            shape, device, amplitudes[0], positions[0], velocities[0], dtype
        )
        rho2, rho_t2 = self.create_moving_soliton(
            shape, device, amplitudes[1], positions[1], velocities[1], dtype
        )

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
            f"rk4={self.use_rk4}"
        )
