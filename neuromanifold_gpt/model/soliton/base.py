# neuromanifold_gpt/model/soliton/base.py
"""
Base classes for PDE-based soliton dynamics.

Implements physics-informed neural network layers for soliton propagation.
Solitons are stable, localized wave packets that maintain their shape during
propagation and survive collisions - ideal for modeling semantic units.

Key PDE types implemented in subclasses:
- Sine-Gordon: phi_tt - phi_xx + sin(phi) = 0 (topological solitons)
- KdV: u_t + 6u*u_x + u_xxx = 0 (dispersive waves)
- Heimburg-Jackson: Thermodynamic soliton model for neural membranes

The base class provides:
- Abstract interface for PDE solvers
- Common numerical methods (finite difference, spectral)
- Stability controls and clamping
- Device/dtype handling

Reference: Ablowitz & Segur, "Solitons and the Inverse Scattering Transform"
"""

from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PDEConfig:
    """
    Configuration for PDE solver modules.

    Provides default hyperparameters for soliton physics layers.
    """

    dim: int = 64
    dt: float = 0.1
    dx: float = 1.0
    n_steps: int = 5
    use_spectral: bool = True
    clamp_min: float = -10.0
    clamp_max: float = 10.0
    dropout: float = 0.0


class PDESolver(nn.Module, ABC):
    """
    Abstract base class for PDE-based soliton solvers.

    Provides the interface for physics-informed neural network layers
    that evolve wave dynamics according to specific PDEs.

    Subclasses implement specific equations:
    - SineGordonSolver: Topological solitons with sin nonlinearity
    - KdVSolver: Dispersive waves with cubic nonlinearity
    - HeimburgJacksonSolver: Thermodynamic neural membrane model

    The continuous PDE is discretized using:
    - Finite differences (spatial derivatives)
    - IMEX time stepping (stability for stiff terms)
    - Optional spectral methods (FFT-based derivatives)
    """

    def __init__(
        self,
        dim: int,
        dt: float = 0.1,
        dx: float = 1.0,
        n_steps: int = 5,
        use_spectral: bool = True,
        clamp_min: float = -10.0,
        clamp_max: float = 10.0,
        dropout: float = 0.0,
    ):
        """
        Initialize PDE solver base class.

        Args:
            dim: Feature dimension for the wave field
            dt: Time step for numerical integration
            dx: Spatial step for finite differences
            n_steps: Number of integration steps per forward pass
            use_spectral: Use FFT-based spectral derivatives (more accurate)
            clamp_min: Minimum value for clamping (numerical stability)
            clamp_max: Maximum value for clamping (numerical stability)
            dropout: Dropout probability for regularization
        """
        super().__init__()
        self.dim = dim
        self.dt_init = dt
        self.dx = dx
        self.n_steps = n_steps
        self.use_spectral = use_spectral
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        # Learnable time step (can adapt during training)
        self.dt = nn.Parameter(torch.tensor(dt))

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    @abstractmethod
    def forward(
        self,
        u: torch.Tensor,
        n_steps: Optional[int] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Evolve the wave field according to the PDE.

        Args:
            u: Wave field tensor of shape (B, T, D) or (B, H, T, D)
               B = batch, T = sequence/spatial, D = feature dimension
            n_steps: Override number of integration steps (optional)

        Returns:
            Tuple of (u_evolved, info) where:
            - u_evolved: Evolved wave field, same shape as input
            - info: Dictionary with solver statistics (energy, stability metrics)
        """
        pass

    @abstractmethod
    def compute_rhs(self, u: torch.Tensor, u_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the right-hand side of the PDE.

        This defines the specific physics of each equation type.
        Called by the time-stepping method.

        Args:
            u: Current wave field
            u_t: Time derivative (for second-order equations)

        Returns:
            Time derivative du/dt (or d^2u/dt^2 for second-order)
        """
        pass

    def spatial_derivative(
        self,
        u: torch.Tensor,
        order: int = 1,
        dim: int = -2,
    ) -> torch.Tensor:
        """
        Compute spatial derivatives using finite differences or spectral method.

        Args:
            u: Wave field tensor
            order: Derivative order (1 = first, 2 = second, 3 = third)
            dim: Dimension along which to differentiate (default: sequence dim)

        Returns:
            Spatial derivative of order `order`
        """
        if self.use_spectral:
            return self._spectral_derivative(u, order, dim)
        else:
            return self._finite_difference(u, order, dim)

    def _spectral_derivative(
        self,
        u: torch.Tensor,
        order: int,
        dim: int,
    ) -> torch.Tensor:
        """
        Compute derivative using FFT (spectral method).

        More accurate for smooth functions, especially for higher orders.
        """
        # Handle BFloat16 for FFT (not supported natively)
        orig_dtype = u.dtype
        if u.dtype == torch.bfloat16:
            u = u.float()

        # Get size along derivative dimension
        n = u.shape[dim]

        # Compute wavenumbers: k = 2*pi * [0, 1, ..., n/2, -n/2+1, ..., -1] / (n * dx)
        k = torch.fft.fftfreq(n, d=self.dx, device=u.device, dtype=u.dtype)
        k = 2 * torch.pi * k

        # Reshape k for broadcasting
        shape = [1] * u.ndim
        shape[dim] = n
        k = k.view(*shape)

        # FFT along specified dimension
        u_hat = torch.fft.fft(u, dim=dim)

        # Multiply by (ik)^order for derivative
        # For real input, need to handle Nyquist frequency carefully
        ik_power = (1j * k) ** order

        # Apply derivative in frequency domain
        du_hat = u_hat * ik_power

        # Inverse FFT
        du = torch.fft.ifft(du_hat, dim=dim)

        # Return real part (input is real)
        return du.real.to(orig_dtype)

    def _finite_difference(
        self,
        u: torch.Tensor,
        order: int,
        dim: int,
    ) -> torch.Tensor:
        """
        Compute derivative using finite differences.

        Uses central differences for accuracy, with periodic boundary conditions.
        """
        # Normalize negative dim
        if dim < 0:
            dim = u.ndim + dim

        # Get sequence length
        n = u.shape[dim]

        if order == 1:
            # Central difference: (u[i+1] - u[i-1]) / (2*dx)
            u_plus = torch.roll(u, shifts=-1, dims=dim)
            u_minus = torch.roll(u, shifts=1, dims=dim)
            return (u_plus - u_minus) / (2 * self.dx)

        elif order == 2:
            # Central difference: (u[i+1] - 2*u[i] + u[i-1]) / dx^2
            u_plus = torch.roll(u, shifts=-1, dims=dim)
            u_minus = torch.roll(u, shifts=1, dims=dim)
            return (u_plus - 2 * u + u_minus) / (self.dx**2)

        elif order == 3:
            # Central difference for third derivative
            # (u[i+2] - 2*u[i+1] + 2*u[i-1] - u[i-2]) / (2*dx^3)
            u_p2 = torch.roll(u, shifts=-2, dims=dim)
            u_p1 = torch.roll(u, shifts=-1, dims=dim)
            u_m1 = torch.roll(u, shifts=1, dims=dim)
            u_m2 = torch.roll(u, shifts=2, dims=dim)
            return (u_p2 - 2 * u_p1 + 2 * u_m1 - u_m2) / (2 * self.dx**3)

        else:
            raise ValueError(f"Derivative order {order} not implemented")

    def euler_step(
        self,
        u: torch.Tensor,
        u_t: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Single explicit Euler time step.

        Args:
            u: Current wave field
            u_t: Time derivative (for second-order systems)

        Returns:
            Tuple of (u_new, u_t_new)
        """
        dt = self.dt.abs()  # Ensure positive dt

        rhs = self.compute_rhs(u, u_t)

        if u_t is not None:
            # Second-order system: u_tt = rhs
            # u_new = u + dt * u_t
            # u_t_new = u_t + dt * rhs
            u_new = u + dt * u_t
            u_t_new = u_t + dt * rhs
            return u_new, u_t_new
        else:
            # First-order system: u_t = rhs
            u_new = u + dt * rhs
            return u_new, None

    def rk4_step(
        self,
        u: torch.Tensor,
        u_t: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Single RK4 (Runge-Kutta 4th order) time step.

        More accurate than Euler, but more expensive.

        Args:
            u: Current wave field
            u_t: Time derivative (for second-order systems)

        Returns:
            Tuple of (u_new, u_t_new)
        """
        dt = self.dt.abs()

        if u_t is not None:
            # Second-order system converted to first-order system
            # State: [u, u_t], derivatives: [u_t, rhs]
            k1_u = u_t
            k1_v = self.compute_rhs(u, u_t)

            k2_u = u_t + 0.5 * dt * k1_v
            k2_v = self.compute_rhs(u + 0.5 * dt * k1_u, k2_u)

            k3_u = u_t + 0.5 * dt * k2_v
            k3_v = self.compute_rhs(u + 0.5 * dt * k2_u, k3_u)

            k4_u = u_t + dt * k3_v
            k4_v = self.compute_rhs(u + dt * k3_u, k4_u)

            u_new = u + (dt / 6) * (k1_u + 2 * k2_u + 2 * k3_u + k4_u)
            u_t_new = u_t + (dt / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

            return u_new, u_t_new
        else:
            # First-order system
            k1 = self.compute_rhs(u, None)
            k2 = self.compute_rhs(u + 0.5 * dt * k1, None)
            k3 = self.compute_rhs(u + 0.5 * dt * k2, None)
            k4 = self.compute_rhs(u + dt * k3, None)

            u_new = u + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            return u_new, None

    def clamp_field(self, u: torch.Tensor) -> torch.Tensor:
        """
        Clamp wave field values for numerical stability.

        Args:
            u: Wave field tensor

        Returns:
            Clamped tensor
        """
        return torch.clamp(u, self.clamp_min, self.clamp_max)

    def compute_energy(self, u: torch.Tensor, u_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute total energy of the wave field.

        For conservative systems, energy should be preserved.
        Useful for monitoring numerical stability.

        Args:
            u: Wave field
            u_t: Time derivative (for kinetic energy)

        Returns:
            Total energy (scalar per batch)
        """
        # Kinetic energy: 0.5 * u_t^2
        if u_t is not None:
            kinetic = 0.5 * (u_t**2).sum(dim=(-2, -1))
        else:
            kinetic = torch.zeros(u.shape[0], device=u.device)

        # Gradient energy: 0.5 * (du/dx)^2
        u_x = self.spatial_derivative(u, order=1)
        gradient = 0.5 * (u_x**2).sum(dim=(-2, -1))

        # Potential energy is system-specific (computed in subclasses)
        potential = self.compute_potential_energy(u)

        return kinetic + gradient + potential

    def compute_potential_energy(self, u: torch.Tensor) -> torch.Tensor:
        """
        Compute potential energy (system-specific).

        Override in subclasses for specific potentials:
        - Sine-Gordon: 1 - cos(u)
        - KdV: u^3 / 3
        - Heimburg-Jackson: thermodynamic potential

        Args:
            u: Wave field

        Returns:
            Potential energy (per batch)
        """
        # Default: quadratic potential
        return 0.5 * (u**2).sum(dim=(-2, -1))

    def init_state(
        self,
        shape: tuple,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize wave field and velocity to zeros.

        Args:
            shape: Shape of the wave field (B, T, D) or (B, H, T, D)
            device: Device to create tensors on
            dtype: Data type for tensors

        Returns:
            Tuple of (u, u_t) initialized to zeros
        """
        u = torch.zeros(shape, device=device, dtype=dtype)
        u_t = torch.zeros(shape, device=device, dtype=dtype)
        return u, u_t

    def extra_repr(self) -> str:
        """Extra representation for module printing."""
        return (
            f"dim={self.dim}, "
            f"dt={self.dt_init:.3f}, "
            f"dx={self.dx:.3f}, "
            f"n_steps={self.n_steps}, "
            f"spectral={self.use_spectral}"
        )
