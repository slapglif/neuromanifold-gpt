import torch
import torch.nn as nn
from typing import Callable, Tuple, Optional


class SymplecticIntegrator(nn.Module):
    """
    Base class for symplectic integrators that preserve Hamiltonian structure.

    Symplectic methods ensure energy conservation during long-term integration,
    critical for stable physics-informed neural networks.
    """

    def __init__(
        self,
        potential_fn: Optional[Callable] = None,
        kinetic_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.V = potential_fn
        self.T = kinetic_fn if kinetic_fn else lambda p: 0.5 * torch.sum(p**2, dim=-1)

    def get_grad_v(self, q: torch.Tensor) -> torch.Tensor:
        """Compute gradient of potential energy wrt position."""
        with torch.enable_grad():
            q = q.detach().requires_grad_(True)
            v = self.V(q).sum()
            grad = torch.autograd.grad(v, q, create_graph=True)[0]
        return grad

    def get_grad_t(self, p: torch.Tensor) -> torch.Tensor:
        """Compute gradient of kinetic energy wrt momentum."""
        with torch.enable_grad():
            p = p.detach().requires_grad_(True)
            t = self.T(p).sum()
            grad = torch.autograd.grad(t, p, create_graph=True)[0]
        return grad

    def forward(
        self, q0: torch.Tensor, p0: torch.Tensor, dt: float, steps: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class StormerVerlet(SymplecticIntegrator):
    """
    Störmer-Verlet (Leapfrog) integrator.

    Second-order accurate, symplectic method using kick-drift-kick sequence:
    1. p_{n+1/2} = p_n - (dt/2) * grad_V(q_n)
    2. q_{n+1} = q_n + dt * grad_T(p_{n+1/2})
    3. p_{n+1} = p_{n+1/2} - (dt/2) * grad_V(q_{n+1})

    This preserves the symplectic 2-form and ensures bounded energy drift.
    """

    def step(
        self, q: torch.Tensor, p: torch.Tensor, dt: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        p = p - 0.5 * dt * self.get_grad_v(q)
        q = q + dt * self.get_grad_t(p)
        p = p - 0.5 * dt * self.get_grad_v(q)
        return q, p

    def forward(
        self, q0: torch.Tensor, p0: torch.Tensor, dt: float, steps: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q, p = q0, p0
        qs, ps = [q0], [p0]
        for _ in range(steps):
            q, p = self.step(q, p, dt)
            qs.append(q)
            ps.append(p)
        return torch.stack(qs), torch.stack(ps)


class Ruth4(SymplecticIntegrator):
    """
    Ruth's 4th-order symplectic integrator using Yoshida coefficients.

    Provides higher precision energy conservation for long-term integration
    at the cost of 4x more force evaluations per step.

    Reference: Ruth (1983), "A Canonical Integration Technique"
    """

    def __init__(
        self,
        potential_fn: Optional[Callable] = None,
        kinetic_fn: Optional[Callable] = None,
    ):
        super().__init__(potential_fn, kinetic_fn)
        cbrt2 = 2.0 ** (1.0 / 3.0)
        c1 = c4 = 1.0 / (2.0 * (2.0 - cbrt2))
        c2 = c3 = (1.0 - cbrt2) / (2.0 * (2.0 - cbrt2))
        d1 = d3 = 1.0 / (2.0 - cbrt2)
        d2 = -cbrt2 / (2.0 - cbrt2)
        d4 = 0.0
        self.coeffs_c = [c1, c2, c3, c4]
        self.coeffs_d = [d1, d2, d3, d4]

    def step(
        self, q: torch.Tensor, p: torch.Tensor, dt: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for c, d in zip(self.coeffs_c, self.coeffs_d):
            q = q + c * dt * self.get_grad_t(p)
            p = p - d * dt * self.get_grad_v(q)
        return q, p

    def forward(
        self, q0: torch.Tensor, p0: torch.Tensor, dt: float, steps: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q, p = q0, p0
        qs, ps = [q0], [p0]
        for _ in range(steps):
            q, p = self.step(q, p, dt)
            qs.append(q)
            ps.append(p)
        return torch.stack(qs), torch.stack(ps)


class SolitonSymplecticWrapper(nn.Module):
    """
    Wrapper to apply symplectic integration to Hamiltonian PDE systems.

    Converts first-order PDE systems into Hamiltonian form (q, p)
    and evolves them using energy-conserving symplectic methods.
    """

    def __init__(
        self,
        hamiltonian_fn: Callable,
        integrator: str = "verlet",
        dt: float = 0.1,
        n_steps: int = 5,
    ):
        super().__init__()
        self.H = hamiltonian_fn
        self.dt = dt
        self.n_steps = n_steps

        potential_fn = lambda q: self.H(q, torch.zeros_like(q))
        kinetic_fn = lambda p: self.H(torch.zeros_like(p), p)

        if integrator == "verlet":
            self.integrator = StormerVerlet(potential_fn, kinetic_fn)
        elif integrator == "ruth4":
            self.integrator = Ruth4(potential_fn, kinetic_fn)
        else:
            raise ValueError(f"Unknown integrator: {integrator}")

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        original_shape = state.shape

        if len(state.shape) == 2:
            state = state.unsqueeze(1)

        q0 = state
        p0 = torch.zeros_like(state)

        qs, ps = self.integrator(q0, p0, self.dt, self.n_steps)

        result = qs[-1]

        if len(original_shape) == 2:
            result = result.squeeze(1)

        return result
