import torch
import torch.nn as nn

class SymplecticIntegrator(nn.Module):
    """
    Base class for Symplectic Integrators.
    """
    def __init__(self, potential_fn, kinetic_fn=None):
        super().__init__()
        self.V = potential_fn
        # Default kinetic energy: T(p) = 0.5 * p^2
        self.T = kinetic_fn if kinetic_fn else lambda p: 0.5 * torch.sum(p**2, dim=-1)

    def get_grad_v(self, q):
        with torch.enable_grad():
            q = q.detach().requires_grad_(True)
            v = self.V(q).sum()
            grad = torch.autograd.grad(v, q, create_graph=True)[0]
        return grad

    def get_grad_t(self, p):
        with torch.enable_grad():
            p = p.detach().requires_grad_(True)
            t = self.T(p).sum()
            grad = torch.autograd.grad(t, p, create_graph=True)[0]
        return grad

    def forward(self, q0, p0, dt, steps):
        raise NotImplementedError("Subclasses must implement forward/step logic.")

class StormerVerlet(SymplecticIntegrator):
    """
    Störmer-Verlet (Leapfrog) Integrator.
    Second-order accurate, symplectic.
    """
    def step(self, q, p, dt):
        # 1. Half-kick p
        p = p - 0.5 * dt * self.get_grad_v(q)
        # 2. Full-drift q
        q = q + dt * self.get_grad_t(p)
        # 3. Half-kick p
        p = p - 0.5 * dt * self.get_grad_v(q)
        return q, p

    def forward(self, q0, p0, dt, steps):
        q, p = q0, p0
        qs, ps = [q0], [p0]
        for _ in range(steps):
            q, p = self.step(q, p, dt)
            qs.append(q)
            ps.append(p)
        return torch.stack(qs), torch.stack(ps)

class Ruth4(SymplecticIntegrator):
    """
    Ruth's 4th-order Symplectic Integrator.
    Higher precision for long-term stability.
    """
    def __init__(self, potential_fn, kinetic_fn=None):
        super().__init__(potential_fn, kinetic_fn)
        # Coefficients for 4th order Ruth/Yoshida integrator
        c1 = c4 = 1.0 / (2.0 * (2.0 - 2.0**(1.0/3.0)))
        c2 = c3 = (1.0 - 2.0**(1.0/3.0)) / (2.0 * (2.0 - 2.0**(1.0/3.0)))
        d1 = d3 = 1.0 / (2.0 - 2.0**(1.0/3.0))
        d2 = -2.0**(1.0/3.0) / (2.0 - 2.0**(1.0/3.0))
        d4 = 0.0
        self.coeffs_c = [c1, c2, c3, c4]
        self.coeffs_d = [d1, d2, d3, d4]

    def step(self, q, p, dt):
        for c, d in zip(self.coeffs_c, self.coeffs_d):
            q = q + c * dt * self.get_grad_t(p)
            p = p - d * dt * self.get_grad_v(q)
        return q, p

    def forward(self, q0, p0, dt, steps):
        q, p = q0, p0
        qs, ps = [q0], [p0]
        for _ in range(steps):
            q, p = self.step(q, p, dt)
            qs.append(q)
            ps.append(p)
        return torch.stack(qs), torch.stack(ps)
