import pytest
import torch

from neuromanifold_gpt.model.physics.symplectic import (
    Ruth4,
    SolitonSymplecticWrapper,
    StormerVerlet,
)
from neuromanifold_gpt.model.soliton.kdv import KdVSolver
from neuromanifold_gpt.model.soliton.sine_gordon import SineGordonSolver


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestSymplecticSolitonIntegration:
    def test_sine_gordon_hamiltonian(self, device):
        def sine_gordon_H(q, p):
            c = 1.0
            kinetic = 0.5 * (p**2).sum(dim=-1)
            potential = 0.5 * c**2 * (q**2).sum(dim=-1) + (1 - torch.cos(q)).sum(
                dim=-1
            )
            return kinetic + potential

        wrapper = SolitonSymplecticWrapper(
            sine_gordon_H, integrator="verlet", dt=0.05, n_steps=10
        ).to(device)

        state = torch.randn(2, 8, 16, device=device) * 0.1

        output = wrapper(state)

        assert output.shape == state.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_kdv_hamiltonian(self, device):
        def kdv_H(q, p):
            kinetic = 0.5 * (p**2).sum(dim=-1)
            potential = (q**3).sum(dim=-1)
            return kinetic + potential

        wrapper = SolitonSymplecticWrapper(
            kdv_H, integrator="ruth4", dt=0.01, n_steps=20
        ).to(device)

        state = torch.randn(2, 8, 16, device=device) * 0.1

        output = wrapper(state)

        assert output.shape == state.shape

    def test_energy_conservation_vs_standard(self, device):
        def simple_H(q, p):
            return 0.5 * ((q**2).sum(dim=-1) + (p**2).sum(dim=-1))

        verlet = StormerVerlet(
            lambda q: 0.5 * (q**2).sum(dim=-1), lambda p: 0.5 * (p**2).sum(dim=-1)
        ).to(device)

        q0 = torch.randn(4, 32, device=device)
        p0 = torch.zeros_like(q0)

        qs, ps = verlet(q0, p0, 0.01, 100)

        E0 = simple_H(qs[0], ps[0]).mean()
        E_final = simple_H(qs[-1], ps[-1]).mean()

        energy_drift = torch.abs(E_final - E0) / (E0 + 1e-8)

        assert energy_drift < 0.01

    def test_gradient_flow(self, device):
        def simple_H(q, p):
            return 0.5 * ((q**2).sum(dim=-1) + (p**2).sum(dim=-1))

        wrapper = SolitonSymplecticWrapper(
            simple_H, integrator="verlet", dt=0.01, n_steps=10
        ).to(device)

        state = torch.randn(2, 8, 16, device=device, requires_grad=True)

        output = wrapper(state)
        loss = output.sum()
        loss.backward()

        assert state.grad is not None
        assert state.grad.abs().max() > 0


class TestSymplecticPerformance:
    def test_verlet_vs_ruth4_accuracy(self, device):
        L = 1.0
        g = 9.81
        m = 1.0

        def pendulum_V(q):
            return m * g * L * (1 - torch.cos(q)).sum(dim=-1)

        def pendulum_T(p):
            return 0.5 / m * (p**2).sum(dim=-1)

        verlet = StormerVerlet(pendulum_V, pendulum_T).to(device)
        ruth4 = Ruth4(pendulum_V, pendulum_T).to(device)

        q0 = torch.tensor([[0.1]], device=device)
        p0 = torch.tensor([[0.0]], device=device)
        dt = 0.01
        steps = 100

        qs_v, ps_v = verlet(q0, p0, dt, steps)
        qs_r, ps_r = ruth4(q0, p0, dt, steps)

        E0 = pendulum_V(q0) + pendulum_T(p0)
        E_verlet = pendulum_V(qs_v[-1]) + pendulum_T(ps_v[-1])
        E_ruth4 = pendulum_V(qs_r[-1]) + pendulum_T(ps_r[-1])

        drift_verlet = torch.abs(E_verlet - E0)
        drift_ruth4 = torch.abs(E_ruth4 - E0)

        assert drift_ruth4 < drift_verlet


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
