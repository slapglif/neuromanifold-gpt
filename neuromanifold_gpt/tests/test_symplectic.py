import pytest
import torch
from neuromanifold_gpt.model.physics.symplectic import (
    StormerVerlet,
    Ruth4,
    SolitonSymplecticWrapper,
)


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def harmonic_oscillator_V(q):
    """Simple harmonic oscillator potential: V(q) = 0.5 * k * q^2"""
    k = 1.0
    return 0.5 * k * (q**2).sum(dim=-1)


def harmonic_oscillator_T(p):
    """Kinetic energy: T(p) = 0.5 * p^2 / m"""
    m = 1.0
    return 0.5 / m * (p**2).sum(dim=-1)


class TestStormerVerlet:
    def test_single_step(self, device):
        integrator = StormerVerlet(harmonic_oscillator_V, harmonic_oscillator_T).to(
            device
        )

        q0 = torch.tensor([[1.0, 0.0]], device=device)
        p0 = torch.tensor([[0.0, 1.0]], device=device)
        dt = 0.01

        q1, p1 = integrator.step(q0, p0, dt)

        assert q1.shape == q0.shape
        assert p1.shape == p0.shape

    def test_energy_conservation(self, device):
        integrator = StormerVerlet(harmonic_oscillator_V, harmonic_oscillator_T).to(
            device
        )

        q0 = torch.tensor([[1.0]], device=device)
        p0 = torch.tensor([[0.0]], device=device)
        dt = 0.01
        steps = 1000

        qs, ps = integrator(q0, p0, dt, steps)

        E0 = harmonic_oscillator_V(qs[0]) + harmonic_oscillator_T(ps[0])
        E_final = harmonic_oscillator_V(qs[-1]) + harmonic_oscillator_T(ps[-1])

        energy_drift = torch.abs(E_final - E0) / E0

        assert energy_drift < 0.01

    def test_differentiability(self, device):
        integrator = StormerVerlet(harmonic_oscillator_V, harmonic_oscillator_T).to(
            device
        )

        q0 = torch.tensor([[1.0]], device=device, requires_grad=True)
        p0 = torch.tensor([[0.0]], device=device)
        dt = 0.01
        steps = 10

        qs, ps = integrator(q0, p0, dt, steps)

        loss = qs[-1].sum()
        loss.backward()

        assert q0.grad is not None
        assert q0.grad.abs().max() > 0

    def test_batch_processing(self, device):
        integrator = StormerVerlet(harmonic_oscillator_V, harmonic_oscillator_T).to(
            device
        )

        batch_size = 4
        q0 = torch.randn(batch_size, 2, device=device)
        p0 = torch.randn(batch_size, 2, device=device)
        dt = 0.01
        steps = 10

        qs, ps = integrator(q0, p0, dt, steps)

        assert qs.shape == (steps + 1, batch_size, 2)
        assert ps.shape == (steps + 1, batch_size, 2)


class TestRuth4:
    def test_higher_order_accuracy(self, device):
        verlet = StormerVerlet(harmonic_oscillator_V, harmonic_oscillator_T).to(device)
        ruth4 = Ruth4(harmonic_oscillator_V, harmonic_oscillator_T).to(device)

        q0 = torch.tensor([[1.0]], device=device)
        p0 = torch.tensor([[0.0]], device=device)
        dt = 0.1
        steps = 500

        qs_v, ps_v = verlet(q0, p0, dt, steps)
        qs_r, ps_r = ruth4(q0, p0, dt, steps)

        E0 = harmonic_oscillator_V(q0) + harmonic_oscillator_T(p0)
        E_verlet = harmonic_oscillator_V(qs_v[-1]) + harmonic_oscillator_T(ps_v[-1])
        E_ruth4 = harmonic_oscillator_V(qs_r[-1]) + harmonic_oscillator_T(ps_r[-1])

        drift_verlet = torch.abs(E_verlet - E0)
        drift_ruth4 = torch.abs(E_ruth4 - E0)

        assert drift_ruth4 < drift_verlet

    def test_differentiability(self, device):
        integrator = Ruth4(harmonic_oscillator_V, harmonic_oscillator_T).to(device)

        q0 = torch.tensor([[1.0]], device=device, requires_grad=True)
        p0 = torch.tensor([[0.0]], device=device)
        dt = 0.01
        steps = 10

        qs, ps = integrator(q0, p0, dt, steps)

        loss = qs[-1].sum()
        loss.backward()

        assert q0.grad is not None


class TestSolitonSymplecticWrapper:
    def test_wrapper_integration(self, device):
        def simple_hamiltonian(q, p):
            return 0.5 * ((q**2).sum(dim=-1) + (p**2).sum(dim=-1))

        wrapper = SolitonSymplecticWrapper(
            simple_hamiltonian, integrator="verlet", dt=0.01, n_steps=10
        ).to(device)

        state = torch.randn(2, 8, 16, device=device)

        output = wrapper(state)

        assert output.shape == state.shape

    def test_wrapper_ruth4(self, device):
        def simple_hamiltonian(q, p):
            return 0.5 * ((q**2).sum(dim=-1) + (p**2).sum(dim=-1))

        wrapper = SolitonSymplecticWrapper(
            simple_hamiltonian, integrator="ruth4", dt=0.01, n_steps=10
        ).to(device)

        state = torch.randn(2, 8, 16, device=device)

        output = wrapper(state)

        assert output.shape == state.shape

    def test_invalid_integrator(self):
        def simple_hamiltonian(q, p):
            return 0.5 * ((q**2).sum(dim=-1) + (p**2).sum(dim=-1))

        with pytest.raises(ValueError):
            SolitonSymplecticWrapper(simple_hamiltonian, integrator="invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
