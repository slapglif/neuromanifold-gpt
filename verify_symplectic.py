import torch
import torch.nn as nn
from symplectic_integrator import StormerVerlet, Ruth4
import matplotlib.pyplot as plt
import numpy as np

# 1. Define Physics: Simple Harmonic Oscillator
# H(q, p) = 0.5 * p^2 + 0.5 * q^2
class SHOPotential(nn.Module):
    def forward(self, q):
        return 0.5 * torch.sum(q**2, dim=-1)

def compute_energy(q, p):
    return 0.5 * torch.sum(q**2, dim=-1) + 0.5 * torch.sum(p**2, dim=-1)

# Non-symplectic Euler for comparison
def euler_step(q, p, dt, grad_v_fn, grad_t_fn):
    p_new = p - dt * grad_v_fn(q)
    q_new = q + dt * grad_t_fn(p)
    return q_new, p_new

def test_energy_conservation():
    potential = SHOPotential()
    leapfrog = StormerVerlet(potential)
    ruth4 = Ruth4(potential)
    
    dt = 0.1
    steps = 500
    q0 = torch.tensor([1.0])
    p0 = torch.tensor([0.0])
    
    # 1. Leapfrog
    q_lf, p_lf = leapfrog(q0, p0, dt, steps)
    energy_lf = compute_energy(q_lf, p_lf).detach().numpy()
    
    # 2. Ruth4
    q_r4, p_r4 = ruth4(q0, p0, dt, steps)
    energy_r4 = compute_energy(q_r4, p_r4).detach().numpy()
    
    # 3. Euler
    q_e, p_e = q0, p0
    energy_e = [compute_energy(q0, p0).item()]
    for _ in range(steps):
        q_e, p_e = euler_step(q_e, p_e, dt, leapfrog.get_grad_v, leapfrog.get_grad_t)
        energy_e.append(compute_energy(q_e, p_e).item())
    energy_e = np.array(energy_e)

    print(f"Initial Energy: {energy_lf[0]:.6f}")
    print(f"Final Energy (Euler): {energy_e[-1]:.6f} (Drift: {abs(energy_e[-1]-energy_e[0]):.6f})")
    print(f"Final Energy (Leapfrog): {energy_lf[-1]:.6f} (Drift: {abs(energy_lf[-1]-energy_lf[0]):.6f})")
    print(f"Final Energy (Ruth4): {energy_r4[-1]:.6f} (Drift: {abs(energy_r4[-1]-energy_r4[0]):.6f})")

    # Success Criteria Check
    assert abs(energy_lf[-1] - energy_lf[0]) < 0.01, "Leapfrog energy drift too high!"
    assert abs(energy_r4[-1] - energy_r4[0]) < 1e-5, "Ruth4 energy drift too high!"
    assert abs(energy_e[-1] - energy_e[0]) > 0.1, "Euler should have drifted more!"
    
    print("\nSUCCESS: Symplectic integrators show bounded energy error.")

def test_differentiability():
    potential = SHOPotential()
    leapfrog = StormerVerlet(potential)
    
    q0 = torch.tensor([1.0], requires_grad=True)
    p0 = torch.tensor([0.0], requires_grad=True)
    
    q_traj, p_traj = leapfrog(q0, p0, 0.1, 10)
    loss = torch.sum(q_traj[-1]**2)
    loss.backward()
    
    print(f"Gradient dq_final/dq0: {q0.grad.item():.6f}")
    assert q0.grad is not None, "Gradients should not be None"
    print("SUCCESS: Integrator is differentiable.")

if __name__ == "__main__":
    test_energy_conservation()
    test_differentiability()
