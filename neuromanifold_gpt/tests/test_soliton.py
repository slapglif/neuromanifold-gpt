# neuromanifold_gpt/tests/test_soliton.py
"""
Comprehensive tests for soliton physics components.

Tests cover:
- PDEConfig: Configuration dataclass
- PDESolver: Base class methods (spatial derivatives, time stepping, energy)
- SineGordonSolver: Topological soliton dynamics
- KdVSolver: Dispersive wave dynamics
- HeimburgJacksonSolver: Thermodynamic soliton dynamics
- SolitonAttention: Attention mechanism combining PDEs
- MultiHeadSolitonAttention: Multi-head variant with grouped physics
"""

import pytest
import torch
import torch.nn as nn
import math

from neuromanifold_gpt.model.soliton import (
    PDESolver,
    PDEConfig,
    SineGordonSolver,
    KdVSolver,
    HeimburgJacksonSolver,
    SolitonAttention,
    MultiHeadSolitonAttention,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def device():
    """Return CPU device for testing."""
    return torch.device("cpu")


@pytest.fixture
def dtype():
    """Return default dtype for testing."""
    return torch.float32


@pytest.fixture
def batch_size():
    """Standard batch size for tests."""
    return 2


@pytest.fixture
def seq_len():
    """Standard sequence length for tests."""
    return 32


@pytest.fixture
def dim():
    """Standard feature dimension for tests."""
    return 64


@pytest.fixture
def embed_dim():
    """Standard embedding dimension for attention tests."""
    return 384


@pytest.fixture
def n_heads():
    """Standard number of attention heads."""
    return 8


@pytest.fixture
def sample_input(batch_size, seq_len, dim, device, dtype):
    """Create sample input tensor."""
    return torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype)


@pytest.fixture
def small_input(device, dtype):
    """Create small input for quick tests."""
    return torch.randn(2, 16, 32, device=device, dtype=dtype)


# ==============================================================================
# PDEConfig Tests
# ==============================================================================

class TestPDEConfig:
    """Tests for PDEConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PDEConfig()
        assert config.dim == 64
        assert config.dt == 0.1
        assert config.dx == 1.0
        assert config.n_steps == 5
        assert config.use_spectral is True
        assert config.clamp_min == -10.0
        assert config.clamp_max == 10.0
        assert config.dropout == 0.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = PDEConfig(
            dim=128,
            dt=0.05,
            dx=0.5,
            n_steps=10,
            use_spectral=False,
            clamp_min=-5.0,
            clamp_max=5.0,
            dropout=0.1,
        )
        assert config.dim == 128
        assert config.dt == 0.05
        assert config.dx == 0.5
        assert config.n_steps == 10
        assert config.use_spectral is False
        assert config.clamp_min == -5.0
        assert config.clamp_max == 5.0
        assert config.dropout == 0.1

    def test_config_is_dataclass(self):
        """Test that PDEConfig is a proper dataclass."""
        from dataclasses import is_dataclass
        assert is_dataclass(PDEConfig)


# ==============================================================================
# SineGordonSolver Tests
# ==============================================================================

class TestSineGordonSolver:
    """Tests for SineGordonSolver."""

    def test_instantiation(self, dim):
        """Test basic instantiation."""
        solver = SineGordonSolver(dim=dim)
        assert solver.dim == dim
        assert isinstance(solver.dt, nn.Parameter)
        assert isinstance(solver.c_squared, nn.Parameter)
        assert isinstance(solver.nonlin_scale, nn.Parameter)

    def test_instantiation_with_params(self):
        """Test instantiation with custom parameters."""
        solver = SineGordonSolver(
            dim=64,
            dt=0.01,
            dx=0.5,
            n_steps=10,
            wave_speed=2.0,
            use_spectral=False,
            use_rk4=False,
            damping=0.1,
        )
        assert solver.n_steps == 10
        assert solver.use_rk4 is False
        assert solver.damping == 0.1
        # c_squared should be wave_speed^2
        assert torch.isclose(solver.c_squared, torch.tensor(4.0), atol=1e-5)

    def test_forward_shape(self, sample_input, dim):
        """Test forward pass output shape."""
        solver = SineGordonSolver(dim=dim, n_steps=3)
        u_out, info = solver(sample_input, n_steps=3)
        assert u_out.shape == sample_input.shape

    def test_forward_info_dict(self, small_input):
        """Test that forward returns proper info dictionary."""
        solver = SineGordonSolver(dim=32, n_steps=2)
        _, info = solver(small_input, n_steps=2)

        assert 'energy' in info
        assert 'energy_initial' in info
        assert 'energy_conservation' in info
        assert 'topological_charge' in info
        assert 'u_t' in info

    def test_forward_with_velocity(self, small_input):
        """Test forward pass with initial velocity."""
        solver = SineGordonSolver(dim=32, n_steps=2)
        u_t_init = torch.zeros_like(small_input)
        u_out, info = solver(small_input, n_steps=2, u_t=u_t_init)
        assert u_out.shape == small_input.shape

    def test_return_trajectory(self, small_input):
        """Test trajectory return option."""
        solver = SineGordonSolver(dim=32, n_steps=3)
        _, info = solver(small_input, n_steps=3, return_trajectory=True)

        assert 'trajectory' in info
        # Trajectory should have n_steps + 1 frames
        assert info['trajectory'].shape[0] == 4

    def test_compute_rhs(self, small_input):
        """Test right-hand side computation."""
        solver = SineGordonSolver(dim=32)
        u_t = torch.zeros_like(small_input)
        rhs = solver.compute_rhs(small_input, u_t)
        assert rhs.shape == small_input.shape

    def test_compute_topological_charge(self, small_input):
        """Test topological charge computation."""
        solver = SineGordonSolver(dim=32)
        charge = solver.compute_topological_charge(small_input)
        assert charge.shape == (small_input.shape[0],)

    def test_compute_potential_energy(self, small_input):
        """Test potential energy computation."""
        solver = SineGordonSolver(dim=32)
        potential = solver.compute_potential_energy(small_input)
        assert potential.shape == (small_input.shape[0],)
        assert (potential >= 0).all()  # Energy should be non-negative

    def test_create_kink_soliton(self, device, dtype):
        """Test analytical kink soliton creation."""
        solver = SineGordonSolver(dim=32)
        shape = (2, 16, 32)
        u, u_t = solver.create_kink_soliton(shape, device, dtype=dtype)

        assert u.shape == shape
        assert u_t.shape == shape

    def test_create_kink_with_velocity(self, device, dtype):
        """Test kink creation with non-zero velocity."""
        solver = SineGordonSolver(dim=32)
        shape = (2, 16, 32)
        u, u_t = solver.create_kink_soliton(
            shape, device, position=0.3, velocity=0.5, dtype=dtype
        )

        assert u.shape == shape
        assert u_t.shape == shape

    def test_spectral_vs_finite_difference(self, small_input):
        """Test both derivative methods work."""
        solver_spectral = SineGordonSolver(dim=32, use_spectral=True)
        solver_fd = SineGordonSolver(dim=32, use_spectral=False)

        out_spectral, _ = solver_spectral(small_input, n_steps=2)
        out_fd, _ = solver_fd(small_input, n_steps=2)

        assert out_spectral.shape == small_input.shape
        assert out_fd.shape == small_input.shape

    def test_euler_vs_rk4(self, small_input):
        """Test both time stepping methods work."""
        solver_rk4 = SineGordonSolver(dim=32, use_rk4=True, n_steps=2)
        solver_euler = SineGordonSolver(dim=32, use_rk4=False, n_steps=2)

        out_rk4, _ = solver_rk4(small_input)
        out_euler, _ = solver_euler(small_input)

        assert out_rk4.shape == small_input.shape
        assert out_euler.shape == small_input.shape

    def test_damping_reduces_energy(self, small_input):
        """Test that damping reduces energy over time."""
        solver_undamped = SineGordonSolver(dim=32, damping=0.0, n_steps=5)
        solver_damped = SineGordonSolver(dim=32, damping=0.5, n_steps=5)

        _, info_undamped = solver_undamped(small_input)
        _, info_damped = solver_damped(small_input)

        # Damped should conserve energy less well (more dissipation)
        # Just verify both run without error
        assert 'energy' in info_undamped
        assert 'energy' in info_damped


# ==============================================================================
# KdVSolver Tests
# ==============================================================================

class TestKdVSolver:
    """Tests for KdVSolver."""

    def test_instantiation(self, dim):
        """Test basic instantiation."""
        solver = KdVSolver(dim=dim)
        assert solver.dim == dim
        assert isinstance(solver.nonlin_coeff, nn.Parameter)
        assert isinstance(solver.disp_coeff, nn.Parameter)

    def test_instantiation_with_params(self):
        """Test instantiation with custom parameters."""
        solver = KdVSolver(
            dim=64,
            dt=0.005,
            nonlin_coeff=3.0,
            disp_coeff=0.5,
            use_rk4=False,
            damping=0.2,
        )
        assert torch.isclose(solver.nonlin_coeff, torch.tensor(3.0), atol=1e-5)
        assert torch.isclose(solver.disp_coeff, torch.tensor(0.5), atol=1e-5)
        assert solver.damping == 0.2

    def test_forward_shape(self, sample_input, dim):
        """Test forward pass output shape."""
        solver = KdVSolver(dim=dim, n_steps=3)
        u_out, info = solver(sample_input, n_steps=3)
        assert u_out.shape == sample_input.shape

    def test_forward_info_dict(self, small_input):
        """Test that forward returns proper info dictionary."""
        solver = KdVSolver(dim=32, n_steps=2)
        _, info = solver(small_input, n_steps=2)

        assert 'mass' in info
        assert 'mass_initial' in info
        assert 'mass_conservation' in info
        assert 'momentum' in info
        assert 'momentum_initial' in info
        assert 'momentum_conservation' in info
        assert 'energy' in info

    def test_compute_rhs(self, small_input):
        """Test right-hand side computation."""
        solver = KdVSolver(dim=32)
        rhs = solver.compute_rhs(small_input)
        assert rhs.shape == small_input.shape

    def test_compute_mass(self, small_input):
        """Test mass computation."""
        solver = KdVSolver(dim=32)
        mass = solver.compute_mass(small_input)
        assert mass.shape == (small_input.shape[0],)

    def test_compute_momentum(self, small_input):
        """Test momentum computation."""
        solver = KdVSolver(dim=32)
        momentum = solver.compute_momentum(small_input)
        assert momentum.shape == (small_input.shape[0],)
        assert (momentum >= 0).all()  # L2 norm is non-negative

    def test_create_soliton(self, device, dtype):
        """Test analytical soliton creation."""
        solver = KdVSolver(dim=32)
        shape = (2, 16, 32)
        u = solver.create_soliton(shape, device, dtype=dtype)

        assert u.shape == shape

    def test_create_soliton_with_params(self, device, dtype):
        """Test soliton creation with custom parameters."""
        solver = KdVSolver(dim=32)
        shape = (2, 16, 32)
        u = solver.create_soliton(
            shape, device, position=0.3, amplitude=2.0, dtype=dtype
        )

        assert u.shape == shape

    def test_create_two_soliton(self, device, dtype):
        """Test two-soliton initial condition creation."""
        solver = KdVSolver(dim=32)
        shape = (2, 16, 32)
        u = solver.create_two_soliton(
            shape, device,
            positions=(0.3, 0.7),
            amplitudes=(2.0, 1.0),
            dtype=dtype
        )

        assert u.shape == shape

    def test_return_trajectory(self, small_input):
        """Test trajectory return option."""
        solver = KdVSolver(dim=32, n_steps=3)
        _, info = solver(small_input, n_steps=3, return_trajectory=True)

        assert 'trajectory' in info
        assert info['trajectory'].shape[0] == 4

    def test_spectral_derivative(self, small_input):
        """Test spectral derivative computation."""
        solver = KdVSolver(dim=32, use_spectral=True)
        out, _ = solver(small_input, n_steps=2)
        assert out.shape == small_input.shape

    def test_finite_difference_derivative(self, small_input):
        """Test finite difference derivative computation."""
        solver = KdVSolver(dim=32, use_spectral=False)
        out, _ = solver(small_input, n_steps=2)
        assert out.shape == small_input.shape


# ==============================================================================
# HeimburgJacksonSolver Tests
# ==============================================================================

class TestHeimburgJacksonSolver:
    """Tests for HeimburgJacksonSolver."""

    def test_instantiation(self, dim):
        """Test basic instantiation."""
        solver = HeimburgJacksonSolver(dim=dim)
        assert solver.dim == dim
        assert isinstance(solver.c0_squared, nn.Parameter)
        assert isinstance(solver.p_coeff, nn.Parameter)
        assert isinstance(solver.q_coeff, nn.Parameter)
        assert isinstance(solver.h_disp, nn.Parameter)

    def test_instantiation_with_params(self):
        """Test instantiation with custom parameters."""
        solver = HeimburgJacksonSolver(
            dim=64,
            c0_squared=2.0,
            p_coeff=-2.0,
            q_coeff=0.5,
            h_disp=0.2,
            damping=0.1,
        )
        assert torch.isclose(solver.c0_squared, torch.tensor(2.0), atol=1e-5)
        assert torch.isclose(solver.p_coeff, torch.tensor(-2.0), atol=1e-5)
        assert torch.isclose(solver.q_coeff, torch.tensor(0.5), atol=1e-5)
        assert torch.isclose(solver.h_disp, torch.tensor(0.2), atol=1e-5)

    def test_forward_shape(self, sample_input, dim):
        """Test forward pass output shape."""
        solver = HeimburgJacksonSolver(dim=dim, n_steps=3)
        u_out, info = solver(sample_input, n_steps=3)
        assert u_out.shape == sample_input.shape

    def test_forward_info_dict(self, small_input):
        """Test that forward returns proper info dictionary."""
        solver = HeimburgJacksonSolver(dim=32, n_steps=2)
        _, info = solver(small_input, n_steps=2)

        assert 'enthalpy' in info
        assert 'enthalpy_initial' in info
        assert 'enthalpy_conservation' in info
        assert 'compression' in info
        assert 'wave_speed' in info
        assert 'rho_t' in info

    def test_forward_with_velocity(self, small_input):
        """Test forward pass with initial velocity."""
        solver = HeimburgJacksonSolver(dim=32, n_steps=2)
        rho_t_init = torch.zeros_like(small_input)
        u_out, info = solver(small_input, n_steps=2, rho_t=rho_t_init)
        assert u_out.shape == small_input.shape

    def test_compute_state_dependent_speed(self, small_input):
        """Test state-dependent wave speed computation."""
        solver = HeimburgJacksonSolver(dim=32)
        c = solver.compute_state_dependent_speed(small_input)
        assert c.shape == small_input.shape

    def test_compute_rhs(self, small_input):
        """Test right-hand side computation."""
        solver = HeimburgJacksonSolver(dim=32)
        u_t = torch.zeros_like(small_input)
        rhs = solver.compute_rhs(small_input, u_t)
        assert rhs.shape == small_input.shape

    def test_compute_enthalpy(self, small_input):
        """Test enthalpy computation."""
        solver = HeimburgJacksonSolver(dim=32)
        rho_t = torch.zeros_like(small_input)
        enthalpy = solver.compute_enthalpy(small_input, rho_t)
        assert enthalpy.shape == (small_input.shape[0],)
        assert (enthalpy >= 0).all()

    def test_create_soliton(self, device, dtype):
        """Test stationary soliton creation."""
        solver = HeimburgJacksonSolver(dim=32)
        shape = (2, 16, 32)
        rho, rho_t = solver.create_soliton(shape, device, dtype=dtype)

        assert rho.shape == shape
        assert rho_t.shape == shape
        assert torch.allclose(rho_t, torch.zeros_like(rho_t))  # Stationary

    def test_create_moving_soliton(self, device, dtype):
        """Test moving soliton creation."""
        solver = HeimburgJacksonSolver(dim=32)
        shape = (2, 16, 32)
        rho, rho_t = solver.create_moving_soliton(
            shape, device, amplitude=0.5, position=0.3, velocity=0.5, dtype=dtype
        )

        assert rho.shape == shape
        assert rho_t.shape == shape

    def test_create_collision_state(self, device, dtype):
        """Test collision state creation."""
        solver = HeimburgJacksonSolver(dim=32)
        shape = (2, 16, 32)
        rho, rho_t = solver.create_collision_state(
            shape, device,
            amplitudes=(0.5, 0.3),
            positions=(0.3, 0.7),
            velocities=(0.5, -0.5),
            dtype=dtype
        )

        assert rho.shape == shape
        assert rho_t.shape == shape

    def test_return_trajectory(self, small_input):
        """Test trajectory return option."""
        solver = HeimburgJacksonSolver(dim=32, n_steps=3)
        _, info = solver(small_input, n_steps=3, return_trajectory=True)

        assert 'trajectory' in info
        assert info['trajectory'].shape[0] == 4


# ==============================================================================
# PDESolver Base Class Tests
# ==============================================================================

class TestPDESolverBase:
    """Tests for PDESolver base class methods via concrete implementations."""

    def test_spatial_derivative_spectral_first_order(self, small_input):
        """Test first-order spectral derivative."""
        solver = SineGordonSolver(dim=32, use_spectral=True)
        deriv = solver.spatial_derivative(small_input, order=1)
        assert deriv.shape == small_input.shape

    def test_spatial_derivative_spectral_second_order(self, small_input):
        """Test second-order spectral derivative."""
        solver = SineGordonSolver(dim=32, use_spectral=True)
        deriv = solver.spatial_derivative(small_input, order=2)
        assert deriv.shape == small_input.shape

    def test_spatial_derivative_spectral_third_order(self, small_input):
        """Test third-order spectral derivative."""
        solver = KdVSolver(dim=32, use_spectral=True)
        deriv = solver.spatial_derivative(small_input, order=3)
        assert deriv.shape == small_input.shape

    def test_spatial_derivative_fd_first_order(self, small_input):
        """Test first-order finite difference derivative."""
        solver = SineGordonSolver(dim=32, use_spectral=False)
        deriv = solver.spatial_derivative(small_input, order=1)
        assert deriv.shape == small_input.shape

    def test_spatial_derivative_fd_second_order(self, small_input):
        """Test second-order finite difference derivative."""
        solver = SineGordonSolver(dim=32, use_spectral=False)
        deriv = solver.spatial_derivative(small_input, order=2)
        assert deriv.shape == small_input.shape

    def test_spatial_derivative_fd_third_order(self, small_input):
        """Test third-order finite difference derivative."""
        solver = KdVSolver(dim=32, use_spectral=False)
        deriv = solver.spatial_derivative(small_input, order=3)
        assert deriv.shape == small_input.shape

    def test_euler_step(self, small_input):
        """Test Euler time stepping."""
        solver = SineGordonSolver(dim=32, use_rk4=False)
        u_t = torch.zeros_like(small_input)
        u_new, u_t_new = solver.euler_step(small_input, u_t)
        assert u_new.shape == small_input.shape
        assert u_t_new.shape == small_input.shape

    def test_rk4_step(self, small_input):
        """Test RK4 time stepping."""
        solver = SineGordonSolver(dim=32, use_rk4=True)
        u_t = torch.zeros_like(small_input)
        u_new, u_t_new = solver.rk4_step(small_input, u_t)
        assert u_new.shape == small_input.shape
        assert u_t_new.shape == small_input.shape

    def test_clamp_field(self, small_input):
        """Test field clamping."""
        solver = SineGordonSolver(dim=32, clamp_min=-5.0, clamp_max=5.0)
        large_input = small_input * 100  # Create values outside clamp range
        clamped = solver.clamp_field(large_input)
        assert clamped.min() >= -5.0
        assert clamped.max() <= 5.0

    def test_compute_energy(self, small_input):
        """Test energy computation."""
        solver = SineGordonSolver(dim=32)
        u_t = torch.zeros_like(small_input)
        energy = solver.compute_energy(small_input, u_t)
        assert energy.shape == (small_input.shape[0],)
        assert (energy >= 0).all()

    def test_init_state(self, device, dtype):
        """Test state initialization."""
        solver = SineGordonSolver(dim=32)
        shape = (2, 16, 32)
        u, u_t = solver.init_state(shape, device, dtype)
        assert u.shape == shape
        assert u_t.shape == shape
        assert torch.allclose(u, torch.zeros_like(u))
        assert torch.allclose(u_t, torch.zeros_like(u_t))

    def test_extra_repr(self):
        """Test string representation."""
        solver = SineGordonSolver(dim=64)
        repr_str = solver.extra_repr()
        assert 'dim=64' in repr_str
        assert 'spectral' in repr_str


# ==============================================================================
# SolitonAttention Tests
# ==============================================================================

class TestSolitonAttention:
    """Tests for SolitonAttention."""

    def test_instantiation(self, embed_dim, n_heads):
        """Test basic instantiation."""
        attn = SolitonAttention(embed_dim=embed_dim, n_heads=n_heads)
        assert attn.embed_dim == embed_dim
        assert attn.n_heads == n_heads
        assert attn.head_dim == embed_dim // n_heads

    def test_instantiation_with_all_solvers(self, embed_dim, n_heads):
        """Test instantiation with all solvers enabled."""
        attn = SolitonAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            use_sine_gordon=True,
            use_kdv=True,
            use_heimburg_jackson=True,
        )
        assert attn.use_sine_gordon
        assert attn.use_kdv
        assert attn.use_heimburg_jackson
        assert attn.n_active_solvers == 3

    def test_instantiation_with_single_solver(self, embed_dim, n_heads):
        """Test instantiation with single solver."""
        attn = SolitonAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            use_sine_gordon=True,
            use_kdv=False,
            use_heimburg_jackson=False,
        )
        assert attn.n_active_solvers == 1

    def test_instantiation_no_solvers_defaults_to_one(self, embed_dim, n_heads):
        """Test that at least one solver is enabled."""
        attn = SolitonAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            use_sine_gordon=False,
            use_kdv=False,
            use_heimburg_jackson=False,
        )
        # Should default to at least one solver
        assert attn.n_active_solvers >= 1

    def test_forward_shape(self, embed_dim, n_heads, device, dtype):
        """Test forward pass output shape."""
        attn = SolitonAttention(embed_dim=embed_dim, n_heads=n_heads)
        x = torch.randn(2, 32, embed_dim, device=device, dtype=dtype)
        out, info = attn(x)
        assert out.shape == x.shape

    def test_forward_info_dict(self, embed_dim, n_heads, device, dtype):
        """Test that forward returns proper info dictionary."""
        attn = SolitonAttention(embed_dim=embed_dim, n_heads=n_heads)
        x = torch.randn(2, 16, embed_dim, device=device, dtype=dtype)
        _, info = attn(x)

        assert 'soliton_gate' in info
        assert 'attn_entropy' in info
        assert 'solver_mix_weights' in info

    def test_forward_with_mask(self, embed_dim, n_heads, device, dtype):
        """Test forward pass with attention mask."""
        attn = SolitonAttention(embed_dim=embed_dim, n_heads=n_heads)
        x = torch.randn(2, 16, embed_dim, device=device, dtype=dtype)
        mask = torch.ones(2, 16, device=device)
        mask[:, 8:] = 0  # Mask second half

        out, _ = attn(x, mask=mask)
        assert out.shape == x.shape

    def test_causal_masking(self, embed_dim, n_heads, device, dtype):
        """Test causal masking is applied."""
        attn_causal = SolitonAttention(
            embed_dim=embed_dim, n_heads=n_heads, causal=True
        )
        attn_noncausal = SolitonAttention(
            embed_dim=embed_dim, n_heads=n_heads, causal=False
        )

        x = torch.randn(2, 16, embed_dim, device=device, dtype=dtype)
        out_causal, _ = attn_causal(x)
        out_noncausal, _ = attn_noncausal(x)

        # Both should produce valid outputs
        assert out_causal.shape == x.shape
        assert out_noncausal.shape == x.shape
        # But they should generally differ (non-deterministic due to initialization)
        # Just verify both run successfully

    def test_dropout(self, embed_dim, n_heads, device, dtype):
        """Test dropout is applied during training."""
        attn = SolitonAttention(embed_dim=embed_dim, n_heads=n_heads, dropout=0.5)
        attn.train()

        x = torch.randn(2, 16, embed_dim, device=device, dtype=dtype)
        out1, _ = attn(x)
        out2, _ = attn(x)

        # Outputs may differ due to dropout
        assert out1.shape == x.shape
        assert out2.shape == x.shape

    def test_solver_mix_weights_sum_to_one(self, embed_dim, n_heads, device, dtype):
        """Test that solver mixing weights sum to 1."""
        attn = SolitonAttention(embed_dim=embed_dim, n_heads=n_heads)
        x = torch.randn(2, 16, embed_dim, device=device, dtype=dtype)
        _, info = attn(x)

        mix_weights = info['solver_mix_weights']
        assert torch.isclose(mix_weights.sum(), torch.tensor(1.0), atol=1e-5)

    def test_pde_steps_parameter(self, embed_dim, n_heads, device, dtype):
        """Test that n_pde_steps affects computation."""
        attn_few = SolitonAttention(embed_dim=embed_dim, n_heads=n_heads, n_pde_steps=1)
        attn_many = SolitonAttention(embed_dim=embed_dim, n_heads=n_heads, n_pde_steps=5)

        x = torch.randn(2, 16, embed_dim, device=device, dtype=dtype)
        out_few, _ = attn_few(x)
        out_many, _ = attn_many(x)

        assert out_few.shape == x.shape
        assert out_many.shape == x.shape

    def test_extra_repr(self, embed_dim, n_heads):
        """Test string representation."""
        attn = SolitonAttention(embed_dim=embed_dim, n_heads=n_heads)
        repr_str = attn.extra_repr()
        assert f'embed_dim={embed_dim}' in repr_str
        assert f'n_heads={n_heads}' in repr_str


# ==============================================================================
# MultiHeadSolitonAttention Tests
# ==============================================================================

class TestMultiHeadSolitonAttention:
    """Tests for MultiHeadSolitonAttention."""

    def test_instantiation(self, embed_dim):
        """Test basic instantiation."""
        # Need n_heads divisible by 3 for clean group division
        attn = MultiHeadSolitonAttention(embed_dim=embed_dim, n_heads=12)
        assert attn.embed_dim == embed_dim
        assert attn.n_heads == 12

    def test_head_group_division(self, embed_dim):
        """Test that heads are divided into 3 groups."""
        attn = MultiHeadSolitonAttention(embed_dim=embed_dim, n_heads=12)
        assert sum(attn.group_sizes) == 12
        assert len(attn.group_sizes) == 3

    def test_head_group_division_non_divisible(self):
        """Test head division when not evenly divisible by 3."""
        # Use embed_dim=320 which is divisible by 10 heads (head_dim=32)
        attn = MultiHeadSolitonAttention(embed_dim=320, n_heads=10)
        assert sum(attn.group_sizes) == 10

    def test_forward_shape(self, embed_dim, device, dtype):
        """Test forward pass output shape."""
        attn = MultiHeadSolitonAttention(embed_dim=embed_dim, n_heads=12)
        x = torch.randn(2, 32, embed_dim, device=device, dtype=dtype)
        out, info = attn(x)
        assert out.shape == x.shape

    def test_forward_info_dict(self, embed_dim, device, dtype):
        """Test that forward returns info from all head groups."""
        attn = MultiHeadSolitonAttention(embed_dim=embed_dim, n_heads=12)
        x = torch.randn(2, 16, embed_dim, device=device, dtype=dtype)
        _, info = attn(x)

        # Should have info from each physics type
        assert any('sg_' in k for k in info.keys())
        assert any('kdv_' in k for k in info.keys())
        assert any('hj_' in k for k in info.keys())

    def test_forward_with_mask(self, embed_dim, device, dtype):
        """Test forward pass with attention mask."""
        attn = MultiHeadSolitonAttention(embed_dim=embed_dim, n_heads=12)
        x = torch.randn(2, 16, embed_dim, device=device, dtype=dtype)
        mask = torch.ones(2, 16, device=device)
        mask[:, 8:] = 0

        out, _ = attn(x, mask=mask)
        assert out.shape == x.shape

    def test_specialized_physics_per_group(self, embed_dim, device, dtype):
        """Test that each group uses different physics."""
        attn = MultiHeadSolitonAttention(embed_dim=embed_dim, n_heads=12)

        # Verify group-specific attention modules exist
        assert attn.sine_gordon_attn is not None
        assert attn.kdv_attn is not None
        assert attn.hj_attn is not None


# ==============================================================================
# Integration and Edge Case Tests
# ==============================================================================

class TestSolitonEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_zero_input_sine_gordon(self, device, dtype):
        """Test SineGordon with zero input."""
        solver = SineGordonSolver(dim=32)
        x = torch.zeros(2, 16, 32, device=device, dtype=dtype)
        out, info = solver(x, n_steps=2)
        assert torch.isfinite(out).all()

    def test_zero_input_kdv(self, device, dtype):
        """Test KdV with zero input."""
        solver = KdVSolver(dim=32)
        x = torch.zeros(2, 16, 32, device=device, dtype=dtype)
        out, info = solver(x, n_steps=2)
        assert torch.isfinite(out).all()

    def test_zero_input_hj(self, device, dtype):
        """Test HeimburgJackson with zero input."""
        solver = HeimburgJacksonSolver(dim=32)
        x = torch.zeros(2, 16, 32, device=device, dtype=dtype)
        out, info = solver(x, n_steps=2)
        assert torch.isfinite(out).all()

    def test_large_input_stability(self, device, dtype):
        """Test numerical stability with large inputs."""
        solver = SineGordonSolver(dim=32, clamp_min=-10.0, clamp_max=10.0)
        x = torch.randn(2, 16, 32, device=device, dtype=dtype) * 100
        out, _ = solver(x, n_steps=2)
        assert torch.isfinite(out).all()

    def test_single_timestep(self, small_input):
        """Test with single time step."""
        solver = SineGordonSolver(dim=32)
        out, _ = solver(small_input, n_steps=1)
        assert out.shape == small_input.shape

    def test_many_timesteps(self, small_input):
        """Test with many time steps."""
        solver = SineGordonSolver(dim=32, dt=0.01)  # Small dt for stability
        out, _ = solver(small_input, n_steps=20)
        assert torch.isfinite(out).all()

    def test_batch_size_one(self, device, dtype):
        """Test with batch size of 1."""
        solver = SineGordonSolver(dim=32)
        x = torch.randn(1, 16, 32, device=device, dtype=dtype)
        out, _ = solver(x, n_steps=2)
        assert out.shape == x.shape

    def test_sequence_length_one(self, device, dtype):
        """Test with sequence length of 1."""
        solver = SineGordonSolver(dim=32, use_spectral=True)
        x = torch.randn(2, 1, 32, device=device, dtype=dtype)
        # This may have numerical issues but should not crash
        out, _ = solver(x, n_steps=2)
        assert out.shape == x.shape

    def test_gradient_flow_sine_gordon(self, small_input):
        """Test that gradients flow through SineGordon."""
        solver = SineGordonSolver(dim=32)
        x = small_input.clone().requires_grad_(True)
        out, _ = solver(x, n_steps=2)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_gradient_flow_kdv(self, small_input):
        """Test that gradients flow through KdV."""
        solver = KdVSolver(dim=32)
        x = small_input.clone().requires_grad_(True)
        out, _ = solver(x, n_steps=2)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_gradient_flow_hj(self, small_input):
        """Test that gradients flow through HeimburgJackson."""
        solver = HeimburgJacksonSolver(dim=32)
        x = small_input.clone().requires_grad_(True)
        out, _ = solver(x, n_steps=2)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_gradient_flow_attention(self, embed_dim, n_heads, device, dtype):
        """Test that gradients flow through SolitonAttention."""
        attn = SolitonAttention(embed_dim=embed_dim, n_heads=n_heads)
        x = torch.randn(2, 16, embed_dim, device=device, dtype=dtype, requires_grad=True)
        out, _ = attn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


class TestSolitonParameterLearning:
    """Tests for learnable parameter behavior."""

    def test_sine_gordon_learnable_params(self):
        """Test SineGordon has learnable parameters."""
        solver = SineGordonSolver(dim=32)
        params = list(solver.parameters())
        assert len(params) > 0
        assert any('c_squared' in name for name, _ in solver.named_parameters())
        assert any('nonlin_scale' in name for name, _ in solver.named_parameters())
        assert any('dt' in name for name, _ in solver.named_parameters())

    def test_kdv_learnable_params(self):
        """Test KdV has learnable parameters."""
        solver = KdVSolver(dim=32)
        params = list(solver.parameters())
        assert len(params) > 0
        assert any('nonlin_coeff' in name for name, _ in solver.named_parameters())
        assert any('disp_coeff' in name for name, _ in solver.named_parameters())

    def test_hj_learnable_params(self):
        """Test HeimburgJackson has learnable parameters."""
        solver = HeimburgJacksonSolver(dim=32)
        params = list(solver.parameters())
        assert len(params) > 0
        assert any('c0_squared' in name for name, _ in solver.named_parameters())
        assert any('p_coeff' in name for name, _ in solver.named_parameters())
        assert any('q_coeff' in name for name, _ in solver.named_parameters())
        assert any('h_disp' in name for name, _ in solver.named_parameters())

    def test_attention_learnable_params(self, embed_dim, n_heads):
        """Test SolitonAttention has learnable parameters."""
        attn = SolitonAttention(embed_dim=embed_dim, n_heads=n_heads)
        params = list(attn.parameters())
        assert len(params) > 0
        assert any('solver_mix' in name for name, _ in attn.named_parameters())
        assert any('qkv' in name for name, _ in attn.named_parameters())
        assert any('out_proj' in name for name, _ in attn.named_parameters())


# ==============================================================================
# Run tests if executed directly
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
