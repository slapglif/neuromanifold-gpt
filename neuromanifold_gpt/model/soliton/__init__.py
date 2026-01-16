# neuromanifold_gpt/model/soliton/__init__.py
"""Soliton Physics for NeuroManifoldGPT.

Exports:
    PDESolver: Abstract base class for PDE-based soliton solvers
    PDEConfig: Configuration dataclass for PDE solver hyperparameters

Solitons are stable, self-reinforcing wave packets that propagate without
changing shape. They are ideal for modeling semantic units in language:
- Maintain coherent meaning over long sequences
- Interact (collide) without destroying each other
- Exhibit threshold behavior (activation above noise)

The soliton physics module implements:
- Sine-Gordon equation (topological solitons)
- KdV equation (dispersive wave solitons)
- Heimburg-Jackson model (thermodynamic neural membrane solitons)

These PDE solvers provide physics-informed inductive biases for attention
mechanisms, replacing purely data-driven patterns with principled wave dynamics.

Key concepts:
- Wave field u(x,t): Represents semantic activation over sequence positions
- Soliton: Localized wave that maintains shape during propagation
- Collision: Two solitons can pass through each other unchanged
- Dispersion: Wave spreading prevented by nonlinear focusing

Usage:
    from neuromanifold_gpt.model.soliton import PDESolver, PDEConfig

    # PDESolver is abstract - use concrete implementations:
    # - SineGordonSolver (topological solitons)
    # - KdVSolver (dispersive waves)
    # - HeimburgJacksonSolver (neural membrane model)

    # Example with SineGordonSolver:
    >>> from neuromanifold_gpt.model.soliton.sine_gordon import SineGordonSolver
    >>> solver = SineGordonSolver(dim=64)
    >>> u = torch.randn(2, 32, 64)
    >>> u_evolved, info = solver(u, n_steps=5)
"""

from neuromanifold_gpt.model.soliton.base import PDESolver, PDEConfig

__all__ = [
    "PDESolver",
    "PDEConfig",
]
