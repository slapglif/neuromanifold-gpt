# neuromanifold_gpt/model/ssm/__init__.py
"""State Space Models for NeuroManifoldGPT.

Exports:
    SSMBase: Abstract base class for state space models
    SSMConfig: Configuration dataclass for SSM hyperparameters
    HiPPO: Memory-optimal matrix initialization
    DiagonalHiPPO: Efficient diagonal approximation
    SelectiveScan: Mamba-style selective scan mechanism

State Space Models (SSMs) provide an alternative to attention mechanisms
by modeling sequence relationships as continuous-time dynamical systems.
This enables linear-time complexity O(n) compared to attention's O(n^2).

Key advantages:
- Linear time complexity for long sequences
- Continuous-time formulation matches neural dynamics
- Efficient parallel training via associative scan
- Fast autoregressive generation with constant memory

The SSM backbone implements Mamba-style selective state spaces:
- HiPPO initialization for optimal memory
- Input-dependent selection mechanism
- Parallel scan for efficient training
- Single-step mode for generation

Usage:
    from neuromanifold_gpt.model.ssm import SSMBase, SSMConfig, HiPPO, SelectiveScan

    # SSMBase is abstract - use concrete implementations:
    # - MambaBlock (full Mamba layer with conv + SSM)
    # - SelectiveScan (core selective scan mechanism)
    # - HiPPO (memory-optimal matrix initialization)
"""

from neuromanifold_gpt.model.ssm.base import SSMBase, SSMConfig
from neuromanifold_gpt.model.ssm.hippo import HiPPO, DiagonalHiPPO
from neuromanifold_gpt.model.ssm.selective_scan import SelectiveScan, ParallelSelectiveScan

__all__ = [
    "SSMBase",
    "SSMConfig",
    "HiPPO",
    "DiagonalHiPPO",
    "SelectiveScan",
    "ParallelSelectiveScan",
]
