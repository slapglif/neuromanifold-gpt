# neuromanifold_gpt/model/__init__.py
"""
Model components for NeuroManifold GPT.

This package contains the core model architectures:
- continuous: Continuous generation with diffusion and RL policies
"""

from neuromanifold_gpt.model.continuous import LatentDiffusion

__all__ = [
    "LatentDiffusion",
]
