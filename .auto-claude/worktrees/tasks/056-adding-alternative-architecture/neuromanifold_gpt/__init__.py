# neuromanifold_gpt/__init__.py
"""
NeuroManifold GPT - Neural network architectures with manifold-based operations.

This package provides:
- model: Core model architectures
- tests: Unit tests for all components
"""

from neuromanifold_gpt.model.continuous import LatentDiffusion

__all__ = [
    "LatentDiffusion",
]
