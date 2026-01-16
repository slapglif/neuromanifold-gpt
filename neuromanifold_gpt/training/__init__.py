# neuromanifold_gpt/training/__init__.py
"""Training utilities for NeuroManifoldGPT."""

from .lightning_module import NeuroManifoldLitModule  # Original
from .wave_lightning_module import WaveManifoldLightning  # New

__all__ = [
    "NeuroManifoldLitModule",
    "WaveManifoldLightning",
]