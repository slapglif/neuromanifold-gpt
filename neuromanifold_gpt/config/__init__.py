"""Configuration module for NeuroManifoldGPT.

This module exports the main configuration dataclasses:
- NeuroManifoldConfig: Full configuration with all options
- NeuroManifoldConfigNano: Minimal preset for testing/development
"""

from .base import NeuroManifoldConfig, NeuroManifoldConfigNano

__all__ = ["NeuroManifoldConfig", "NeuroManifoldConfigNano"]
