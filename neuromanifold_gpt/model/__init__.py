# neuromanifold_gpt/model/__init__.py
"""Core model components for NeuroManifoldGPT."""

# Core components
# Attention mechanisms
from .attention.fhn import FHNAttention
from .block import NeuroManifoldBlock
from .gpt import NeuroManifoldGPT

# Manifold & Spectral
from .manifold import ManifoldProjection

# Memory & SDR
from .memory.engram import SDREngramMemory
from .sdr_ops import SDROperations
from .semantic_folding import SemanticFoldingEncoder
from .spectral import SpectralDecomposition
from .wave_manifold_block import WaveManifoldBlock

# New Wave Manifold components
from .wave_manifold_gpt import WaveManifoldGPT

__all__ = [
    # Core
    "NeuroManifoldGPT",
    "NeuroManifoldBlock",
    # Attention
    "FHNAttention",
    # Manifold
    "ManifoldProjection",
    "SpectralDecomposition",
    # Memory
    "SDREngramMemory",
    "SDROperations",
    "SemanticFoldingEncoder",
    # New Architecture
    "WaveManifoldGPT",
    "WaveManifoldBlock",
]
