# neuromanifold_gpt/model/__init__.py
"""Core model components for NeuroManifoldGPT."""

# Core components
from .gpt import NeuroManifoldGPT
from .block import NeuroManifoldBlock

# Attention mechanisms
from .attention.fhn import FHNAttention

# Manifold & Spectral
from .manifold import ManifoldProjection
from .spectral import SpectralDecomposition

# Memory & SDR
from .memory.engram import SDREngramMemory
from .sdr_ops import SDROperations
from .semantic_folding import SemanticFoldingEncoder

# New Wave Manifold components
from .wave_manifold_gpt import WaveManifoldGPT
from .wave_manifold_block import WaveManifoldBlock

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