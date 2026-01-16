"""Position embedding modules for NeuroManifoldGPT.

This module exports various position embedding strategies:

Position Embeddings:
    RamanujanPositionalEmbedding: Ramanujan graph-based position encoding
    RotaryPositionalEmbedding: Rotary position embeddings (RoPE)
    ALiBiPositionalBias: Attention with Linear Biases (ALiBi)

Spectral Embeddings:
    SpectralTokenEmbedding: Spectral graph-based token embeddings
    AFNOTokenMixer: Adaptive Fourier Neural Operator token mixer
    WaveInputEncoder: Wave-based input encoder
"""

# Position embeddings
from neuromanifold_gpt.model.embeddings.ramanujan import RamanujanPositionalEmbedding
from neuromanifold_gpt.model.embeddings.rotary import RotaryPositionalEmbedding
from neuromanifold_gpt.model.embeddings.alibi import ALiBiPositionalBias

# Spectral embeddings
from neuromanifold_gpt.model.embeddings.spectral import (
    SpectralTokenEmbedding,
    AFNOTokenMixer,
    WaveInputEncoder,
)

__all__ = [
    # Position embeddings
    "RamanujanPositionalEmbedding",
    "RotaryPositionalEmbedding",
    "ALiBiPositionalBias",
    # Spectral embeddings
    "SpectralTokenEmbedding",
    "AFNOTokenMixer",
    "WaveInputEncoder",
]
