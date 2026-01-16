"""WaveKAN subpackage."""
from .activation import wavekan_activation
from .linear import WaveKANLinear
from .ffn import WaveKANFFN

__all__ = ["wavekan_activation", "WaveKANLinear", "WaveKANFFN"]
