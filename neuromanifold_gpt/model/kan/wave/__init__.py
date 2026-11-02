"""WaveKAN subpackage."""
from .activation import wavekan_activation
from .ffn import WaveKANFFN
from .linear import WaveKANLinear

__all__ = ["wavekan_activation", "WaveKANLinear", "WaveKANFFN"]
