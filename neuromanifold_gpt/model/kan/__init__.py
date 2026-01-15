# neuromanifold_gpt/model/kan/__init__.py
"""Kolmogorov-Arnold Network (KAN) layers for NeuroManifoldGPT.

KAN layers replace standard MLPs with learnable activation functions
based on the Kolmogorov-Arnold representation theorem.

Available Variants:
    - FasterKAN (default): RSWAF basis, ~1.5x MLP overhead, most stable
    - WaveKAN: Wavelet basis (mexican_hat, morlet, dog), learnable shape
    - ChebyKAN: Chebyshev polynomial basis, mathematically elegant

Example:
    from neuromanifold_gpt.model.kan import FasterKANFFN, WaveKANFFN

    # Default: FasterKAN (recommended)
    ffn = FasterKANFFN(embed_dim=384, hidden_dim=1024)

    # Alternative: WaveKAN
    ffn = WaveKANFFN(embed_dim=384, hidden_dim=1024, wavelet_type="dog")
"""

# Chebyshev KAN
from .cheby import ChebyKANLinear, ChebyKANFFN

# Wavelet KAN
from .wave import WaveKANLinear, WaveKANFFN

# FasterKAN (RSWAF basis) - Default/Recommended
from .faster import (
    FasterKANLayer,
    FasterKANFFN,
    FasterKANLinear,
    RSWAFBasis,
    replace_linear_with_fasterkan,
)

# EMA (Exponential Moving Average)
from .ema import MultiHeadDampedEMA, CEMA

__all__ = [
    # FasterKAN (default)
    "FasterKANLayer",
    "FasterKANFFN",
    "FasterKANLinear",
    "RSWAFBasis",
    "replace_linear_with_fasterkan",
    # Wavelet KAN
    "WaveKANLinear",
    "WaveKANFFN",
    # Chebyshev KAN
    "ChebyKANLinear",
    "ChebyKANFFN",
    # EMA
    "MultiHeadDampedEMA",
    "CEMA",
]
