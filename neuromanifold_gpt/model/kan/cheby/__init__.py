# neuromanifold_gpt/model/kan/cheby/__init__.py
"""
ChebyKAN subpackage.

Implementation using Chebyshev polynomial basis functions.
Mathematically elegant, stable for normalized inputs in [-1, 1].
"""

from .ffn import ChebyKANFFN
from .linear import ChebyKANLinear

__all__ = ["ChebyKANLinear", "ChebyKANFFN"]
