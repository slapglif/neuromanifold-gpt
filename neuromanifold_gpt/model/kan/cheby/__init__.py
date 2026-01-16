# neuromanifold_gpt/model/kan/cheby/__init__.py
"""
ChebyKAN subpackage.

Implementation using Chebyshev polynomial basis functions.
Mathematically elegant, stable for normalized inputs in [-1, 1].
"""

from .linear import ChebyKANLinear
from .ffn import ChebyKANFFN

__all__ = ["ChebyKANLinear", "ChebyKANFFN"]
