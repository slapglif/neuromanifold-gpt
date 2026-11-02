# neuromanifold_gpt/model/kan/faster/__init__.py
"""
FasterKAN subpackage.

Implementation using RSWAF basis functions.
Based on: https://github.com/AthanasiosDelis/faster-kan
"""

from .basis import RSWAFBasis
from .ffn import FasterKANFFN
from .layer import FasterKANLayer
from .linear import FasterKANLinear
from .utils import replace_linear_with_fasterkan

__all__ = [
    "RSWAFBasis",
    "FasterKANLayer",
    "FasterKANFFN",
    "FasterKANLinear",
    "replace_linear_with_fasterkan",
]
