# neuromanifold_gpt/model/kan/faster/__init__.py
"""
FasterKAN subpackage.

Implementation using RSWAF basis functions.
Based on: https://github.com/AthanasiosDelis/faster-kan
"""

from .basis import RSWAFBasis
from .layer import FasterKANLayer
from .ffn import FasterKANFFN

__all__ = ['RSWAFBasis', 'FasterKANLayer', 'FasterKANFFN']
