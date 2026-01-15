# neuromanifold_gpt/model/kan/faster/__init__.py
"""
FasterKAN subpackage.

Implementation using RSWAF basis functions.
Based on: https://github.com/AthanasiosDelis/faster-kan
"""

from .basis import RSWAFBasis

__all__ = ['RSWAFBasis']
