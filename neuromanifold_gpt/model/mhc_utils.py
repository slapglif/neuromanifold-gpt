"""Utility functions for Manifold-Constrained Hyper-Connections (mHC).

This module contains utility functions used by the mHC implementation:
- **exists**: Check if a value is not None
- **default**: Return value or default if None
- **get_expand_reduce_stream_functions**: Create stream expansion/reduction functions

These utilities are extracted from mhc.py to improve modularity and code organization.

Author: DeepSeek Team (original), adapted for NeuroManifold GPT
"""

from torch import nn
from einops.layers.torch import Reduce


def exists(v):
    """Check if a value is not None.

    Args:
        v: Value to check

    Returns:
        True if v is not None, False otherwise
    """
    return v is not None


def default(v, d):
    """Return value v if it exists, otherwise return default d.

    Args:
        v: Value to check
        d: Default value to return if v is None

    Returns:
        v if v is not None, else d
    """
    return v if exists(v) else d


def get_expand_reduce_stream_functions(num_streams: int, disable: bool = False):
    """Get functions to expand input to multiple streams and reduce back.

    Args:
        num_streams: Number of parallel residual streams
        disable: If True, return identity functions

    Returns:
        (expand_fn, reduce_fn) tuple
    """
    if num_streams == 1 or disable:
        return (nn.Identity(), nn.Identity())

    expand_fn = Reduce(
        pattern="b ... -> (b s) ...", reduction="repeat", s=num_streams
    )
    reduce_fn = Reduce(
        pattern="(b s) ... -> b ...", reduction="sum", s=num_streams
    )

    return expand_fn, reduce_fn
