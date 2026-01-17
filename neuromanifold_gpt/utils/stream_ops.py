"""
Stream expansion and reduction operations for multi-stream processing.

Provides utilities for expanding batch dimensions to multiple parallel streams
and reducing them back. This is commonly used in multi-stream architectures like
manifold-constrained hyper-connections (mHC) where computations are performed
across multiple residual streams and then aggregated.

The stream expansion/reduction pattern enables parallelization of residual
pathways while maintaining the ability to aggregate results back to the
original batch size.
"""
from typing import Tuple

import torch.nn as nn
from einops.layers.torch import Reduce


def get_expand_reduce_stream_functions(
    num_streams: int, disable: bool = False
) -> Tuple[nn.Module, nn.Module]:
    """Get functions to expand input to multiple streams and reduce back.

    This creates a pair of operations for multi-stream processing:
    - expand_fn: Repeats each batch element across num_streams streams
    - reduce_fn: Sums across streams to reduce back to original batch size

    Args:
        num_streams: Number of parallel residual streams to create.
            The batch dimension will be multiplied by this factor during expansion.
        disable: If True, return identity functions (no expansion/reduction).
            Useful for disabling multi-stream processing without code changes.

    Returns:
        Tuple of (expand_fn, reduce_fn):
            - expand_fn: Module that expands batch from (b, ...) to (b*num_streams, ...)
            - reduce_fn: Module that reduces batch from (b*num_streams, ...) to (b, ...)

    Example:
        >>> import torch
        >>> from neuromanifold_gpt.utils.stream_ops import get_expand_reduce_stream_functions
        >>>
        >>> # Create expand/reduce functions for 4 streams
        >>> expand, reduce = get_expand_reduce_stream_functions(4)
        >>>
        >>> # Input tensor with batch_size=2
        >>> x = torch.randn(2, 8, 16)  # (batch, seq, dim)
        >>>
        >>> # Expand to 4 streams: batch becomes 2*4=8
        >>> x_expanded = expand(x)
        >>> print(x_expanded.shape)  # (8, 8, 16)
        >>>
        >>> # Process across streams (e.g., apply a model)
        >>> # ... some computation on x_expanded ...
        >>>
        >>> # Reduce back to original batch size by summing across streams
        >>> x_reduced = reduce(x_expanded)
        >>> print(x_reduced.shape)  # (2, 8, 16)
        >>>
        >>> # Disabled mode returns identity functions
        >>> expand_id, reduce_id = get_expand_reduce_stream_functions(4, disable=True)
        >>> y = torch.randn(2, 8, 16)
        >>> assert (expand_id(y) == y).all()
        >>> assert (reduce_id(y) == y).all()

    Note:
        When num_streams=1 or disable=True, both functions are identity operations
        (nn.Identity), which means they pass inputs through unchanged with no
        computational overhead.
    """
    if num_streams == 1 or disable:
        return (nn.Identity(), nn.Identity())

    expand_fn = Reduce(pattern="b ... -> (b s) ...", reduction="repeat", s=num_streams)
    reduce_fn = Reduce(pattern="(b s) ... -> b ...", reduction="sum", s=num_streams)

    return expand_fn, reduce_fn
