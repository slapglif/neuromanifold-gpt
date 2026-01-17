"""Utility functions for MHC module.

Common helper functions used throughout the MHC (Manifold Harmonic Coupling) implementation.
These utilities handle optional values and provide clean default value semantics.
"""
from typing import TypeVar, Optional

import torch
from torch import nn
from einops.layers.torch import Reduce


T = TypeVar("T")


def exists(val: Optional[T]) -> bool:
    """Check if a value exists (is not None).

    Args:
        val: Value to check for existence

    Returns:
        True if val is not None, False otherwise
    """
    return val is not None


def default(val: Optional[T], d: T) -> T:
    """Return value if it exists, otherwise return default.

    Args:
        val: Value to return if it exists
        d: Default value to return if val is None

    Returns:
        val if val is not None, otherwise d
    """
    return val if exists(val) else d


def sinkhorn_log(
    logits: torch.Tensor,
    num_iters: int = 10,
    tau: float = 0.05,
    convergence_tol: Optional[float] = 1e-6,
) -> torch.Tensor:
    """Project matrix onto Birkhoff polytope via Sinkhorn-Knopp in log space.

    The Birkhoff polytope is the set of doubly stochastic matrices:
    - All entries >= 0
    - All rows sum to 1
    - All columns sum to 1

    This uses the numerically stable log-space algorithm from DeepSeek.

    Args:
        logits: Raw logits matrix (n, n)
        num_iters: Number of full Sinkhorn iterations (each iteration does
            both row and column normalization)
        tau: Temperature for softmax (lower = sharper, closer to permutation)
        convergence_tol: Optional convergence threshold for early stopping.
            Stops when ||u_new - u_old|| < convergence_tol.
            Default 1e-6 enables convergence-based early stopping.

    Returns:
        Doubly stochastic matrix on Birkhoff polytope
    """
    n = logits.shape[-1]
    Z = logits / tau
    log_marginal = torch.zeros((n,), device=logits.device, dtype=logits.dtype)

    u = torch.zeros(logits.shape[:-1], device=Z.device, dtype=Z.dtype)
    v = torch.zeros_like(u)

    for i in range(num_iters):
        u_prev = u.clone() if convergence_tol is not None else None

        u = log_marginal - torch.logsumexp(Z + v.unsqueeze(-2), dim=-1)
        v = log_marginal - torch.logsumexp(Z + u.unsqueeze(-1), dim=-2)

        # Early stopping check
        if convergence_tol is not None and i > 0:  # Skip first iteration
            u_change = torch.norm(u - u_prev)
            if u_change < convergence_tol:
                break

    return torch.exp(Z + u.unsqueeze(-1) + v.unsqueeze(-2))


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

    expand_fn = Reduce(pattern="b ... -> (b s) ...", reduction="repeat", s=num_streams)
    reduce_fn = Reduce(pattern="(b s) ... -> b ...", reduction="sum", s=num_streams)

    return expand_fn, reduce_fn
