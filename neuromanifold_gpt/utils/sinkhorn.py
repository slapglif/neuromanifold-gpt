"""
Sinkhorn-Knopp algorithm for optimal transport and doubly stochastic matrix projection.

Provides numerically stable log-space implementation for projecting matrices onto
the Birkhoff polytope (set of doubly stochastic matrices where all rows and columns
sum to 1).
"""
from typing import Optional

import torch


def sinkhorn_log(
    logits: torch.Tensor,
    num_iters: int = 10,
    tau: float = 0.05,
    convergence_tol: Optional[float] = None,
) -> torch.Tensor:
    """
    Project matrix onto Birkhoff polytope via Sinkhorn-Knopp in log space.

    The Birkhoff polytope is the set of doubly stochastic matrices:
    - All entries >= 0
    - All rows sum to 1
    - All columns sum to 1

    This uses the numerically stable log-space algorithm from DeepSeek.

    Args:
        logits: Raw logits matrix (n, n)
        num_iters: Number of alternating normalization iterations
        tau: Temperature for softmax (lower = sharper, closer to permutation)
        convergence_tol: Optional convergence threshold for early stopping.
            If provided, stops when ||u_new - u_old|| < convergence_tol.
            Default None uses all num_iters iterations.

    Returns:
        Doubly stochastic matrix on Birkhoff polytope

    Example:
        >>> import torch
        >>> logits = torch.randn(4, 4)
        >>> result = sinkhorn_log(logits)
        >>> # Check doubly stochastic properties
        >>> assert torch.allclose(result.sum(dim=0), torch.ones(4), atol=1e-3)
        >>> assert torch.allclose(result.sum(dim=1), torch.ones(4), atol=1e-3)
    """
    n = logits.shape[-1]
    Z = logits / tau
    log_marginal = torch.zeros((n,), device=logits.device, dtype=logits.dtype)

    u = torch.zeros(logits.shape[:-1], device=Z.device, dtype=Z.dtype)
    v = torch.zeros_like(u)

    for _ in range(num_iters):
        u_prev = u.clone() if convergence_tol is not None else None

        u = log_marginal - torch.logsumexp(Z + v.unsqueeze(-2), dim=-1)
        v = log_marginal - torch.logsumexp(Z + u.unsqueeze(-1), dim=-2)

        # Early stopping check
        if convergence_tol is not None:
            u_change = torch.norm(u - u_prev)
            if u_change < convergence_tol:
                break

    return torch.exp(Z + u.unsqueeze(-1) + v.unsqueeze(-2))
