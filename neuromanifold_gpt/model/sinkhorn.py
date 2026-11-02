# neuromanifold_gpt/model/sinkhorn.py
"""
Sinkhorn-Knopp algorithm for projecting matrices onto the Birkhoff polytope.

This module implements the numerically stable log-space Sinkhorn-Knopp algorithm
used to project arbitrary matrices onto the Birkhoff polytope (the set of doubly
stochastic matrices). This is a core component of manifold-constrained hyper-connections
(mHC), where it ensures that residual connection matrices maintain the identity
mapping property for stable gradient flow.

The Birkhoff Polytope:
    A doubly stochastic matrix is one where:
    - All entries are non-negative (>= 0)
    - All rows sum to 1
    - All columns sum to 1

    The Birkhoff polytope is the convex hull of all permutation matrices, and
    doubly stochastic matrices form a manifold constraint that preserves the
    identity mapping property crucial for residual networks.

Algorithm:
    The Sinkhorn-Knopp algorithm iteratively normalizes rows and columns in
    log-space for numerical stability. Each iteration:
    1. Normalizes rows (via logsumexp)
    2. Normalizes columns (via logsumexp)

    After sufficient iterations, the matrix converges to a doubly stochastic
    matrix on the Birkhoff polytope.

Key Features:
    - Log-space implementation for numerical stability
    - Optional convergence-based early stopping
    - Temperature parameter (tau) for controlling sharpness
    - Efficient PyTorch implementation with GPU support

References:
    - Sinkhorn Distance: https://arxiv.org/abs/1306.0895
    - DeepSeek mHC: https://arxiv.org/abs/2512.24880
    - Optimal Transport: https://arxiv.org/abs/1803.00567

Author: DeepSeek Team (original), adapted for NeuroManifold GPT
"""

from typing import Optional

import torch


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
