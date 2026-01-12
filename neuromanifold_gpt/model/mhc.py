# neuromanifold_gpt/model/mhc.py
"""
Manifold-Constrained Hyper-Connections (mHC) from DeepSeek.

Reference: https://arxiv.org/abs/2512.24880
Implementation based on: https://github.com/tokenbender/mHC-manifold-constrained-hyper-connections

Key idea: Project residual connection matrices onto the Birkhoff polytope
(doubly stochastic matrices) via Sinkhorn-Knopp to restore identity mapping
property and improve training stability.

Architecture:
    x_{l+1} = H_res @ x_l + H_post^T @ F(H_pre @ x_l)

Where:
    - H_res is doubly stochastic (Birkhoff polytope via Sinkhorn-Knopp)
    - H_pre, H_post are non-negative (softmax over streams)
    - The mHC WRAPS sublayers, computing H_pre @ input before passing to sublayer
"""

from typing import Callable, Optional
from functools import partial
from random import randrange

import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
from einops import rearrange, einsum
from einops.layers.torch import Reduce


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def sinkhorn_log(logits: torch.Tensor, num_iters: int = 10, tau: float = 0.05) -> torch.Tensor:
    """Project matrix onto Birkhoff polytope via Sinkhorn-Knopp in log space.

    The Birkhoff polytope is the set of doubly stochastic matrices:
    - All entries >= 0
    - All rows sum to 1
    - All columns sum to 1

    This uses the numerically stable log-space algorithm from DeepSeek.

    Args:
        logits: Raw logits matrix (n, n)
        num_iters: Number of alternating normalization iterations
        tau: Temperature for softmax (lower = sharper, closer to permutation)

    Returns:
        Doubly stochastic matrix on Birkhoff polytope
    """
    n = logits.shape[-1]
    Z = logits / tau
    log_marginal = torch.zeros((n,), device=logits.device, dtype=logits.dtype)

    u = torch.zeros(logits.shape[:-1], device=Z.device, dtype=Z.dtype)
    v = torch.zeros_like(u)

    for _ in range(num_iters):
        u = log_marginal - torch.logsumexp(Z + v.unsqueeze(-2), dim=-1)
        v = log_marginal - torch.logsumexp(Z + u.unsqueeze(-1), dim=-2)

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

    expand_fn = Reduce(
        pattern="b ... -> (b s) ...", reduction="repeat", s=num_streams
    )
    reduce_fn = Reduce(
        pattern="(b s) ... -> b ...", reduction="sum", s=num_streams
    )

    return expand_fn, reduce_fn


class HyperConnections(Module):
    """Manifold-Constrained Hyper-Connection layer (Full DeepSeek mHC).

    This WRAPS a sublayer (attention, MLP) and applies mHC residual connections.

    Key architecture points:
    1. H_res is projected to doubly stochastic via Sinkhorn-Knopp
    2. H_pre computes the input to the sublayer: H_pre @ residuals
    3. H_post mixes the sublayer output back: H_post @ branch_output

    Usage:
        hc = HyperConnections(num_streams, dim=embed_dim, branch=sublayer)
        output = hc(residuals)  # wraps sublayer internally

    Or decorator style:
        hc = HyperConnections(num_streams, dim=embed_dim)
        wrapped_fn = hc.decorate_branch(sublayer.forward)
        output = wrapped_fn(residuals)
    """

    def __init__(
        self,
        num_residual_streams: int,
        *,
        dim: int,
        branch: Optional[Module] = None,
        layer_index: Optional[int] = None,
        dropout: float = 0.0,
        sinkhorn_iters: int = 10,
        sinkhorn_tau: float = 0.05,
    ):
        """
        Args:
            num_residual_streams: Number of parallel residual streams (width expansion)
            dim: Feature dimension
            branch: Sublayer module to wrap (attention, MLP, etc.)
            layer_index: Layer index for initialization (defaults to random)
            dropout: Dropout probability
            sinkhorn_iters: Number of Sinkhorn-Knopp iterations
            sinkhorn_tau: Temperature for Sinkhorn softmax
        """
        super().__init__()

        self.branch = branch
        self.num_residual_streams = num_residual_streams
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_tau = sinkhorn_tau

        # Choose initial residual stream index
        init_residual_index = (
            default(layer_index, randrange(num_residual_streams)) % num_residual_streams
        )

        # H_res: Residual mixing matrix (doubly stochastic via Sinkhorn)
        # Initialize with -8.0 off-diagonal, 0.0 on diagonal -> near-identity after Sinkhorn
        init_h_res = torch.full((num_residual_streams, num_residual_streams), -8.0)
        init_h_res.fill_diagonal_(0.0)
        self.H_res_logits = nn.Parameter(init_h_res)

        # H_pre: Pre-mixing matrix (softmax over streams)
        # Initialize to select initial stream
        init_h_pre = torch.full((1, num_residual_streams), -8.0)
        init_h_pre[:, init_residual_index] = 0.0
        self.H_pre_logits = nn.Parameter(init_h_pre)

        # H_post: Post-mixing matrix (softmax over streams)
        # Initialize to zeros (equal distribution)
        self.H_post_logits = nn.Parameter(
            torch.zeros(1, num_residual_streams)
        )

        self.dropout = nn.Dropout(dropout)

    def width_connection(self, residuals: torch.Tensor):
        """Compute width connection: residual mixing and branch input.

        Args:
            residuals: (B*S, T, D) tensor where S is num_streams

        Returns:
            branch_input: (B, T, D) input for the sublayer
            residuals_out: (B*S, T, D) mixed residuals
            beta: H_post matrix for depth connection
        """
        streams = self.num_residual_streams

        # Reshape to separate streams: (B, T, S, D)
        residuals = rearrange(residuals, "(b s) t d -> b t s d", s=streams)

        # Get doubly stochastic H_res via Sinkhorn-Knopp
        h_res = sinkhorn_log(
            self.H_res_logits,
            num_iters=self.sinkhorn_iters,
            tau=self.sinkhorn_tau
        )

        # Apply H_res to residuals: H_res @ residuals
        # einsum: (s_out, s_in), (batch, time, s_in, dim) -> (batch, time, s_out, dim)
        residuals_out = einsum(h_res, residuals, "i j, b t j d -> b t i d")

        # Get H_pre (softmax over streams)
        h_pre = self.H_pre_logits.softmax(dim=-1)  # (1, S)

        # Compute branch input: H_pre @ residuals (weighted average over streams)
        # einsum: (1, s), (batch, time, s, dim) -> (batch, time, 1, dim)
        branch_input = einsum(h_pre, residuals, "v s, b t s d -> b t v d")
        branch_input = branch_input[..., 0, :]  # Remove view dimension: (B, T, D)

        # Get H_post for depth connection
        h_post = self.H_post_logits.softmax(dim=-1)  # (1, S)

        # Reshape residuals_out back to (B*S, T, D)
        residuals_out = rearrange(residuals_out, "b t s d -> (b s) t d")

        return branch_input, residuals_out, h_post

    def depth_connection(
        self,
        branch_output: torch.Tensor,
        residuals: torch.Tensor,
        beta: torch.Tensor
    ) -> torch.Tensor:
        """Add branch output to residuals via H_post mixing.

        Args:
            branch_output: (B, T, D) output from sublayer
            residuals: (B*S, T, D) mixed residuals from width_connection
            beta: H_post matrix (1, S)

        Returns:
            (B*S, T, D) final output
        """
        # Expand branch output to all streams via H_post
        # beta: (1, S), branch_output: (B, T, D)
        # Output: (B, T, S, D)
        if beta.ndim == 2:
            beta = beta[0]  # (S,)

        output = einsum(branch_output, beta, "b t d, s -> b t s d")
        output = rearrange(output, "b t s d -> (b s) t d")

        # Add to residuals
        residuals = output + residuals

        return self.dropout(residuals)

    def decorate_branch(self, branch: Callable):
        """Decorator to wrap a branch function with mHC.

        Usage:
            hc = HyperConnections(num_streams, dim=dim)
            wrapped = hc.decorate_branch(attention.forward)
            output = wrapped(residuals)
        """
        def forward_and_add_residual(residual, *args, **kwargs):
            branch_input, add_residual = self.forward(residual)
            branch_output = branch(branch_input, *args, **kwargs)
            residual = add_residual(branch_output)
            return residual

        return forward_and_add_residual

    def forward(self, residuals: torch.Tensor, *branch_args, **branch_kwargs):
        """Forward pass through mHC.

        If branch is set, computes: H_res @ x + H_post @ branch(H_pre @ x)
        Otherwise, returns (branch_input, add_residual_fn) for external use.

        Args:
            residuals: (B*S, T, D) input tensor (S streams stacked in batch)

        Returns:
            If branch is set: (B*S, T, D) output
            If branch is None: (branch_input, add_residual_fn)
        """
        branch_input, residuals_out, beta = self.width_connection(residuals)

        def add_residual_fn(branch_out):
            """Closure to add branch output to residuals."""
            # Handle tuple outputs (e.g., (output, info))
            if isinstance(branch_out, tuple):
                branch_out, *rest = branch_out
                result = self.depth_connection(branch_out, residuals_out, beta)
                return (result, *rest)
            return self.depth_connection(branch_out, residuals_out, beta)

        if not exists(self.branch):
            return branch_input, add_residual_fn

        # Call branch and add residual
        branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)
        return add_residual_fn(branch_output)


class Residual(Module):
    """Simple residual connection (identity fallback when mHC disabled)."""

    def __init__(self, *args, branch: Optional[Module] = None, **kwargs):
        super().__init__()
        self.branch = branch

    def forward(self, x, *args, **kwargs):
        if not exists(self.branch):
            def add_residual_fn(branch_out):
                if isinstance(branch_out, tuple):
                    branch_out, *rest = branch_out
                    return (x + branch_out, *rest)
                return x + branch_out
            return x, add_residual_fn

        branch_out = self.branch(x, *args, **kwargs)
        if isinstance(branch_out, tuple):
            branch_out, *rest = branch_out
            return (x + branch_out, *rest)
        return x + branch_out


def get_init_and_expand_reduce_stream_functions(
    num_streams: int,
    dim: Optional[int] = None,
    disable: bool = False,
    sinkhorn_iters: int = 10,
    sinkhorn_tau: float = 0.05,
):
    """Get mHC initializer and stream expand/reduce functions.

    Args:
        num_streams: Number of parallel residual streams
        dim: Feature dimension (required for HyperConnections)
        disable: If True, use simple Residual instead of mHC
        sinkhorn_iters: Sinkhorn-Knopp iterations
        sinkhorn_tau: Sinkhorn temperature

    Returns:
        (init_hc_fn, expand_fn, reduce_fn)
    """
    disable = disable or (num_streams == 1)

    hyper_conn_klass = HyperConnections if not disable else Residual

    init_hyper_conn_fn = partial(
        hyper_conn_klass,
        num_streams,
        sinkhorn_iters=sinkhorn_iters,
        sinkhorn_tau=sinkhorn_tau,
    )

    expand_fn, reduce_fn = get_expand_reduce_stream_functions(
        num_streams, disable=disable
    )

    if exists(dim):
        init_hyper_conn_fn = partial(init_hyper_conn_fn, dim=dim)

    return (init_hyper_conn_fn, expand_fn, reduce_fn)


# Backwards compatibility aliases
SimplifiedMHC = Residual  # Deprecated, use Residual
HyperConnection = HyperConnections  # Alias
