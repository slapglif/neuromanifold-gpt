# neuromanifold_gpt/model/mhc.py
"""
Manifold-Constrained Hyper-Connections (mHC) from DeepSeek.

This module implements manifold-constrained hyper-connections (mHC), a novel
residual connection architecture that improves training stability and gradient
flow in deep neural networks by projecting connection matrices onto the Birkhoff
polytope (set of doubly stochastic matrices).

The Problem:
    Standard residual connections can suffer from gradient flow issues in very
    deep networks. The identity mapping property (gradient = 1) is crucial for
    training stability, but multi-stream or learnable residual connections can
    violate this property.

The Solution:
    mHC uses manifold constraints to preserve the identity mapping property:
    1. H_res is constrained to the Birkhoff polytope (doubly stochastic)
    2. This ensures eigenvalues centered around 1, preserving gradient scale
    3. Sinkhorn-Knopp algorithm provides efficient projection to this manifold
    4. H_pre/H_post use softmax for non-negative mixing across streams

Mathematical Framework:
    The mHC transformation is:
        x_{l+1} = H_res @ x_l + H_post^T @ F(H_pre @ x_l)

    Where:
        - H_res ∈ Birkhoff polytope (doubly stochastic matrix)
          * All entries >= 0
          * All rows sum to 1
          * All columns sum to 1
          * Projected via Sinkhorn-Knopp from learned logits
        - H_pre ∈ ℝ^(1×S), softmax normalized (non-negative, sums to 1)
        - H_post ∈ ℝ^(1×S), softmax normalized (non-negative, sums to 1)
        - F: Sublayer (attention, MLP, etc.)
        - S: Number of residual streams

Components:
    - **HyperConnections**: Full mHC layer wrapping a sublayer
    - **Residual**: Simple fallback for single-stream or disabled mHC
    - **sinkhorn_log**: Numerically stable Birkhoff projection algorithm
    - **get_expand_reduce_stream_functions**: Utilities for stream management
    - **get_init_and_expand_reduce_stream_functions**: Factory for mHC setup

Module Organization:
    For improved maintainability, this implementation is organized across multiple files:
    - **sinkhorn.py**: Sinkhorn-Knopp algorithm for projecting matrices onto the
      Birkhoff polytope (doubly stochastic constraint)
    - **mhc_utils.py**: Utility functions including stream expansion/reduction,
      existence checks, and default value helpers
    - **mhc.py**: Core HyperConnections and Residual classes (this file)
    - **mhc_fused.py**: Optional Triton-fused kernels for GPU acceleration

Usage Patterns:
    1. Direct wrapping:
        >>> hc = HyperConnections(num_streams=4, dim=384, branch=attention)
        >>> output = hc(residuals)

    2. Decorator style:
        >>> hc = HyperConnections(num_streams=4, dim=384)
        >>> wrapped_attn = hc.decorate_branch(attention.forward)
        >>> output = wrapped_attn(residuals)

    3. Manual control:
        >>> hc = HyperConnections(num_streams=4, dim=384)
        >>> branch_input, add_residual_fn = hc(residuals)
        >>> branch_output = custom_layer(branch_input)
        >>> output = add_residual_fn(branch_output)

    4. Factory pattern (recommended for models):
        >>> init_hc, expand, reduce = get_init_and_expand_reduce_stream_functions(
        ...     num_streams=4, dim=384
        ... )
        >>> x = expand(embeddings)  # (B, T, D) -> (B*4, T, D)
        >>> x = init_hc(branch=attention, layer_index=0)(x)
        >>> x = init_hc(branch=mlp, layer_index=1)(x)
        >>> output = reduce(x)  # (B*4, T, D) -> (B, T, D)

Performance Considerations:
    - Each mHC layer performs multiple sequential operations (rearrange, einsum)
    - Each operation launches a separate CUDA kernel with associated overhead
    - For GPU training with large models, consider using fused Triton kernels
      (see mhc_fused.py) for 1.5-2.5x speedup on width_connection
    - Memory usage scales linearly with num_streams (e.g., 4x for num_streams=4)

References:
    - Paper: https://arxiv.org/abs/2512.24880 (DeepSeek mHC)
    - Implementation: https://github.com/tokenbender/mHC-manifold-constrained-hyper-connections
    - Sinkhorn-Knopp: Optimal transport and doubly stochastic matrix projection

Author: DeepSeek Team (original), adapted for NeuroManifold GPT
"""

from functools import partial
from random import randrange
from typing import Callable, Optional

import torch
from einops import einsum, rearrange
from torch import nn
from torch.nn import Module

from .mhc_utils import default, exists, get_expand_reduce_stream_functions

# Import from extracted modules
from .sinkhorn import sinkhorn_log

# Try to import fused kernel for GPU acceleration
try:
    from .mhc_fused import fused_mhc_width_connection

    HAS_TRITON = torch.cuda.is_available()
except (ImportError, RuntimeError):
    HAS_TRITON = False
    fused_mhc_width_connection = None


class HyperConnections(Module):
    """Manifold-Constrained Hyper-Connection layer (Full DeepSeek mHC).

    This module WRAPS a sublayer (attention, MLP, etc.) and applies manifold-constrained
    hyper-connections for improved training stability and gradient flow. The key insight
    is to project residual connection matrices onto the Birkhoff polytope (doubly
    stochastic matrices) via Sinkhorn-Knopp, which preserves the identity mapping
    property and ensures stable residual flow.

    Architecture:
        x_{l+1} = H_res @ x_l + H_post^T @ F(H_pre @ x_l)

    Where:
        - H_res: Doubly stochastic matrix (Birkhoff polytope via Sinkhorn-Knopp)
                 Mixes residual streams while preserving identity mapping property
        - H_pre: Non-negative mixing matrix (softmax over streams)
                 Computes weighted input to sublayer from multiple residual streams
        - H_post: Non-negative mixing matrix (softmax over streams)
                  Distributes sublayer output back across residual streams
        - F: The wrapped sublayer (attention, MLP, etc.)

    Key Components:
        1. **Width Connection**: Computes H_res @ x and H_pre @ x before sublayer
           - Mixes residual streams with doubly stochastic H_res
           - Creates branch input as weighted combination via H_pre
        2. **Depth Connection**: Computes H_post @ F(·) and adds to residuals
           - Distributes sublayer output across streams via H_post
           - Adds result to mixed residuals from width connection

    Performance Notes:
        - The width_connection method performs 4 sequential operations:
          rearrange, einsum (H_res), einsum (H_pre), rearrange
        - Each operation launches a separate CUDA kernel
        - For GPU training with large models, consider using fused Triton kernels
          via the mhc_fused module for 1.5-2.5x speedup

    Usage Examples:
        # Basic usage with sublayer
        >>> hc = HyperConnections(
        ...     num_residual_streams=4,
        ...     dim=384,
        ...     branch=attention_layer
        ... )
        >>> residuals = torch.randn(batch * 4, seq_len, 384)  # 4 streams
        >>> output = hc(residuals)  # Returns (batch*4, seq_len, 384)

        # Decorator style for wrapping functions
        >>> hc = HyperConnections(num_residual_streams=4, dim=384)
        >>> wrapped_fn = hc.decorate_branch(attention_layer.forward)
        >>> output = wrapped_fn(residuals, mask=mask)

        # Manual usage with add_residual_fn
        >>> branch_input, add_residual_fn = hc(residuals)
        >>> branch_output = custom_layer(branch_input)
        >>> output = add_residual_fn(branch_output)

    Args:
        num_residual_streams: Number of parallel residual streams (width expansion)
        dim: Feature dimension
        branch: Optional sublayer module to wrap (attention, MLP, etc.)
        layer_index: Optional layer index for initialization (defaults to random)
        dropout: Dropout probability applied after depth connection
        sinkhorn_iters: Number of Sinkhorn-Knopp iterations for H_res projection
        sinkhorn_tau: Temperature for Sinkhorn softmax (lower = sharper)
        sinkhorn_convergence_tol: Optional convergence threshold for early stopping
        use_fused: Use Triton-fused kernel for GPU acceleration (auto-enabled if available)

    Shape:
        - Input: (B*S, T, D) where S is num_residual_streams
        - Output: (B*S, T, D)

    References:
        - DeepSeek mHC paper: https://arxiv.org/abs/2512.24880
        - Implementation: https://github.com/tokenbender/mHC-manifold-constrained-hyper-connections
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
        sinkhorn_convergence_tol: Optional[float] = 1e-6,
        use_fused: Optional[
            bool
        ] = None,  # Use Triton-fused kernel for GPU acceleration (auto-detect)
    ):
        """Initialize the HyperConnections layer.

        This initializes the three learnable mixing matrices:
        - H_res_logits: Residual mixing (projected to doubly stochastic)
        - H_pre_logits: Input mixing (projected to non-negative via softmax)
        - H_post_logits: Output mixing (projected to non-negative via softmax)

        Initialization Strategy:
            H_res: Near-identity initialization (-8.0 off-diagonal, 0.0 on diagonal)
                   After Sinkhorn-Knopp, this yields ~identity matrix for stable training
            H_pre: One-hot initialization selecting a single stream (based on layer_index)
                   Ensures each layer initially focuses on one specific residual stream
            H_post: Uniform initialization (zeros -> equal distribution after softmax)
                    Allows gradients to shape output distribution during training

        Args:
            num_residual_streams: Number of parallel residual streams (width expansion)
                Typical values: 2, 4, 8 for moderate to aggressive width expansion
            dim: Feature dimension (embedding dimension)
                Must match the sublayer's expected input/output dimension
            branch: Optional sublayer module to wrap (attention, MLP, etc.)
                If None, forward() returns (branch_input, add_residual_fn) for manual use
            layer_index: Optional layer index for H_pre initialization (defaults to random)
                Used to select which residual stream this layer initially focuses on
            dropout: Dropout probability applied after depth connection
                Applied to final output to prevent overfitting
            sinkhorn_iters: Number of Sinkhorn-Knopp iterations for H_res projection
                Typical values: 5-20. More iterations = closer to doubly stochastic
            sinkhorn_tau: Temperature for Sinkhorn softmax
                Lower values (0.01-0.05) = sharper, closer to permutation matrices
                Higher values (0.1-1.0) = softer, more diffuse mixing
            sinkhorn_convergence_tol: Optional convergence threshold for early stopping
                If provided, Sinkhorn-Knopp stops when ||u_new - u_old|| < threshold
                Default None uses all sinkhorn_iters iterations
                Recommended: 1e-6 to 1e-4 for adaptive convergence
            use_fused: Use Triton-optimized fused kernel for GPU acceleration
                When True and CUDA is available, width_connection uses a single
                fused kernel instead of 4 sequential operations. Provides 1.5-2.5x
                speedup on GPU. Gracefully falls back to unfused path on CPU.
                Default: None (auto-enabled when Triton is available).
        """
        super().__init__()

        self.branch = branch
        self.num_residual_streams = num_residual_streams
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_tau = sinkhorn_tau
        self.sinkhorn_convergence_tol = sinkhorn_convergence_tol
        self.use_fused = (
            default(use_fused, HAS_TRITON) and HAS_TRITON
        )  # Enable fusion by default if available

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
        self.H_post_logits = nn.Parameter(torch.zeros(1, num_residual_streams))

        self.dropout = nn.Dropout(dropout)

    def width_connection(self, residuals: torch.Tensor):
        """Apply width connection: mix residuals via H_res and compute branch input.

        This method performs the "width" part of mHC, which operates on the residual
        streams before passing input to the sublayer. It serves two purposes:
        1. Mix residual streams using doubly stochastic H_res (preserves gradient flow)
        2. Compute weighted input for sublayer using H_pre

        The width connection ensures that by constraining H_res to be
        a doubly stochastic H_res, we preserve the identity mapping property
        while allowing flexible inter-stream communication.

        Performance Note:
            The unfused path performs 4 sequential operations (2 rearrange, 2 einsum).
            Each launches a separate CUDA kernel. The fused path (use_fused=True)
            combines these into a single Triton kernel for 1.5-2.5x speedup on GPU.

        Args:
            residuals: Input tensor with shape (B*S, T, D) where:
                B = batch size
                S = num_residual_streams (streams are stacked in batch dimension)
                T = sequence length
                D = feature dimension

        Returns:
            Tuple of (branch_input, residuals_out, beta):
            - branch_input: (B, T, D) - Weighted input for sublayer (H_pre @ residuals)
            - residuals_out: (B*S, T, D) - Mixed residuals (H_res @ residuals)
            - beta: (1, S) - H_post matrix for depth connection

        Shape Transformations:
            residuals (B*S, T, D)
            -> rearrange -> (B, T, S, D)
            -> H_res @ -> (B, T, S, D)
            -> H_pre @ -> (B, T, 1, D) -> (B, T, D) [branch_input]
            -> rearrange -> (B*S, T, D) [residuals_out]
        """
        streams = self.num_residual_streams

        # Get H matrices (needed for both paths)
        h_res = sinkhorn_log(
            self.H_res_logits,
            num_iters=self.sinkhorn_iters,
            tau=self.sinkhorn_tau,
            convergence_tol=self.sinkhorn_convergence_tol,
        )
        h_pre = self.H_pre_logits.softmax(dim=-1)  # (1, S)
        h_post = self.H_post_logits.softmax(dim=-1)  # (1, S)

        # === Fused Path (Single Triton Kernel) ===
        if self.use_fused:
            # Call fused kernel: combines all 4 operations into one GPU kernel
            # Input: (B*S, T, D) -> Output: branch_input (B, T, D), residuals_out (B*S, T, D)
            branch_input, residuals_out = fused_mhc_width_connection(
                residuals, h_res, h_pre
            )
            return branch_input, residuals_out, h_post

        # === Standard Unfused Path (4 Sequential Operations) ===
        # This is the fallback implementation using einops operations.
        # Each operation launches a separate CUDA kernel:
        # 1. rearrange: (B*S, T, D) -> (B, T, S, D)
        # 2. einsum (H_res): matrix multiply over streams
        # 3. einsum (H_pre): weighted average for branch input
        # 4. rearrange: (B, T, S, D) -> (B*S, T, D)

        # Operation 1: Reshape to separate streams: (B*S, T, D) -> (B, T, S, D)
        residuals = rearrange(residuals, "(b s) t d -> b t s d", s=streams)

        # Operation 2: Apply H_res to residuals: H_res @ residuals
        # einsum: (s_out, s_in), (batch, time, s_in, dim) -> (batch, time, s_out, dim)
        residuals_out = einsum(h_res, residuals, "i j, b t j d -> b t i d")

        # Operation 3: Compute branch input: H_pre @ residuals (weighted average over streams)
        # einsum: (1, s), (batch, time, s, dim) -> (batch, time, 1, dim)
        branch_input = einsum(h_pre, residuals, "v s, b t s d -> b t v d")
        branch_input = branch_input[..., 0, :]  # Remove view dimension: (B, T, D)

        # Operation 4: Reshape residuals_out back to (B*S, T, D)
        residuals_out = rearrange(residuals_out, "b t s d -> (b s) t d")

        return branch_input, residuals_out, h_post

    def depth_connection(
        self, branch_output: torch.Tensor, residuals: torch.Tensor, beta: torch.Tensor
    ) -> torch.Tensor:
        """Add branch output to residuals via H_post mixing (depth connection).

        This method implements the "depth" part of mHC, which distributes the
        sublayer output back across all residual streams and adds it to the
        mixed residuals from width_connection.

        The depth connection completes the mHC transformation:
            output = H_res @ x + H_post @ F(H_pre @ x)
        where this method computes the second term and adds it to residuals_out.

        Steps:
            1. Expand branch_output (B, T, D) to all streams via H_post
            2. Reshape to (B*S, T, D) to match residuals
            3. Add to residuals: residuals + H_post @ branch_output
            4. Apply dropout to final output

        Args:
            branch_output: Sublayer output with shape (B, T, D) where:
                B = batch size
                T = sequence length
                D = feature dimension
            residuals: Mixed residuals from width_connection with shape (B*S, T, D)
                These are the H_res @ x term from the mHC equation
            beta: H_post mixing matrix with shape (1, S) or (S,) where:
                S = num_residual_streams
                Controls how branch_output is distributed across streams

        Returns:
            Final output tensor with shape (B*S, T, D) after:
            - Distributing branch_output across streams via H_post
            - Adding to mixed residuals
            - Applying dropout

        Shape Transformations:
            branch_output (B, T, D)
            -> einsum with beta (1, S) -> (B, T, S, D)
            -> rearrange -> (B*S, T, D)
            -> add residuals (B*S, T, D) -> (B*S, T, D)
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
        """Decorator to wrap a branch function with mHC residual connections.

        This method provides a functional interface to wrap any callable
        (typically a forward method) with mHC width and depth connections.
        The wrapped function will automatically handle residual mixing.

        The decorator pattern is useful when you want to apply mHC to a
        function without modifying its implementation or when you need
        dynamic wrapping behavior.

        Args:
            branch: Callable (function or method) that takes (input, *args, **kwargs)
                and returns output tensor. Typically a sublayer's forward method.

        Returns:
            Wrapped function that:
            1. Applies width_connection to compute branch input
            2. Calls the original branch function with branch input
            3. Applies depth_connection to add result to residuals
            4. Returns final output

        Example:
            >>> hc = HyperConnections(num_streams=4, dim=384)
            >>> # Wrap attention layer's forward method
            >>> wrapped_attention = hc.decorate_branch(attention.forward)
            >>> # Use like normal function with mHC applied automatically
            >>> output = wrapped_attention(residuals, mask=attention_mask)
            >>> # residuals shape: (batch*4, seq_len, 384)
            >>> # output shape: (batch*4, seq_len, 384)

        Note:
            The wrapped function preserves the signature of the original branch
            function, accepting arbitrary positional and keyword arguments.
        """

        def forward_and_add_residual(residual, *args, **kwargs):
            branch_input, add_residual = self.forward(residual)
            branch_output = branch(branch_input, *args, **kwargs)
            residual = add_residual(branch_output)
            return residual

        return forward_and_add_residual

    def forward(self, residuals: torch.Tensor, *branch_args, **branch_kwargs):
        """Forward pass through manifold-constrained hyper-connections.

        This method implements the full mHC transformation:
            output = H_res @ x + H_post^T @ F(H_pre @ x)

        Behavior depends on whether a branch sublayer was provided at initialization:

        1. **If branch is set** (provided in __init__):
           - Computes full mHC transformation automatically
           - Returns final output tensor with residuals added

        2. **If branch is None**:
           - Returns (branch_input, add_residual_fn) for manual use
           - Caller must apply sublayer and call add_residual_fn
           - Useful for custom control flow or dynamic sublayers

        The method handles tuple outputs from the branch (e.g., (output, attention_weights))
        and preserves additional return values through the residual addition.

        Args:
            residuals: Input tensor with shape (B*S, T, D) where:
                B = batch size
                S = num_residual_streams (streams are stacked in batch dimension)
                T = sequence length
                D = feature dimension
            *branch_args: Additional positional arguments passed to branch (if set)
            **branch_kwargs: Additional keyword arguments passed to branch (if set)

        Returns:
            If branch is set:
                Output tensor (B*S, T, D) - or tuple (output, *rest) if branch returns tuple
            If branch is None:
                Tuple of (branch_input, add_residual_fn):
                - branch_input: (B, T, D) - Input for sublayer (H_pre @ residuals)
                - add_residual_fn: Closure that takes branch_output and returns final output

        Example (with branch):
            >>> hc = HyperConnections(num_streams=4, dim=384, branch=attention)
            >>> residuals = torch.randn(32*4, 128, 384)  # batch=32, streams=4, seq=128
            >>> output = hc(residuals)
            >>> output.shape  # (128, 128, 384) = (32*4, 128, 384)

        Example (without branch):
            >>> hc = HyperConnections(num_streams=4, dim=384)  # No branch
            >>> residuals = torch.randn(32*4, 128, 384)
            >>> branch_input, add_residual_fn = hc(residuals)
            >>> branch_input.shape  # (32, 128, 384) - streams collapsed
            >>> branch_output = custom_layer(branch_input)
            >>> output = add_residual_fn(branch_output)
            >>> output.shape  # (128, 128, 384) = (32*4, 128, 384)

        Example (tuple outputs):
            >>> # If branch returns (output, attention_weights)
            >>> hc = HyperConnections(num_streams=4, dim=384, branch=attention)
            >>> residuals = torch.randn(32*4, 128, 384)
            >>> output, attn_weights = hc(residuals)  # Tuple preserved
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
    """Simple residual connection (identity fallback when mHC is disabled).

    This class provides a drop-in replacement for HyperConnections when
    mHC is disabled (e.g., num_streams=1 or explicit disable flag). It
    implements standard residual connections without the manifold-constrained
    mixing matrices, reducing to the classic transformer residual:
        output = x + F(x)

    The interface matches HyperConnections to enable seamless switching:
    - Accepts same arguments (but ignores mHC-specific parameters)
    - Returns same output shapes
    - Supports both branch module and manual usage patterns

    This allows architectural flexibility without code changes: models can
    use mHC for multi-stream training or fall back to standard residuals
    for single-stream deployment or baseline comparisons.

    Args:
        *args: Ignored (accepts HyperConnections args for compatibility)
        branch: Optional sublayer module to wrap (attention, MLP, etc.)
        **kwargs: Ignored (accepts HyperConnections kwargs for compatibility)

    Example:
        >>> # Same interface as HyperConnections
        >>> residual = Residual(branch=attention_layer)
        >>> output = residual(x)  # Returns x + attention_layer(x)
        >>>
        >>> # Or manual usage
        >>> residual = Residual()
        >>> branch_input, add_residual_fn = residual(x)
        >>> output = add_residual_fn(attention_layer(branch_input))

    Note:
        This class is also aliased as SimplifiedMHC for backwards compatibility.
    """

    def __init__(self, *args, branch: Optional[Module] = None, **kwargs):
        super().__init__()
        self.branch = branch

    def forward(self, x, *args, **kwargs):
        """Forward pass with simple residual connection.

        Args:
            x: Input tensor of shape (B, T, D)
            *args: Additional positional arguments passed to branch (if set)
            **kwargs: Additional keyword arguments passed to branch (if set)

        Returns:
            If branch is set: x + branch(x, *args, **kwargs)
            If branch is None: (x, add_residual_fn) for manual use
        """
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
    sinkhorn_convergence_tol: Optional[float] = 1e-6,
):
    """Get mHC initializer and stream expand/reduce functions.

    This is a convenience factory function that returns three components needed
    for multi-stream mHC architectures:
    1. init_hc_fn: Partially applied HyperConnections (or Residual) constructor
    2. expand_fn: Function to expand batch dimension to multiple streams
    3. reduce_fn: Function to reduce streams back to batch dimension

    The function handles the disable logic automatically:
    - If disable=True or num_streams=1: Returns Residual class with identity expand/reduce
    - Otherwise: Returns HyperConnections class with proper stream expansion/reduction

    This pattern allows models to seamlessly switch between mHC and standard residuals
    based on configuration, without changing the model code.

    Typical Usage:
        >>> # In model __init__
        >>> init_hc, expand, reduce = get_init_and_expand_reduce_stream_functions(
        ...     num_streams=4, dim=384, sinkhorn_iters=10
        ... )
        >>> # Use expand at input
        >>> x = expand(x)  # (B, T, D) -> (B*4, T, D)
        >>>
        >>> # Wrap each sublayer with mHC
        >>> self.attn_hc = init_hc(branch=attention, layer_index=0)
        >>> self.mlp_hc = init_hc(branch=mlp, layer_index=1)
        >>>
        >>> # Use reduce at output
        >>> x = reduce(x)  # (B*4, T, D) -> (B, T, D)

    Args:
        num_streams: Number of parallel residual streams
            Typical values: 1 (disabled), 2, 4, 8 for increasing width expansion
        dim: Feature dimension (embedding dimension)
            Required for HyperConnections (passed to partial constructor)
            Can be None if you'll provide it later when calling init_hc_fn
        disable: If True, force use of simple Residual instead of mHC
            Useful for ablation studies or single-stream deployment
        sinkhorn_iters: Number of Sinkhorn-Knopp iterations for H_res projection
            Default 10. Increase for more accurate doubly stochastic projection
        sinkhorn_tau: Temperature for Sinkhorn softmax
            Default 0.05. Lower = sharper (closer to permutation), higher = softer
        sinkhorn_convergence_tol: Optional early stopping threshold
            If provided, stops when ||u_new - u_old|| < threshold
            Default None uses all sinkhorn_iters iterations

    Returns:
        Tuple of (init_hc_fn, expand_fn, reduce_fn):
        - init_hc_fn: Partially applied constructor for HyperConnections or Residual
                      Call with (branch=layer, layer_index=i) to create an instance
        - expand_fn: nn.Module that expands (B, ...) to (B*S, ...) where S=num_streams
                     Identity if disabled
        - reduce_fn: nn.Module that reduces (B*S, ...) to (B, ...) via summation
                     Identity if disabled

    Example (with mHC enabled):
        >>> init_hc, expand, reduce = get_init_and_expand_reduce_stream_functions(
        ...     num_streams=4, dim=384
        ... )
        >>> isinstance(init_hc.func, type) and issubclass(init_hc.func, HyperConnections)
        True
        >>> x = torch.randn(8, 128, 384)  # batch=8
        >>> x_expanded = expand(x)
        >>> x_expanded.shape  # (32, 128, 384) = (8*4, 128, 384)
        >>> x_reduced = reduce(x_expanded)
        >>> x_reduced.shape  # (8, 128, 384) - back to original

    Example (with mHC disabled):
        >>> init_hc, expand, reduce = get_init_and_expand_reduce_stream_functions(
        ...     num_streams=4, dim=384, disable=True
        ... )
        >>> isinstance(init_hc.func, type) and issubclass(init_hc.func, Residual)
        True
        >>> # expand and reduce are identity functions
    """
    disable = disable or (num_streams == 1)

    hyper_conn_klass = HyperConnections if not disable else Residual

    init_hyper_conn_fn = partial(
        hyper_conn_klass,
        num_streams,
        sinkhorn_iters=sinkhorn_iters,
        sinkhorn_tau=sinkhorn_tau,
        sinkhorn_convergence_tol=sinkhorn_convergence_tol,
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
