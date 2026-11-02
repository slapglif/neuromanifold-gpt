"""
Fused Triton kernel for mHC width_connection operation.

Fuses the following operations from mhc.py:
1. Reshape residuals: (B*S, T, D) -> (B, T, S, D)
2. Apply H_res matrix multiply: residuals_out = H_res @ residuals
3. Compute branch_input: branch_input = H_pre @ residuals
4. Reshape residuals_out: (B, T, S, D) -> (B*S, T, D)

This eliminates 3 intermediate tensor allocations and 3 kernel launches.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def mhc_width_forward_kernel(
    # Input pointers
    residuals_ptr,  # (B*S, T, D) input
    h_res_ptr,  # (S, S) doubly stochastic matrix
    h_pre_ptr,  # (1, S) pre-mixing weights
    # Output pointers
    residuals_out_ptr,  # (B*S, T, D) output
    branch_input_ptr,  # (B, T, D) output
    # Dimensions
    B,
    T,
    D,
    S,
    # Strides for residuals (B*S, T, D)
    residuals_stride_bs,
    residuals_stride_t,
    residuals_stride_d,
    # Strides for outputs
    residuals_out_stride_bs,
    residuals_out_stride_t,
    residuals_out_stride_d,
    branch_input_stride_b,
    branch_input_stride_t,
    branch_input_stride_d,
    # Strides for H matrices
    h_res_stride_0,
    h_res_stride_1,
    h_pre_stride_0,
    h_pre_stride_1,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused forward kernel for mHC width_connection.

    Each thread block processes a chunk of (B, T, D) elements.
    For each element, we compute:
    - residuals_out[b, t, s_out, d] = sum_s_in(H_res[s_out, s_in] * residuals[b, t, s_in, d])
    - branch_input[b, t, d] = sum_s(H_pre[0, s] * residuals[b, t, s, d])
    """
    # Get program ID
    pid = tl.program_id(0)

    # Calculate which (b, t, d) we're processing
    n_elements = B * T * D
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Decompose linear offset into (b, t, d) indices
    td = T * D
    b = offsets // td
    remainder = offsets % td
    t = remainder // D
    d = remainder % D

    # Compute branch_input[b, t, d] = sum_s(H_pre[0, s] * residuals[b, t, s, d])
    branch_val = 0.0
    for s_in in range(S):
        # Calculate index in residuals: (b*S + s_in, t, d)
        residual_idx = (
            (b * S + s_in) * residuals_stride_bs
            + t * residuals_stride_t
            + d * residuals_stride_d
        )
        residual_val = tl.load(residuals_ptr + residual_idx, mask=mask, other=0.0)

        # Load H_pre[0, s_in]
        h_pre_idx = 0 * h_pre_stride_0 + s_in * h_pre_stride_1
        h_pre_val = tl.load(h_pre_ptr + h_pre_idx)

        branch_val += h_pre_val * residual_val

    # Store branch_input[b, t, d]
    branch_out_idx = (
        b * branch_input_stride_b
        + t * branch_input_stride_t
        + d * branch_input_stride_d
    )
    tl.store(branch_input_ptr + branch_out_idx, branch_val, mask=mask)

    # Compute residuals_out for all output streams
    for s_out in range(S):
        residual_out_val = 0.0
        for s_in in range(S):
            # Calculate index in residuals: (b*S + s_in, t, d)
            residual_idx = (
                (b * S + s_in) * residuals_stride_bs
                + t * residuals_stride_t
                + d * residuals_stride_d
            )
            residual_val = tl.load(residuals_ptr + residual_idx, mask=mask, other=0.0)

            # Load H_res[s_out, s_in]
            h_res_idx = s_out * h_res_stride_0 + s_in * h_res_stride_1
            h_res_val = tl.load(h_res_ptr + h_res_idx)

            residual_out_val += h_res_val * residual_val

        # Store residuals_out[b*S + s_out, t, d]
        residual_out_idx = (
            (b * S + s_out) * residuals_out_stride_bs
            + t * residuals_out_stride_t
            + d * residuals_out_stride_d
        )
        tl.store(residuals_out_ptr + residual_out_idx, residual_out_val, mask=mask)


@triton.jit
def mhc_width_backward_kernel(
    # Gradient inputs (from next layer)
    grad_residuals_out_ptr,  # (B*S, T, D)
    grad_branch_input_ptr,  # (B, T, D)
    # Saved from forward
    residuals_ptr,  # (B*S, T, D)
    h_res_ptr,  # (S, S)
    h_pre_ptr,  # (1, S)
    # Gradient outputs (to previous layer)
    grad_residuals_ptr,  # (B*S, T, D)
    grad_h_res_ptr,  # (S, S) - accumulated
    grad_h_pre_ptr,  # (1, S) - accumulated
    # Dimensions
    B,
    T,
    D,
    S,
    # Strides
    residuals_stride_bs,
    residuals_stride_t,
    residuals_stride_d,
    grad_residuals_out_stride_bs,
    grad_residuals_out_stride_t,
    grad_residuals_out_stride_d,
    grad_branch_input_stride_b,
    grad_branch_input_stride_t,
    grad_branch_input_stride_d,
    grad_residuals_stride_bs,
    grad_residuals_stride_t,
    grad_residuals_stride_d,
    h_res_stride_0,
    h_res_stride_1,
    h_pre_stride_0,
    h_pre_stride_1,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused backward kernel for mHC width_connection.

    Backpropagates gradients through:
    1. branch_input = H_pre @ residuals
    2. residuals_out = H_res @ residuals

    For each thread block processing (b, t, d):
    - Accumulate grad_residuals from both paths
    - Accumulate grad_H_res and grad_H_pre (use atomics)
    """
    pid = tl.program_id(0)

    # Calculate which (b, t, d) we're processing
    n_elements = B * T * D
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Decompose linear offset into (b, t, d)
    td = T * D
    b = offsets // td
    remainder = offsets % td
    t = remainder // D
    d = remainder % D

    # Load grad_branch_input[b, t, d]
    grad_branch_idx = (
        b * grad_branch_input_stride_b
        + t * grad_branch_input_stride_t
        + d * grad_branch_input_stride_d
    )
    grad_branch_val = tl.load(
        grad_branch_input_ptr + grad_branch_idx, mask=mask, other=0.0
    )

    # Backprop through branch_input = H_pre @ residuals
    # grad_residuals[b, t, s, d] += H_pre[0, s] * grad_branch_input[b, t, d]
    # grad_H_pre[0, s] += residuals[b, t, s, d] * grad_branch_input[b, t, d]
    for s in range(S):
        # Load H_pre[0, s]
        h_pre_idx = 0 * h_pre_stride_0 + s * h_pre_stride_1
        h_pre_val = tl.load(h_pre_ptr + h_pre_idx)

        # Load residuals[b, t, s, d]
        residual_idx = (
            (b * S + s) * residuals_stride_bs
            + t * residuals_stride_t
            + d * residuals_stride_d
        )
        residual_val = tl.load(residuals_ptr + residual_idx, mask=mask, other=0.0)

        # Accumulate grad_residuals[b, t, s, d]
        grad_res_val = h_pre_val * grad_branch_val
        grad_res_idx = (
            (b * S + s) * grad_residuals_stride_bs
            + t * grad_residuals_stride_t
            + d * grad_residuals_stride_d
        )
        tl.atomic_add(grad_residuals_ptr + grad_res_idx, grad_res_val, mask=mask)

        # Accumulate grad_H_pre[0, s] - need atomic across threads
        grad_h_pre_val = residual_val * grad_branch_val
        grad_h_pre_idx = 0 * h_pre_stride_0 + s * h_pre_stride_1
        tl.atomic_add(grad_h_pre_ptr + grad_h_pre_idx, grad_h_pre_val, mask=mask)

    # Backprop through residuals_out = H_res @ residuals
    # grad_residuals[b, t, s_in, d] += sum_s_out(H_res[s_out, s_in] * grad_residuals_out[b, t, s_out, d])
    # grad_H_res[s_out, s_in] += residuals[b, t, s_in, d] * grad_residuals_out[b, t, s_out, d]
    for s_out in range(S):
        # Load grad_residuals_out[b, t, s_out, d]
        grad_res_out_idx = (
            (b * S + s_out) * grad_residuals_out_stride_bs
            + t * grad_residuals_out_stride_t
            + d * grad_residuals_out_stride_d
        )
        grad_res_out_val = tl.load(
            grad_residuals_out_ptr + grad_res_out_idx, mask=mask, other=0.0
        )

        for s_in in range(S):
            # Load H_res[s_out, s_in]
            h_res_idx = s_out * h_res_stride_0 + s_in * h_res_stride_1
            h_res_val = tl.load(h_res_ptr + h_res_idx)

            # Load residuals[b, t, s_in, d]
            residual_idx = (
                (b * S + s_in) * residuals_stride_bs
                + t * residuals_stride_t
                + d * residuals_stride_d
            )
            residual_val = tl.load(residuals_ptr + residual_idx, mask=mask, other=0.0)

            # Accumulate grad_residuals[b, t, s_in, d]
            grad_res_val = h_res_val * grad_res_out_val
            grad_res_idx = (
                (b * S + s_in) * grad_residuals_stride_bs
                + t * grad_residuals_stride_t
                + d * grad_residuals_stride_d
            )
            tl.atomic_add(grad_residuals_ptr + grad_res_idx, grad_res_val, mask=mask)

            # Accumulate grad_H_res[s_out, s_in]
            grad_h_res_val = residual_val * grad_res_out_val
            grad_h_res_idx = s_out * h_res_stride_0 + s_in * h_res_stride_1
            tl.atomic_add(grad_h_res_ptr + grad_h_res_idx, grad_h_res_val, mask=mask)


class FusedMHCWidthConnection(torch.autograd.Function):
    """
    Fused implementation of mHC width_connection using Triton.

    Replaces the sequential operations in mhc.py:
    1. rearrange(residuals, "(b s) t d -> b t s d", s=S)
    2. residuals_out = einsum(h_res, residuals, "i j, b t j d -> b t i d")
    3. branch_input = einsum(h_pre, residuals, "v s, b t s d -> b t v d")
    4. rearrange(residuals_out, "b t s d -> (b s) t d")

    With a single fused kernel that computes both outputs in one pass.
    """

    @staticmethod
    def forward(ctx, residuals, h_res, h_pre):
        """
        Args:
            residuals: (B*S, T, D) input tensor
            h_res: (S, S) doubly stochastic matrix
            h_pre: (1, S) pre-mixing weights

        Returns:
            branch_input: (B, T, D)
            residuals_out: (B*S, T, D)
        """
        # Ensure contiguous
        residuals = residuals.contiguous()
        h_res = h_res.contiguous()
        h_pre = h_pre.contiguous()

        # Extract dimensions
        BS, T, D = residuals.shape
        S = h_res.shape[0]
        B = BS // S

        assert BS % S == 0, f"Batch size {BS} must be divisible by num_streams {S}"
        assert h_res.shape == (S, S), f"H_res shape {h_res.shape} must be ({S}, {S})"
        assert h_pre.shape == (1, S), f"H_pre shape {h_pre.shape} must be (1, {S})"

        # Save for backward
        ctx.save_for_backward(residuals, h_res, h_pre)
        ctx.B = B
        ctx.T = T
        ctx.D = D
        ctx.S = S

        # Allocate outputs
        residuals_out = torch.empty_like(residuals)
        branch_input = torch.empty(
            B, T, D, device=residuals.device, dtype=residuals.dtype
        )

        # Launch kernel
        BLOCK_SIZE = 1024
        n_elements = B * T * D
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        mhc_width_forward_kernel[grid](
            residuals,
            h_res,
            h_pre,
            residuals_out,
            branch_input,
            B,
            T,
            D,
            S,
            # Residuals strides (B*S, T, D)
            residuals.stride(0),
            residuals.stride(1),
            residuals.stride(2),
            # Residuals_out strides (B*S, T, D)
            residuals_out.stride(0),
            residuals_out.stride(1),
            residuals_out.stride(2),
            # Branch_input strides (B, T, D)
            branch_input.stride(0),
            branch_input.stride(1),
            branch_input.stride(2),
            # H_res strides (S, S)
            h_res.stride(0),
            h_res.stride(1),
            # H_pre strides (1, S)
            h_pre.stride(0),
            h_pre.stride(1),
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return branch_input, residuals_out

    @staticmethod
    def backward(ctx, grad_branch_input, grad_residuals_out):
        """
        Args:
            grad_branch_input: (B, T, D)
            grad_residuals_out: (B*S, T, D)

        Returns:
            grad_residuals: (B*S, T, D)
            grad_h_res: (S, S)
            grad_h_pre: (1, S)
        """
        residuals, h_res, h_pre = ctx.saved_tensors
        B, T, D, S = ctx.B, ctx.T, ctx.D, ctx.S

        # Ensure contiguous
        grad_branch_input = grad_branch_input.contiguous()
        grad_residuals_out = grad_residuals_out.contiguous()

        # Allocate gradient outputs (zero-initialized for atomic adds)
        grad_residuals = torch.zeros_like(residuals)
        grad_h_res = torch.zeros_like(h_res)
        grad_h_pre = torch.zeros_like(h_pre)

        # Launch kernel
        BLOCK_SIZE = 1024
        n_elements = B * T * D
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        mhc_width_backward_kernel[grid](
            grad_residuals_out,
            grad_branch_input,
            residuals,
            h_res,
            h_pre,
            grad_residuals,
            grad_h_res,
            grad_h_pre,
            B,
            T,
            D,
            S,
            # Residuals strides
            residuals.stride(0),
            residuals.stride(1),
            residuals.stride(2),
            # Grad residuals_out strides
            grad_residuals_out.stride(0),
            grad_residuals_out.stride(1),
            grad_residuals_out.stride(2),
            # Grad branch_input strides
            grad_branch_input.stride(0),
            grad_branch_input.stride(1),
            grad_branch_input.stride(2),
            # Grad residuals strides
            grad_residuals.stride(0),
            grad_residuals.stride(1),
            grad_residuals.stride(2),
            # H_res strides
            h_res.stride(0),
            h_res.stride(1),
            # H_pre strides
            h_pre.stride(0),
            h_pre.stride(1),
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return grad_residuals, grad_h_res, grad_h_pre


def fused_mhc_width_connection(residuals, h_res, h_pre):
    """
    Fused mHC width connection operation.

    Args:
        residuals: (B*S, T, D) input tensor
        h_res: (S, S) doubly stochastic matrix
        h_pre: (1, S) pre-mixing weights

    Returns:
        branch_input: (B, T, D)
        residuals_out: (B*S, T, D)
    """
    return FusedMHCWidthConnection.apply(residuals, h_res, h_pre)
