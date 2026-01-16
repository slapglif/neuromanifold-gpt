# neuromanifold_gpt/model/attention/fhn_triton.py
"""
Triton-accelerated FHN Dynamics Kernel.

Custom Triton implementation for FitzHugh-Nagumo dynamics that provides
GPU-accelerated computation with automatic differentiation support.
Optimized for FHN attention patterns where n_steps is typically small (1-4).
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    triton = None
    tl = None


if TRITON_AVAILABLE:
    @triton.jit
    def fhn_imex_forward_kernel(
        # Input pointers
        v_in_ptr, w_in_ptr, I_ptr,
        # Output pointers
        v_out_ptr, w_out_ptr,
        # Parameters
        dt_ptr, a_ptr, b_ptr,
        # Constants
        tau: tl.constexpr,
        n_elements: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
    ):
        """
        Triton kernel for one FHN IMEX step (forward pass).

        Implements the IMEX (Implicit-Explicit) scheme:
        - v update (Explicit Euler): v_new = v + dt * (v - v^3/3 - w + I)
        - w update (Implicit Euler): w_new = (w + dt/tau * (v_new + a)) / (1 + dt*b/tau)

        Args:
            v_in_ptr: Input membrane potential
            w_in_ptr: Input recovery variable
            I_ptr: External stimulus/input current
            v_out_ptr: Output membrane potential
            w_out_ptr: Output recovery variable
            dt_ptr: Time step (learnable parameter)
            a_ptr: FHN parameter a (learnable)
            b_ptr: FHN parameter b (learnable)
            tau: Time constant for slow variable (fixed)
            n_elements: Total number of elements to process
            BLOCK_SIZE: Block size for parallel processing
        """
        # Get program ID and compute offsets
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Load inputs
        v = tl.load(v_in_ptr + offsets, mask=mask, other=0.0)
        w = tl.load(w_in_ptr + offsets, mask=mask, other=0.0)
        I = tl.load(I_ptr + offsets, mask=mask, other=0.0)

        # Load parameters (scalars)
        dt = tl.load(dt_ptr)
        a = tl.load(a_ptr)
        b = tl.load(b_ptr)

        # Explicit Euler for v
        # dv/dt = v - v^3/3 - w + I
        v3 = v * v * v
        dv = v - (v3 / 3.0) - w + I
        v_new = v + dt * dv

        # Clamp v to prevent numerical instability
        v_new = tl.where(v_new > 3.0, 3.0, v_new)
        v_new = tl.where(v_new < -3.0, -3.0, v_new)

        # Implicit Euler for w
        # w_new = (w + dt/tau * (v_new + a)) / (1 + dt*b/tau)
        alpha = dt / tau
        denom = 1.0 + alpha * b
        w_step = alpha * (v_new + a)
        w_new = (w + w_step) / denom

        # Clamp w to prevent numerical instability
        w_new = tl.where(w_new > 3.0, 3.0, w_new)
        w_new = tl.where(w_new < -3.0, -3.0, w_new)

        # Store outputs
        tl.store(v_out_ptr + offsets, v_new, mask=mask)
        tl.store(w_out_ptr + offsets, w_new, mask=mask)


    @triton.jit
    def fhn_imex_backward_kernel(
        # Incoming gradients (from next layer)
        grad_v_out_ptr, grad_w_out_ptr,
        # Saved states from forward
        v_ptr, w_ptr, I_ptr,
        # Outgoing gradients (to previous layer)
        grad_v_in_ptr, grad_w_in_ptr, grad_I_ptr,
        # Parameters
        dt_ptr, a_ptr, b_ptr,
        # Constants
        tau: tl.constexpr,
        n_elements: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
    ):
        """
        Triton kernel for FHN backward pass.

        Computes gradients using chain rule:
        - grad_v, grad_w, grad_I from grad_v_new, grad_w_new

        Forward equations:
        - v_new = v + dt*(v - v^3/3 - w + I)
        - w_new = (w + alpha*(v_new + a)) / beta
        where alpha = dt/tau, beta = 1 + dt*b/tau
        """
        # Get program ID and compute offsets
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Load saved states
        v = tl.load(v_ptr + offsets, mask=mask, other=0.0)
        w = tl.load(w_ptr + offsets, mask=mask, other=0.0)

        # Load incoming gradients
        grad_v_new = tl.load(grad_v_out_ptr + offsets, mask=mask, other=0.0)
        grad_w_new = tl.load(grad_w_out_ptr + offsets, mask=mask, other=0.0)

        # Load parameters
        dt = tl.load(dt_ptr)
        a = tl.load(a_ptr)
        b = tl.load(b_ptr)

        # Precompute constants
        alpha = dt / tau
        beta = 1.0 + alpha * b

        # Chain rule for w_new
        # w_new depends on w and v_new
        # dL/dw = dL/dw_new * dw_new/dw = grad_w_new * (1/beta)
        # dL/dv_new += dL/dw_new * dw_new/dv_new = grad_w_new * (alpha/beta)
        grad_w_from_w = grad_w_new / beta
        grad_v_new_total = grad_v_new + (grad_w_new * alpha / beta)

        # Chain rule for v_new
        # v_new depends on v, w, I
        # dv_new/dv = 1 + dt*(1 - v^2)
        # dv_new/dw = -dt
        # dv_new/dI = dt
        term_dv = 1.0 + dt * (1.0 - v * v)

        grad_v = grad_v_new_total * term_dv
        grad_w_from_v = grad_v_new_total * (-dt)
        grad_I = grad_v_new_total * dt

        # Total grad_w
        grad_w = grad_w_from_w + grad_w_from_v

        # Store gradients
        tl.store(grad_v_in_ptr + offsets, grad_v, mask=mask)
        tl.store(grad_w_in_ptr + offsets, grad_w, mask=mask)
        tl.store(grad_I_ptr + offsets, grad_I, mask=mask)


class FHNTritonSolver(torch.autograd.Function):
    """
    Autograd function for FHN dynamics using Triton kernels.

    Runs a single IMEX step with automatic differentiation support.
    For multiple steps, call this function in a loop (allows autograd
    to track intermediate states correctly).
    """

    @staticmethod
    def forward(ctx, v, w, I, a, b, tau, dt):
        """
        Forward pass: one IMEX step of FHN dynamics.

        Args:
            v: Membrane potential tensor (B, H, T, D)
            w: Recovery variable tensor (B, H, T, D)
            I: External stimulus tensor (B, H, T, D)
            a: FHN parameter a (scalar tensor)
            b: FHN parameter b (scalar tensor)
            tau: Time constant (float)
            dt: Time step (scalar tensor)

        Returns:
            Tuple of (v_new, w_new)
        """
        # Ensure contiguous memory layout
        v = v.contiguous()
        w = w.contiguous()
        I = I.contiguous()

        # Convert scalar parameters to tensors if needed
        if isinstance(a, float):
            a = torch.tensor(a, device=v.device, dtype=v.dtype)
        if isinstance(b, float):
            b = torch.tensor(b, device=v.device, dtype=v.dtype)
        if isinstance(dt, float):
            dt = torch.tensor(dt, device=v.device, dtype=v.dtype)

        # Save for backward
        ctx.save_for_backward(v, w, I, a, b, dt)
        ctx.tau = tau

        # Allocate output tensors
        v_out = torch.empty_like(v)
        w_out = torch.empty_like(w)

        # Compute grid size
        n_elements = v.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        # Launch kernel
        fhn_imex_forward_kernel[grid](
            v, w, I,
            v_out, w_out,
            dt, a, b,
            tau=tau,
            n_elements=n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )

        return v_out, w_out

    @staticmethod
    def backward(ctx, grad_v_out, grad_w_out):
        """
        Backward pass: compute gradients for inputs.

        Returns gradients for: v, w, I, a, b, tau, dt
        (tau and parameter gradients set to None for simplicity)
        """
        v, w, I, a, b, dt = ctx.saved_tensors
        tau = ctx.tau

        # Allocate gradient tensors
        grad_v = torch.empty_like(v)
        grad_w = torch.empty_like(w)
        grad_I = torch.empty_like(I)

        # Compute grid size
        n_elements = v.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        # Launch backward kernel
        fhn_imex_backward_kernel[grid](
            grad_v_out, grad_w_out,
            v, w, I,
            grad_v, grad_w, grad_I,
            dt, a, b,
            tau=tau,
            n_elements=n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )

        # Return gradients (None for a, b, tau, dt - not computing parameter grads here)
        # In practice, parameter gradients could be computed if needed
        return grad_v, grad_w, grad_I, None, None, None, None


def fhn_triton_kernel(
    v: torch.Tensor,
    w: torch.Tensor,
    I: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    tau: float,
    dt: torch.Tensor,
    n_steps: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run FHN dynamics using Triton kernel.

    Args:
        v: Initial membrane potential (B, H, T, D)
        w: Initial recovery variable (B, H, T, D)
        I: External stimulus (B, H, T, D)
        a: FHN parameter a (scalar tensor)
        b: FHN parameter b (scalar tensor)
        tau: Time constant (float)
        dt: Time step (scalar tensor)
        n_steps: Number of integration steps (default: 1)

    Returns:
        Tuple of (v_final, w_final) after n_steps

    Raises:
        ImportError: If Triton is not installed
    """
    if not TRITON_AVAILABLE:
        raise ImportError(
            "Triton is not installed. Install it with: pip install triton"
        )

    # Run n_steps sequentially (allows autograd to track intermediate states)
    for _ in range(n_steps):
        v, w = FHNTritonSolver.apply(v, w, I, a, b, tau, dt)

    return v, w


class FHNTritonAttention(nn.Module):
    """
    FHN Attention using Triton-accelerated dynamics.

    Drop-in replacement for FHNAttention that uses Triton kernels
    for FHN dynamics computation. Provides same API but with GPU-optimized
    kernel execution.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 8,
        threshold: float = 0.5,
        tau: float = 12.5,
        dropout: float = 0.0,
        n_fhn_steps: int = 2,
    ):
        super().__init__()

        if not TRITON_AVAILABLE:
            raise ImportError(
                "Triton is not installed. Install it with: pip install triton"
            )

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.threshold = threshold
        self.tau = tau
        self.n_fhn_steps = n_fhn_steps

        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        # Q, K, V projections
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # FHN parameters (learnable)
        self.a = nn.Parameter(torch.tensor(0.7))
        self.b = nn.Parameter(torch.tensor(0.8))
        self.dt = nn.Parameter(torch.tensor(1.0))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, spectral_basis: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass with Triton-accelerated FHN dynamics.

        Args:
            x: Input tensor of shape (B, T, embed_dim)
            spectral_basis: Optional spectral basis (unused, for API compatibility)

        Returns:
            Tuple of (output, info) where:
            - output: Attention output of shape (B, T, embed_dim)
            - info: Dictionary with attention statistics
        """
        from einops import rearrange, einsum

        B, T, D = x.shape

        # QKV projection
        qkv = self.qkv(x)
        q, k, v = rearrange(
            qkv, "b t (three h d) -> three b h t d", three=3, h=self.n_heads
        )

        # Standard scaled dot-product attention
        attn_weights = einsum(q, k, "b h t d, b h s d -> b h t s")
        attn_weights = attn_weights / (self.head_dim ** 0.5)

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn_weights = attn_weights.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )

        # Softmax
        attn_probs = torch.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Apply FHN dynamics using Triton kernel
        attn_energy = attn_probs.sum(dim=-1)  # (B, H, T)

        # Normalize stimulus
        stim_scale = attn_energy.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        stimulus_normed = attn_energy / stim_scale

        # Soft threshold
        threshold_gate = torch.sigmoid((attn_energy.abs() - self.threshold) * 10.0)
        I = stimulus_normed * (0.1 + 0.9 * threshold_gate)
        I = I.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)  # (B, H, T, D)

        # Initialize FHN state
        v_init = torch.zeros_like(I)
        w_init = torch.zeros_like(I)

        # Run Triton kernel
        v_final, w_final = fhn_triton_kernel(
            v_init, w_init, I,
            self.a, self.b, self.tau, self.dt,
            n_steps=self.n_fhn_steps
        )

        # FHN gate modulation
        fhn_gate = torch.sigmoid(v_final.mean(dim=-1)).unsqueeze(-1)  # (B, H, T, 1)
        attn_probs = attn_probs * (0.5 + 0.5 * fhn_gate)
        attn_probs = attn_probs / (attn_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # Apply attention to values
        out = einsum(attn_probs, v, "b h t s, b h s d -> b h t d")
        out = rearrange(out, "b h t d -> b t (h d)")
        out = self.out_proj(out)

        info = {
            "attention_type": "fhn_triton",
            "backend": "triton",
            "fhn_state": v_final.mean(),
            "attn_probs": attn_probs,
        }

        return out, info
