# neuromanifold_gpt/model/attention/fhn.py
"""
Soliton Attention based on Kaufmann-Heimburg Model.

Action potentials as acoustic solitons in lipid membranes (NOT electrical).
Key properties:
- Threshold behavior (all-or-none)
- Stable wave propagation
- Collision without annihilation

Reference: https://www.nbi.ku.dk/membranes/Kaufmann/publications.html
"""

import torch
import torch.nn as nn
from einops import rearrange, einsum
from .partitioning import SpectralPartitioner


@torch.jit.script
def fhn_update_step(
    v: torch.Tensor,
    w: torch.Tensor,
    I: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    tau: float,
    dt: torch.Tensor,
    n_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    JIT-compiled FHN update loop.
    Replaces Python loop with optimized TorchScript (C++).
    """
    # implicit_denom depends on dt, b, tau. b is tensor?
    # b is parameter tensor. JIT can handle tensor ops.
    # We should keep it as tensor op if b is tensor.
    # But for division, we need scalar if we want scalar float arithmetic?
    # No, keep tensors.

    # Precompute constants
    # Note: JIT script needs explicit types or tensor ops

    for _ in range(n_steps):
        # Explicit Euler for v
        # v, w, I are tensors (B, H, T, D)
        # a, b, dt are tensors (0-d or 1-d)

        # dv = v - v^3/3 - w + I
        v3 = v * v * v
        dv_explicit = v - (v3 / 3.0) - w + I

        # Implicit Euler for w
        # w_new = (w + dt/tau * (v + a)) / (1 + dt*b/tau)

        # We need to ensure shapes broadcast correctly. a, b are scalars (0-d tensors)

        alpha = dt / tau
        denom = 1.0 + alpha * b

        w_step = alpha * (v + a)
        w_new = (w + w_step) / denom

        # Update v
        # v_new = v + dt * dv_explicit
        # Use v_new calculation
        # To match IMEX, we usually update v using OLD w, then update w using NEW v?
        # My previous implementation was:
        # u_new = u + du * dt (Explicit u)
        # w_new = (w + ...) (Implicit w using u_new)

        # Let's follow that order for consistency
        v_next = v + dt * dv_explicit

        # Recompute w using v_next
        w_step_next = alpha * (v_next + a)
        w_next = (w + w_step_next) / denom

        # Update state
        v = torch.clamp(v_next, -3.0, 3.0)
        w = torch.clamp(w_next, -3.0, 3.0)

    return v, w


class FHNDynamics(nn.Module):
    """
    Excitable dynamics layer (FitzHugh-Nagumo).

    Simulates wave propagation in excitable media (like axons).
    - u: membrane potential (fast variable)
    - w: recovery variable (slow variable)
    """

    def __init__(
        self,
        dim: int,
        tau: float = 12.5,  # Fixed: proper slow-fast separation
        threshold: float = 0.5,
        use_imex: bool = True,  # Use semi-implicit scheme for efficiency
        use_fused: bool = False,  # Deprecated/Ignored (JIT is standard now)
    ):
        super().__init__()
        self.dim = dim
        self.tau = tau
        self.threshold = threshold
        self.use_imex = use_imex

        # Learnable parameters (standard FHN values)
        self.a = nn.Parameter(torch.tensor(0.7))
        self.b = nn.Parameter(torch.tensor(0.8))
        # Use dt=1.0 by default (matches original IMEX design)
        self.dt = nn.Parameter(torch.tensor(1.0))

    def forward(
        self, stimulus: torch.Tensor, n_steps: int = 2
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evolve soliton dynamics using JIT-compiled IMEX scheme.
        """
        # Normalize stimulus to prevent numerical explosion
        stim_scale = stimulus.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        stimulus_normed = stimulus / stim_scale

        # Soft threshold with sigmoid (differentiable, no gradient discontinuity)
        # Replaces hard threshold for better gradient flow
        threshold_gate = torch.sigmoid((stimulus.abs() - self.threshold) * 10.0)
        I = stimulus_normed * (0.1 + 0.9 * threshold_gate)  # Soft transition

        # Initialize state
        v = torch.zeros_like(stimulus)
        w = torch.zeros_like(stimulus)

        # Execute dynamics using JIT-compiled kernel
        if self.use_imex:
            v, w = fhn_update_step(v, w, I, self.a, self.b, self.tau, self.dt, n_steps)
        else:
            # Fallback for ablation studies (Explicit Euler)
            dt_val = 1.0 / max(n_steps, 5)
            dt = torch.tensor(dt_val, device=v.device)
            for _ in range(max(n_steps, 5)):
                dv = v - v.pow(3) / 3 - w + I
                dw = (v + self.a - self.b * w) / self.tau
                v = (v + dt * dv).clamp(-3.0, 3.0)
                w = (w + dt * dw).clamp(-3.0, 3.0)

        # Scale response by original stimulus magnitude
        response = v * stim_scale

        return response, v


class FHNAttention(nn.Module):
    """
    Attention via FHN excitable wave propagation.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 8,
        threshold: float = 0.5,
        tau: float = 12.5,
        pulse_width_base: int = 4,
        dropout: float = 0.0,
        n_fhn_steps: int = 2,
        use_imex: bool = True,
        use_partitioning: bool = True,
        use_fused: bool = False,  # Deprecated
        use_flash_fhn_fusion: bool = False,  # Use Flash Attention + output modulation
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.threshold = threshold
        self.pulse_width_base = pulse_width_base
        self.n_fhn_steps = n_fhn_steps
        self.use_partitioning = use_partitioning
        self.use_flash_fhn_fusion = use_flash_fhn_fusion

        assert embed_dim % n_heads == 0

        # Q, K, V projections
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # FHN dynamics per head
        self.fhn = FHNDynamics(self.head_dim, tau, threshold, use_imex)

        # Pulse width modulation (content-dependent)
        self.pulse_width_net = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim // 2),
            nn.SiLU(),
            nn.Linear(self.head_dim // 2, 1),
            nn.Softplus(),
        )

        # Spectral filtering
        self.spectral_filter = nn.Parameter(torch.ones(n_heads, 32) * 0.5)

        # Partitioning for FHN stability
        self.use_partitioning = use_partitioning
        if self.use_partitioning:
            self.partitioner = SpectralPartitioner(32)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, spectral_basis: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Causal FHN attention: Standard causal attention with FHN modulation.

        CRITICAL FIX: Previous implementation had no causal masking, causing
        information leakage from future positions during training.
        """
        B, T, D = x.shape

        # QKV projection
        qkv = self.qkv(x)
        q, key, v = rearrange(
            qkv, "b t (three h d) -> three b h t d", three=3, h=self.n_heads
        )

        # Compute pulse widths (content-dependent)
        pulse_widths = self.pulse_width_net(q.mean(dim=2))  # (B, H, 1)
        pulse_widths = self.pulse_width_base + pulse_widths.squeeze(-1)  # (B, H)

        # === Flash Attention Fusion Path ===
        # Use Flash Attention with output modulation when fusion is enabled
        if self.use_flash_fhn_fusion and self.n_fhn_steps > 0:
            # Use PyTorch's optimized scaled_dot_product_attention (Flash Attention)
            out = torch.nn.functional.scaled_dot_product_attention(
                q, key, v,
                attn_mask=None,  # Use is_causal=True instead
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True,
            )

            # Compute output variance/statistics as FHN stimulus proxy
            # Use output statistics (cheaper than computing full attention weights)
            out_variance = out.var(dim=-1, keepdim=True)  # (B, H, T, 1)
            out_std = out.std(dim=-1, keepdim=True)  # (B, H, T, 1)

            # Use output variance as FHN stimulus (proxy for attention "focus")
            # Higher variance indicates more dynamic/focused attention
            fhn_stimulus = out_variance.expand(-1, -1, -1, self.head_dim)
            fhn_out, fhn_state = self.fhn(
                fhn_stimulus,
                n_steps=self.n_fhn_steps,
            )

            # Modulate output with FHN response
            # Use sigmoid to bound the modulation gate
            fhn_gate = torch.sigmoid(fhn_out)  # (B, H, T, D)

            # Apply modulation (0.5 baseline + 0.5 * gate for stability)
            out = out * (0.5 + 0.5 * fhn_gate)

            fhn_state_val = fhn_state.mean()
            attn_probs = None  # Not computed in flash attention

            out_mean = out.mean(dim=-1, keepdim=True)
            output_stats = {
                "variance": out_variance.mean().item(),
                "std": out_std.mean().item(),
                "mean": out_mean.abs().mean().item(),
            }
        # === Fast Path: Use Flash Attention when no FHN modulation ===
        elif self.n_fhn_steps == 0:
            # Use PyTorch's optimized scaled_dot_product_attention (Flash Attention)
            # This is MUCH faster than manual einsum implementation
            out = torch.nn.functional.scaled_dot_product_attention(
                q, key, v,
                attn_mask=None,  # Use is_causal=True instead
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True,
            )

            # Compute output variance/statistics as FHN stimulus proxy
            # This provides a measure of attention "activity" without explicit attention weights
            out_variance = out.var(dim=-1, keepdim=True)  # (B, H, T, 1)
            out_std = out.std(dim=-1, keepdim=True)  # (B, H, T, 1)
            out_mean = out.mean(dim=-1, keepdim=True)  # (B, H, T, 1)

            fhn_state_val = torch.tensor(0.0, device=x.device)
            attn_probs = None  # Not computed in flash attention
            output_stats = {
                "variance": out_variance.mean().item(),
                "std": out_std.mean().item(),
                "mean": out_mean.abs().mean().item(),
            }
        else:
            # === Standard Causal Scaled Dot-Product Attention (for FHN modulation) ===
            # Legacy path: manual attention computation with FHN weight modulation
            # (B, H, T, D) @ (B, H, D, T) -> (B, H, T, T)
            attn_weights = einsum(q, key, "b h t d, b h s d -> b h t s")
            attn_weights = attn_weights / (self.head_dim**0.5)

            # Causal mask: prevent attending to future positions
            causal_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
            attn_weights = attn_weights.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

            # Apply softmax
            attn_probs = torch.softmax(attn_weights, dim=-1)
            attn_probs = self.dropout(attn_probs)

            # Apply FHN dynamics to modulate attention patterns along query dimension
            attn_energy = attn_probs.sum(dim=-1)  # (B, H, T) - attention "energy" per query
            fhn_out, fhn_state = self.fhn(
                attn_energy.unsqueeze(-1).expand(-1, -1, -1, self.head_dim),
                n_steps=self.n_fhn_steps,
            )
            fhn_gate = torch.sigmoid(fhn_out.mean(dim=-1)).unsqueeze(-1)  # (B, H, T, 1)

            # Modulate attention with FHN response (0.5 baseline + 0.5 * gate for stability)
            attn_probs = attn_probs * (0.5 + 0.5 * fhn_gate)
            # Renormalize (preserve causal structure)
            attn_probs = attn_probs / (attn_probs.sum(dim=-1, keepdim=True) + 1e-8)
            fhn_state_val = fhn_state.mean()

            # Apply attention to values
            out = einsum(attn_probs, v, "b h t s, b h s d -> b h t d")

            # Compute output variance/statistics as FHN stimulus proxy
            out_variance = out.var(dim=-1, keepdim=True)  # (B, H, T, 1)
            out_std = out.std(dim=-1, keepdim=True)  # (B, H, T, 1)
            out_mean = out.mean(dim=-1, keepdim=True)  # (B, H, T, 1)
            output_stats = {
                "variance": out_variance.mean().item(),
                "std": out_std.mean().item(),
                "mean": out_mean.abs().mean().item(),
            }

        out = rearrange(out, "b h t d -> b t (h d)")
        out = self.out_proj(out)

        info = {
            "pulse_widths": pulse_widths,
            "fhn_state": fhn_state_val,
            "attn_probs": attn_probs,
            "output_stats": output_stats,
        }

        return out, info
