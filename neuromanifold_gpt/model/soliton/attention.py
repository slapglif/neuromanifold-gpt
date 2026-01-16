# neuromanifold_gpt/model/soliton/attention.py
"""
Soliton Attention: Attention mechanism using PDE-based soliton dynamics.

Combines three PDE solvers to model different aspects of wave dynamics:
1. Sine-Gordon: Topological solitons for semantic boundaries
2. KdV: Dispersive waves for information propagation
3. Heimburg-Jackson: Thermodynamic solitons for stable coherence

The attention mechanism uses soliton dynamics to:
- Compute attention weights through wave interference
- Propagate information via stable soliton solutions
- Maintain coherence through conserved quantities

Reference:
- Kaufmann, "Action Potentials and Electrochemical Coupling" (1989)
- Heimburg & Jackson, "On soliton propagation in biomembranes" (2005)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .sine_gordon import SineGordonSolver
from .kdv import KdVSolver
from .heimburg_jackson import HeimburgJacksonSolver


class SolitonAttention(nn.Module):
    """
    Attention mechanism based on soliton PDE dynamics.

    Combines three types of soliton physics:
    1. Sine-Gordon (topological): Detects semantic boundaries/transitions
    2. KdV (dispersive): Propagates information across sequence
    3. Heimburg-Jackson (thermodynamic): Maintains stable coherence

    The mechanism:
    - Projects input to Q, K, V
    - Computes attention scores via dot product
    - Applies soliton dynamics to modulate attention patterns
    - Uses wave interference for attention aggregation

    Example:
        >>> attn = SolitonAttention(384, n_heads=8)
        >>> x = torch.randn(2, 32, 384)
        >>> out, info = attn(x)
        >>> assert out.shape == x.shape
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 8,
        dropout: float = 0.0,
        n_pde_steps: int = 3,
        use_sine_gordon: bool = True,
        use_kdv: bool = True,
        use_heimburg_jackson: bool = True,
        dt: float = 0.05,
        causal: bool = True,
    ):
        """
        Initialize SolitonAttention.

        Args:
            embed_dim: Model embedding dimension
            n_heads: Number of attention heads
            dropout: Dropout probability for attention weights
            n_pde_steps: Number of PDE integration steps per forward pass
            use_sine_gordon: Enable Sine-Gordon dynamics (topological)
            use_kdv: Enable KdV dynamics (dispersive)
            use_heimburg_jackson: Enable Heimburg-Jackson dynamics (thermodynamic)
            dt: Time step for PDE integration
            causal: Use causal masking (for autoregressive models)
        """
        super().__init__()
        assert embed_dim % n_heads == 0, f"embed_dim {embed_dim} must be divisible by n_heads {n_heads}"

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.n_pde_steps = n_pde_steps
        self.causal = causal
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Feature flags for ablation studies
        self.use_sine_gordon = use_sine_gordon
        self.use_kdv = use_kdv
        self.use_heimburg_jackson = use_heimburg_jackson

        # Count active solvers for mixing weights
        self.n_active_solvers = sum([use_sine_gordon, use_kdv, use_heimburg_jackson])
        if self.n_active_solvers == 0:
            # Default to at least one solver
            self.use_sine_gordon = True
            self.n_active_solvers = 1

        # Q, K, V projections
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # PDE Solvers - one per head group
        # We use shared solvers across heads for efficiency
        if self.use_sine_gordon:
            self.sine_gordon = SineGordonSolver(
                dim=self.head_dim,
                dt=dt,
                n_steps=n_pde_steps,
                wave_speed=1.0,
                use_rk4=True,
                damping=0.01,  # Small damping for stability
            )

        if self.use_kdv:
            self.kdv = KdVSolver(
                dim=self.head_dim,
                dt=dt * 0.5,  # KdV needs smaller dt for stability
                n_steps=n_pde_steps,
                nonlin_coeff=6.0,
                disp_coeff=1.0,
                use_rk4=True,
                damping=0.01,
            )

        if self.use_heimburg_jackson:
            self.heimburg_jackson = HeimburgJacksonSolver(
                dim=self.head_dim,
                dt=dt * 0.5,
                n_steps=n_pde_steps,
                c0_squared=1.0,
                p_coeff=-1.0,  # Scaled down from biophysical values
                q_coeff=1.0,
                h_disp=0.1,
                use_rk4=True,
                damping=0.01,
            )

        # Learnable mixing weights for combining solver outputs
        self.solver_mix = nn.Parameter(torch.ones(self.n_active_solvers) / self.n_active_solvers)

        # Soliton gating: controls how much PDE dynamics affects attention
        self.soliton_gate = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim // 2),
            nn.SiLU(),
            nn.Linear(self.head_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def _apply_soliton_dynamics(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Apply soliton PDE dynamics to input tensor.

        Args:
            x: Input tensor (B, H, T, D) where D is head_dim

        Returns:
            Tuple of (evolved_x, info_dict)
        """
        B, H, T, D = x.shape
        info = {}

        # Normalize input to prevent numerical instability
        x_norm = x / (x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) + 1e-6)
        x_norm = x_norm * 2.0  # Scale to [-2, 2] range for PDE stability

        # Flatten batch and heads for PDE processing: (B*H, T, D)
        x_flat = rearrange(x_norm, 'b h t d -> (b h) t d')

        # Collect outputs from active solvers
        outputs = []
        mix_weights = F.softmax(self.solver_mix, dim=0)
        mix_idx = 0

        if self.use_sine_gordon:
            # Sine-Gordon: topological solitons
            sg_out, sg_info = self.sine_gordon(x_flat, n_steps=self.n_pde_steps)
            outputs.append(sg_out * mix_weights[mix_idx])
            info['sine_gordon_energy'] = sg_info.get('energy', 0.0)
            info['sine_gordon_charge'] = sg_info.get('topological_charge', 0.0)
            mix_idx += 1

        if self.use_kdv:
            # KdV: dispersive wave dynamics
            kdv_out, kdv_info = self.kdv(x_flat, n_steps=self.n_pde_steps)
            outputs.append(kdv_out * mix_weights[mix_idx])
            info['kdv_mass'] = kdv_info.get('mass', 0.0)
            info['kdv_momentum'] = kdv_info.get('momentum', 0.0)
            mix_idx += 1

        if self.use_heimburg_jackson:
            # Heimburg-Jackson: thermodynamic solitons
            hj_out, hj_info = self.heimburg_jackson(x_flat, n_steps=self.n_pde_steps)
            outputs.append(hj_out * mix_weights[mix_idx])
            info['hj_enthalpy'] = hj_info.get('enthalpy', 0.0)
            info['hj_wave_speed'] = hj_info.get('wave_speed', 0.0)

        # Combine solver outputs (weighted sum)
        combined = sum(outputs)

        # Reshape back to (B, H, T, D)
        combined = rearrange(combined, '(b h) t d -> b h t d', b=B, h=H)

        # Store mixing weights in info
        info['solver_mix_weights'] = mix_weights.detach()

        return combined, info

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Forward pass with soliton-modulated attention.

        Args:
            x: Input tensor (B, T, D)
            mask: Optional attention mask (B, T) or (B, 1, T, T)

        Returns:
            Tuple of (output, info) where:
            - output: Tensor (B, T, D) same shape as input
            - info: Dictionary with soliton dynamics statistics
        """
        B, T, D = x.shape
        H = self.n_heads

        # QKV projection: (B, T, 3*D) -> Q, K, V each (B, H, T, head_dim)
        qkv = self.qkv(x)
        q, k, v = rearrange(
            qkv, 'b t (three h d) -> three b h t d',
            three=3, h=self.n_heads
        )

        # Apply soliton dynamics to values
        # This modulates the "information content" being attended to
        v_evolved, pde_info = self._apply_soliton_dynamics(v)

        # Compute soliton gate: how much PDE evolution to use
        # (B, H, T, 1) gating factor
        gate = self.soliton_gate(v.mean(dim=2, keepdim=True))  # Global gate per head
        gate = gate.expand(-1, -1, T, -1)

        # Blend original values with evolved values
        v_blended = gate * v_evolved + (1 - gate) * v

        # Standard scaled dot-product attention
        # (B, H, T, D) @ (B, H, D, T) -> (B, H, T, T)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Causal mask: prevent attending to future positions
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            attn_scores = attn_scores.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0),
                float('-inf')
            )

        # Optional external mask
        if mask is not None:
            if mask.dim() == 2:
                # (B, T) -> (B, 1, 1, T)
                mask = mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to soliton-modulated values
        # (B, H, T, T) @ (B, H, T, D) -> (B, H, T, D)
        out = torch.matmul(attn_weights, v_blended)

        # Reshape and project output: (B, H, T, D) -> (B, T, H*D)
        out = rearrange(out, 'b h t d -> b t (h d)')
        out = self.out_proj(out)

        # Compile info dictionary
        info = {
            'soliton_gate': gate.mean().item(),
            'attn_entropy': -(attn_weights * (attn_weights + 1e-8).log()).sum(dim=-1).mean().item(),
            **pde_info,
        }

        return out, info

    def extra_repr(self) -> str:
        """Extra representation for module printing."""
        solvers = []
        if self.use_sine_gordon:
            solvers.append('SineGordon')
        if self.use_kdv:
            solvers.append('KdV')
        if self.use_heimburg_jackson:
            solvers.append('HeimburgJackson')

        return (
            f"embed_dim={self.embed_dim}, "
            f"n_heads={self.n_heads}, "
            f"n_pde_steps={self.n_pde_steps}, "
            f"solvers=[{', '.join(solvers)}], "
            f"causal={self.causal}"
        )


class MultiHeadSolitonAttention(nn.Module):
    """
    Multi-head variant with separate soliton dynamics per head group.

    Divides heads into groups, each with its own PDE solver type:
    - Group 1: Sine-Gordon (topological)
    - Group 2: KdV (dispersive)
    - Group 3: Heimburg-Jackson (thermodynamic)

    This allows different aspects of the input to be processed
    by different physics simultaneously.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 12,
        dropout: float = 0.0,
        n_pde_steps: int = 3,
        dt: float = 0.05,
        causal: bool = True,
    ):
        """
        Initialize MultiHeadSolitonAttention.

        Args:
            embed_dim: Model embedding dimension
            n_heads: Number of attention heads (should be divisible by 3)
            dropout: Dropout probability
            n_pde_steps: Number of PDE steps per forward pass
            dt: Time step for PDE integration
            causal: Use causal masking
        """
        super().__init__()
        assert embed_dim % n_heads == 0

        # Divide heads into 3 groups (or as close as possible)
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // n_heads

        # Head group assignments
        base_group_size = n_heads // 3
        remainder = n_heads % 3

        self.group_sizes = [
            base_group_size + (1 if i < remainder else 0)
            for i in range(3)
        ]

        # Create attention modules for each physics type
        # Each group has group_size heads, so embed_dim = group_size * head_dim
        self.sine_gordon_attn = SolitonAttention(
            embed_dim=self.group_sizes[0] * self.head_dim,
            n_heads=self.group_sizes[0],
            dropout=dropout,
            n_pde_steps=n_pde_steps,
            use_sine_gordon=True,
            use_kdv=False,
            use_heimburg_jackson=False,
            dt=dt,
            causal=causal,
        ) if self.group_sizes[0] > 0 else None

        self.kdv_attn = SolitonAttention(
            embed_dim=self.group_sizes[1] * self.head_dim,
            n_heads=self.group_sizes[1],
            dropout=dropout,
            n_pde_steps=n_pde_steps,
            use_sine_gordon=False,
            use_kdv=True,
            use_heimburg_jackson=False,
            dt=dt,
            causal=causal,
        ) if self.group_sizes[1] > 0 else None

        self.hj_attn = SolitonAttention(
            embed_dim=self.group_sizes[2] * self.head_dim,
            n_heads=self.group_sizes[2],
            dropout=dropout,
            n_pde_steps=n_pde_steps,
            use_sine_gordon=False,
            use_kdv=False,
            use_heimburg_jackson=True,
            dt=dt,
            causal=causal,
        ) if self.group_sizes[2] > 0 else None

        # Projection layers to split/merge
        self.in_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Forward pass with head-grouped soliton dynamics.

        Args:
            x: Input tensor (B, T, D)
            mask: Optional attention mask

        Returns:
            Tuple of (output, info)
        """
        B, T, D = x.shape

        # Project input
        x = self.in_proj(x)

        # Split by head groups
        group_dims = [g * self.head_dim for g in self.group_sizes]
        x_groups = x.split(group_dims, dim=-1)

        outputs = []
        info = {}

        # Process each group with its physics
        if self.sine_gordon_attn is not None and self.group_sizes[0] > 0:
            sg_out, sg_info = self.sine_gordon_attn(x_groups[0], mask)
            outputs.append(sg_out)
            info.update({f'sg_{k}': v for k, v in sg_info.items()})

        if self.kdv_attn is not None and self.group_sizes[1] > 0:
            kdv_out, kdv_info = self.kdv_attn(x_groups[1], mask)
            outputs.append(kdv_out)
            info.update({f'kdv_{k}': v for k, v in kdv_info.items()})

        if self.hj_attn is not None and self.group_sizes[2] > 0:
            hj_out, hj_info = self.hj_attn(x_groups[2], mask)
            outputs.append(hj_out)
            info.update({f'hj_{k}': v for k, v in hj_info.items()})

        # Concatenate outputs
        out = torch.cat(outputs, dim=-1)
        out = self.out_proj(out)

        return out, info
