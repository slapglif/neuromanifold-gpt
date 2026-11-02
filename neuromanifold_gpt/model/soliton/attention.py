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

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .heimburg_jackson import HeimburgJacksonSolver
from .kdv import KdVSolver
from .sine_gordon import SineGordonSolver


class SolitonInteractionLayer(nn.Module):
    """
    Pure Soliton Interaction Layer (Linear O(N)).

    Replaces quadratic attention with physics-based elastic scattering.

    Mechanism:
    1. Input U represents the wave field state.
    2. Hyena/SSM block (external) moves information globally.
    3. This layer applies the local non-linear PDE dynamics (collision logic).
    4. Solitons interact via the non-linearity (e.g. sin(u)) but preserve shape.

    Complexity: O(N) (Linear)
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
        super().__init__()
        # No head division needed for pure interaction, but kept for compatibility
        self.embed_dim = embed_dim
        self.n_pde_steps = n_pde_steps

        # Feature flags
        self.use_sine_gordon = use_sine_gordon
        self.use_kdv = use_kdv
        self.use_heimburg_jackson = use_heimburg_jackson

        # Solvers
        if self.use_sine_gordon:
            self.sine_gordon = SineGordonSolver(
                dim=embed_dim,  # Full dimension
                dt=dt,
                n_steps=n_pde_steps,
                wave_speed=1.0,
                use_rk4=True,
                damping=0.5,  # High damping for stability/locality
                causal=causal,
            )

        if self.use_kdv:
            self.kdv = KdVSolver(
                dim=embed_dim,
                dt=dt * 0.5,
                n_steps=n_pde_steps,
                use_rk4=True,
                damping=0.5,
                causal=causal,
            )

        if self.use_heimburg_jackson:
            self.heimburg_jackson = HeimburgJacksonSolver(
                dim=embed_dim,
                dt=dt * 0.5,
                n_steps=n_pde_steps,
                use_rk4=True,
                damping=0.5,
                causal=causal,
            )

        # Gating/Mixing weights
        n_solvers = sum([use_sine_gordon, use_kdv, use_heimburg_jackson])
        self.solver_mix = nn.Parameter(
            torch.ones(max(1, n_solvers)) / max(1, n_solvers)
        )

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,  # Unused, PDE is local/causal
    ) -> tuple[torch.Tensor, dict]:
        """
        Apply soliton elastic scattering.

        Args:
            x: (B, T, D) Wave field

        Returns:
            out: (B, T, D) Evolved field
        """
        B, T, D = x.shape
        info = {}

        # 1. Prepare Field (Normalize for PDE stability range [-pi, pi])
        # Soft clamp to keep dynamics valid
        x_field = torch.tanh(x) * 2.0

        outputs = []
        mix_weights = F.softmax(self.solver_mix, dim=0)
        idx = 0

        # 2. Parallel Physics Evolution
        if self.use_sine_gordon:
            sg_out, sg_info = self.sine_gordon(x_field)
            outputs.append(sg_out * mix_weights[idx])
            idx += 1

        if self.use_kdv:
            kdv_out, kdv_info = self.kdv(x_field)
            outputs.append(kdv_out * mix_weights[idx])
            idx += 1

        if self.use_heimburg_jackson:
            hj_out, hj_info = self.heimburg_jackson(x_field)
            outputs.append(hj_out * mix_weights[idx])
            idx += 1

        # 3. Combine Results
        if outputs:
            evolved = sum(outputs)
        else:
            evolved = x_field

        # 4. Project and Residual
        # Note: The block usually handles residual, but we project here
        out = self.out_proj(evolved)
        out = self.dropout(out)

        return out, info

    def extra_repr(self) -> str:
        """Extra representation for module printing."""
        solvers = []
        if self.use_sine_gordon:
            solvers.append("SineGordon")
        if self.use_kdv:
            solvers.append("KdV")
        if self.use_heimburg_jackson:
            solvers.append("HeimburgJackson")

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
            base_group_size + (1 if i < remainder else 0) for i in range(3)
        ]

        # Create attention modules for each physics type
        # Each group has group_size heads, so embed_dim = group_size * head_dim
        self.sine_gordon_attn = (
            SolitonAttention(
                embed_dim=self.group_sizes[0] * self.head_dim,
                n_heads=self.group_sizes[0],
                dropout=dropout,
                n_pde_steps=n_pde_steps,
                use_sine_gordon=True,
                use_kdv=False,
                use_heimburg_jackson=False,
                dt=dt,
                causal=causal,
            )
            if self.group_sizes[0] > 0
            else None
        )

        self.kdv_attn = (
            SolitonAttention(
                embed_dim=self.group_sizes[1] * self.head_dim,
                n_heads=self.group_sizes[1],
                dropout=dropout,
                n_pde_steps=n_pde_steps,
                use_sine_gordon=False,
                use_kdv=True,
                use_heimburg_jackson=False,
                dt=dt,
                causal=causal,
            )
            if self.group_sizes[1] > 0
            else None
        )

        self.hj_attn = (
            SolitonAttention(
                embed_dim=self.group_sizes[2] * self.head_dim,
                n_heads=self.group_sizes[2],
                dropout=dropout,
                n_pde_steps=n_pde_steps,
                use_sine_gordon=False,
                use_kdv=False,
                use_heimburg_jackson=True,
                dt=dt,
                causal=causal,
            )
            if self.group_sizes[2] > 0
            else None
        )

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
            info.update({f"sg_{k}": v for k, v in sg_info.items()})

        if self.kdv_attn is not None and self.group_sizes[1] > 0:
            kdv_out, kdv_info = self.kdv_attn(x_groups[1], mask)
            outputs.append(kdv_out)
            info.update({f"kdv_{k}": v for k, v in kdv_info.items()})

        if self.hj_attn is not None and self.group_sizes[2] > 0:
            hj_out, hj_info = self.hj_attn(x_groups[2], mask)
            outputs.append(hj_out)
            info.update({f"hj_{k}": v for k, v in hj_info.items()})

        # Concatenate outputs
        out = torch.cat(outputs, dim=-1)
        out = self.out_proj(out)

        return out, info
