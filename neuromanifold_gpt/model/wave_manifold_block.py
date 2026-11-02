"""
Wave Manifold Block.

The fundamental building block of NS-WMN.
Combines:
1. Mamba (SSM) for longitudinal time-evolution (sequence mixing).
2. Soliton Interaction for transverse feature mixing (physics-informed).
3. Topological Gating (optional).
"""

import torch
import torch.nn as nn

from neuromanifold_gpt.config.wave_manifold_config import WaveManifoldConfig
from neuromanifold_gpt.model.soliton.attention import SolitonInteractionLayer
from neuromanifold_gpt.model.ssm.hyena_operator import HyenaOperator
from neuromanifold_gpt.model.ssm.mamba import MambaBlock
from neuromanifold_gpt.model.ssm.mixture_of_mamba import (
    AdaptiveMixtureOfMamba,
)


class WaveManifoldBlock(nn.Module):
    """
    Hybrid block integrating SSM/Hyena dynamics and Soliton physics.

    Structure:
    x -> LayerNorm -> Mamba/Hyena -> Residual
    x -> LayerNorm -> Soliton -> Residual
    """

    def __init__(self, config: WaveManifoldConfig, max_seq_len: int = 1024):
        super().__init__()
        self.config = config
        self.embed_dim = config.n_embd

        # 1. SSM/Hyena Backbone (Time Evolution)
        self.norm1 = nn.LayerNorm(self.embed_dim)

        use_mom = getattr(config, "use_mixture_of_mamba", False)

        if config.backbone_type == "hyena":
            self.mixer = HyenaOperator(
                d_model=self.embed_dim,
                l_max=max_seq_len,
                order=2,
                filter_order=64,
                dropout=config.dropout,
            )
        elif use_mom:
            self.mixer = AdaptiveMixtureOfMamba(
                embed_dim=self.embed_dim,
                num_experts=getattr(config, "mom_num_experts", 8),
                top_k=getattr(config, "mom_top_k", 2),
                state_dim=config.mamba_state_dim,
                use_spectral_routing=True,
                expand_factor=config.mamba_expand,
                dropout=config.dropout,
                memory_efficient=False,
            )
        else:
            self.mixer = MambaBlock(
                embed_dim=self.embed_dim,
                state_dim=config.mamba_state_dim,
                expand_factor=config.mamba_expand,
                dropout=config.dropout,
                memory_efficient=False,
            )

        # 2. Soliton Interaction (Feature Physics)
        self.norm2 = nn.LayerNorm(self.embed_dim)

        # Map string config to boolean flags
        use_sg = config.soliton_type == "sine_gordon"
        use_kdv = config.soliton_type == "kdv"
        use_hj = config.soliton_type == "heimburg_jackson"

        # Fallback if "all" or unknown
        if config.soliton_type == "all" or not (use_sg or use_kdv or use_hj):
            use_sg = use_kdv = use_hj = True

        self.soliton = SolitonInteractionLayer(
            embed_dim=self.embed_dim,
            n_heads=config.n_head,
            use_sine_gordon=use_sg,
            use_kdv=use_kdv,
            use_heimburg_jackson=use_hj,
        )

        # Optional: Topological features could be integrated here or at model level
        # For block level, we keep it simple.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        info = {}

        if self.config.use_mamba_backbone or self.config.backbone_type in [
            "hyena",
            "mamba",
        ]:
            mixer_input = self.norm1(x)

            if isinstance(self.mixer, AdaptiveMixtureOfMamba):
                mixer_out, mom_info = self.mixer(mixer_input)
                info.update(mom_info)
            else:
                mixer_out = self.mixer(mixer_input)

            x = x + mixer_out

        if self.config.use_soliton_mixing:
            soliton_out, soliton_info = self.soliton(self.norm2(x))
            x = x + soliton_out
            info.update(soliton_info)

        return x
