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
from neuromanifold_gpt.model.ssm.mamba import MambaBlock
from neuromanifold_gpt.model.ssm.hyena_operator import HyenaOperator
from neuromanifold_gpt.model.soliton.attention import SolitonInteractionLayer
from neuromanifold_gpt.config.wave_manifold_config import WaveManifoldConfig

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
        
        if config.backbone_type == "hyena":
            self.mixer = HyenaOperator(
                d_model=self.embed_dim,
                l_max=max_seq_len,
                order=2,
                filter_order=64,
                dropout=config.dropout
            )
        else: # Default to Mamba
            self.mixer = MambaBlock(
                embed_dim=self.embed_dim,
                state_dim=config.mamba_state_dim,
                expand_factor=config.mamba_expand,
                dropout=config.dropout
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
            use_heimburg_jackson=use_hj
        )
        
        # Optional: Topological features could be integrated here or at model level
        # For block level, we keep it simple.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Time mixing (Mamba or Hyena)
        if self.config.use_mamba_backbone: # Keep flag name for compatibility for now
            x = x + self.mixer(self.norm1(x))
        
        # Soliton Path (Physics mixing)
        if self.config.use_soliton_mixing:
            soliton_out, _ = self.soliton(self.norm2(x))
            x = x + soliton_out
            
        return x