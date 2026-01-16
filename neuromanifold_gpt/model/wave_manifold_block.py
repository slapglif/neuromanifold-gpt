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
from neuromanifold_gpt.model.soliton.attention import SolitonAttention
from neuromanifold_gpt.config.wave_manifold_config import WaveManifoldConfig

class WaveManifoldBlock(nn.Module):
    """
    Hybrid block integrating SSM dynamics and Soliton physics.
    
    Structure:
    x -> LayerNorm -> Mamba -> Residual
    x -> LayerNorm -> Soliton -> Residual
    """
    def __init__(self, config: WaveManifoldConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.n_embd
        
        # 1. SSM Backbone (Time Evolution)
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.mamba = MambaBlock(
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
            
        self.soliton = SolitonAttention(
            embed_dim=self.embed_dim,
            n_heads=config.n_head,
            use_sine_gordon=use_sg,
            use_kdv=use_kdv,
            use_heimburg_jackson=use_hj
        )
        
        # Optional: Topological features could be integrated here or at model level
        # For block level, we keep it simple.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mamba Path (Time mixing)
        if self.config.use_mamba_backbone:
            x = x + self.mamba(self.norm1(x))
        
        # Soliton Path (Physics mixing)
        if self.config.use_soliton_mixing:
            soliton_out, _ = self.soliton(self.norm2(x))
            x = x + soliton_out
            
        return x