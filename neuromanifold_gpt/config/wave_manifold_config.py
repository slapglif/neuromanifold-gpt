"""
Configuration for Neuro-Symbolic Wave Manifold Network (NS-WMN).

Unified configuration integrating:
- SSM Backbone (Mamba)
- Soliton Physics (Sine-Gordon, KdV)
- Topology (Braid, Jones)
- Continuous Generation (Rectified Flow, KAN)
- FNO Input
"""

from dataclasses import dataclass, field
from neuromanifold_gpt.config.ralph_base import RalphBaseConfig
from neuromanifold_gpt.model.continuous import ContinuousOutputConfig

@dataclass
class WaveManifoldConfig(RalphBaseConfig):
    # --- Wave Manifold Specifics ---
    
    # Model dimensions
    vocab_size: int = 50304 # Default to GPT-2 BPE, typically overridden
    
    # Input Processing
    use_fno_encoder: bool = True
    fno_modes: int = 32
    
    # Backbone
    use_mamba_backbone: bool = True
    mamba_state_dim: int = 16
    mamba_expand: int = 2
    
    # Soliton Dynamics (Latent Physics)
    use_soliton_mixing: bool = True
    soliton_type: str = "sine_gordon"  # "sine_gordon", "kdv"
    
    # Topological Regularization
    use_topological_loss: bool = True
    topology_weight: float = 0.1
    
    # Continuous Generation
    use_continuous_head: bool = True
    continuous_head_config: ContinuousOutputConfig = field(
        default_factory=lambda: ContinuousOutputConfig()
    )
    
    # Legacy compatibility overrides
    model_type: str = "wave_manifold"