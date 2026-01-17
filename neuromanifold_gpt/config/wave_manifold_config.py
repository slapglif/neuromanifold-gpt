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
    vocab_size: int = 50304  # Default to GPT-2 BPE, typically overridden

    # Input Processing
    use_fno_encoder: bool = True
    fno_modes: int = 32

    # Backbone
    backbone_type: str = "hyena"  # "mamba", "hyena", or "mom"
    use_mamba_backbone: bool = False
    mamba_state_dim: int = 16
    mamba_expand: int = 2

    # Mixture-of-Mamba (MoM) settings
    use_mixture_of_mamba: bool = False
    mom_num_experts: int = 8
    mom_top_k: int = 2
    mom_load_balance_weight: float = 0.01
    mom_state_dim: int = 16  # Added for alignment with train.py

    # Hybrid Stack (Disabled for Pure Wave Network)
    use_hybrid_stack: bool = False
    hybrid_ratio: int = 7
    attention_layer_idx: list = field(default_factory=list)

    # Soliton Dynamics (Latent Physics)
    use_soliton_mixing: bool = True
    soliton_type: str = "sine_gordon"  # "sine_gordon", "kdv"

    # Topological Regularization
    use_topological_loss: bool = True
    topology_weight: float = 0.1
    braid_dim: int = 64

    # Continuous Generation
    use_continuous_head: bool = True
    use_sac_output: bool = False
    continuous_head_config: ContinuousOutputConfig = field(
        default_factory=lambda: ContinuousOutputConfig()
    )

    # Hybrid Reasoning (System 1/2)
    use_hybrid_reasoning: bool = False
    n_thinking_layers: int = 2
    thinking_threshold: float = 0.5
    use_e7_prior: bool = False

    # Legacy compatibility overrides
    model_type: str = "wave_manifold"
