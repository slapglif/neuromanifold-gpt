# neuromanifold_gpt/model/continuous/__init__.py
"""Continuous Generation for NeuroManifoldGPT.

Exports:
    LatentDiffusion: Diffusion-based decoder for continuous semantic generation
    DiffusionConfig: Configuration dataclass for diffusion hyperparameters
    SACPolicy: Soft Actor-Critic policy for continuous action spaces
    SACConfig: Configuration dataclass for SAC hyperparameters
    DDPGPolicy: Deep Deterministic Policy Gradient for continuous actions
    DDPGConfig: Configuration dataclass for DDPG hyperparameters
    ContinuousOutputHead: Unified output head combining diffusion and RL
    ContinuousOutputConfig: Configuration dataclass for output head

Continuous generation mechanisms enable output generation on the continuous
semantic manifold, moving beyond discrete token-by-token autoregression.
This module implements two complementary approaches:

1. Diffusion-based Generation:
   - Operates in latent space for efficiency
   - Score-based denoising for high-quality samples
   - Controllable via guidance mechanisms
   - Natural handling of uncertainty

2. Reinforcement Learning Policies:
   - SAC (Soft Actor-Critic): Maximum entropy RL for exploration
   - DDPG (Deep Deterministic Policy Gradient): Deterministic actions
   - Continuous action space matching semantic manifold
   - Policy optimization for coherent sequences

Key concepts:
- Latent diffusion: Denoise Gaussian noise to meaningful representations
- Score matching: Learn gradient of data distribution
- Policy gradient: Optimize expected return in semantic space
- Entropy regularization: Encourage exploration (SAC)
- Target networks: Stabilize TD learning (DDPG)

The continuous generation module provides:
1. Direct generation in semantic manifold (no quantization)
2. Smooth interpolation between representations
3. Principled uncertainty quantification
4. Integration with wave dynamics from other modules

These components complement the wave-based architecture by:
- Enabling continuous output generation (vs discrete tokens)
- Providing principled exploration of semantic space
- Supporting multi-step planning via RL
- Integrating with FNO/Soliton continuous representations

Usage:
    from neuromanifold_gpt.model.continuous import LatentDiffusion

    # Diffusion-based generation:
    >>> ld = LatentDiffusion(embed_dim=384, n_steps=100)
    >>> z_noise = torch.randn(2, 32, 384)
    >>> z_clean = ld.sample(z_noise.shape, z_noise.device)

    # SAC policy for continuous actions:
    from neuromanifold_gpt.model.continuous import SACPolicy
    >>> sac = SACPolicy(state_dim=384, action_dim=384)
    >>> state = torch.randn(2, 32, 384)
    >>> action, log_prob = sac.get_action(state)

    # DDPG policy:
    from neuromanifold_gpt.model.continuous import DDPGPolicy
    >>> ddpg = DDPGPolicy(state_dim=384, action_dim=384)
    >>> action = ddpg.get_action(state)

    # Unified output head:
    from neuromanifold_gpt.model.continuous import ContinuousOutputHead
    >>> head = ContinuousOutputHead(embed_dim=384, vocab_size=50304)
    >>> h = torch.randn(2, 32, 384)
    >>> logits, continuous_out = head(h)
"""

# Diffusion decoder
# from neuromanifold_gpt.model.continuous.diffusion import (
#     LatentDiffusion,
#     DiffusionConfig,
#     DDPMScheduler,
#     ScoreNetwork,
# )

# SAC policy
# from neuromanifold_gpt.model.continuous.sac_policy import (
#     SACPolicy,
#     SACConfig,
#     SACCritic,
#     SquashedGaussianPolicy,
# )

# DDPG policy
# from neuromanifold_gpt.model.continuous.ddpg_policy import (
#     DDPGPolicy,
#     DDPGConfig,
#     DDPGCritic,
#     DDPGActor,
# )

# Continuous output head
# from neuromanifold_gpt.model.continuous.output_head import (
#     ContinuousOutputHead,
#     ContinuousOutputConfig,
# )

__all__ = [
    # Diffusion decoder
    # "LatentDiffusion",
    # "DiffusionConfig",
    # "DDPMScheduler",
    # "ScoreNetwork",
    # SAC policy
    # "SACPolicy",
    # "SACConfig",
    # "SACCritic",
    # "SquashedGaussianPolicy",
    # DDPG policy
    # "DDPGPolicy",
    # "DDPGConfig",
    # "DDPGCritic",
    # "DDPGActor",
    # Continuous output head
    # "ContinuousOutputHead",
    # "ContinuousOutputConfig",
]
