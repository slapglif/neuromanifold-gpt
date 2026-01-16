# neuromanifold_gpt/model/continuous/__init__.py
"""
Continuous Generation Module.

This module provides continuous generation mechanisms for the wave-based architecture:

1. Latent Diffusion Decoder:
   - LatentDiffusion: DDPM-based latent space generation
   - DDPMScheduler: Noise scheduling for diffusion process
   - DiffusionConfig: Configuration for diffusion models

2. RL Policies:
   - SACPolicy: Soft Actor-Critic for continuous control
   - SACConfig: Configuration for SAC policy
   - DDPGPolicy: Deep Deterministic Policy Gradient (TODO)

3. Output Head (TODO):
   - ContinuousOutputHead: Unified output combining discrete and continuous

Key Concepts:
- Diffusion models learn to reverse a noise process
- Forward process adds Gaussian noise over T steps
- Reverse process learns to denoise, generating samples
- Latent diffusion operates in compressed latent space for efficiency
- SAC enables maximum entropy RL for continuous action spaces

Example:
    >>> from neuromanifold_gpt.model.continuous import LatentDiffusion, SACPolicy
    >>> import torch
    >>> # Create diffusion model
    >>> diffusion = LatentDiffusion(dim=384, n_steps=100)
    >>> # Sample from the model
    >>> z_samples = diffusion.sample((2, 32, 384), device='cuda')
    >>> # Create SAC policy
    >>> sac = SACPolicy(state_dim=384, action_dim=384)
    >>> state = torch.randn(2, 32, 384)
    >>> action, log_prob = sac.get_action(state)
"""

from neuromanifold_gpt.model.continuous.diffusion import (
    LatentDiffusion,
    DDPMScheduler,
    DiffusionConfig,
    DenoisingNetwork,
)

from neuromanifold_gpt.model.continuous.sac_policy import (
    SACPolicy,
    SACConfig,
    GaussianPolicy,
    TwinQNetwork,
)

__all__ = [
    # Diffusion
    "LatentDiffusion",
    "DDPMScheduler",
    "DiffusionConfig",
    "DenoisingNetwork",
    # SAC Policy
    "SACPolicy",
    "SACConfig",
    "GaussianPolicy",
    "TwinQNetwork",
]
