# neuromanifold_gpt/model/continuous/__init__.py
"""
Continuous Generation Module.

This module provides continuous generation mechanisms for the wave-based architecture:

1. Latent Diffusion Decoder:
   - LatentDiffusion: DDPM-based latent space generation
   - DDPMScheduler: Noise scheduling for diffusion process
   - DiffusionConfig: Configuration for diffusion models

2. RL Policies (TODO):
   - SACPolicy: Soft Actor-Critic for continuous control
   - DDPGPolicy: Deep Deterministic Policy Gradient

3. Output Head (TODO):
   - ContinuousOutputHead: Unified output combining discrete and continuous

Key Concepts:
- Diffusion models learn to reverse a noise process
- Forward process adds Gaussian noise over T steps
- Reverse process learns to denoise, generating samples
- Latent diffusion operates in compressed latent space for efficiency

Example:
    >>> from neuromanifold_gpt.model.continuous import LatentDiffusion
    >>> import torch
    >>> # Create diffusion model
    >>> diffusion = LatentDiffusion(dim=384, n_steps=100)
    >>> # Sample from the model
    >>> z_samples = diffusion.sample((2, 32, 384), device='cuda')
    >>> # Train by computing loss
    >>> z = torch.randn(2, 32, 384)
    >>> loss = diffusion.compute_loss(z)
"""

from neuromanifold_gpt.model.continuous.diffusion import (
    LatentDiffusion,
    DDPMScheduler,
    DiffusionConfig,
    DenoisingNetwork,
)

__all__ = [
    # Diffusion
    "LatentDiffusion",
    "DDPMScheduler",
    "DiffusionConfig",
    "DenoisingNetwork",
]
