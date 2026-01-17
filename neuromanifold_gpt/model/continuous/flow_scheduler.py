"""
Rectified Flow Scheduler (Flow Matching).

Implements the straight-line ODE trajectory for continuous generation.
Ref: "Flow Matching for Generative Modeling" (Lipman et al., 2023)

Equation: dX_t/dt = v(X_t, t)
Forward (Training): X_t = t * X_1 + (1 - t) * X_0
Target Velocity: v = X_1 - X_0

This provides faster generation (1-step to few-step) compared to DDPM.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class FlowConfig:
    num_train_steps: int = 1000
    num_inference_steps: int = 10
    sigma_min: float = 1e-4  # Numerical stability for t=0/1


class RectifiedFlowScheduler(nn.Module):
    """
    Handles the noise scheduling and ODE solving for Rectified Flow.
    """

    def __init__(self, config: FlowConfig = None):
        super().__init__()
        self.config = config or FlowConfig()

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward process: Interpolate between noise (X_0) and data (X_1).

        Args:
            original_samples (X_1): Target data (batch, dim)
            noise (X_0): Gaussian noise (batch, dim)
            timesteps (t): Time values in [0, 1] (batch,)

        Returns:
            noisy_samples (X_t): Interpolated state
            target_velocity (v): Target for the network to predict (X_1 - X_0)
        """
        # Expand t for broadcasting: (B,) -> (B, 1, 1...) matches x shape
        t = timesteps.view(-1, *([1] * (original_samples.ndim - 1)))

        # Rectified Flow: Linear interpolation
        # X_t = t * X_1 + (1 - t) * X_0
        noisy_samples = t * original_samples + (1.0 - t) * noise

        # Target velocity is the slope of the line
        # v = d/dt (tX_1 + (1-t)X_0) = X_1 - X_0
        target_velocity = original_samples - noise

        return noisy_samples, target_velocity

    def step(
        self, model_output: torch.Tensor, sample: torch.Tensor, dt: float
    ) -> torch.Tensor:
        """
        Single Euler step for ODE solver: z_{t+dt} = z_t + v * dt

        Args:
            model_output (v): Predicted velocity
            sample (z_t): Current latent state
            dt: Time step size

        Returns:
            prev_sample (z_{t+dt}): Updated state
        """
        # Euler integration
        prev_sample = sample + model_output * dt
        return prev_sample
