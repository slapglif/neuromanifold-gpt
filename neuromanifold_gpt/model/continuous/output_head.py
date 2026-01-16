"""
Continuous Output Head.

Unified module combining the Flow Scheduler and Velocity Field.
Handles training (loss computation) and inference (generation).
"""

import torch
import torch.nn as nn
from dataclasses import dataclass

from .flow_scheduler import RectifiedFlowScheduler, FlowConfig
from .velocity_field import KANVelocityField

@dataclass
class ContinuousOutputConfig:
    embed_dim: int = 384
    hidden_dim: int = 512
    num_kan_layers: int = 3
    num_kan_centers: int = 8
    num_inference_steps: int = 10

class ContinuousOutputHead(nn.Module):
    """
    Head for generating continuous outputs via Rectified Flow.
    
    Training:
        1. Sample t ~ U[0, 1]
        2. Interpolate x_t = t * x_1 + (1-t) * x_0
        3. Predict v = model(x_t, t, c)
        4. Loss = MSE(v, x_1 - x_0)
        
    Inference:
        1. Sample x_0 ~ N(0, 1)
        2. Solve ODE dx/dt = v over t=[0, 1]
    """
    def __init__(self, config: ContinuousOutputConfig):
        super().__init__()
        self.config = config
        
        self.scheduler = RectifiedFlowScheduler(
            FlowConfig(num_inference_steps=config.num_inference_steps)
        )
        
        self.velocity_model = KANVelocityField(
            input_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_kan_layers,
            num_centers=config.num_kan_centers
        )

    def compute_loss(
        self, 
        target_embeddings: torch.Tensor, 
        condition: torch.Tensor
    ) -> dict:
        """
        Compute Flow Matching loss.
        
        Args:
            target_embeddings (x_1): Ground truth next-token embeddings (..., D)
            condition (c): Context from backbone (..., D)
            
        Returns:
            dict: {'loss': scalar_loss}
        """
        # Handle arbitrary leading dimensions (B, T, D) -> (N, D)
        # We process all tokens in the batch/sequence as independent samples for flow matching
        orig_shape = target_embeddings.shape
        D = orig_shape[-1]
        
        # Flatten to (N, D)
        target_embeddings = target_embeddings.view(-1, D)
        condition = condition.view(-1, D)
        
        B = target_embeddings.shape[0]
        device = target_embeddings.device
        dtype = target_embeddings.dtype
        
        # 1. Sample Noise (x_0)
        noise = torch.randn_like(target_embeddings)
        
        # 2. Sample Timesteps t ~ U[0, 1]
        # (B,)
        timesteps = torch.rand(B, device=device, dtype=dtype)
        
        # 3. Add Noise (Forward Process)
        # x_t = t * x_1 + (1-t) * x_0
        # target_v = x_1 - x_0
        noisy_samples, target_velocity = self.scheduler.add_noise(
            target_embeddings, noise, timesteps
        )
        
        # 4. Predict Velocity
        pred_velocity = self.velocity_model(noisy_samples, timesteps, condition)
        
        # 5. MSE Loss
        loss = torch.mean((pred_velocity - target_velocity) ** 2)
        
        return {
            'continuous_loss': loss,
            'velocity_norm': pred_velocity.norm(dim=-1).mean()
        }

    @torch.no_grad()
    def generate(
        self, 
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate samples from noise (Inference).
        
        Args:
            condition: Context (..., D)
            
        Returns:
            generated_samples: (..., D)
        """
        orig_shape = condition.shape
        D = orig_shape[-1]
        
        # Flatten to (N, D)
        condition = condition.view(-1, D)
        B = condition.shape[0]
        device = condition.device
        
        # 1. Start from Gaussian Noise (x_0)
        x_t = torch.randn(B, D, device=device, dtype=condition.dtype)
        
        # 2. ODE Solver Loop
        num_steps = self.config.num_inference_steps
        dt = 1.0 / num_steps
        
        # Generate time grid [0, ..., 1]
        # Note: Rectified Flow usually goes 0 -> 1 (noise -> data)
        # or 1 -> 0 depending on formulation. 
        # Here we defined x_t = t*x_1 + (1-t)x_0.
        # So t=0 is noise, t=1 is data. We integrate 0 to 1.
        
        for step in range(num_steps):
            t_val = step / num_steps
            # Tensorize t for batch
            t_batch = torch.full((B,), t_val, device=device, dtype=condition.dtype)
            
            # Predict velocity
            velocity = self.velocity_model(x_t, t_batch, condition)
            
            # Euler Step
            x_t = self.scheduler.step(velocity, x_t, dt)
            
        # Reshape back to original
        return x_t.view(*orig_shape)
