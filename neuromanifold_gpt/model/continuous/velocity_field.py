"""
KAN Velocity Field (Score Network).

Predicts the flow velocity v(x, t, c) using Kolmogorov-Arnold Networks.
Uses FasterKAN for efficient spline-based activation functions.
"""

import torch
import torch.nn as nn
import math
from neuromanifold_gpt.model.kan.faster.layer import FasterKANLayer

class SinusoidalTimeEmbeddings(nn.Module):
    """Standard sinusoidal time embeddings."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class KANVelocityField(nn.Module):
    """
    Predicts velocity field v = dx/dt using a KAN backbone.
    
    Architecture:
    [Latent + Time + Cond] -> KAN -> KAN -> KAN -> Velocity
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        num_centers: int = 8,  # KAN spline centers
        time_dim: int = 256
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )

        # Input dimension = Latent(D) + Time(T) + Cond(D)
        # We assume Condition has same dim as Input for now
        concat_dim = input_dim + time_dim + input_dim 
        
        layers = []
        # Input layer
        layers.append(
            FasterKANLayer(concat_dim, hidden_dim, num_centers=num_centers)
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(
                FasterKANLayer(hidden_dim, hidden_dim, num_centers=num_centers)
            )
            
        # Output layer (project back to input_dim)
        # Note: Last layer typically linear for regression, but KAN is fine too
        # provided the range isn't bounded restrictedly. FasterKAN is linear basis.
        layers.append(
            nn.Linear(hidden_dim, input_dim) 
        )
        
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Noisy latent (B, D)
            t: Timestep (B,)
            cond: Conditioning vector (B, D) e.g. from SSM backbone
        """
        t_emb = self.time_mlp(t) # (B, time_dim)
        
        # Concatenate inputs
        # (B, D+T+D)
        h = torch.cat([x, t_emb, cond], dim=-1)
        
        # Predict velocity
        velocity = self.net(h)
        
        return velocity
