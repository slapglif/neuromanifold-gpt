"""
Soft Actor-Critic (SAC) Policy for Continuous Latent Control.

Adapts SAC for high-dimensional latent spaces using KAN backbones.
Ref: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
Innovation: FasterKAN Actor and Critic networks for manifold-aware value estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from dataclasses import dataclass
from neuromanifold_gpt.model.kan.faster.layer import FasterKANLayer

@dataclass
class SACConfig:
    state_dim: int = 384
    action_dim: int = 384  # Actions act directly on latent state
    hidden_dim: int = 256
    num_layers: int = 2
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2  # Entropy regularization coefficient

class KANNetwork(nn.Module):
    """Generic KAN-based MLP."""
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super().__init__()
        layers = []
        layers.append(FasterKANLayer(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            layers.append(FasterKANLayer(hidden_dim, hidden_dim))
        layers.append(nn.Linear(hidden_dim, output_dim)) # Final linear projection
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class SACCritic(nn.Module):
    """Twin Q-Networks using KAN."""
    def __init__(self, config: SACConfig):
        super().__init__()
        # Q(s, a)
        input_dim = config.state_dim + config.action_dim
        
        self.q1 = KANNetwork(input_dim, 1, config.hidden_dim, config.num_layers)
        self.q2 = KANNetwork(input_dim, 1, config.hidden_dim, config.num_layers)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        return self.q1(xu), self.q2(xu)

class SACActor(nn.Module):
    """Gaussian Policy using KAN."""
    def __init__(self, config: SACConfig):
        super().__init__()
        self.net = KANNetwork(
            config.state_dim, 
            config.hidden_dim,  # Intermediate output 
            config.hidden_dim, 
            config.num_layers
        )
        
        # Mean and LogStd heads
        self.mean = nn.Linear(config.hidden_dim, config.action_dim)
        self.log_std = nn.Linear(config.hidden_dim, config.action_dim)
        
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        
        # Enforce action bounds [-1, 1] usually, but latent space is unbounded.
        # However, for stability, we often use tanh.
        y_t = torch.tanh(x_t)
        action = y_t
        
        # Enforcing bounds affects log_prob density
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob, mean

class SACPolicy(nn.Module):
    """Container for Actor-Critic."""
    def __init__(self, config: SACConfig):
        super().__init__()
        self.config = config
        self.actor = SACActor(config)
        self.critic = SACCritic(config)
        self.critic_target = SACCritic(config)
        
        # Copy target parameters
        self.critic_target.load_state_dict(self.critic.state_dict())
        for param in self.critic_target.parameters():
            param.requires_grad = False

    def get_action(self, state, deterministic=False):
        with torch.no_grad():
            mean, log_std = self.actor(state)
            if deterministic:
                return torch.tanh(mean)
            else:
                action, _, _ = self.actor.sample(state)
                return action
