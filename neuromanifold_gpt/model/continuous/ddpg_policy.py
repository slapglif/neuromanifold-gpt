"""
Deep Deterministic Policy Gradient (DDPG) for Continuous Latent Control.

Deterministic counterpart to SAC. Useful when exploration is handled externally
or for fine-tuning specific trajectories.
Innovation: FasterKAN Actor and Critic.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from neuromanifold_gpt.model.kan.faster.layer import FasterKANLayer


@dataclass
class DDPGConfig:
    state_dim: int = 384
    action_dim: int = 384
    hidden_dim: int = 256
    num_layers: int = 2
    gamma: float = 0.99
    tau: float = 0.005


class KANNetwork(nn.Module):
    """Generic KAN-based MLP."""

    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super().__init__()
        layers = []
        layers.append(FasterKANLayer(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            layers.append(FasterKANLayer(hidden_dim, hidden_dim))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DDPGCritic(nn.Module):
    """Q(s, a) using KAN."""

    def __init__(self, config: DDPGConfig):
        super().__init__()
        input_dim = config.state_dim + config.action_dim
        self.net = KANNetwork(input_dim, 1, config.hidden_dim, config.num_layers)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        return self.net(xu)


class DDPGActor(nn.Module):
    """Deterministic Policy mu(s) using KAN."""

    def __init__(self, config: DDPGConfig):
        super().__init__()
        self.net = KANNetwork(
            config.state_dim, config.action_dim, config.hidden_dim, config.num_layers
        )

    def forward(self, state):
        x = self.net(state)
        return torch.tanh(x)


class DDPGPolicy(nn.Module):
    """Container for DDPG."""

    def __init__(self, config: DDPGConfig):
        super().__init__()
        self.config = config

        self.actor = DDPGActor(config)
        self.actor_target = DDPGActor(config)

        self.critic = DDPGCritic(config)
        self.critic_target = DDPGCritic(config)

        # Copy weights
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        for p in self.actor_target.parameters():
            p.requires_grad = False
        for p in self.critic_target.parameters():
            p.requires_grad = False

    def get_action(self, state):
        with torch.no_grad():
            return self.actor(state)
