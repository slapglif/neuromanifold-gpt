"""
Soft Actor-Critic (SAC) Continuous Control Output Layer.

Implements continuous generation via RL-based policy optimization:
- Actor network: Generates continuous action (next embedding vector)
- Critic network: Evaluates state-action value
- Temperature parameter: Balances exploration vs exploitation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple, Dict, Optional


class GaussianActor(nn.Module):
    """
    Gaussian policy network for continuous action generation.

    Outputs mean and log_std for a diagonal Gaussian distribution
    over the continuous embedding space.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            state: [batch, seq_len, state_dim]
            deterministic: If True, return mean without sampling

        Returns:
            action: [batch, seq_len, action_dim]
            log_prob: [batch, seq_len, 1]
            mean: [batch, seq_len, action_dim]
        """
        features = self.net(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        if deterministic:
            action = torch.tanh(mean)
            log_prob = torch.zeros_like(mean[..., :1])
        else:
            dist = Normal(mean, std)
            action_pre_tanh = dist.rsample()
            log_prob = dist.log_prob(action_pre_tanh).sum(dim=-1, keepdim=True)
            log_prob = log_prob - (
                2
                * (
                    torch.log(torch.tensor(2.0))
                    - action_pre_tanh
                    - F.softplus(-2 * action_pre_tanh)
                )
            ).sum(dim=-1, keepdim=True)
            action = torch.tanh(action_pre_tanh)

        return action, log_prob, mean


class TwinQNetwork(nn.Module):
    """
    Twin Q-networks for critic evaluation (reduces overestimation bias).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: [batch, seq_len, state_dim]
            action: [batch, seq_len, action_dim]

        Returns:
            q1: [batch, seq_len, 1]
            q2: [batch, seq_len, 1]
        """
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)


class SACOutputHead(nn.Module):
    """
    SAC-based continuous control output for NS-WMN.

    Generates next token embedding via RL policy instead of discrete softmax.
    Enables true continuous generation on the semantic manifold.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha_init: float = 0.2,
        target_entropy: Optional[float] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.gamma = gamma
        self.tau = tau

        self.actor = GaussianActor(
            state_dim=embed_dim,
            action_dim=embed_dim,
            hidden_dim=hidden_dim,
        )

        self.critic = TwinQNetwork(
            state_dim=embed_dim,
            action_dim=embed_dim,
            hidden_dim=hidden_dim,
        )

        self.critic_target = TwinQNetwork(
            state_dim=embed_dim,
            action_dim=embed_dim,
            hidden_dim=hidden_dim,
        )
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.log_alpha = nn.Parameter(torch.tensor([alpha_init]).log())
        self.target_entropy = target_entropy or -embed_dim

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        """
        Args:
            state: [batch, seq_len, embed_dim]
            deterministic: If True, use mean action

        Returns:
            action: [batch, seq_len, embed_dim]
        """
        with torch.no_grad():
            action, _, _ = self.actor(state, deterministic=deterministic)
        return action

    def compute_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            state: [batch, seq_len, embed_dim]
            action: [batch, seq_len, embed_dim] (target embedding)
            reward: [batch, seq_len, 1]
            next_state: [batch, seq_len, embed_dim]
            done: [batch, seq_len, 1]

        Returns:
            dict with 'actor_loss', 'critic_loss', 'alpha_loss'
        """
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor(next_state)
            q1_next, q2_next = self.critic_target(next_state, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            q_target = reward + (1 - done) * self.gamma * q_next

        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        new_action, log_prob, _ = self.actor(state)
        q1_new, q2_new = self.critic(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_prob - q_new).mean()

        alpha_loss = -(
            self.log_alpha * (log_prob + self.target_entropy).detach()
        ).mean()

        return {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "alpha_loss": alpha_loss,
            "q_value": q_new.mean(),
            "log_prob": log_prob.mean(),
            "alpha": self.alpha,
        }

    def update_targets(self):
        """Soft update of target networks."""
        with torch.no_grad():
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.mul_(1 - self.tau).add_(param.data, alpha=self.tau)

    def forward(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Generate next embedding via policy.

        Args:
            state: [batch, seq_len, embed_dim]
            deterministic: Use mean action if True

        Returns:
            action: [batch, seq_len, embed_dim]
            info: Dict with policy statistics
        """
        action, log_prob, mean = self.actor(state, deterministic=deterministic)

        info = {
            "action_mean": mean.mean().item(),
            "action_std": (action - mean).std().item(),
            "log_prob": log_prob.mean().item(),
        }

        return action, info
