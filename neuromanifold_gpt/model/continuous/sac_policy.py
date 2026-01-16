# neuromanifold_gpt/model/continuous/sac_policy.py
"""
Soft Actor-Critic (SAC) Policy for continuous action space.

Implements SAC for continuous control in the semantic manifold, enabling
learned policies for continuous generation alongside diffusion models.

SAC key features:
1. Maximum entropy reinforcement learning
2. Stochastic policy outputs (Gaussian with learned mean and std)
3. Twin Q-networks to reduce overestimation bias
4. Automatic entropy coefficient tuning

The policy outputs actions as transformations on the continuous semantic
manifold, which can be used for guided generation or refinement.

References:
- Haarnoja et al. "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL
  with a Stochastic Actor" (2018)
- Haarnoja et al. "Soft Actor-Critic Algorithms and Applications" (2019)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


@dataclass
class SACConfig:
    """Configuration for SAC policy.

    Args:
        state_dim: Dimension of state/input space
        action_dim: Dimension of action/output space
        hidden_dim: Hidden layer dimension
        n_layers: Number of hidden layers
        log_std_min: Minimum log standard deviation
        log_std_max: Maximum log standard deviation
        init_temperature: Initial entropy temperature
        learnable_temperature: Whether temperature is learnable
        dropout: Dropout rate
        use_layer_norm: Whether to use layer normalization
    """
    state_dim: int = 384
    action_dim: int = 384
    hidden_dim: int = 512
    n_layers: int = 3
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    init_temperature: float = 0.2
    learnable_temperature: bool = True
    dropout: float = 0.0
    use_layer_norm: bool = True


class MLP(nn.Module):
    """Multi-layer perceptron with optional layer normalization."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
        output_activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = []
        in_dim = input_dim

        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, output_dim))
        if output_activation is not None:
            layers.append(output_activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [*, input_dim]

        Returns:
            Output tensor [*, output_dim]
        """
        return self.net(x)


class GaussianPolicy(nn.Module):
    """Gaussian policy network for SAC.

    Outputs mean and log_std of a Gaussian distribution over actions.
    Uses tanh squashing to bound actions to [-1, 1].
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Shared feature extractor
        self.features = MLP(
            input_dim=state_dim,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers - 1 if n_layers > 1 else 1,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
            output_activation=nn.SiLU(),
        )

        # Separate heads for mean and log_std
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        # Initialize output layers to small values
        nn.init.xavier_uniform_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.xavier_uniform_(self.log_std_head.weight, gain=0.01)
        nn.init.zeros_(self.log_std_head.bias)

    def forward(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to get mean and log_std.

        Args:
            state: State tensor [B, L, state_dim] or [B, state_dim]

        Returns:
            Tuple of (mean, log_std), each [B, L, action_dim] or [B, action_dim]
        """
        features = self.features(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)

        # Clamp log_std to prevent numerical issues
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action and compute log probability.

        Uses reparameterization trick for differentiable sampling.
        Applies tanh squashing and computes corrected log probability.

        Args:
            state: State tensor [B, L, state_dim] or [B, state_dim]
            deterministic: If True, return mean action (for evaluation)

        Returns:
            Tuple of (action, log_prob, mean):
                - action: Sampled action [B, L, action_dim] or [B, action_dim]
                - log_prob: Log probability [B, L, 1] or [B, 1]
                - mean: Mean action (before tanh) for reference
        """
        mean, log_std = self(state)
        std = log_std.exp()

        if deterministic:
            # Deterministic action (for evaluation)
            action_pre_tanh = mean
        else:
            # Reparameterization trick
            normal = Normal(mean, std)
            action_pre_tanh = normal.rsample()

        # Tanh squashing
        action = torch.tanh(action_pre_tanh)

        # Compute log probability with tanh squashing correction
        # log π(a|s) = log π(u|s) - sum(log(1 - tanh²(u)))
        # where u is the pre-tanh action
        if deterministic:
            log_prob = torch.zeros(
                *action.shape[:-1], 1,
                device=action.device, dtype=action.dtype
            )
        else:
            normal = Normal(mean, std)
            log_prob = normal.log_prob(action_pre_tanh)
            # Correction for tanh squashing
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            # Sum over action dimensions, keep batch dims
            log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, mean


class QNetwork(nn.Module):
    """Q-network for SAC (estimates Q(s, a)).

    Takes state and action as input, outputs Q-value.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.net = MLP(
            input_dim=state_dim + action_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            state: State tensor [B, L, state_dim] or [B, state_dim]
            action: Action tensor [B, L, action_dim] or [B, action_dim]

        Returns:
            Q-value [B, L, 1] or [B, 1]
        """
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class TwinQNetwork(nn.Module):
    """Twin Q-network for SAC (reduces overestimation bias).

    Contains two independent Q-networks and returns the minimum.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        self.q1 = QNetwork(
            state_dim, action_dim, hidden_dim, n_layers, dropout, use_layer_norm
        )
        self.q2 = QNetwork(
            state_dim, action_dim, hidden_dim, n_layers, dropout, use_layer_norm
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both Q-values.

        Args:
            state: State tensor [B, L, state_dim] or [B, state_dim]
            action: Action tensor [B, L, action_dim] or [B, action_dim]

        Returns:
            Tuple of (q1_value, q2_value), each [B, L, 1] or [B, 1]
        """
        return self.q1(state, action), self.q2(state, action)

    def min_q(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Get minimum Q-value (used for policy updates).

        Args:
            state: State tensor
            action: Action tensor

        Returns:
            Minimum Q-value
        """
        q1, q2 = self(state, action)
        return torch.min(q1, q2)


class SACPolicy(nn.Module):
    """Soft Actor-Critic Policy for continuous action space.

    Implements the SAC algorithm for maximum entropy reinforcement learning
    in continuous action spaces. Suitable for learning policies on the
    semantic manifold for guided generation.

    Key components:
    - Stochastic Gaussian policy with tanh squashing
    - Twin Q-networks to reduce overestimation
    - Automatic entropy temperature tuning

    Example:
        >>> sac = SACPolicy(state_dim=384, action_dim=384)
        >>> state = torch.randn(2, 32, 384)
        >>> action, log_prob = sac.get_action(state)
        >>> assert action.shape == state.shape
        >>> # For training
        >>> q_loss, policy_loss, alpha_loss, info = sac.compute_losses(
        ...     state, action, reward, next_state, done
        ... )
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        n_layers: int = 3,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        init_temperature: float = 0.2,
        learnable_temperature: bool = True,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
        gamma: float = 0.99,
        tau: float = 0.005,
        target_entropy: Optional[float] = None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        # Policy network
        self.policy = GaussianPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )

        # Twin Q-networks
        self.q_networks = TwinQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )

        # Target Q-networks (for stable training)
        self.target_q_networks = TwinQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )
        # Initialize targets to match main networks
        self.target_q_networks.load_state_dict(self.q_networks.state_dict())
        # Freeze target networks (updated via soft updates)
        for param in self.target_q_networks.parameters():
            param.requires_grad = False

        # Entropy temperature (alpha)
        if learnable_temperature:
            # Log alpha for numerical stability
            self.log_alpha = nn.Parameter(
                torch.tensor(math.log(init_temperature), dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "log_alpha",
                torch.tensor(math.log(init_temperature), dtype=torch.float32)
            )

        # Target entropy (typically -action_dim for continuous spaces)
        if target_entropy is None:
            target_entropy = -float(action_dim)
        self.target_entropy = target_entropy

    @property
    def alpha(self) -> torch.Tensor:
        """Entropy temperature coefficient."""
        return self.log_alpha.exp()

    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action from policy.

        This is the main interface for action selection.

        Args:
            state: State tensor [B, L, state_dim] or [B, state_dim]
            deterministic: If True, return mean action (for evaluation)

        Returns:
            Tuple of (action, log_prob):
                - action: Action tensor [B, L, action_dim] or [B, action_dim]
                - log_prob: Log probability [B, L, 1] or [B, 1]
        """
        action, log_prob, _ = self.policy.sample(state, deterministic)
        return action, log_prob

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass - alias for get_action.

        Args:
            state: State tensor
            deterministic: If True, return mean action

        Returns:
            Tuple of (action, log_prob)
        """
        return self.get_action(state, deterministic)

    def get_q_values(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get Q-values from both critics.

        Args:
            state: State tensor
            action: Action tensor

        Returns:
            Tuple of (q1, q2) values
        """
        return self.q_networks(state, action)

    def compute_losses(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Compute SAC losses for training.

        Args:
            state: Current state [B, state_dim]
            action: Taken action [B, action_dim]
            reward: Received reward [B, 1]
            next_state: Next state [B, state_dim]
            done: Done flag [B, 1]

        Returns:
            Tuple of (q_loss, policy_loss, alpha_loss, info_dict)
        """
        # Compute target Q-value
        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy.sample(next_state)
            target_q = self.target_q_networks.min_q(next_state, next_action)
            target_value = target_q - self.alpha * next_log_prob
            q_target = reward + (1 - done) * self.gamma * target_value

        # Q-network losses
        q1, q2 = self.q_networks(state, action)
        q1_loss = F.mse_loss(q1, q_target)
        q2_loss = F.mse_loss(q2, q_target)
        q_loss = q1_loss + q2_loss

        # Policy loss (maximize Q - alpha * log_prob)
        new_action, log_prob, _ = self.policy.sample(state)
        q_new = self.q_networks.min_q(state, new_action)
        policy_loss = (self.alpha.detach() * log_prob - q_new).mean()

        # Alpha (temperature) loss
        alpha_loss = -(
            self.log_alpha * (log_prob + self.target_entropy).detach()
        ).mean()

        info = {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
            "q1_mean": q1.mean().item(),
            "q2_mean": q2.mean().item(),
            "log_prob_mean": log_prob.mean().item(),
        }

        return q_loss, policy_loss, alpha_loss, info

    @torch.no_grad()
    def soft_update_targets(self):
        """Soft update target networks.

        τ * θ_target + (1 - τ) * θ
        """
        for target_param, param in zip(
            self.target_q_networks.parameters(),
            self.q_networks.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def extra_repr(self) -> str:
        return (
            f"state_dim={self.state_dim}, action_dim={self.action_dim}, "
            f"gamma={self.gamma}, tau={self.tau}"
        )
