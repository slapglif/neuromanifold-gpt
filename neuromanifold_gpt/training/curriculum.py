import torch
import torch.nn as nn
from typing import Dict, Callable, Optional
from dataclasses import dataclass
from enum import Enum


class GenerationMode(Enum):
    DISCRETE = "discrete"
    HYBRID = "hybrid"
    CONTINUOUS = "continuous"


@dataclass
class CurriculumConfig:
    discrete_steps: int = 5000
    hybrid_steps: int = 5000
    continuous_steps: int = 10000

    discrete_temp: float = 1.0
    hybrid_alpha_start: float = 0.0
    hybrid_alpha_end: float = 1.0
    continuous_temp: float = 0.5

    warmup_steps: int = 500


class CurriculumScheduler:
    """
    Manages curriculum learning from discrete tokens to continuous representations.

    Three-stage progression:
    1. Discrete: Standard token-based generation with softmax
    2. Hybrid: Weighted combination of discrete and continuous
    3. Continuous: Pure continuous embedding generation via RL/diffusion
    """

    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.step = 0

    def get_stage(self) -> GenerationMode:
        if self.step < self.config.discrete_steps:
            return GenerationMode.DISCRETE
        elif self.step < self.config.discrete_steps + self.config.hybrid_steps:
            return GenerationMode.HYBRID
        else:
            return GenerationMode.CONTINUOUS

    def get_alpha(self) -> float:
        """
        Get interpolation weight for hybrid mode.
        alpha=0: fully discrete, alpha=1: fully continuous
        """
        if self.step < self.config.discrete_steps:
            return 0.0
        elif self.step >= self.config.discrete_steps + self.config.hybrid_steps:
            return 1.0
        else:
            progress = (
                self.step - self.config.discrete_steps
            ) / self.config.hybrid_steps
            alpha = self.config.hybrid_alpha_start + progress * (
                self.config.hybrid_alpha_end - self.config.hybrid_alpha_start
            )
            return alpha

    def get_temperature(self) -> float:
        stage = self.get_stage()
        if stage == GenerationMode.DISCRETE:
            return self.config.discrete_temp
        elif stage == GenerationMode.HYBRID:
            alpha = self.get_alpha()
            return (
                1 - alpha
            ) * self.config.discrete_temp + alpha * self.config.continuous_temp
        else:
            return self.config.continuous_temp

    def advance(self):
        self.step += 1

    def state_dict(self) -> Dict:
        return {"step": self.step}

    def load_state_dict(self, state: Dict):
        self.step = state["step"]


class HybridOutput(nn.Module):
    """
    Hybrid output layer that interpolates between discrete and continuous generation.
    """

    def __init__(
        self,
        embed_dim: int,
        vocab_size: int,
        continuous_head: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        self.discrete_head = nn.Linear(embed_dim, vocab_size)
        self.continuous_head = continuous_head

        if continuous_head is None:
            self.continuous_head = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mode: GenerationMode,
        alpha: float = 0.5,
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            hidden_states: [batch, seq_len, embed_dim]
            mode: Generation mode
            alpha: Interpolation weight (0=discrete, 1=continuous)
            temperature: Sampling temperature

        Returns:
            Dictionary with 'logits' and/or 'embeddings'
        """
        batch, seq_len, _ = hidden_states.shape

        if mode == GenerationMode.DISCRETE:
            logits = self.discrete_head(hidden_states) / temperature
            return {"logits": logits, "mode": "discrete"}

        elif mode == GenerationMode.CONTINUOUS:
            embeddings = self.continuous_head(hidden_states)
            return {"embeddings": embeddings, "mode": "continuous"}

        else:
            logits = self.discrete_head(hidden_states) / temperature
            embeddings = self.continuous_head(hidden_states)

            discrete_probs = torch.softmax(logits, dim=-1)
            discrete_samples = torch.multinomial(
                discrete_probs.view(-1, self.vocab_size), num_samples=1
            ).view(batch, seq_len)

            blended = (1 - alpha) * embeddings + alpha * embeddings

            return {
                "logits": logits,
                "embeddings": blended,
                "discrete_samples": discrete_samples,
                "alpha": alpha,
                "mode": "hybrid",
            }


class CurriculumLoss(nn.Module):
    """
    Loss function that adapts based on curriculum stage.
    """

    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        mode: GenerationMode,
        alpha: float = 0.5,
    ) -> torch.Tensor:
        """
        Args:
            outputs: Model outputs from HybridOutput
            targets: [batch, seq_len] target token IDs
            mode: Current generation mode
            alpha: Interpolation weight

        Returns:
            loss: Scalar loss tensor
        """
        if mode == GenerationMode.DISCRETE:
            logits = outputs["logits"]
            batch, seq_len, vocab_size = logits.shape
            loss = self.cross_entropy(
                logits.reshape(-1, vocab_size), targets.reshape(-1)
            )
            return loss

        elif mode == GenerationMode.CONTINUOUS:
            embeddings = outputs["embeddings"]
            return torch.tensor(0.0, device=embeddings.device)

        else:
            logits = outputs["logits"]
            batch, seq_len, vocab_size = logits.shape

            discrete_loss = self.cross_entropy(
                logits.reshape(-1, vocab_size), targets.reshape(-1)
            )

            continuous_loss = torch.tensor(0.0, device=logits.device)

            loss = (1 - alpha) * discrete_loss + alpha * continuous_loss
            return loss
