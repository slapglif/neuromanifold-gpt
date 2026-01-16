# neuromanifold_gpt/model/imagination.py
"""
Consistency Imagination Module.

Implements lightweight diffusion-based counterfactual exploration for System 2 reasoning.
Uses manifold-guided denoising to generate diverse alternative trajectories.

Key features:
- Lightweight diffusion (2-4 denoising steps) for efficiency
- Manifold-guided exploration preserves semantic coherence
- Goal-directed optimization when target is provided
- Returns multiple alternatives with quality scores
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum

from .manifold import ManifoldProjection


class ConsistencyImaginationModule(nn.Module):
    """
    Generate diverse alternative trajectories via lightweight diffusion.

    This module implements counterfactual imagination for System 2 reasoning:
    1. Project input to manifold space for semantic guidance
    2. Add controlled noise to create initial alternatives
    3. Denoise via lightweight diffusion (2-4 steps)
    4. Score alternatives for quality/coherence
    5. Return alternatives, scores, and best candidate

    Args:
        embed_dim: Dimension of embedding space (e.g., 384)
        manifold_dim: Dimension of manifold coordinates (e.g., 64)
        n_imagination_steps: Number of denoising steps (2-4, default: 4)
        dropout: Dropout probability (default: 0.0)
    """

    def __init__(
        self,
        embed_dim: int,
        manifold_dim: int,
        n_imagination_steps: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.manifold_dim = manifold_dim
        self.n_imagination_steps = n_imagination_steps

        # Manifold projection for semantic guidance
        # We'll use a dummy sdr_size = embed_dim for now since we're working with embeddings
        self.manifold_proj = ManifoldProjection(embed_dim, manifold_dim)

        # Denoising network: predicts noise to remove at each step
        # Input: (noisy_x, timestep, manifold_coords, goal?)
        # Output: predicted noise
        hidden_dim = embed_dim * 2
        self.denoiser = nn.Sequential(
            nn.Linear(embed_dim + manifold_dim + 1, hidden_dim),  # +1 for timestep
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Goal conditioning network (if goal is provided)
        self.goal_encoder = nn.Sequential(
            nn.Linear(embed_dim, manifold_dim),
            nn.SiLU(),
            nn.Linear(manifold_dim, manifold_dim),
        )

        # Quality scoring network
        # Scores alternatives based on coherence, diversity, goal alignment
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim + manifold_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Noise schedule: linearly decreasing noise from 1.0 to 0.0
        self.register_buffer(
            "noise_schedule",
            torch.linspace(1.0, 0.0, n_imagination_steps + 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Kaiming normal for better gradient flow."""
        for module in [self.denoiser, self.goal_encoder, self.scorer]:
            for m in module:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        goal: torch.Tensor | None = None,
        n_alternatives: int = 4,
    ) -> dict[str, torch.Tensor]:
        """
        Generate alternative trajectories via diffusion-based imagination.

        Args:
            x: (B, T, embed_dim) input embeddings
            goal: (B, T, embed_dim) optional goal embeddings to optimize toward
            n_alternatives: Number of alternative trajectories to generate

        Returns:
            Dictionary containing:
                - alternatives: (B, n_alt, T, embed_dim) generated alternatives
                - scores: (B, n_alt) quality scores for each alternative
                - best: (B, T, embed_dim) best alternative (highest score)
        """
        B, T, D = x.shape
        device = x.device

        # Project to manifold for semantic guidance
        coords, metric = self.manifold_proj(x)  # (B, T, manifold_dim), (B, T, D_m, D_m)

        # Encode goal if provided
        goal_embedding = None
        if goal is not None:
            goal_embedding = self.goal_encoder(goal.mean(dim=1))  # (B, manifold_dim)

        # Generate initial noisy alternatives
        # Start from x + noise
        noise_level = self.noise_schedule[0]  # Start at max noise
        noise = torch.randn(B, n_alternatives, T, D, device=device)

        # (B, 1, T, D) + noise_level * (B, n_alt, T, D) -> (B, n_alt, T, D)
        alternatives = x.unsqueeze(1) + noise_level * noise

        # Diffusion denoising loop
        for step in range(self.n_imagination_steps):
            t_curr = self.noise_schedule[step]
            t_next = self.noise_schedule[step + 1]

            # Denoise each alternative
            alternatives = self._denoise_step(
                alternatives, coords, metric, goal_embedding, t_curr, t_next
            )

        # Score alternatives
        scores = self._score_alternatives(alternatives, coords, goal_embedding)  # (B, n_alt)

        # Find best alternative
        best_idx = scores.argmax(dim=1)  # (B,)
        best = alternatives[torch.arange(B, device=device), best_idx]  # (B, T, D)

        return {
            "alternatives": alternatives,  # (B, n_alt, T, D)
            "scores": scores,              # (B, n_alt)
            "best": best,                  # (B, T, D)
        }

    def _denoise_step(
        self,
        alternatives: torch.Tensor,
        coords: torch.Tensor,
        metric: torch.Tensor,
        goal_embedding: torch.Tensor | None,
        t_curr: float,
        t_next: float,
    ) -> torch.Tensor:
        """
        Single denoising step: predict and remove noise.

        Args:
            alternatives: (B, n_alt, T, D) current noisy alternatives
            coords: (B, T, manifold_dim) manifold coordinates
            metric: (B, T, manifold_dim, manifold_dim) metric tensor
            goal_embedding: (B, manifold_dim) optional goal embedding
            t_curr: Current noise level
            t_next: Next noise level

        Returns:
            (B, n_alt, T, D) denoised alternatives
        """
        B, n_alt, T, D = alternatives.shape

        # Expand coords for broadcasting with alternatives
        coords_expanded = coords.unsqueeze(1).expand(B, n_alt, T, -1)  # (B, n_alt, T, D_m)

        # Add goal embedding if provided
        if goal_embedding is not None:
            # (B, 1, 1, D_m) -> broadcast to (B, n_alt, T, D_m)
            goal_expanded = goal_embedding.unsqueeze(1).unsqueeze(1).expand(B, n_alt, T, -1)
            coords_expanded = coords_expanded + 0.1 * goal_expanded  # Small goal influence

        # Timestep embedding (same for all alternatives)
        timestep = torch.full((B, n_alt, T, 1), t_curr, device=alternatives.device)

        # Concatenate inputs for denoiser
        denoiser_input = torch.cat([alternatives, coords_expanded, timestep], dim=-1)

        # Predict noise
        predicted_noise = self.denoiser(denoiser_input)  # (B, n_alt, T, D)

        # Remove predicted noise
        # Use DDPM-style update: x_{t-1} = x_t - (t_curr - t_next) * noise
        denoised = alternatives - (t_curr - t_next) * predicted_noise

        return denoised

    def _score_alternatives(
        self,
        alternatives: torch.Tensor,
        coords: torch.Tensor,
        goal_embedding: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Score alternatives for quality and goal alignment.

        Args:
            alternatives: (B, n_alt, T, D) generated alternatives
            coords: (B, T, manifold_dim) manifold coordinates
            goal_embedding: (B, manifold_dim) optional goal embedding

        Returns:
            (B, n_alt) scores for each alternative
        """
        B, n_alt, T, D = alternatives.shape

        # Pool alternatives along time dimension
        alt_pooled = alternatives.mean(dim=2)  # (B, n_alt, D)

        # Expand coords for scoring
        coords_pooled = coords.mean(dim=1)  # (B, manifold_dim)
        coords_expanded = coords_pooled.unsqueeze(1).expand(B, n_alt, -1)  # (B, n_alt, D_m)

        # Concatenate alternative and manifold coords
        scorer_input = torch.cat([alt_pooled, coords_expanded], dim=-1)  # (B, n_alt, D + D_m)

        # Compute base scores
        scores = self.scorer(scorer_input).squeeze(-1)  # (B, n_alt)

        # Add goal alignment bonus if goal provided
        if goal_embedding is not None:
            # Compute similarity to goal
            goal_expanded = goal_embedding.unsqueeze(1).expand(B, n_alt, -1)  # (B, n_alt, D_m)
            goal_similarity = F.cosine_similarity(coords_expanded, goal_expanded, dim=-1)  # (B, n_alt)
            scores = scores + 0.5 * goal_similarity  # Add goal bonus

        return scores
