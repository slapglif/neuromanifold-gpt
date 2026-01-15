"""
Consistency Imagination Module - Counterfactual exploration via consistency models.

Uses consistency models (fast diffusion) to explore alternative reasoning paths
in manifold space. This implements a "mental whiteboard" where the model can
imagine different ways a thought sequence could unfold.

Key insights:
- Consistency models: Fast sampling via learned consistency function
- Manifold exploration: Alternatives lie on semantic manifold
- Goal-directed: Can optionally optimize toward a goal representation
- Parallel generation: Multiple alternatives explored simultaneously

Inspired by human counterfactual reasoning and mental simulation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum


class ConsistencyImaginationModule(nn.Module):
    """
    Consistency model for exploring alternative reasoning paths.

    This module implements a lightweight diffusion-inspired process for
    counterfactual exploration. Given a current hidden state, it generates
    multiple alternative continuations by:

    1. Projecting to manifold space for semantic exploration
    2. Adding controlled noise to explore nearby regions
    3. Iteratively denoising via consistency function
    4. Scoring alternatives by coherence and goal alignment

    The "mental whiteboard" metaphor: the model sketches multiple rough
    ideas (noisy samples), then refines them (denoising), and selects
    the most promising path.

    Args:
        embed_dim: Dimension of input embeddings (typically 384)
        manifold_dim: Dimension of manifold space for exploration (typically 64)
        n_imagination_steps: Number of denoising steps (default 4)
        noise_scale: Initial noise magnitude for exploration (default 1.0)
    """

    def __init__(
        self,
        embed_dim: int,
        manifold_dim: int,
        n_imagination_steps: int = 4,
        noise_scale: float = 1.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.manifold_dim = manifold_dim
        self.n_imagination_steps = n_imagination_steps
        self.noise_scale = noise_scale

        # Project embeddings to manifold space for exploration
        self.to_manifold = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.SiLU(),
            nn.Linear(embed_dim // 2, manifold_dim),
        )

        # Consistency function: predicts clean sample from noisy input
        # Takes (noisy_sample, time_step, optional_goal) -> clean_sample
        self.consistency_net = nn.Sequential(
            nn.Linear(manifold_dim + 1, manifold_dim * 2),  # +1 for time embedding
            nn.LayerNorm(manifold_dim * 2),
            nn.SiLU(),
            nn.Linear(manifold_dim * 2, manifold_dim * 2),
            nn.LayerNorm(manifold_dim * 2),
            nn.SiLU(),
            nn.Linear(manifold_dim * 2, manifold_dim),
        )

        # Optional goal conditioning network
        self.goal_encoder = nn.Sequential(
            nn.Linear(embed_dim, manifold_dim),
            nn.SiLU(),
        )

        # Project manifold samples back to embedding space
        self.from_manifold = nn.Sequential(
            nn.Linear(manifold_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.SiLU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )

        # Score alternatives by coherence
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.SiLU(),
            nn.Linear(embed_dim // 4, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        # Initialize projections with Kaiming normal
        for module in [self.to_manifold, self.from_manifold, self.goal_encoder]:
            for m in module:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # Initialize consistency net to output near zero (near-identity at init)
        for m in self.consistency_net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Initialize scorer
        for m in self.scorer:
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
        Generate alternative reasoning paths via consistency-based exploration.

        Args:
            x: Input hidden states (B, T, embed_dim)
            goal: Optional goal embedding (B, embed_dim) to guide exploration
            n_alternatives: Number of alternative paths to generate

        Returns:
            dict containing:
                - alternatives: (B, n_alternatives, T, embed_dim) generated alternatives
                - scores: (B, n_alternatives) coherence scores for each alternative
                - best_idx: (B,) index of highest-scoring alternative per batch
                - best_alternative: (B, T, embed_dim) best alternative for each batch
        """
        B, T, D = x.shape
        device = x.device

        # Pool sequence to get a single representation per batch
        # Use mean pooling over time dimension
        x_pooled = x.mean(dim=1)  # (B, embed_dim)

        # Project to manifold space
        x_manifold = self.to_manifold(x_pooled)  # (B, manifold_dim)

        # Encode goal if provided
        goal_manifold = None
        if goal is not None:
            goal_manifold = self.goal_encoder(goal)  # (B, manifold_dim)

        # Generate multiple alternatives in parallel
        # Expand to (B, n_alternatives, manifold_dim)
        x_manifold_expanded = x_manifold.unsqueeze(1).expand(B, n_alternatives, self.manifold_dim)

        # Add noise to explore nearby regions in manifold space
        noise = torch.randn_like(x_manifold_expanded) * self.noise_scale
        noisy_samples = x_manifold_expanded + noise

        # Iterative denoising via consistency function
        samples = noisy_samples
        for step in range(self.n_imagination_steps):
            # Time embedding (decreasing from 1 to 0 as we denoise)
            t = 1.0 - (step / self.n_imagination_steps)
            t_emb = torch.full(
                (B, n_alternatives, 1),
                t,
                device=device,
                dtype=samples.dtype,
            )

            # Concatenate sample with time embedding
            samples_with_t = torch.cat([samples, t_emb], dim=-1)  # (B, n_alt, manifold_dim + 1)

            # Apply consistency function
            denoised = self.consistency_net(samples_with_t)  # (B, n_alt, manifold_dim)

            # If goal provided, guide toward it
            if goal_manifold is not None:
                # Blend denoised sample toward goal (stronger as we denoise)
                goal_weight = (step + 1) / self.n_imagination_steps  # 0 -> 1
                goal_expanded = goal_manifold.unsqueeze(1).expand_as(denoised)
                denoised = (1 - goal_weight) * denoised + goal_weight * goal_expanded

            samples = denoised

        # Project back to embedding space
        alternatives_pooled = self.from_manifold(samples)  # (B, n_alt, embed_dim)

        # Expand back to sequence length by broadcasting
        # Each alternative gets the same representation repeated across time
        alternatives = alternatives_pooled.unsqueeze(2).expand(B, n_alternatives, T, D)

        # Score alternatives by coherence
        # First flatten to (B * n_alternatives, embed_dim)
        alternatives_flat = alternatives_pooled.view(B * n_alternatives, D)
        scores_flat = self.scorer(alternatives_flat).squeeze(-1)  # (B * n_alternatives,)
        scores = scores_flat.view(B, n_alternatives)  # (B, n_alternatives)

        # Find best alternative per batch
        best_idx = scores.argmax(dim=1)  # (B,)

        # Gather best alternatives
        # best_idx: (B,) -> (B, 1, 1, 1) for gather
        best_idx_expanded = best_idx.view(B, 1, 1, 1).expand(B, 1, T, D)
        best_alternative = alternatives.gather(1, best_idx_expanded).squeeze(1)  # (B, T, D)

        return {
            "alternatives": alternatives,
            "scores": scores,
            "best_idx": best_idx,
            "best_alternative": best_alternative,
        }

    def consistency_loss(
        self,
        x: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute consistency training loss.

        The consistency loss trains the network to map noisy samples at any
        noise level to the same clean sample. This enables one-step sampling
        at inference time.

        Args:
            x: Input samples (B, T, embed_dim)
            target: Optional target clean samples (B, T, embed_dim)
                   If None, uses x as target (self-consistency)

        Returns:
            loss: Scalar consistency loss
        """
        B, T, D = x.shape
        device = x.device

        if target is None:
            target = x

        # Pool to single representation
        x_pooled = x.mean(dim=1)  # (B, embed_dim)
        target_pooled = target.mean(dim=1)  # (B, embed_dim)

        # Project to manifold
        x_manifold = self.to_manifold(x_pooled)
        target_manifold = self.to_manifold(target_pooled)

        # Sample random noise levels
        t = torch.rand(B, 1, device=device)

        # Add noise scaled by t
        noise = torch.randn_like(x_manifold)
        noisy_x = x_manifold + t * noise

        # Predict clean sample from noisy
        noisy_with_t = torch.cat([noisy_x, t], dim=-1)
        pred_clean = self.consistency_net(noisy_with_t)

        # Consistency loss: predicted clean should match target
        loss = F.mse_loss(pred_clean, target_manifold.detach())

        return loss
