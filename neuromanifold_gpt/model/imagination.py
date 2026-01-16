# neuromanifold_gpt/model/imagination.py
"""
Consistency Imagination Module.

Lightweight diffusion-based counterfactual exploration for System 2 reasoning.
Uses consistency model approach for fast sampling of alternative trajectories.

Reference:
- Consistency Models (Song et al., 2023)
- Denoising Diffusion for "mental whiteboard" exploration
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConsistencyImaginationModule(nn.Module):
    """Consistency model for fast counterfactual generation.

    This module enables the model to explore alternative reasoning paths
    by generating diverse trajectory samples in manifold space.

    Architecture:
    - Encoder: Projects input to latent imagination space
    - Denoiser: Iterative refinement network (consistency model)
    - Decoder: Projects back to embedding space

    Unlike full diffusion models, consistency models enable single-step
    or few-step generation, making them practical for inference-time reasoning.

    Args:
        embed_dim: Embedding dimension from transformer
        manifold_dim: Dimension of manifold latent space
        n_imagination_steps: Number of denoising steps (2-4 recommended)
        dropout: Dropout probability for regularization
        bias: Whether to use bias in linear layers
    """

    def __init__(
        self,
        embed_dim: int,
        manifold_dim: int = 64,
        n_imagination_steps: int = 4,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.manifold_dim = manifold_dim
        self.n_imagination_steps = n_imagination_steps

        # Encoder: embedding -> manifold latent space
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, manifold_dim * 2, bias=bias),
            nn.SiLU(),
            nn.Linear(manifold_dim * 2, manifold_dim, bias=bias),
        )

        # Denoiser: iterative refinement in latent space
        # Conditioned on timestep and input context
        self.denoiser = nn.Sequential(
            nn.Linear(manifold_dim * 2, manifold_dim * 2, bias=bias),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(manifold_dim * 2, manifold_dim, bias=bias),
        )

        # Decoder: manifold latent -> embedding space
        self.decoder = nn.Sequential(
            nn.Linear(manifold_dim, manifold_dim * 2, bias=bias),
            nn.SiLU(),
            nn.Linear(manifold_dim * 2, embed_dim, bias=bias),
        )

        # Learnable timestep embeddings
        self.time_embed = nn.Embedding(n_imagination_steps, manifold_dim)

    def forward(
        self,
        x: torch.Tensor,
        n_alternatives: int = 4,
    ) -> dict[str, torch.Tensor]:
        """Generate counterfactual alternatives via consistency sampling.

        Args:
            x: Input embeddings (B, T, embed_dim)
            n_alternatives: Number of alternative trajectories to generate

        Returns:
            Dictionary containing:
                - alternatives: (B, n_alternatives, T, embed_dim) alternative trajectories
                - latents: (B, n_alternatives, T, manifold_dim) latent codes
                - diversity: (B,) diversity score across alternatives
        """
        B, T, _ = x.shape

        # Encode to manifold latent space
        z_context = self.encoder(x)  # (B, T, manifold_dim)

        # Generate alternatives via iterative denoising
        alternatives = []
        latents = []

        for _ in range(n_alternatives):
            # Initialize with noise
            z = torch.randn(B, T, self.manifold_dim, device=x.device)

            # Iterative denoising
            for step in range(self.n_imagination_steps):
                # Timestep embedding
                t_emb = self.time_embed(torch.tensor([step], device=x.device))
                t_emb = t_emb.expand(B, T, -1)

                # Concatenate noisy latent with context and timestep
                z_input = torch.cat([z, z_context], dim=-1)  # (B, T, manifold_dim*2)

                # Denoise step
                z_denoised = self.denoiser(z_input)

                # Update latent (consistency model: direct prediction)
                z = z_denoised

            # Decode to embedding space
            alternative = self.decoder(z)  # (B, T, embed_dim)

            alternatives.append(alternative)
            latents.append(z)

        # Stack alternatives
        alternatives = torch.stack(alternatives, dim=1)  # (B, n_alternatives, T, embed_dim)
        latents = torch.stack(latents, dim=1)  # (B, n_alternatives, T, manifold_dim)

        # Compute diversity metric (variance across alternatives)
        diversity = torch.var(latents, dim=1).mean(dim=[1, 2])  # (B,)

        return {
            'alternatives': alternatives,
            'latents': latents,
            'diversity': diversity,
        }

    def sample_single(
        self,
        x: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Sample a single alternative trajectory.

        Args:
            x: Input embeddings (B, T, embed_dim)
            temperature: Sampling temperature (higher = more diverse)

        Returns:
            alternative: (B, T, embed_dim) single alternative trajectory
        """
        B, T, _ = x.shape

        # Encode context
        z_context = self.encoder(x)

        # Initialize with scaled noise
        z = torch.randn(B, T, self.manifold_dim, device=x.device) * temperature

        # Iterative denoising
        for step in range(self.n_imagination_steps):
            t_emb = self.time_embed(torch.tensor([step], device=x.device))
            t_emb = t_emb.expand(B, T, -1)

            z_input = torch.cat([z, z_context], dim=-1)
            z = self.denoiser(z_input)

        # Decode
        alternative = self.decoder(z)

        return alternative
