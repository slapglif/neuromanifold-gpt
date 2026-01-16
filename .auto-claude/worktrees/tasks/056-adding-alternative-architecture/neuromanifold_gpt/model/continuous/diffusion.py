# neuromanifold_gpt/model/continuous/diffusion.py
"""
Latent Diffusion Decoder for continuous generation.

Implements DDPM (Denoising Diffusion Probabilistic Models) operating in
latent space for efficient high-quality generation.

The diffusion process:
1. Forward process: q(z_t | z_{t-1}) = N(sqrt(1-beta_t) * z_{t-1}, beta_t * I)
2. Reverse process: p_theta(z_{t-1} | z_t) learned by neural network
3. Sampling: Start from z_T ~ N(0, I) and iteratively denoise

Key features:
- Cosine noise schedule for stable training
- Learnable denoising network (Transformer-based)
- Classifier-free guidance support
- Configurable number of diffusion steps
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DiffusionConfig:
    """Configuration for latent diffusion model.

    Args:
        dim: Latent dimension
        n_steps: Number of diffusion timesteps
        beta_start: Starting noise level
        beta_end: Ending noise level
        schedule_type: Noise schedule type ('linear', 'cosine', 'quadratic')
        n_layers: Number of denoising network layers
        n_heads: Number of attention heads in denoiser
        dropout: Dropout rate
        clip_denoised: Whether to clip denoised values
        clip_range: Range for clipping denoised values
    """
    dim: int = 384
    n_steps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule_type: str = "cosine"
    n_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.0
    clip_denoised: bool = True
    clip_range: float = 1.0


class DDPMScheduler(nn.Module):
    """DDPM noise scheduler.

    Computes and stores the noise schedule parameters:
    - betas: noise levels at each timestep
    - alphas: 1 - betas
    - alpha_cumprod: cumulative product of alphas
    - sqrt_alpha_cumprod: for adding noise
    - sqrt_one_minus_alpha_cumprod: for noise scaling
    """

    def __init__(
        self,
        n_steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule_type: str = "cosine",
    ):
        super().__init__()
        self.n_steps = n_steps
        self.schedule_type = schedule_type

        # Compute beta schedule
        if schedule_type == "linear":
            betas = torch.linspace(beta_start, beta_end, n_steps)
        elif schedule_type == "quadratic":
            betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, n_steps) ** 2
        elif schedule_type == "cosine":
            # Cosine schedule from "Improved DDPM" paper
            steps = torch.linspace(0, n_steps, n_steps + 1)
            alpha_bar = torch.cos(((steps / n_steps) + 0.008) / 1.008 * math.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
            betas = torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        # Compute derived quantities
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        alpha_cumprod_prev = F.pad(alpha_cumprod[:-1], (1, 0), value=1.0)

        # Register as buffers (not parameters)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("alpha_cumprod_prev", alpha_cumprod_prev)
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1.0 - alpha_cumprod))
        self.register_buffer("sqrt_recip_alpha", torch.sqrt(1.0 / alphas))
        self.register_buffer("sqrt_recip_alpha_cumprod", torch.sqrt(1.0 / alpha_cumprod))
        self.register_buffer("sqrt_recipm1_alpha_cumprod", torch.sqrt(1.0 / alpha_cumprod - 1))

        # Posterior variance for q(z_{t-1} | z_t, z_0)
        posterior_variance = betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped",
                           torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer("posterior_mean_coef1",
                           betas * torch.sqrt(alpha_cumprod_prev) / (1.0 - alpha_cumprod))
        self.register_buffer("posterior_mean_coef2",
                           (1.0 - alpha_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alpha_cumprod))

    def add_noise(
        self,
        z_0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """Add noise to clean sample z_0 at timestep t.

        q(z_t | z_0) = N(sqrt(alpha_bar_t) * z_0, (1 - alpha_bar_t) * I)

        Args:
            z_0: Clean samples [B, ...]
            noise: Gaussian noise [B, ...]
            t: Timesteps [B]

        Returns:
            Noisy samples z_t [B, ...]
        """
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t]

        # Reshape for broadcasting
        while sqrt_alpha_cumprod_t.dim() < z_0.dim():
            sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.unsqueeze(-1)

        return sqrt_alpha_cumprod_t * z_0 + sqrt_one_minus_alpha_cumprod_t * noise

    def get_posterior_mean_variance(
        self,
        z_0: torch.Tensor,
        z_t: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute posterior q(z_{t-1} | z_t, z_0).

        Args:
            z_0: Predicted clean sample
            z_t: Current noisy sample
            t: Current timestep

        Returns:
            Tuple of (mean, variance, log_variance)
        """
        coef1 = self.posterior_mean_coef1[t]
        coef2 = self.posterior_mean_coef2[t]
        variance = self.posterior_variance[t]
        log_variance = self.posterior_log_variance_clipped[t]

        # Reshape for broadcasting
        while coef1.dim() < z_0.dim():
            coef1 = coef1.unsqueeze(-1)
            coef2 = coef2.unsqueeze(-1)
            variance = variance.unsqueeze(-1)
            log_variance = log_variance.unsqueeze(-1)

        mean = coef1 * z_0 + coef2 * z_t
        return mean, variance, log_variance


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed timesteps.

        Args:
            t: Timesteps [B]

        Returns:
            Embeddings [B, dim]
        """
        device = t.device
        half_dim = self.dim // 2

        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t.unsqueeze(1) * embeddings.unsqueeze(0)
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        return embeddings


class DenoisingBlock(nn.Module):
    """Single block of the denoising network."""

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        dropout: float = 0.0,
        time_dim: int = 256,
    ):
        super().__init__()
        self.dim = dim

        # Self-attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Time conditioning (AdaLN-style)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, dim * 2),
        )

        # Feed-forward
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input [B, L, D]
            t_emb: Time embedding [B, time_dim]

        Returns:
            Output [B, L, D]
        """
        # Time conditioning: get scale and shift
        time_cond = self.time_mlp(t_emb)  # [B, D*2]
        scale, shift = time_cond.chunk(2, dim=-1)  # [B, D] each
        scale = scale.unsqueeze(1)  # [B, 1, D]
        shift = shift.unsqueeze(1)  # [B, 1, D]

        # Self-attention with residual
        h = self.norm1(x)
        h = h * (1 + scale) + shift  # AdaLN conditioning
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out

        # Feed-forward with residual
        h = self.norm2(x)
        x = x + self.ff(h)

        return x


class DenoisingNetwork(nn.Module):
    """Neural network for predicting noise in diffusion process.

    Uses a Transformer-style architecture with time conditioning.
    """

    def __init__(
        self,
        dim: int,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.0,
        time_dim: int = 256,
    ):
        super().__init__()
        self.dim = dim
        self.time_dim = time_dim

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Input projection
        self.input_proj = nn.Linear(dim, dim)

        # Denoising blocks
        self.blocks = nn.ModuleList([
            DenoisingBlock(dim, n_heads, dropout, time_dim)
            for _ in range(n_layers)
        ])

        # Output projection
        self.output_norm = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, dim)

        # Initialize output to zero for stability
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise from noisy input and timestep.

        Args:
            z_t: Noisy input [B, L, D] or [B, D]
            t: Timesteps [B]

        Returns:
            Predicted noise [B, L, D] or [B, D]
        """
        # Handle 2D input (no sequence dimension)
        squeeze_output = False
        if z_t.dim() == 2:
            z_t = z_t.unsqueeze(1)
            squeeze_output = True

        # Time embedding
        t_emb = self.time_embed(t.float())  # [B, time_dim]

        # Input projection
        h = self.input_proj(z_t)

        # Apply denoising blocks
        for block in self.blocks:
            h = block(h, t_emb)

        # Output projection
        h = self.output_norm(h)
        noise_pred = self.output_proj(h)

        if squeeze_output:
            noise_pred = noise_pred.squeeze(1)

        return noise_pred


class LatentDiffusion(nn.Module):
    """Latent Diffusion Model for continuous generation.

    Implements DDPM in latent space with configurable denoising network
    and noise schedule.

    Example:
        >>> diffusion = LatentDiffusion(dim=384, n_steps=100)
        >>> # Training: compute loss
        >>> z = torch.randn(2, 32, 384)
        >>> loss = diffusion.compute_loss(z)
        >>> # Sampling: generate new samples
        >>> samples = diffusion.sample((2, 32, 384), device='cuda')
    """

    def __init__(
        self,
        dim: int,
        n_steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule_type: str = "cosine",
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.0,
        clip_denoised: bool = True,
        clip_range: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self.clip_denoised = clip_denoised
        self.clip_range = clip_range

        # Noise scheduler
        self.scheduler = DDPMScheduler(
            n_steps=n_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            schedule_type=schedule_type,
        )

        # Denoising network
        self.denoiser = DenoisingNetwork(
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
        )

    def forward(
        self,
        z: torch.Tensor,
        return_loss: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass - compute diffusion loss.

        Args:
            z: Clean latent samples [B, L, D]
            return_loss: Whether to return loss

        Returns:
            Tuple of (loss or predicted noise, info dict)
        """
        if return_loss:
            loss = self.compute_loss(z)
            return loss, {"loss": loss.item()}
        else:
            # Just predict noise at random timestep
            B = z.shape[0]
            t = torch.randint(0, self.n_steps, (B,), device=z.device)
            noise = torch.randn_like(z)
            z_t = self.scheduler.add_noise(z, noise, t)
            noise_pred = self.denoiser(z_t, t)
            return noise_pred, {"t": t}

    def compute_loss(
        self,
        z: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute diffusion training loss.

        Uses simple MSE loss between predicted and actual noise.

        Args:
            z: Clean latent samples [B, L, D] or [B, D]
            noise: Optional pre-sampled noise
            t: Optional pre-sampled timesteps

        Returns:
            Scalar loss
        """
        B = z.shape[0]
        device = z.device

        # Sample timesteps
        if t is None:
            t = torch.randint(0, self.n_steps, (B,), device=device)

        # Sample noise
        if noise is None:
            noise = torch.randn_like(z)

        # Add noise to get z_t
        z_t = self.scheduler.add_noise(z, noise, t)

        # Predict noise
        noise_pred = self.denoiser(z_t, t)

        # MSE loss
        loss = F.mse_loss(noise_pred, noise)

        return loss

    def predict_start_from_noise(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Predict z_0 from z_t and predicted noise.

        z_0 = (z_t - sqrt(1-alpha_bar) * noise) / sqrt(alpha_bar)

        Args:
            z_t: Noisy sample
            t: Timestep
            noise: Predicted noise

        Returns:
            Predicted clean sample z_0
        """
        sqrt_recip_alpha_cumprod = self.scheduler.sqrt_recip_alpha_cumprod[t]
        sqrt_recipm1_alpha_cumprod = self.scheduler.sqrt_recipm1_alpha_cumprod[t]

        # Reshape for broadcasting
        while sqrt_recip_alpha_cumprod.dim() < z_t.dim():
            sqrt_recip_alpha_cumprod = sqrt_recip_alpha_cumprod.unsqueeze(-1)
            sqrt_recipm1_alpha_cumprod = sqrt_recipm1_alpha_cumprod.unsqueeze(-1)

        z_0 = sqrt_recip_alpha_cumprod * z_t - sqrt_recipm1_alpha_cumprod * noise

        if self.clip_denoised:
            z_0 = torch.clamp(z_0, -self.clip_range, self.clip_range)

        return z_0

    @torch.no_grad()
    def p_sample(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Single denoising step p(z_{t-1} | z_t).

        Args:
            z_t: Noisy sample at timestep t
            t: Current timestep (same for all batch elements)

        Returns:
            Denoised sample z_{t-1}
        """
        B = z_t.shape[0]
        device = z_t.device

        # Expand t to batch size if needed
        if t.dim() == 0:
            t = t.expand(B)

        # Predict noise
        noise_pred = self.denoiser(z_t, t)

        # Predict z_0
        z_0_pred = self.predict_start_from_noise(z_t, t, noise_pred)

        # Get posterior mean and variance
        mean, variance, log_variance = self.scheduler.get_posterior_mean_variance(
            z_0_pred, z_t, t
        )

        # Sample z_{t-1}
        noise = torch.randn_like(z_t)

        # No noise for t=0
        nonzero_mask = (t != 0).float()
        while nonzero_mask.dim() < z_t.dim():
            nonzero_mask = nonzero_mask.unsqueeze(-1)

        z_prev = mean + nonzero_mask * torch.sqrt(variance) * noise

        return z_prev

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """Generate samples via iterative denoising.

        Starts from pure noise z_T ~ N(0, I) and iteratively denoises
        to produce clean samples z_0.

        Args:
            shape: Shape of samples to generate (B, L, D) or (B, D)
            device: Device to generate on
            return_trajectory: Whether to return full denoising trajectory

        Returns:
            Generated samples [shape] or trajectory [(T+1, *shape)]
        """
        # Start from pure noise
        z = torch.randn(shape, device=device)

        trajectory = [z] if return_trajectory else None

        # Iterative denoising
        for i in reversed(range(self.n_steps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            z = self.p_sample(z, t)

            if return_trajectory:
                trajectory.append(z)

        if return_trajectory:
            return torch.stack(trajectory, dim=0)

        return z

    @torch.no_grad()
    def sample_ddim(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        n_steps: int = 50,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """DDIM sampling for faster generation.

        Uses deterministic sampling (eta=0) or stochastic (eta=1).

        Args:
            shape: Shape of samples to generate
            device: Device to generate on
            n_steps: Number of DDIM steps (can be less than training steps)
            eta: Stochasticity parameter (0 = deterministic, 1 = DDPM)

        Returns:
            Generated samples
        """
        # Create timestep schedule
        step_ratio = self.n_steps // n_steps
        timesteps = torch.arange(0, self.n_steps, step_ratio, device=device)
        timesteps = torch.flip(timesteps, dims=[0])

        # Start from noise
        z = torch.randn(shape, device=device)

        for i, t in enumerate(timesteps):
            t_batch = t.expand(shape[0])

            # Predict noise
            noise_pred = self.denoiser(z, t_batch)

            # Predict z_0
            z_0_pred = self.predict_start_from_noise(z, t_batch, noise_pred)

            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]

                # Get alpha values
                alpha_t = self.scheduler.alpha_cumprod[t]
                alpha_next = self.scheduler.alpha_cumprod[t_next]

                # DDIM update
                sigma = eta * torch.sqrt(
                    (1 - alpha_next) / (1 - alpha_t) * (1 - alpha_t / alpha_next)
                )

                # Predicted direction
                pred_dir = torch.sqrt(1 - alpha_next - sigma ** 2) * noise_pred

                # DDIM step
                z = torch.sqrt(alpha_next) * z_0_pred + pred_dir

                if eta > 0:
                    z = z + sigma * torch.randn_like(z)
            else:
                z = z_0_pred

        return z

    def extra_repr(self) -> str:
        return f"dim={self.dim}, n_steps={self.n_steps}"
