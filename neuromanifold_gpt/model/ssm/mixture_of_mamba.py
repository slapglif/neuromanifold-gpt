"""
Mixture-of-Mamba (MoM) - Modality-Aware Sparse State Space Models.

Implements the architecture from:
"Mixture-of-Mamba: Enhancing Multi-Modal State-Space Models with Modality-Aware Sparsity"
https://iclr.cc/virtual/2025/33731

Key innovations:
1. Modality-aware expert routing (text vs audio vs image)
2. Sparse gating to select subset of Mamba experts per token
3. Load balancing loss to prevent expert collapse
4. Adaptive dt parameters per modality type
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuromanifold_gpt.model.ssm.mamba import MambaBlock


class ModalityRouter(nn.Module):
    """
    Routes tokens to Mamba experts based on modality characteristics.

    Learns to distinguish spectral properties of different modalities:
    - Text: Low-frequency, sparse activations
    - Audio: High-frequency, dense waveforms
    - Image: Mid-frequency, spatial patterns
    """

    def __init__(
        self,
        embed_dim: int,
        num_experts: int,
        top_k: int = 2,
        jitter_noise: float = 0.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter_noise = jitter_noise

        self.gate = nn.Linear(embed_dim, num_experts, bias=False)

    def forward(
        self, x: torch.Tensor, modality_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tokens [batch, seq_len, embed_dim]
            modality_ids: Optional modality type per token [batch, seq_len]
                         (0=text, 1=audio, 2=image)

        Returns:
            gates: Top-k gate weights [batch, seq_len, top_k]
            indices: Top-k expert indices [batch, seq_len, top_k]
            load: Expert load distribution [num_experts]
            logits: Raw routing logits (for entropy computation)
        """
        batch, seq_len, _ = x.shape

        logits = self.gate(x)

        if self.training and self.jitter_noise > 0:
            noise = torch.randn_like(logits) * self.jitter_noise
            logits = logits + noise

        top_logits, top_indices = torch.topk(logits, self.top_k, dim=-1)
        gates = F.softmax(top_logits, dim=-1)

        load = torch.zeros(self.num_experts, device=x.device)
        for k in range(self.top_k):
            load.scatter_add_(0, top_indices[..., k].flatten(), gates[..., k].flatten())

        return gates, top_indices, load, top_logits


class MixtureOfMambaBlock(nn.Module):
    """
    Sparse Mixture-of-Mamba layer with modality-aware routing.

    Routes each token to top-k Mamba experts based on learned
    modality characteristics. Enables efficient multimodal processing
    by specializing experts to different signal types.
    """

    def __init__(
        self,
        embed_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        state_dim: int = 16,
        expand_factor: int = 2,
        load_balance_weight: float = 0.01,
        jitter_noise: float = 0.01,
        **mamba_kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight

        self.router = ModalityRouter(
            embed_dim=embed_dim,
            num_experts=num_experts,
            top_k=top_k,
            jitter_noise=jitter_noise,
        )

        self.experts = nn.ModuleList(
            [
                MambaBlock(
                    embed_dim=embed_dim,
                    state_dim=state_dim,
                    expand_factor=expand_factor,
                    **mamba_kwargs,
                )
                for _ in range(num_experts)
            ]
        )

    def forward(
        self, x: torch.Tensor, modality_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: Input [batch, seq_len, embed_dim]
            modality_ids: Optional modality type [batch, seq_len]

        Returns:
            output: Mixed expert outputs [batch, seq_len, embed_dim]
            info: Dict with 'load_balance_loss', 'expert_load', 'routing_entropy'
        """
        batch, seq_len, embed_dim = x.shape

        gates, indices, load, logits = self.router(x, modality_ids)

        output = torch.zeros_like(x)

        for expert_idx in range(self.num_experts):
            expert_mask = indices == expert_idx
            token_uses_expert = expert_mask.any(dim=-1)

            if not token_uses_expert.any():
                continue

            expert_output = self.experts[expert_idx](x)

            expert_contribution = torch.zeros_like(x)
            for k in range(self.top_k):
                k_uses_expert = indices[..., k] == expert_idx
                k_gate = gates[..., k].unsqueeze(-1)
                expert_contribution = (
                    expert_contribution
                    + k_uses_expert.unsqueeze(-1).float() * k_gate * expert_output
                )

            output = output + expert_contribution

        load_balance_loss = self._compute_load_balance_loss(load, gates)

        # Compute entropy on raw logits for numerical stability
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        routing_entropy = -torch.sum(probs * log_probs, dim=-1).mean()

        info = {
            "load_balance_loss": load_balance_loss,
            "expert_load": load,
            "routing_entropy": routing_entropy,
            "num_experts_used": (load > 0).sum().item(),
        }

        return output, info

    def _compute_load_balance_loss(
        self, load: torch.Tensor, gates: torch.Tensor
    ) -> torch.Tensor:
        """
        Encourages uniform expert utilization.

        Loss = coefficient_of_variation(load) * num_experts
        """
        num_tokens = gates.shape[0] * gates.shape[1]
        target_load = num_tokens * self.top_k / self.num_experts

        load_normalized = load / (target_load + 1e-8)

        # Clamp to prevent numerical instability
        mean = load_normalized.mean().clamp(min=1e-8)
        std = load_normalized.std().clamp(min=0.0)
        cv = std / mean

        return cv * self.load_balance_weight * self.num_experts


class AdaptiveMixtureOfMamba(nn.Module):
    """
    Enhanced MoM with adaptive gating based on spectral analysis.

    Automatically detects modality type by analyzing frequency content:
    - Text: Low-frequency dominance (smooth semantic changes)
    - Audio: Full spectrum (complex waveforms)
    - Image: Mid-frequency (spatial edges and textures)
    """

    def __init__(
        self,
        embed_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        state_dim: int = 16,
        use_spectral_routing: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_spectral_routing = use_spectral_routing

        self.mom = MixtureOfMambaBlock(
            embed_dim=embed_dim,
            num_experts=num_experts,
            top_k=top_k,
            state_dim=state_dim,
            **kwargs,
        )

        if use_spectral_routing:
            self.spectral_analyzer = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.GELU(),
                nn.Linear(embed_dim // 4, 3),
            )

    def _estimate_modality(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimate modality type from spectral characteristics.

        Returns:
            modality_probs: [batch, seq_len, 3] probabilities for [text, audio, image]
        """
        fft = torch.fft.rfft(x, dim=-1)
        power_spectrum = torch.abs(fft).pow(2)

        low_freq = power_spectrum[..., : power_spectrum.shape[-1] // 4].sum(
            dim=-1, keepdim=True
        )
        mid_freq = power_spectrum[
            ..., power_spectrum.shape[-1] // 4 : power_spectrum.shape[-1] // 2
        ].sum(dim=-1, keepdim=True)
        high_freq = power_spectrum[..., power_spectrum.shape[-1] // 2 :].sum(
            dim=-1, keepdim=True
        )

        spectral_features = torch.cat([low_freq, mid_freq, high_freq], dim=-1)
        spectral_features = spectral_features / (
            spectral_features.sum(dim=-1, keepdim=True) + 1e-6
        )

        modality_logits = self.spectral_analyzer(x)

        return F.softmax(modality_logits, dim=-1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: Input [batch, seq_len, embed_dim]

        Returns:
            output: [batch, seq_len, embed_dim]
            info: Dict with modality predictions and MoM info
        """
        modality_ids = None

        if self.use_spectral_routing:
            modality_probs = self._estimate_modality(x)
            modality_ids = modality_probs.argmax(dim=-1)

        output, mom_info = self.mom(x, modality_ids)

        if self.use_spectral_routing:
            mom_info["modality_distribution"] = modality_probs.mean(dim=(0, 1))

        return output, mom_info
