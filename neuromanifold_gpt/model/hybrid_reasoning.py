"""
Hybrid Reasoning Module - System 1/2 adaptive processing.

Implements adaptive routing between fast inference (System 1) and deep thinking (System 2)
based on input complexity. Uses learned gating to decide when to engage multi-step reasoning.

Architecture:
    Input -> Complexity Estimator -> Router -> [Fast Path | Thinking Path] -> Output

Key components:
- ThinkingLayer: Multi-head self-attention with residual for iterative refinement
- ComplexityEstimator: Learns to predict when thinking is needed
- AdaptiveRouter: Gated routing with optional E7 curriculum bias
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class ThinkingLayer(nn.Module):
    """
    Single thinking step with self-attention and FFN.

    Enables iterative refinement of representations through
    recurrent application of attention mechanisms.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads

        assert embed_dim % n_heads == 0, (
            f"embed_dim {embed_dim} must be divisible by n_heads {n_heads}"
        )
        self.head_dim = embed_dim // n_heads

        self.attn = nn.MultiheadAttention(
            embed_dim,
            n_heads,
            dropout=dropout,
            bias=bias,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim, bias=bias),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim, bias=bias),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, embed_dim]
        Returns:
            [batch, seq_len, embed_dim]
        """
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.norm2(x)

        return x


class ComplexityEstimator(nn.Module):
    """
    Estimates input complexity to guide thinking vs fast-path routing.

    Uses attention entropy and activation magnitudes as complexity proxies.
    Returns raw logits (pre-sigmoid).
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.complexity_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, embed_dim]
        Returns:
            complexity_logits: [batch] - logit of needing thinking
        """
        pooled = x.mean(dim=1)
        complexity = self.complexity_head(pooled).squeeze(-1)
        return complexity


class HybridReasoningModule(nn.Module):
    """
    Adaptive System 1/2 reasoning with learned routing.

    Routes inputs between:
    - Fast path (System 1): Direct feedforward
    - Thinking path (System 2): Iterative refinement with multiple thinking layers

    Uses complexity estimation and optional E7 curriculum bias to determine routing.
    """

    def __init__(
        self,
        embed_dim: int,
        n_thinking_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.0,
        use_e7_prior: bool = False,
        e7_bias_strength: float = 0.5,
        thinking_threshold: float = 0.5,
        disable_thinking: bool = False,
        bias: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_thinking_layers = n_thinking_layers
        self.use_e7_prior = use_e7_prior
        self.e7_bias_strength = e7_bias_strength
        self.thinking_threshold = thinking_threshold
        self.disable_thinking = disable_thinking

        self.complexity_estimator = ComplexityEstimator(embed_dim)

        self.thinking_layers = nn.ModuleList(
            [
                ThinkingLayer(embed_dim, n_heads, dropout, bias)
                for _ in range(n_thinking_layers)
            ]
        )

        self.fast_path = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=bias),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        if use_e7_prior:
            self.e7_embedding = nn.Embedding(7, embed_dim)
            self.e7_gate = nn.Linear(embed_dim, 1)

    def _compute_complexity_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute statistical features for complexity estimation.

        Args:
            x: [batch, seq_len, embed_dim]
        Returns:
            features: [batch, embed_dim * 3] concatenation of mean, max, std
        """
        mean_features = x.mean(dim=1)
        max_features = x.max(dim=1)[0]
        std_features = x.std(dim=1)
        return torch.cat([mean_features, max_features, std_features], dim=-1)

    @torch.compiler.disable
    def forward(
        self,
        x: torch.Tensor,
        e7_tier: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            x: [batch, seq_len, embed_dim]
            e7_tier: Optional [batch] - E7 curriculum tier (0-6)

        Returns:
            output: [batch, seq_len, embed_dim]
            info: Dict with 'thinking_probs' and 'mode_selections'
        """
        batch_size = x.shape[0]

        if self.disable_thinking:
            output = self.fast_path(x)
            thinking_probs = torch.zeros(batch_size, device=x.device)
            mode_selections = torch.zeros(batch_size, device=x.device)
        else:
            # Get logits from estimator
            thinking_logits = self.complexity_estimator(x)

            if self.use_e7_prior and e7_tier is not None:
                e7_bias = self.e7_embedding(e7_tier.clamp(0, 6))
                e7_score = self.e7_gate(e7_bias).squeeze(-1)
                thinking_logits = thinking_logits + self.e7_bias_strength * e7_score

            thinking_probs = thinking_logits.sigmoid()

            if self.training:
                mode_selections = torch.bernoulli(thinking_probs)
            else:
                mode_selections = (thinking_probs > self.thinking_threshold).float()

            thinking_mask = mode_selections.bool()
            fast_mask = ~thinking_mask

            output = torch.zeros_like(x)

            if thinking_mask.any():
                thinking_x = x[thinking_mask]
                for layer in self.thinking_layers:
                    thinking_x = layer(thinking_x)
                output[thinking_mask] = thinking_x.to(output.dtype)

            if fast_mask.any():
                fast_x = x[fast_mask]
                fast_out = self.fast_path(fast_x)
                output[fast_mask] = fast_out.to(output.dtype)

        info = {
            "thinking_probs": thinking_probs,
            "mode_selections": mode_selections,
            "thinking_ratio": mode_selections.float().mean(),
        }

        return output, info
