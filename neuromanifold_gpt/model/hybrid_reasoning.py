# neuromanifold_gpt/model/hybrid_reasoning.py
"""
Hybrid Reasoning Module (Qwen3-style thinking/non-thinking routing).

Implements dual-mode reasoning for System 2 processing:
- Fast path: Direct passthrough for simple queries
- Slow path: Additional transformer layers for complex reasoning

Router network predicts whether to engage "thinking mode" based on:
- Input complexity (learned from embeddings)
- Optional curriculum tier (E7 prior) for progressive training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ThinkingLayer(nn.Module):
    """
    Simple transformer layer for slow/thinking path.

    Uses standard multi-head attention + FFN without the full NeuroManifold complexity.
    Designed to be lightweight but powerful for deliberate reasoning.
    """

    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feed-forward network
        ffn_dim = embed_dim * 4
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # Layer norms (pre-norm architecture)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, embed_dim)

        Returns:
            (B, T, embed_dim)
        """
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out

        # FFN with residual
        x = x + self.ffn(self.norm2(x))

        return x


class HybridReasoningModule(nn.Module):
    """
    Hybrid reasoning with thinking/non-thinking mode routing.

    Inspired by Qwen3's approach to selective deep reasoning:
    - Router predicts complexity from input features
    - Fast path: Direct passthrough for simple inputs
    - Slow path: Additional transformer layers for complex reasoning
    - E7 curriculum integration: Use tier as prior for progressive training

    Args:
        embed_dim: Dimension of embedding space (e.g., 384)
        n_thinking_layers: Number of extra layers for slow/thinking path (default: 2)
        n_heads: Number of attention heads for thinking layers (default: 8)
        dropout: Dropout probability (default: 0.0)
        use_e7_prior: Whether to use E7 curriculum tier as routing prior (default: True)
        thinking_threshold: Threshold for engaging thinking mode (default: 0.5)
    """

    def __init__(
        self,
        embed_dim: int,
        n_thinking_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.0,
        use_e7_prior: bool = True,
        thinking_threshold: float = 0.5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_thinking_layers = n_thinking_layers
        self.use_e7_prior = use_e7_prior
        self.thinking_threshold = thinking_threshold

        # Router network: predicts thinking probability
        # Takes: sequence features (mean, max, std) + optional e7_tier
        router_input_dim = embed_dim * 3  # mean, max, std
        if use_e7_prior:
            router_input_dim += 1  # +1 for e7_tier scalar

        self.router = nn.Sequential(
            nn.Linear(router_input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),  # Single logit for thinking probability
            nn.Sigmoid(),  # Map to [0, 1] probability
        )

        # Thinking layers (slow path)
        self.thinking_layers = nn.ModuleList([
            ThinkingLayer(embed_dim, n_heads, dropout)
            for _ in range(n_thinking_layers)
        ])

        # Fast path: just identity (passthrough)
        # No parameters needed

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize router weights for better gradient flow."""
        for module in self.router:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _compute_complexity_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract complexity features from input sequence.

        Args:
            x: (B, T, embed_dim)

        Returns:
            (B, embed_dim * 3) features: [mean, max, std] across time
        """
        # Mean pooling
        x_mean = x.mean(dim=1)  # (B, embed_dim)

        # Max pooling
        x_max = x.max(dim=1)[0]  # (B, embed_dim)

        # Std pooling (measure of variability/complexity)
        x_std = x.std(dim=1)  # (B, embed_dim)

        # Concatenate
        features = torch.cat([x_mean, x_max, x_std], dim=-1)  # (B, embed_dim * 3)

        return features

    def forward(
        self,
        x: torch.Tensor,
        e7_tier: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Forward pass with thinking/non-thinking routing.

        Args:
            x: (B, T, embed_dim) input embeddings
            e7_tier: (B,) optional curriculum tier (0-6 for E7 levels)

        Returns:
            x_out: (B, T, embed_dim) output embeddings
            info_dict: Dictionary containing:
                - thinking_probs: (B,) probability of thinking mode
                - mode_selections: (B,) binary mask (1 = thinking, 0 = fast)
        """
        B, T, D = x.shape
        device = x.device

        # Compute complexity features
        features = self._compute_complexity_features(x)  # (B, embed_dim * 3)

        # Add E7 tier prior if provided
        if self.use_e7_prior and e7_tier is not None:
            # Normalize tier to [0, 1] (E7 has 7 tiers: 0-6)
            e7_normalized = e7_tier.float() / 6.0  # (B,)
            e7_normalized = e7_normalized.unsqueeze(-1)  # (B, 1)
            features = torch.cat([features, e7_normalized], dim=-1)  # (B, embed_dim*3 + 1)
        elif self.use_e7_prior and e7_tier is None:
            # If use_e7_prior is True but tier not provided, use default (mid-tier)
            default_tier = torch.full((B, 1), 0.5, device=device)  # Mid-tier default
            features = torch.cat([features, default_tier], dim=-1)

        # Predict thinking probability
        thinking_probs = self.router(features).squeeze(-1)  # (B,)

        # Mode selection: probabilistic during training, threshold during eval
        if self.training:
            # Sample from Bernoulli during training (differentiable via Gumbel-Softmax)
            # Using Gumbel-Softmax for differentiable sampling
            logits = torch.stack([
                torch.log(1 - thinking_probs + 1e-8),  # log P(fast)
                torch.log(thinking_probs + 1e-8),      # log P(thinking)
            ], dim=-1)  # (B, 2)

            # Gumbel-Softmax with temperature=1.0
            mode_probs = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)  # (B, 2)
            use_thinking = mode_probs[:, 1]  # (B,) - one-hot selected
        else:
            # Threshold-based selection during eval
            use_thinking = (thinking_probs > self.thinking_threshold).float()  # (B,)

        # Apply fast or slow path
        # Fast path: passthrough
        x_fast = x  # (B, T, D)

        # Slow path: thinking layers
        x_slow = x
        for layer in self.thinking_layers:
            x_slow = layer(x_slow)  # (B, T, D)

        # Blend based on mode selection
        # use_thinking: (B,) -> (B, 1, 1) for broadcasting
        use_thinking_broadcast = use_thinking.unsqueeze(-1).unsqueeze(-1)
        x_out = (1 - use_thinking_broadcast) * x_fast + use_thinking_broadcast * x_slow

        # Return output + info
        info_dict = {
            "thinking_probs": thinking_probs,     # (B,)
            "mode_selections": use_thinking,      # (B,) - 1 = thinking, 0 = fast
        }

        return x_out, info_dict
