# neuromanifold_gpt/model/gpt.py
"""
NeuroManifoldGPT - Complete model.

Combines:
- Semantic Folding Encoder (SDR)
- Manifold-Spectral-Soliton Attention Blocks
- SDR Engram Memory
- Language Model Head
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuromanifold_gpt.config import NeuroManifoldConfig
from neuromanifold_gpt.model.semantic_folding import SemanticFoldingEncoder
from neuromanifold_gpt.model.block import NeuroManifoldBlock
from neuromanifold_gpt.model.memory.engram import SDREngramMemory


class NeuroManifoldGPT(nn.Module):
    """Complete NeuroManifoldGPT model.

    Architecture:
    1. SemanticFoldingEncoder: tokens -> SDRs (2048 bits, 2% sparse)
    2. NeuroManifoldBlock (n_layer times):
       - SDR -> manifold coordinates + learned metric
       - Manifold -> spectral eigenvectors (O(n) attention)
       - Soliton attention (Kaufmann-Heimburg wave dynamics)
       - MLP with residual
    3. SDREngramMemory: stores breadcrumbs for infinite context
    4. LM Head: embedding -> vocab logits

    Args:
        config: NeuroManifoldConfig with all hyperparameters
    """

    def __init__(self, config: NeuroManifoldConfig):
        super().__init__()
        self.config = config

        # Semantic Folding Encoder
        self.encoder = SemanticFoldingEncoder(
            vocab_size=config.vocab_size,
            sdr_size=config.sdr_size,
            n_active=config.sdr_n_active,
            embed_dim=config.sdr_embed_dim,
            context_size=config.sdr_context_size,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            NeuroManifoldBlock(
                sdr_size=config.sdr_size,
                embed_dim=config.n_embd,
                manifold_dim=config.manifold_dim,
                n_eigenvectors=config.n_eigenvectors,
                n_heads=config.n_heads,
                dropout=config.dropout,
            )
            for _ in range(config.n_layer)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Language model head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Engram memory
        self.memory = SDREngramMemory(
            sdr_size=config.sdr_size,
            capacity=config.engram_capacity,
            n_active=config.sdr_n_active,
            content_dim=config.n_embd,
            threshold=config.engram_threshold,
        )

        # Projection for feeding block output back as SDR-like input
        # (used for residual connections after first block)
        self.embed_to_sdr = nn.Linear(config.n_embd, config.sdr_size)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize linear and embedding weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict]:
        """
        Forward pass.

        Args:
            tokens: (B, T) input token indices
            targets: (B, T) optional target indices for loss

        Returns:
            logits: (B, T, vocab_size)
            loss: scalar if targets provided, else None
            info: diagnostic dict
        """
        B, T = tokens.shape

        # Semantic folding to SDR
        sdr, sdr_scores = self.encoder(tokens)

        # Through blocks
        x = None
        block_infos = []
        for i, block in enumerate(self.blocks):
            if i == 0:
                # First block takes raw SDR
                x, info = block(sdr)
            else:
                # Subsequent blocks: project embeddings to SDR-like space
                # and add residual from SDR
                x_as_sdr = self.embed_to_sdr(x)
                # Combine with original SDR for residual information
                x_as_sdr = x_as_sdr + sdr
                x_new, info = block(x_as_sdr)
                x = x + x_new  # Residual connection
            block_infos.append(info)

        # Final norm
        x = self.ln_f(x)

        # Store final representations as engrams (during training only)
        if self.training:
            for b in range(B):
                for t in range(T):
                    self.memory.store(sdr[b, t], x[b, t].detach())

        # Compute logits
        logits = self.lm_head(x)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        info = {
            "sdr": sdr,
            "sdr_scores": sdr_scores,
            "block_infos": block_infos,
            "memory_size": len(self.memory),
        }

        return logits, loss, info

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation.

        Args:
            idx: (B, T) starting tokens
            max_new_tokens: how many to generate
            temperature: sampling temperature
            top_k: optional top-k filtering

        Returns:
            idx: (B, T + max_new_tokens) generated sequence
        """
        for _ in range(max_new_tokens):
            # Crop to block_size if needed
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )

            # Forward
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        return idx

    def num_parameters(self, non_embedding: bool = True) -> int:
        """Count model parameters.

        Args:
            non_embedding: If True, exclude token embeddings from count

        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.encoder.token_embed.weight.numel()
        return n_params

    def configure_optimizers(
        self,
        weight_decay: float = 0.1,
        learning_rate: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        device_type: str = "cpu",
    ) -> torch.optim.AdamW:
        """Configure optimizer with weight decay for non-bias/norm params.

        Args:
            weight_decay: L2 regularization strength
            learning_rate: Learning rate
            betas: Adam beta parameters
            device_type: For fused Adam (CUDA only)

        Returns:
            Configured AdamW optimizer
        """
        # Separate params into decay and no-decay groups
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # Biases, LayerNorm weights, and embedding weights don't decay
            if param.ndim < 2 or "ln" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        # Use fused Adam on CUDA if available
        fused = device_type == "cuda" and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        return torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            fused=fused,
        )
