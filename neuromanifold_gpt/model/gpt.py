# neuromanifold_gpt/model/gpt.py
"""
NeuroManifoldGPT - Complete model.

Combines:
- Semantic Folding Encoder (SDR)
- Manifold-Spectral-Soliton Attention Blocks
- SDR Engram Memory
- Language Model Head
- Ramanujan Periodic Positional Embeddings (New!)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuromanifold_gpt.config import NeuroManifoldConfig
from neuromanifold_gpt.model.semantic_folding import SemanticFoldingEncoder
from neuromanifold_gpt.model.block import NeuroManifoldBlock
from neuromanifold_gpt.model.memory.engram import SDREngramMemory
from neuromanifold_gpt.model.embeddings.ramanujan import RamanujanPositionalEmbedding
from neuromanifold_gpt.model.mhc import get_expand_reduce_stream_functions
from neuromanifold_gpt.model.kan.faster import replace_linear_with_fasterkan


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

        # Override SDR size if SDR is disabled (Block compatibility)
        if not config.use_sdr:
            block_sdr_size = config.n_embd
        else:
            block_sdr_size = config.sdr_size

        # Token Embedding (SDR or Standard)
        self.use_sdr = config.use_sdr
        if self.use_sdr:
            self.encoder = SemanticFoldingEncoder(
                vocab_size=config.vocab_size,
                sdr_size=config.sdr_size,
                n_active=config.sdr_n_active,
                embed_dim=config.sdr_embed_dim,
                context_size=config.sdr_context_size,
            )
            # Projection for feeding block output back as SDR-like input
            self.embed_to_sdr = nn.Linear(config.n_embd, config.sdr_size)

            # Ramanujan Positional Embedding (Add to SDR projection output in loop)
            # We initialize it with n_embd dimension, as it will be added to the embedding space
            self.ramanujan_pos = RamanujanPositionalEmbedding(
                config.block_size, config.n_embd
            )
        else:
            # Standard embedding (direct to n_embd)
            self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
            # Use Ramanujan for position instead of standard Learned/Sinusoidal
            self.position_embedding = RamanujanPositionalEmbedding(
                config.block_size, config.n_embd
            )
            self.embed_to_sdr = None

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                NeuroManifoldBlock(
                    sdr_size=config.n_embd,
                    embed_dim=config.n_embd,
                    manifold_dim=config.manifold_dim,
                    n_eigenvectors=config.n_eigenvectors,
                    n_heads=config.n_heads,
                    dropout=config.dropout,
                    # FHN dynamics with semi-implicit IMEX scheme
                    fhn_threshold=config.fhn_threshold,
                    fhn_tau=config.fhn_tau,
                    pulse_width_base=config.pulse_width_base,
                    n_fhn_steps=config.n_fhn_steps,
                    use_fhn_imex=config.use_fhn_imex,
                    use_fhn_partitioning=config.use_fhn_partitioning,
                    use_fhn_fused=config.use_fhn_fused,
                    # KAN configuration
                    use_kan=config.use_kan,
                    kan_type=config.kan_type,
                    kan_degree=config.kan_degree,
                    kan_wavelet=config.kan_wavelet,
                    use_fast_wavekan=config.use_fast_wavekan,
                    kan_num_centers=config.kan_num_centers,
                    # Knot attention
                    use_knot_attention=config.use_knot_attention,
                    use_kaufmann_attention=config.use_kaufmann_attention,
                    # mHC (Manifold-Constrained Hyper-Connections)
                    use_mhc=config.use_mhc,
                    use_full_mhc=config.use_full_mhc,
                    mhc_n_streams=config.mhc_n_streams,
                    mhc_residual_weight=config.mhc_residual_weight,
                    # Speed optimization
                    skip_manifold_spectral=config.skip_manifold_spectral,
                )
                for _ in range(config.n_layer)
            ]
        )

        # mHC stream expansion/reduction (for multi-stream mHC)
        # These expand input to num_streams copies and reduce back after blocks
        mhc_disable = not config.use_mhc or config.mhc_n_streams <= 1
        self.expand_stream, self.reduce_stream = get_expand_reduce_stream_functions(
            config.mhc_n_streams, disable=mhc_disable
        )
        self.mhc_enabled = config.use_mhc and config.mhc_n_streams > 1

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

        # Initialize weights
        self.apply(self._init_weights)

        # Replace ALL nn.Linear with FasterKAN (except lm_head)
        # This applies to: manifold projection, spectral decomposition, attention projections
        if getattr(config, "use_kan_everywhere", False):
            replace_linear_with_fasterkan(
                self,
                num_centers=config.kan_num_centers,
                skip_names={"lm_head"},  # Keep output head as Linear
            )

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
        """
        B, T = tokens.shape
        device = tokens.device

        # Embedding Stage
        topographic_loss = torch.tensor(0.0, device=device)
        discrimination_loss = torch.tensor(0.0, device=device)
        if self.use_sdr:
            # Semantic folding to SDR with topographic and discrimination losses
            sdr, sdr_scores, topographic_loss, discrimination_loss = self.encoder(
                tokens
            )
            x = None  # Initial x comes from first block processing SDR
        else:
            # Standard embedding
            tok_emb = self.token_embedding(tokens)
            # Use Ramanujan Position Embedding
            # Indices are 0..T-1
            pos_emb = self.position_embedding(
                torch.arange(T, device=device).unsqueeze(0)
            )  # (1, T, D)
            x = tok_emb + pos_emb  # (B, T, n_embd)
            sdr = torch.zeros(B, T, self.config.sdr_size, device=device)  # Dummy SDR
            sdr_scores = None

        # Through blocks
        block_infos = []
        total_ortho_loss = torch.tensor(0.0, device=device)

        for i, block in enumerate(self.blocks):
            if self.use_sdr:
                if i == 0:
                    # First block: SDR -> x
                    # Expand for multi-stream mHC if enabled
                    sdr_in = self.expand_stream(sdr) if self.mhc_enabled else sdr
                    x, info = block(sdr_in)
                    # Add Ramanujan Positional Embedding here (to x)
                    # x is (B*S, T, n_embd) if mHC, else (B, T, n_embd)
                    pos_emb = self.ramanujan_pos(torch.zeros(1, T, device=device))
                    x = x + pos_emb
                else:
                    # Block already applies internal residuals
                    x, info = block(x)
            else:
                # Dense Mode: Pass embeddings directly
                if i == 0 and self.mhc_enabled:
                    # Expand for multi-stream mHC
                    x = self.expand_stream(x)
                # Block already applies internal residuals (x + attn_out, x + mlp_out)
                # DO NOT add another residual here - that was doubling the signal
                x, info = block(x)

            block_infos.append(info)
            # Accumulate orthogonality regularization loss
            if "ortho_loss" in info:
                total_ortho_loss = total_ortho_loss + info["ortho_loss"]

        # Reduce multi-stream mHC back to single stream
        if self.mhc_enabled:
            x = self.reduce_stream(x)

        # Final norm
        x = self.ln_f(x)

        # Store final representations as engrams (during training only)
        if self.training:
            # Vectorized storage - critical for performance!
            flat_sdr = sdr.view(-1, sdr.size(-1))
            flat_x = x.view(-1, x.size(-1))
            self.memory.store_batch(flat_sdr, flat_x.detach())

        # Compute logits
        logits = self.lm_head(x)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
            # Add auxiliary losses:
            # - ortho_loss: spectral orthogonality
            # - discrimination_loss: prevent SDR mode collapse (CRITICAL for SDR mode)
            # - topographic_loss: currently disabled for stability
            loss = (
                ce_loss
                + total_ortho_loss
                + 0.5 * discrimination_loss
                + 0.0 * topographic_loss
            )
        else:
            loss = None
            topographic_loss = torch.tensor(0.0, device=device)
            discrimination_loss = torch.tensor(0.0, device=device)

        info = {
            "sdr": sdr,
            "sdr_scores": sdr_scores,
            "block_infos": block_infos,
            "memory_size": self.memory.get_size(),
            "ortho_loss": total_ortho_loss,
            "topographic_loss": topographic_loss,
            "discrimination_loss": discrimination_loss,
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
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.use_sdr:
            n_params -= self.encoder.token_embed.weight.numel()
        return n_params

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        N = self.num_parameters()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_heads, cfg.head_dim, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter / dt
        flops_promised = 312e12
        return flops_achieved / flops_promised

    def configure_optimizers(
        self,
        weight_decay: float = 0.1,
        learning_rate: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        device_type: str = "cpu",
    ) -> torch.optim.AdamW:
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or "ln" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        fused = (
            device_type == "cuda"
            and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        )
        return torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            fused=fused,
        )
