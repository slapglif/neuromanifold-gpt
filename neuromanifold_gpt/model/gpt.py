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
from torch.utils.checkpoint import checkpoint

from neuromanifold_gpt.config import NeuroManifoldConfig
from neuromanifold_gpt.config.block_config import NeuroManifoldBlockConfig
from neuromanifold_gpt.model.semantic_folding import SemanticFoldingEncoder
from neuromanifold_gpt.model.block import NeuroManifoldBlock
from neuromanifold_gpt.model.memory.engram import SDREngramMemory
from neuromanifold_gpt.model.embeddings.ramanujan import RamanujanPositionalEmbedding
from neuromanifold_gpt.model.embeddings.rotary import RotaryPositionalEmbedding
from neuromanifold_gpt.model.embeddings.alibi import ALiBiPositionalBias
from neuromanifold_gpt.model.mhc import get_expand_reduce_stream_functions
from neuromanifold_gpt.model.kan.faster import replace_linear_with_fasterkan
from neuromanifold_gpt.model.attention.mla import RMSNorm  # ~15% faster than LayerNorm
from neuromanifold_gpt.model.system_two_mixin import SystemTwoReasoningMixin


class NeuroManifoldGPT(SystemTwoReasoningMixin, nn.Module):
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

            # Positional Embedding (based on config.pos_emb_type)
            # We initialize it with n_embd dimension, as it will be added to the embedding space
            if config.pos_emb_type == 'learned':
                self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
            elif config.pos_emb_type == 'ramanujan':
                self.position_embedding = RamanujanPositionalEmbedding(
                    config.block_size, config.n_embd
                )
            elif config.pos_emb_type == 'rotary':
                head_dim = config.n_embd // config.n_heads
                self.position_embedding = RotaryPositionalEmbedding(
                    embed_dim=config.n_embd,
                    head_dim=head_dim,
                    max_seq_len=config.block_size
                )
            elif config.pos_emb_type == 'alibi':
                self.position_embedding = ALiBiPositionalBias(
                    n_heads=config.n_heads,
                    embed_dim=config.n_embd,
                    max_seq_len=config.block_size
                )
            else:
                raise ValueError(f"Unsupported pos_emb_type for SDR mode: {config.pos_emb_type}")
        else:
            # Standard embedding (direct to n_embd)
            self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
            # Positional embedding (based on config.pos_emb_type)
            if config.pos_emb_type == 'learned':
                self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
            elif config.pos_emb_type == 'ramanujan':
                self.position_embedding = RamanujanPositionalEmbedding(
                    config.block_size, config.n_embd
                )
            elif config.pos_emb_type == 'rotary':
                head_dim = config.n_embd // config.n_heads
                self.position_embedding = RotaryPositionalEmbedding(
                    embed_dim=config.n_embd,
                    head_dim=head_dim,
                    max_seq_len=config.block_size
                )
            elif config.pos_emb_type == 'alibi':
                self.position_embedding = ALiBiPositionalBias(
                    n_heads=config.n_heads,
                    embed_dim=config.n_embd,
                    max_seq_len=config.block_size
                )
            else:
                raise ValueError(f"Unsupported pos_emb_type: {config.pos_emb_type}")
            self.embed_to_sdr = None

        # Transformer blocks
        # First block receives SDR (if enabled), subsequent blocks receive n_embd
        def make_block(layer_idx):
            # Create block config from model config
            block_cfg = NeuroManifoldBlockConfig.from_model_config(config, layer_idx)

            # Override sdr_size for first block if SDR is enabled
            # First block: receives SDR if enabled, else n_embd
            # Other blocks: always receive n_embd from previous block
            if layer_idx == 0 and config.use_sdr:
                block_cfg.sdr_size = config.sdr_size
            else:
                block_cfg.sdr_size = config.n_embd

            return NeuroManifoldBlock(config=block_cfg, position_embedding=self.position_embedding)
        self.blocks = nn.ModuleList([make_block(i) for i in range(config.n_layer)])

        # mHC stream expansion/reduction (for multi-stream mHC)
        # These expand input to num_streams copies and reduce back after blocks
        mhc_disable = not config.use_mhc or config.mhc_n_streams <= 1
        self.expand_stream, self.reduce_stream = get_expand_reduce_stream_functions(
            config.mhc_n_streams, disable=mhc_disable
        )
        self.mhc_enabled = config.use_mhc and config.mhc_n_streams > 1

        # Final layer norm - RMSNorm is ~15% faster than LayerNorm (no mean computation)
        self.ln_f = RMSNorm(config.n_embd)

        # Language model head
        # FP32 for numerical stability with large vocab (MiniMax/DeepSeek recipe)
        self.lm_head_fp32 = getattr(config, 'lm_head_fp32', True)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if self.lm_head_fp32:
            # Keep lm_head in FP32 even during mixed precision training
            self.lm_head = self.lm_head.float()

        # Multi-Token Prediction (MTP) heads - DeepSeek/Meta style
        # Each head predicts token at position t+k (k=2,3,...,n_predict)
        # Head 1 (t+1) is the main lm_head, auxiliary heads predict further ahead
        self.use_mtp = getattr(config, 'use_mtp', False)
        self.mtp_n_predict = getattr(config, 'mtp_n_predict', 4)
        self.mtp_loss_weight = getattr(config, 'mtp_loss_weight', 0.1)

        # Configurable auxiliary loss weights (allows reducing auxiliary impact)
        self.ortho_loss_weight = getattr(config, 'ortho_loss_weight', 1.0)
        self.discrimination_loss_weight = getattr(config, 'discrimination_loss_weight', 0.5)
        self.contrastive_loss_weight = getattr(config, 'contrastive_loss_weight', 1.0)

        if self.use_mtp and self.mtp_n_predict > 1:
            # Auxiliary prediction heads for t+2, t+3, ..., t+n_predict
            # Each head has a small projection to adapt representations
            self.mtp_projs = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(config.n_embd, config.n_embd),
                    nn.SiLU(),
                )
                for _ in range(self.mtp_n_predict - 1)
            ])
            # Share lm_head weights for efficiency (all heads predict from vocab)
            # The projection adapts the representation, shared head does vocab projection

        # Engram memory
        self.memory = SDREngramMemory(
            sdr_size=config.sdr_size,
            capacity=config.engram_capacity,
            n_active=config.sdr_n_active,
            content_dim=config.n_embd,
            threshold=config.engram_threshold,
        )

        # ========================================
        # System 2 Reasoning Components (via Mixin)
        # ========================================
        # Initialize all System 2 components (hybrid reasoning, DAG planner,
        # hierarchical memory, imagination) through the mixin
        self._init_system_two_components(config)

        # Memory Active Retrieval configuration
        self.memory_active_retrieval = getattr(config, 'memory_active_retrieval', False)
        self.memory_retrieval_top_k = getattr(config, 'memory_retrieval_top_k', 3)
        self.memory_retrieval_weight = getattr(config, 'memory_retrieval_weight', 0.1)

        # Projection to combine retrieved memory with input (only if retrieval enabled)
        if self.memory_active_retrieval:
            # Project retrieved content to embedding dimension for mixing
            self.memory_retrieval_proj = nn.Linear(config.n_embd, config.n_embd)

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
        """Initialize linear and embedding weights.

        Uses DeepSeek-V3 style std=0.006 by default (configurable via init_std).
        Smaller initialization leads to faster early convergence.
        """
        init_std = getattr(self.config, 'init_std', 0.006)
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=init_std)

    def _check_memory_has_content(self) -> bool:
        """Check if memory contains any stored content.

        Returns:
            True if memory has content, False otherwise
        """
        if self.use_hierarchical_memory:
            total_count = len(self.hierarchical_memory.l1) + len(self.hierarchical_memory.l2) + len(self.hierarchical_memory.l3)
            return total_count > 0
        else:
            return self.memory.count.item() > 0

    def _aggregate_retrieved_content(
        self,
        contents: torch.Tensor,
        sims: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Aggregate retrieved memory contents weighted by similarity.

        Args:
            contents: Retrieved content tensors (n_results, content_dim)
            sims: Similarity scores (n_results,)
            device: Device to create tensors on

        Returns:
            Aggregated content tensor (content_dim,)
        """
        if len(contents) > 0:
            # Normalize similarities to sum to 1
            weights = F.softmax(sims, dim=0)  # (n_results,)
            # Weighted sum of contents: (n_results, content_dim) -> (content_dim,)
            aggregated = (weights.unsqueeze(1) * contents).sum(dim=0)  # (content_dim,)
            return aggregated
        else:
            # No match found - use zeros
            return torch.zeros(self.config.n_embd, device=device)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor = None,
        loss_weights: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict]:
        """
        Forward pass.

        Args:
            tokens: Input token IDs (B, T)
            targets: Target token IDs for loss computation (B, T), optional
            loss_weights: Per-token loss weights (B, T), optional
                         Used for soft curriculum weighting instead of hard masking
        """
        B, T = tokens.shape
        device = tokens.device

        # Embedding Stage
        topographic_loss = torch.tensor(0.0, device=device)
        discrimination_loss = torch.tensor(0.0, device=device)
        contrastive_loss = torch.tensor(0.0, device=device)
        if self.use_sdr:
            # Semantic folding to SDR with topographic and discrimination losses
            sdr, sdr_scores, topographic_loss, discrimination_loss, contrastive_loss = self.encoder(
                tokens
            )
            x = None  # Initial x comes from first block processing SDR
        else:
            # Standard embedding
            tok_emb = self.token_embedding(tokens)
            # Positional Embedding
            # Indices are 0..T-1
            if self.config.pos_emb_type == 'learned':
                # Learned position embedding: nn.Embedding expects (T,) and returns (T, D)
                pos_emb = self.position_embedding(
                    torch.arange(T, device=device)
                ).unsqueeze(0)  # (1, T, D)
            elif self.config.pos_emb_type == 'ramanujan':
                # Ramanujan expects (B, T) and returns (1, T, D)
                pos_emb = self.position_embedding(
                    torch.arange(T, device=device).unsqueeze(0)
                )  # (1, T, D)
            else:
                raise ValueError(f"Unsupported pos_emb_type: {self.config.pos_emb_type}")
            x = tok_emb + pos_emb  # (B, T, n_embd)
            sdr = torch.zeros(B, T, self.config.sdr_size, device=device)  # Dummy SDR
            sdr_scores = None

        # ========================================
        # Memory Active Retrieval
        # ========================================
        # Retrieve similar memories and augment representations BEFORE transformer blocks
        # This enables the model to use stored engrams during inference
        #
        # NOTE: Memory retrieval requires SDR representations for similarity matching.
        # In non-SDR mode, retrieval is disabled since there are no SDR patterns to query with.
        memory_retrieval_info = {"retrieved_count": 0, "avg_similarity": 0.0}
        pending_memory_retrieval = None  # Local variable (not instance var) for thread safety

        if self.memory_active_retrieval and self.use_sdr:
            if self._check_memory_has_content():
                # For SDR mode, we need to retrieve based on SDR patterns
                # Strategy: Use mean-pooled SDR as query (captures sequence semantics)
                # Alternative: per-position queries (more expensive but more precise)

                # Mean pool SDR across sequence for query (B, sdr_size)
                query_sdr = sdr.mean(dim=1)  # (B, sdr_size)

                # Retrieve for each batch element
                retrieved_contents_list = []
                total_similarity = 0.0
                total_retrieved = 0

                for b in range(B):
                    # Memory retrieval - via mixin helper
                    retrieval_result = self._get_memory_for_retrieval(
                        query_sdr[b],
                        top_k=self.memory_retrieval_top_k,
                        threshold=self.config.engram_threshold,
                    )
                    # Unpack results (hierarchical returns 3 values, basic returns 2)
                    if self.use_hierarchical_memory:
                        contents, sims, tier = retrieval_result
                    else:
                        contents, sims = retrieval_result

                    # Aggregate retrieved content weighted by similarity
                    aggregated = self._aggregate_retrieved_content(contents, sims, device)
                    retrieved_contents_list.append(aggregated)

                    # Track statistics
                    if len(contents) > 0:
                        total_similarity += sims.mean().item()
                        total_retrieved += len(contents)

                # Stack retrieved contents: (B, n_embd)
                retrieved_contents = torch.stack(retrieved_contents_list, dim=0)

                # Project and expand to sequence length: (B, n_embd) -> (B, T, n_embd)
                retrieved_proj = self.memory_retrieval_proj(retrieved_contents)  # (B, n_embd)
                retrieved_expanded = retrieved_proj.unsqueeze(1).expand(-1, T, -1)  # (B, T, n_embd)

                # Mix with input representations
                # For SDR mode, we'll add to the first block's output later
                # Store in local variable for application after first block produces x
                pending_memory_retrieval = retrieved_expanded

                # Update info
                memory_retrieval_info["retrieved_count"] = total_retrieved
                memory_retrieval_info["avg_similarity"] = (
                    total_similarity / B if B > 0 else 0.0
                )
        # NOTE: Memory retrieval requires SDR mode (use_sdr=True).
        # SDR engram memory stores sparse distributed representations for similarity matching.
        # In non-SDR mode, we use standard dense embeddings without sparse patterns, so there
        # are no valid SDR fingerprints to query with. Config validation now enforces that
        # memory_active_retrieval=True requires use_sdr=True.

        # Through blocks
        block_infos = []
        total_ortho_loss = torch.tensor(0.0, device=device)
        total_fhn_lyapunov_loss = torch.tensor(0.0, device=device)

        for i, block in enumerate(self.blocks):
            if self.use_sdr:
                if i == 0:
                    # First block: SDR -> x
                    # Expand for multi-stream mHC if enabled
                    sdr_in = self.expand_stream(sdr) if self.mhc_enabled else sdr

                    # Apply gradient checkpointing if enabled
                    # Trades compute for memory by recomputing activations during backward pass
                    if getattr(self.config, 'gradient_checkpointing', False) and self.training:
                        x, info = checkpoint(block, sdr_in, use_reentrant=False)
                    else:
                        x, info = block(sdr_in)

                    # Add Positional Embedding here (to x)
                    # x is (B*S, T, n_embd) if mHC, else (B, T, n_embd)
                    if self.config.pos_emb_type == 'learned':
                        # Learned position embedding: nn.Embedding expects (T,) and returns (T, D)
                        pos_emb = self.position_embedding(
                            torch.arange(T, device=device)
                        ).unsqueeze(0)  # (1, T, D)
                    elif self.config.pos_emb_type == 'ramanujan':
                        # Ramanujan expects (B, T) and returns (1, T, D)
                        pos_emb = self.position_embedding(torch.zeros(1, T, device=device))
                    else:
                        raise ValueError(f"Unsupported pos_emb_type for SDR mode: {self.config.pos_emb_type}")
                    x = x + pos_emb

                    # Apply pending memory retrieval after first block produces x
                    # This augments representations with retrieved engrams
                    if pending_memory_retrieval is not None:
                        # Handle mHC multi-stream case: expand retrieval to match
                        if self.mhc_enabled:
                            # x is (B*S, T, n_embd), need to expand retrieval
                            retrieval_expanded = self.expand_stream(pending_memory_retrieval)
                            x = x + self.memory_retrieval_weight * retrieval_expanded
                        else:
                            x = x + self.memory_retrieval_weight * pending_memory_retrieval
                        # Local variable - no need to clear, goes out of scope naturally
                else:
                    # Block already applies internal residuals
                    if getattr(self.config, 'gradient_checkpointing', False) and self.training:
                        x, info = checkpoint(block, x, use_reentrant=False)
                    else:
                        x, info = block(x)
            else:
                # Dense Mode: Pass embeddings directly
                if i == 0 and self.mhc_enabled:
                    # Expand for multi-stream mHC
                    x = self.expand_stream(x)
                # Block already applies internal residuals (x + attn_out, x + mlp_out)
                # DO NOT add another residual here - that was doubling the signal
                if getattr(self.config, 'gradient_checkpointing', False) and self.training:
                    x, info = checkpoint(block, x, use_reentrant=False)
                else:
                    x, info = block(x)

            block_infos.append(info)
            # Accumulate orthogonality regularization loss
            if "ortho_loss" in info:
                total_ortho_loss = total_ortho_loss + info["ortho_loss"]
            # Accumulate FHN Lyapunov stability loss (from theory2 proof)
            if "fhn_lyapunov_loss" in info:
                total_fhn_lyapunov_loss = total_fhn_lyapunov_loss + info["fhn_lyapunov_loss"]

        # Reduce multi-stream mHC back to single stream
        if self.mhc_enabled:
            x = self.reduce_stream(x)

        # Hybrid Reasoning (thinking/non-thinking modes) - via mixin
        x, hybrid_info = self._apply_hybrid_reasoning(x)

        # Final norm
        x = self.ln_f(x)

        # DAG Planner - System 2 task decomposition - via mixin
        dag_info = self._apply_dag_planner(x)

        # Store final representations as engrams (during training only)
        if self.training:
            # Vectorized storage - critical for performance!
            flat_sdr = sdr.view(-1, sdr.size(-1))
            flat_x = x.view(-1, x.size(-1))

            # Memory storage - via mixin helper
            self._store_to_memory(flat_sdr, flat_x.detach())

        # Compute logits
        # Use FP32 computation for lm_head if enabled (MiniMax/DeepSeek recipe)
        if self.lm_head_fp32:
            # Cast input to FP32 and disable autocast for numerical stability
            with torch.amp.autocast(device_type='cuda', enabled=False):
                logits = self.lm_head(x.float())
        else:
            logits = self.lm_head(x)

        # Compute loss if targets provided
        loss = None
        mtp_loss = torch.tensor(0.0, device=device)
        if targets is not None:
            # Get label smoothing from config (critical for large vocab like 151K Qwen3)
            label_smoothing = getattr(self.config, 'label_smoothing', 0.0)

            if loss_weights is not None:
                # Soft curriculum: weighted CE loss (per-token weights)
                per_token_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    reduction='none',  # Per-token loss
                    label_smoothing=label_smoothing,
                )
                # Apply weights and compute weighted mean
                flat_weights = loss_weights.view(-1)
                ce_loss = (per_token_loss * flat_weights).sum() / flat_weights.sum()
            else:
                # Standard CE loss with ignore_index for hard masking
                ce_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-100,
                    label_smoothing=label_smoothing,
                )

            # Multi-Token Prediction (MTP) auxiliary losses
            # Predict tokens at t+2, t+3, ..., t+n_predict
            if self.use_mtp and self.mtp_n_predict > 1 and hasattr(self, 'mtp_projs'):
                B, T, _ = x.shape
                for k, proj in enumerate(self.mtp_projs, start=2):
                    # k=2 means predicting t+2 (2 tokens ahead)
                    if T > k:
                        # Project hidden states for this prediction depth
                        x_proj = proj(x[:, :-k, :])  # (B, T-k, D)
                        # Get logits using shared lm_head (FP32 if enabled)
                        if self.lm_head_fp32:
                            with torch.amp.autocast(device_type='cuda', enabled=False):
                                mtp_logits = self.lm_head(x_proj.float())  # (B, T-k, V)
                        else:
                            mtp_logits = self.lm_head(x_proj)  # (B, T-k, V)
                        # Targets shifted by k (predict k tokens ahead)
                        mtp_targets = targets[:, k:]  # (B, T-k)
                        # Compute loss for this depth
                        mtp_k_loss = F.cross_entropy(
                            mtp_logits.reshape(-1, mtp_logits.size(-1)),
                            mtp_targets.reshape(-1),
                            ignore_index=-100,  # Match E7 curriculum's ignore_index
                            label_smoothing=label_smoothing,
                        )
                        mtp_loss = mtp_loss + mtp_k_loss
                # Average over prediction depths
                mtp_loss = mtp_loss / (self.mtp_n_predict - 1)

            # =================================================================
            # HYBRID LOSS (proven valid in theory2 mathematical proof)
            # L_total = L_CE + α*L_ortho + β*L_disc + γ*L_topo + δ*L_mtp + ε*L_fhn
            # =================================================================
            # - ortho_loss: spectral orthogonality (manifold regularization)
            # - discrimination_loss: prevent SDR mode collapse (CRITICAL for SDR mode)
            # - topographic_loss: E7-inspired manifold organization (with warmup)
            # - mtp_loss: multi-token prediction (DeepSeek/Meta style)
            # - fhn_lyapunov_loss: FHN stability loss (penalize energy increase)
            #
            # Get topographic weight from model attribute (set by trainer)
            topo_weight = getattr(self, '_topo_weight', 0.0)
            # FHN Lyapunov weight (small, for regularization)
            fhn_lyapunov_weight = getattr(self, '_fhn_lyapunov_weight', 0.001)

            loss = (
                ce_loss
                + self.ortho_loss_weight * total_ortho_loss
                + self.discrimination_loss_weight * discrimination_loss
                + topo_weight * topographic_loss
                + self.mtp_loss_weight * mtp_loss
                + fhn_lyapunov_weight * total_fhn_lyapunov_loss
                + self.contrastive_loss_weight * contrastive_loss
            )
        else:
            loss = None
            topographic_loss = torch.tensor(0.0, device=device)
            discrimination_loss = torch.tensor(0.0, device=device)

        # Memory stats (hierarchical or basic)
        if self.use_hierarchical_memory:
            memory_stats = self.hierarchical_memory.get_stats()
        else:
            memory_stats = {"memory_size": self.memory.get_size()}

        info = {
            "sdr": sdr,
            "sdr_scores": sdr_scores,
            "block_infos": block_infos,
            "memory_size": memory_stats.get("memory_size", memory_stats.get("l1_count", 0) + memory_stats.get("l2_count", 0) + memory_stats.get("l3_count", 0)),
            "memory_stats": memory_stats,
            "memory_retrieval": memory_retrieval_info,  # Active retrieval stats
            "ortho_loss": total_ortho_loss,
            "topographic_loss": topographic_loss,
            "discrimination_loss": discrimination_loss,
            "contrastive_loss": contrastive_loss,  # wav2vec-style contrastive embedding loss
            "mtp_loss": mtp_loss,
            "fhn_lyapunov_loss": total_fhn_lyapunov_loss,
            "ce_loss": ce_loss if targets is not None else torch.tensor(0.0, device=device),
            "hybrid_reasoning": hybrid_info,
            "dag_planner": dag_info,
        }

        return logits, loss, info

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.2,
        mirostat: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
    ) -> torch.Tensor:
        """
        Autoregressive generation with multiple sampling strategies.

        Args:
            idx: Input token IDs (B, T)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top-k tokens for sampling
            top_p: Nucleus sampling threshold (cumulative probability)
            repetition_penalty: Penalty for repeating tokens (1.0 = no penalty)
            mirostat: 0=disabled, 2=Mirostat 2 (dynamic perplexity control)
            mirostat_tau: Target surprise/perplexity (lower = more focused)
            mirostat_eta: Learning rate for Mirostat adjustment
        """
        # Mirostat 2 state (per batch element)
        mu = torch.full((idx.size(0),), 2 * mirostat_tau, device=idx.device)

        for _ in range(max_new_tokens):
            # Crop to block_size if needed
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )

            # Forward
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :]

            # Apply repetition penalty to tokens already in sequence
            if repetition_penalty != 1.0:
                # Vectorized repetition penalty using scatter (replaces O(B*T) Python loops)
                # idx shape: (B, T), logits shape: (B, vocab_size)
                batch_size, vocab_size = logits.shape

                # Create a mask for tokens that appear in the sequence (B, vocab_size)
                # scatter_ sets True for any token that appears in idx
                token_mask = torch.zeros(batch_size, vocab_size, device=idx.device, dtype=torch.bool)
                token_mask.scatter_(1, idx, True)

                # Apply penalty: divide if logit > 0 (decrease probability),
                # multiply if logit < 0 (make even more negative)
                # This penalizes repeated tokens
                penalty_factor = torch.where(
                    logits > 0,
                    torch.full_like(logits, 1.0 / repetition_penalty),
                    torch.full_like(logits, repetition_penalty)
                )

                # Apply penalty only to tokens that appeared in sequence
                logits = torch.where(token_mask, logits * penalty_factor, logits)

            # Temperature
            logits = logits / temperature

            if mirostat == 2:
                # Mirostat 2: Dynamic top-k based on target perplexity
                # Sort logits descending
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)

                # Compute surprise (negative log prob) for each token
                surprisals = -torch.log(probs + 1e-10)

                # Find k where cumulative surprise exceeds mu (target)
                # This dynamically selects how many tokens to consider
                idx_next_list = []
                for b in range(idx.size(0)):
                    # Find truncation point
                    cum_surprise = torch.cumsum(probs[b] * surprisals[b], dim=0)
                    k = (cum_surprise < mu[b]).sum().item() + 1
                    k = max(1, min(k, logits.size(-1)))

                    # Truncate and renormalize
                    truncated_probs = probs[b, :k]
                    truncated_probs = truncated_probs / truncated_probs.sum()

                    # Sample
                    sampled_idx = torch.multinomial(truncated_probs, num_samples=1)
                    token_idx = sorted_indices[b, sampled_idx]
                    idx_next_list.append(token_idx)

                    # Update mu based on observed surprise
                    observed_surprise = surprisals[b, sampled_idx].item()
                    mu[b] = mu[b] - mirostat_eta * (observed_surprise - mirostat_tau)

                idx_next = torch.cat(idx_next_list, dim=0).unsqueeze(1)
            else:
                # Standard sampling with top-k and/or top-p
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")

                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    probs = F.softmax(sorted_logits, dim=-1)
                    cum_probs = torch.cumsum(probs, dim=-1)
                    # Remove tokens with cumulative prob above threshold
                    sorted_mask = cum_probs > top_p
                    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                    sorted_mask[..., 0] = False
                    for b in range(logits.size(0)):
                        logits[b, sorted_indices[b][sorted_mask[b]]] = float("-inf")

                # Sample
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, idx_next], dim=1)

        return idx

    # imagine_alternatives() and plan_task() methods are now provided by SystemTwoReasoningMixin

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
