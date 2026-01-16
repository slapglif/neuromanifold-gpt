"""Configuration dataclasses for NeuroManifoldGPT.

This module defines the configuration structure for the NeuroManifoldGPT model,
including settings for:
- SDR (Sparse Distributed Representation) encoding
- Manifold projection and spectral decomposition
- Soliton attention dynamics
- Engram memory hierarchy
- DAG planning and imagination modules
"""

from dataclasses import dataclass, field


@dataclass
class NeuroManifoldConfig:
    """Main configuration for NeuroManifoldGPT model.

    This configuration covers all aspects of the novel architecture:
    - Token embedding via Semantic Folding (SDR)
    - Manifold projection with learned Riemannian metric
    - Spectral attention via eigendecomposition
    - FHN-based attention propagation (Fitzhugh-Nagumo)
    - SDR Engram memory for infinite context
    - Forced DAG planning for System 2 thinking
    - Consistency model imagination module

    Attributes:
        vocab_size: Vocabulary size (default 50304 for GPT-2 + padding)
        block_size: Maximum sequence length (context window)
        sdr_size: Size of SDR binary vectors (bits)
        sdr_sparsity: Target sparsity ratio (~2% for biological plausibility)
        sdr_n_active: Number of active bits (computed from size * sparsity)
        sdr_embed_dim: Embedding dimension for SDR projection
        sdr_context_size: Context window for semantic folding
        manifold_dim: Dimension of the learned manifold space
        n_neighbors: Number of neighbors for manifold construction
        n_eigenvectors: Number of eigenvectors for spectral attention
        spectral_sigma: Bandwidth for spectral kernel
        n_heads: Number of attention heads
        soliton_threshold: Firing threshold for soliton activation
        soliton_tau: Time constant for soliton dynamics
        soliton_velocity: Propagation velocity of soliton waves
        pulse_width_base: Base width of soliton pulses
        n_layer: Number of transformer layers
        n_embd: Embedding dimension (must be divisible by n_heads)
        dropout: Dropout probability
        bias: Whether to use bias in linear layers
        engram_capacity: Maximum engrams in memory
        engram_threshold: SDR overlap threshold for retrieval
        l1_capacity: L1 (working memory) capacity
        l2_capacity: L2 (short-term) capacity
        l3_capacity: L3 (long-term) capacity
        max_dag_depth: Maximum depth of task DAG
        imagination_steps: Number of diffusion steps for imagination
        imagination_dim: Dimension of imagination latent space
        learning_rate: Learning rate for optimizer (default 3e-4)
        weight_decay: Weight decay for AdamW (default 0.1)
        beta1: AdamW beta1 parameter (default 0.9)
        beta2: AdamW beta2 parameter (default 0.95, MiniMax: faster adaptation)
        optimizer_eps: AdamW epsilon (default 1e-15, MiniMax: handles tiny gradients)
        grad_clip: Gradient clipping norm (default 1.0)
    """

    # Vocabulary and sequence
    vocab_size: int = 50304
    block_size: int = 1024

    # Position embedding type: 'learned', 'ramanujan', 'rotary', or 'alibi'
    pos_emb_type: str = 'learned'

    use_sdr: bool = False  # Dense embeddings (faster, less memory) - SDR disabled
    sdr_size: int = 2048
    sdr_sparsity: float = 0.02 # Restored to 2% sparsity for stability
    sdr_n_active: int = field(init=False)  # Computed in __post_init__
    sdr_embed_dim: int = 256
    sdr_context_size: int = 5

    # Manifold projection configuration
    manifold_dim: int = 64
    n_neighbors: int = 15
    n_eigenvectors: int = 32
    spectral_sigma: float = 1.0

    # Phase 2: Multi-scale manifold (E7 subgroup chain: E7 → E6 → D5)
    use_multiscale_manifold: bool = True  # Progressive coarse-to-fine scales
    multiscale_coarse_dim: int = 16  # D5 level (global patterns)
    multiscale_medium_dim: int = 32  # E6 level (phrase patterns)
    multiscale_fine_dim: int = 64  # E7 level (token patterns)

    # Attention configuration
    n_heads: int = 8

    # FHN dynamics (Fitzhugh-Nagumo)
    fhn_threshold: float = 0.5  # Restored: stable excitability threshold
    fhn_tau: float = 12.5  # Fixed: proper slow-fast separation (was 0.1)
    fhn_velocity: float = 1.0
    pulse_width_base: int = 4
    n_fhn_steps: int = 2  # Restored to 2 steps (IMEX), better dynamics
    use_fhn_imex: bool = True  # Use semi-implicit IMEX scheme
    use_fhn_partitioning: bool = True  # Enable energy balancing for stability
    use_fhn_fused: bool = False  # Disabled (Using JIT instead)
    use_fhn_parallel: bool = True  # Use FFT-based Parallel Scan (Linearized FHN) for max speed
    
    # Spectral regularization
    ortho_weight: float = 0.01

    # Loss weights
    ortho_loss_weight: float = 0.01
    discrimination_loss_weight: float = 0.1
    contrastive_loss_weight: float = 0.1

    # Speed optimization
    skip_manifold_spectral: bool = False  # Skip manifold/spectral for faster training

    # mHC (Manifold-Constrained Hyper-Connections) from DeepSeek
    # Reference: https://arxiv.org/abs/2512.24880
    # Architecture: x_{l+1} = H_res @ x_l + H_post^T @ F(H_pre @ x_l)
    # - H_res: Doubly stochastic via Sinkhorn-Knopp (Birkhoff polytope)
    # - H_pre/H_post: Softmax over streams for routing
    use_mhc: bool = True  # Enable mHC for training stability
    use_full_mhc: bool = True  # Use full multi-stream mHC (vs simplified)
    mhc_n_streams: int = 2  # Number of parallel streams for full mHC (2 for efficiency)
    mhc_residual_weight: float = 0.9  # Initial identity mapping bias
    mhc_sinkhorn_iters: int = 5  # Sinkhorn-Knopp iterations (3-5 sufficient for convergence)
    mhc_sinkhorn_tau: float = 0.05  # Sinkhorn temperature for smoothness
    use_mhc_fused: bool = False  # Use Triton-optimized fused mHC operations for performance

    # Attention configuration
    attention_type: str = "fhn"  # Attention mechanism: "fhn", "kaufmann", or "knot"
    use_knot_attention: bool = False  # Enable Knot-Theoretic attention
    use_kaufmann_attention: bool = False  # Enable Kaufmann Trifecta Attention (The Endgame)
    use_qk_norm: bool = True  # Qwen3/GLM-4.5: RMSNorm on Q,K prevents attention logit explosion

    # KAN configuration
    # FFN/MLP uses FasterKAN (RSWAF basis) for speed
    # Wave-based embedding aligns with FHN soliton attention conceptually
    use_kan: bool = True
    kan_type: str = "faster"  # "faster" (RSWAF for FFN), "wave", or "cheby"
    kan_degree: int = 4  # For ChebyKAN
    kan_wavelet: str = "dog"  # "dog" is fastest (linear) and stable
    use_fast_wavekan: bool = True  # Use efficient WaveKAN (shared scale/trans)
    kan_num_centers: int = 3  # For FasterKAN RSWAF basis centers (3 for efficiency)

    # Replace ALL nn.Linear MLP layers with FasterKAN (not just FFN)
    # Applies to: manifold projection, spectral decomposition, attention projections
    # Skips: lm_head (vocab output), embeddings
    # WARNING: This causes parameter bloat - disable for efficiency
    use_kan_everywhere: bool = False  # Keep Linear for projections, FasterKAN only for FFN

    # Model architecture
    n_layer: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = False

    # Engram memory configuration
    engram_capacity: int = 1000
    engram_threshold: float = 0.3
    l1_capacity: int = 100
    l2_capacity: int = 500
    l3_capacity: int = 1000

    # Memory active retrieval (enables using stored engrams during forward pass)
    # When True, retrieves similar memories and augments token representations
    # This transforms memory from "write-only log" to active retrieval-augmented component
    memory_active_retrieval: bool = False  # Off by default for backwards compatibility
    memory_retrieval_top_k: int = 3  # Number of memories to retrieve per query
    memory_retrieval_weight: float = 0.1  # Weight for mixing retrieved content (0-1)

    # DAG planning and imagination
    max_dag_depth: int = 8
    imagination_steps: int = 4
    imagination_dim: int = 256

    # Fast mode for performance optimization
    fast_mode: bool = False  # Enable all fast-path optimizations
    skip_context_encoder: bool = False  # Skip local attention in SDR encoder
    skip_semantic_retina: bool = False  # Skip Gaussian smoothing
    skip_metric_tensor: bool = False  # Skip manifold metric computation

    # Multi-Token Prediction (MTP) - DeepSeek/Meta style
    # Predicting multiple future tokens improves representation learning
    use_mtp: bool = True  # Enable multi-token prediction
    mtp_n_predict: int = 4  # Number of future tokens to predict (1=standard, 4=recommended)
    mtp_loss_weight: float = 0.1  # Weight for auxiliary MTP losses (main loss weight=1.0)

    # MLA (Multi-Head Latent Attention) - DeepSeek style KV compression
    # Compresses KV cache to low-dimensional latent space for 8x memory reduction
    use_mla: bool = False  # Off by default (adds complexity)
    mla_latent_dim: int = 64  # KV compression dimension
    mla_rope_dim: int = 32  # Decoupled RoPE dimension

    # MoE (Mixture of Experts) - DeepSeek style auxiliary-loss-free
    # Uses bias-based load balancing without auxiliary loss terms
    use_moe: bool = False  # Off by default (increases params significantly)
    moe_n_experts: int = 8  # Total number of experts
    moe_n_active: int = 2  # Number of active experts per token
    use_shared_expert: bool = True  # Always-active shared expert (DeepSeek style)
    use_e7_routing: bool = False  # Route by E7 curriculum tier

    # Hybrid Reasoning (Qwen3 style) - Thinking/Non-thinking modes
    # Routes between fast (direct) and slow (thinking) paths based on complexity
    use_hybrid_reasoning: bool = False  # Off by default
    n_thinking_layers: int = 2  # Extra layers for thinking mode
    thinking_threshold: float = 0.5  # Mode selection threshold

    # ========================================
    # System 2 Reasoning Components
    # ========================================

    # ForcedDAGPlanner - Decomposes tasks into DAGs for systematic reasoning
    # Forces deliberate thinking instead of pure pattern matching
    use_dag_planner: bool = False  # Off by default (adds compute)
    dag_max_nodes: int = 32  # Maximum nodes in task DAG
    dag_min_nodes: int = 3  # Minimum nodes (forces decomposition)

    # HierarchicalEngramMemory - L1/L2/L3 tiered memory system
    # Replaces SDREngramMemory for better memory management
    use_hierarchical_memory: bool = False  # Off by default
    hierarchical_l1_capacity: int = 64  # Hot memory (fast)
    hierarchical_l2_capacity: int = 512  # Warm memory (medium)
    hierarchical_l3_capacity: int = 4096  # Cold memory (compressed)

    # ConsistencyImaginationModule - Counterfactual exploration
    # Lightweight diffusion for "mental whiteboard" exploration
    use_imagination: bool = False  # Off by default
    imagination_steps: int = 4  # Denoising steps (2-4 recommended)
    imagination_n_alternatives: int = 4  # Number of counterfactuals to generate

    # Training configuration
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95  # MiniMax: lower than default 0.999 for faster adaptation
    optimizer_eps: float = 1e-15  # MiniMax critical: handles tiny gradients (default 1e-8 masks them)
    grad_clip: float = 1.0
    early_stopping_patience: int = 5
    use_perplexity_stopping: bool = True

    # Gradient checkpointing - trades compute for memory
    gradient_checkpointing: bool = False  # Disabled by default

    # Learning Rate Schedule (MiniMax/DeepSeek WSD)
    # WSD (Warmup-Stable-Decay) provides better final loss than cosine
    # Reference: MiniMax-01 and DeepSeek-V3 technical reports
    lr_schedule: str = "wsd"  # "wsd" or "cosine"
    warmup_ratio: float = 0.05  # 5% warmup phase
    stable_ratio: float = 0.65  # 65% stable at peak LR
    decay_ratio: float = 0.30  # 30% linear decay to min_lr

    # Label smoothing - critical for large vocab (e.g., 151K Qwen3)
    # Softens one-hot targets to prevent overconfident predictions
    # Recommended: 0.1 for 50K vocab, 0.15-0.2 for 150K+ vocab
    label_smoothing: float = 0.0  # Disabled by default for backwards compat

    # FP32 LM Head - MiniMax/DeepSeek style numerical stability
    # Keep lm_head in FP32 for large vocabularies to prevent gradient underflow/overflow
    # Critical for 151K Qwen3 vocab - the final projection layer is most sensitive
    lm_head_fp32: bool = True  # Default True for safety with large vocab

    # Weight initialization (DeepSeek-V3 style)
    # DeepSeek uses std=0.006 instead of typical 0.02 for faster early convergence
    init_std: float = 0.006

    # Initialization strategy selection
    # Options: 'gpt2' (standard), 'gpt3' (scaled residual), 'mup' (maximal update parametrization)
    init_strategy: str = "gpt2"  # Default to GPT-2 style for backward compatibility
    mup_base_width: int = 128  # Base width for muP hyperparameter transfer

    def __post_init__(self) -> None:
        """Validate configuration and compute derived values."""
        # Fast mode enables all fast-path optimizations
        if self.fast_mode:
            self.skip_context_encoder = True
            self.skip_semantic_retina = True
            self.skip_metric_tensor = True
            self.n_fhn_steps = 1  # Reduce FHN steps
            self.sdr_size = min(self.sdr_size, 512)  # Cap SDR size

        # Compute number of active SDR bits
        self.sdr_n_active = int(self.sdr_size * self.sdr_sparsity)

        # Validate n_embd is divisible by n_heads
        if self.n_embd % self.n_heads != 0:
            from neuromanifold_gpt.errors import ConfigurationError
            raise ConfigurationError(
                f"n_embd ({self.n_embd}) must be divisible by n_heads ({self.n_heads})"
            )

        # Memory active retrieval requires SDR mode
        if self.memory_active_retrieval and not self.use_sdr:
            raise ValueError(
                "memory_active_retrieval=True requires use_sdr=True. "
                "Enable use_sdr=True or disable retrieval."
            )

    @property
    def head_dim(self) -> int:
        """Compute dimension per attention head."""
        return self.n_embd // self.n_heads


@dataclass
class NeuroManifoldConfigNano(NeuroManifoldConfig):
    """Nano preset for fast experimentation and testing.

    This is a minimal configuration suitable for:
    - Quick iteration during development
    - Unit testing
    - Resource-constrained environments
    - Debugging model components

    Compared to base config:
    - Reduced block_size (256 vs 1024)
    - Fewer layers (4 vs 6)
    - Smaller embedding (128 vs 384)
    - Fewer heads (4 vs 8)
    - Reduced manifold dimensions
    - Smaller SDR size
    """

    block_size: int = 256
    n_layer: int = 4
    n_heads: int = 4
    n_embd: int = 128
    manifold_dim: int = 32
    n_eigenvectors: int = 16
    sdr_size: int = 1024
