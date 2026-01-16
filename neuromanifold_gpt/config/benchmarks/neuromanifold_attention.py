"""NeuroManifold GPT-2 124M configuration with FHN/SDR for benchmarking.

This configuration enables NeuroManifold features including:
- SDR (Sparse Distributed Representation) encoding
- FHN (Fitzhugh-Nagumo) soliton attention dynamics
- Manifold projection with spectral decomposition
- Multi-scale manifold attention (E7 subgroup chain)

Architecture matches GPT-2 124M scale:
- 12 layers
- 768 embedding dimension
- 12 attention heads (64-dim per head)
- 1024 context length
- ~124M parameters (plus NeuroManifold components)
"""

from neuromanifold_gpt.config.base import NeuroManifoldConfig


# GPT-2 124M with NeuroManifold features
out = dict(
    # Model architecture - GPT-2 124M spec
    n_layer=12,
    n_embd=768,
    n_heads=12,
    block_size=1024,
    vocab_size=50304,  # GPT-2 vocab (50257) padded to nearest multiple of 64

    # Enable SDR (Sparse Distributed Representation)
    use_sdr=True,
    sdr_size=2048,  # SDR vector size (bits)
    sdr_sparsity=0.02,  # 2% sparsity (biological plausibility)
    sdr_embed_dim=256,  # SDR embedding dimension
    sdr_context_size=5,  # Context window for semantic folding

    # Enable multi-scale manifold projection (E7 → E6 → D5)
    use_multiscale_manifold=True,
    multiscale_coarse_dim=16,  # D5 level (global patterns)
    multiscale_medium_dim=32,  # E6 level (phrase patterns)
    multiscale_fine_dim=64,  # E7 level (token patterns)
    manifold_dim=64,
    n_neighbors=15,
    n_eigenvectors=32,
    spectral_sigma=1.0,
    skip_manifold_spectral=False,  # Enable manifold/spectral computation

    # Enable FHN (Fitzhugh-Nagumo) soliton attention dynamics
    fhn_threshold=0.5,  # Excitability threshold
    fhn_tau=12.5,  # Time constant for slow-fast separation
    fhn_velocity=1.0,  # Soliton propagation velocity
    pulse_width_base=4,  # Soliton pulse width
    n_fhn_steps=2,  # IMEX integration steps
    use_fhn_imex=True,  # Semi-implicit IMEX scheme
    use_fhn_partitioning=True,  # Energy balancing for stability
    use_fhn_fused=False,  # Use JIT instead
    use_fhn_parallel=True,  # FFT-based parallel scan for speed

    # Enable mHC (Manifold-Constrained Hyper-Connections)
    use_mhc=True,
    use_full_mhc=True,
    mhc_n_streams=2,
    mhc_residual_weight=0.9,
    mhc_sinkhorn_iters=5,

    # Standard attention settings
    use_knot_attention=False,  # Disable advanced attention variants for baseline
    use_kaufmann_attention=False,
    use_qk_norm=True,  # QK normalization (Qwen3/GLM-4.5 style)

    # Enable KAN (Kolmogorov-Arnold Networks) for FFN
    use_kan=True,
    kan_type="faster",  # FasterKAN (RSWAF basis) for speed
    kan_wavelet="dog",  # Difference of Gaussians (fastest, stable)
    use_fast_wavekan=True,
    kan_num_centers=3,
    use_kan_everywhere=False,  # KAN only in FFN, not projections

    # Enable Multi-Token Prediction (DeepSeek/Meta style)
    use_mtp=True,
    mtp_n_predict=4,  # Predict 4 future tokens
    mtp_loss_weight=0.1,

    # Disable advanced features (for focused FHN/SDR benchmark)
    use_mla=False,  # No KV compression
    use_moe=False,  # No mixture of experts
    use_hybrid_reasoning=False,  # No thinking mode
    use_dag_planner=False,  # No DAG decomposition
    use_hierarchical_memory=False,  # No engram memory
    use_imagination=False,  # No counterfactual exploration
    memory_active_retrieval=False,  # No memory retrieval

    # Fast mode disabled (to use full NeuroManifold features)
    fast_mode=False,
    skip_context_encoder=False,  # Enable SDR context encoding
    skip_semantic_retina=False,  # Enable Gaussian smoothing
    skip_metric_tensor=False,  # Enable manifold metric

    # Standard transformer settings
    dropout=0.0,  # No dropout (modern practice)
    bias=False,  # No bias in linear layers (GPT-2 style)

    # Spectral regularization
    ortho_weight=0.01,

    # Training configuration
    learning_rate=6e-4,  # GPT-2 standard
    weight_decay=1e-1,
    beta1=0.9,
    beta2=0.95,
    optimizer_eps=1e-15,
    grad_clip=1.0,

    # Label smoothing and stability
    label_smoothing=0.0,
    lm_head_fp32=True,  # FP32 head for numerical stability

    # Learning rate schedule
    lr_schedule="cosine",  # Standard cosine decay
    warmup_ratio=0.05,

    # Initialization
    init_std=0.02,  # Standard GPT-2 initialization
)

# Create config instance with all settings
config = NeuroManifoldConfig(**out)
