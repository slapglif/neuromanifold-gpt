"""Standard GPT-2 124M configuration for benchmarking.

This configuration disables all NeuroManifold features and uses standard
transformer attention to establish a baseline for performance comparison.

Architecture matches GPT-2 124M:
- 12 layers
- 768 embedding dimension
- 12 attention heads (64-dim per head)
- 1024 context length
- ~124M parameters
"""

from neuromanifold_gpt.config.base import NeuroManifoldConfig


# GPT-2 124M baseline configuration
# Disables all neuromanifold features for fair comparison
out = dict(
    # Model architecture - GPT-2 124M spec
    n_layer=12,
    n_embd=768,
    n_heads=12,
    block_size=1024,
    vocab_size=50304,  # GPT-2 vocab (50257) padded to nearest multiple of 64

    # Disable all NeuroManifold features
    use_sdr=False,  # Use standard dense embeddings
    use_multiscale_manifold=False,  # No manifold projection
    skip_manifold_spectral=True,  # Skip manifold/spectral computation
    use_fhn_parallel=False,  # No FHN soliton dynamics
    use_mhc=False,  # No manifold-constrained hyper-connections
    use_knot_attention=False,  # No knot-theoretic attention
    use_kaufmann_attention=False,  # No Kaufmann trifecta attention
    use_qk_norm=False,  # Standard attention (no QK normalization)
    use_kan=False,  # Use standard Linear layers (no KAN)
    use_mtp=False,  # Single-token prediction only
    use_mla=False,  # Standard KV cache (no compression)
    use_moe=False,  # Dense model (no mixture of experts)
    use_hybrid_reasoning=False,  # No thinking/non-thinking split
    use_dag_planner=False,  # No DAG decomposition
    use_hierarchical_memory=False,  # No engram memory
    use_imagination=False,  # No counterfactual exploration
    memory_active_retrieval=False,  # No memory retrieval

    # Fast mode settings
    fast_mode=True,  # Enable fast path optimizations
    skip_context_encoder=True,  # No SDR context encoding
    skip_semantic_retina=True,  # No Gaussian smoothing
    skip_metric_tensor=True,  # No manifold metric

    # Standard transformer settings
    dropout=0.0,  # No dropout (modern practice)
    bias=False,  # No bias in linear layers (GPT-2 style)

    # Training configuration (from config/train_gpt2.py)
    learning_rate=6e-4,  # GPT-2 standard
    weight_decay=1e-1,
    beta1=0.9,
    beta2=0.95,
    grad_clip=1.0,

    # Label smoothing and stability
    label_smoothing=0.0,  # No label smoothing for baseline
    lm_head_fp32=True,  # FP32 head for numerical stability

    # Learning rate schedule
    lr_schedule="cosine",  # Standard cosine decay
    warmup_ratio=0.05,

    # Initialization
    init_std=0.02,  # Standard GPT-2 initialization
)

# Create config instance with all settings
config = NeuroManifoldConfig(**out)
