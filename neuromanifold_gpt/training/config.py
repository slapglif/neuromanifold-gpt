"""Training configuration dataclasses for NeuroManifoldGPT.

This module defines the training configuration structure, including settings for:
- I/O and checkpointing
- Data loading and batching
- Model architecture parameters
- Optimization and learning rate scheduling
- Hardware and precision settings
- Logging and monitoring
"""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    """Training configuration with all hyperparameters.

    This configuration covers all aspects of the training process:
    - I/O paths and checkpoint intervals
    - Data loading (batch size, workers, streaming)
    - Model architecture (layers, heads, embedding dimensions)
    - NeuroManifold-specific features (SDR, manifolds, FHN dynamics)
    - Optimization (learning rate, weight decay, gradient clipping)
    - Learning rate schedule (warmup, decay)
    - Early stopping criteria
    - Sampling during training
    - Hardware configuration (devices, precision)
    - Logging and experiment tracking

    Attributes:
        out_dir: Output directory for checkpoints and logs
        eval_interval: Evaluate every N training steps
        log_interval: Log metrics every N training steps
        eval_iters: Number of iterations for evaluation
        save_checkpoints: Whether to save model checkpoints
        save_separate_optimizer: Save optimizer state separately from model
        save_model_only: Save only model weights without optimizer state
        dataset: Dataset name (e.g., "shakespeare_char")
        batch_size: Training batch size per device
        block_size: Maximum sequence length (context window)
        num_workers: Number of data loading workers
        streaming: Use HuggingFace streaming for large datasets
        vocab_size: Vocabulary size (0 = auto-detect)
        model_type: Model architecture ("neuromanifold" or "gpt")
        n_layer: Number of transformer layers
        n_head: Number of attention heads
        n_embd: Embedding dimension
        dropout: Dropout probability
        bias: Whether to use bias in linear layers
        sdr_size: Size of SDR binary vectors (bits)
        manifold_dim: Dimension of the learned manifold space
        n_eigenvectors: Number of eigenvectors for spectral attention
        use_sdr: Enable SDR encoding
        use_kan: Enable KAN (Kolmogorov-Arnold Network) layers
        kan_type: KAN type ("faster", "wave", or "cheby")
        kan_wavelet: Wavelet type for WaveKAN
        use_fast_wavekan: Use efficient WaveKAN implementation
        kan_num_centers: Number of centers for FasterKAN
        fhn_threshold: Firing threshold for FHN dynamics
        fhn_tau: Time constant for FHN dynamics
        n_fhn_steps: Number of FHN integration steps
        use_fhn_imex: Use semi-implicit IMEX scheme for FHN
        use_fhn_partitioning: Enable energy balancing for FHN
        use_fhn_fused: Use fused FHN kernels
        use_mhc: Enable manifold-constrained hyper-connections
        use_full_mhc: Use full multi-stream mHC
        mhc_n_streams: Number of parallel streams for mHC
        use_kaufmann_attention: Enable Kaufmann Trifecta attention
        skip_manifold_spectral: Skip manifold/spectral for faster training
        max_iters: Maximum number of training iterations
        gradient_accumulation_steps: Accumulate gradients over N steps
        learning_rate: Initial learning rate
        min_lr: Minimum learning rate for decay
        weight_decay: Weight decay for AdamW optimizer
        beta1: AdamW beta1 parameter
        beta2: AdamW beta2 parameter
        grad_clip: Gradient clipping norm
        warmup_iters: Number of warmup iterations
        lr_decay_iters: Number of iterations for LR decay
        early_stopping_patience: Patience for early stopping (epochs)
        sample_interval: Generate samples every N steps
        sample_max_tokens: Maximum tokens to generate per sample
        sample_temperature: Sampling temperature
        sample_top_k: Top-k sampling parameter
        devices: Number of GPU devices to use
        precision: Training precision ("bf16-mixed", "fp16-mixed", "32")
        compile_model: Whether to compile model with torch.compile
        wandb_log: Enable Weights & Biases logging
        wandb_project: W&B project name
        wandb_run_name: W&B run name
        enable_sdr_collapse_monitor: Enable SDR collapse detection callback
        sdr_check_interval: Steps between SDR health checks
        sdr_collapse_threshold: Unique pattern ratio threshold for collapse detection
        enable_divergence_rollback: Enable automatic checkpoint rollback on divergence
        divergence_threshold: Loss multiplier to trigger divergence detection
        rollback_checkpoint_interval: Steps between rollback checkpoint saves
        enable_attention_viz: Enable periodic attention pattern visualization
        attention_viz_interval: Steps between attention visualization saves
        attention_viz_max_seq_len: Maximum sequence length to visualize
    """

    # I/O
    out_dir: str = "out-lightning"
    eval_interval: int = 2000
    log_interval: int = 10
    eval_iters: int = 200
    save_checkpoints: bool = True
    save_separate_optimizer: bool = False
    save_model_only: bool = False

    # Data
    dataset: str = "shakespeare_char"
    batch_size: int = 64
    block_size: int = 256
    num_workers: int = 4
    streaming: bool = False
    vocab_size: int = 0

    # Model
    model_type: str = "neuromanifold"
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = False

    # NeuroManifold specific
    sdr_size: int = 2048
    manifold_dim: int = 64
    n_eigenvectors: int = 32
    use_sdr: bool = False
    use_kan: bool = True
    kan_type: str = "faster"
    kan_wavelet: str = "dog"
    use_fast_wavekan: bool = True
    kan_num_centers: int = 3
    fhn_threshold: float = 0.5
    fhn_tau: float = 12.5
    n_fhn_steps: int = 2
    use_fhn_imex: bool = True
    use_fhn_partitioning: bool = True
    use_fhn_fused: bool = False
    use_mhc: bool = True
    use_full_mhc: bool = True
    mhc_n_streams: int = 2
    use_kaufmann_attention: bool = False

    # Speed optimization
    skip_manifold_spectral: bool = False

    # Training
    max_iters: int = 5000
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_iters: int = 100
    lr_decay_iters: int = 5000

    # Early stopping
    early_stopping_patience: int = 50

    # Sampling during training
    sample_interval: int = 500
    sample_max_tokens: int = 200
    sample_temperature: float = 1.0
    sample_top_k: int = 40

    # Hardware
    devices: int = 1
    precision: str = "bf16-mixed"
    compile_model: bool = False

    # Logging
    wandb_log: bool = False
    wandb_project: str = "neuromanifold-gpt"
    wandb_run_name: str = "neuromanifold"

    # Stability Toolkit
    enable_sdr_collapse_monitor: bool = False
    sdr_check_interval: int = 100
    sdr_collapse_threshold: float = 0.3
    enable_divergence_rollback: bool = False
    divergence_threshold: float = 2.0
    rollback_checkpoint_interval: int = 500
    enable_attention_viz: bool = False
    attention_viz_interval: int = 500
    attention_viz_max_seq_len: int = 64
