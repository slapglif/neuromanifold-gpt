"""Ralph Loop base configuration.

This module defines the base configuration for Ralph Loop experiments,
which focus on rapid iteration with tight constraints (val_loss < 1.5, time < 100s).

The configuration captures common parameters across Ralph iterations:
- Consistent dataset (Shakespeare character-level)
- Standard hardware settings (1 GPU, bf16-mixed precision)
- Optimized training parameters for quick convergence
- Configurable model architecture (neuromanifold/baseline)
- NeuroManifold features with sensible defaults
"""

from dataclasses import dataclass


@dataclass
class RalphBaseConfig:
    """Base configuration for Ralph Loop experiments.

    This configuration provides defaults optimized for:
    - Fast training (< 100s on consumer GPU)
    - Character-level language modeling (Shakespeare)
    - Rapid experimentation and iteration
    - Validation loss target < 1.5

    Common across all Ralph iterations:
    - dataset: shakespeare_char (character-level modeling)
    - devices: 1 (single GPU training)
    - precision: bf16-mixed (balanced speed/accuracy)
    - wandb_log: False (no cloud logging overhead)
    - bias: False (modern transformer practice)
    - weight_decay: 0.1 (standard regularization)
    - grad_clip: 1.0 (gradient norm clipping)
    - early_stopping_patience: 0 (disabled for controlled runs)
    - save_checkpoints: False (no checkpoint overhead)

    Attributes:
        dataset: Dataset name (fixed: "shakespeare_char")
        batch_size: Training batch size (default 64)
        block_size: Maximum sequence length (default 128)
        num_workers: Data loader workers (default 4)

        model_type: Model architecture ("neuromanifold" or "baseline")
        n_layer: Number of transformer layers (default 6)
        n_head: Number of attention heads (default 6)
        n_embd: Embedding dimension (default 384)
        dropout: Dropout probability (default 0.1)
        bias: Use bias in linear layers (default False)

        use_sdr: Enable SDR encoding (default False for speed)
        use_kan: Enable KAN layers (default False for speed)
        kan_type: KAN variant ("faster", "wave", "cheby")
        kan_num_centers: Number of RSWAF centers for FasterKAN
        use_mhc: Enable manifold-constrained hyper-connections
        use_full_mhc: Use full multi-stream mHC (vs simplified)
        mhc_n_streams: Number of parallel streams for mHC

        fhn_threshold: FHN firing threshold (default 0.5)
        fhn_tau: FHN time constant (default 12.5)
        n_fhn_steps: Number of FHN dynamics steps (default 0)
        use_fhn_imex: Use semi-implicit IMEX scheme
        use_fhn_partitioning: Enable energy balancing
        use_fhn_fused: Use fused CUDA kernels

        skip_manifold_spectral: Skip expensive manifold/spectral ops

        max_iters: Maximum training iterations (default 1000)
        gradient_accumulation_steps: Gradient accumulation (default 1)
        learning_rate: Peak learning rate (default 2e-3)
        min_lr: Minimum learning rate for decay (default 1e-4)
        weight_decay: AdamW weight decay (default 0.1)
        warmup_iters: Learning rate warmup iterations (default 50)
        lr_decay_iters: LR decay schedule length (default matches max_iters)
        grad_clip: Gradient norm clipping (default 1.0)

        early_stopping_patience: Early stop patience (default 0=disabled)

        eval_interval: Evaluation frequency in iterations (default 100)
        log_interval: Logging frequency in iterations (default 50)
        eval_iters: Number of evaluation iterations (default 10)
        sample_interval: Text sampling frequency (default 0=disabled)

        out_dir: Output directory (default "out-ralph")
        save_checkpoints: Save model checkpoints (default False)

        devices: Number of GPUs (default 1)
        precision: Training precision (default "bf16-mixed")
        compile_model: Use torch.compile (default False)

        wandb_log: Enable Weights & Biases logging (default False)
    """

    # Data configuration
    dataset: str = "shakespeare_char"
    batch_size: int = 64
    block_size: int = 128
    num_workers: int = 4

    # Model architecture
    model_type: str = "neuromanifold"
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = False

    # NeuroManifold features - optimized for speed
    use_sdr: bool = False  # Dense embeddings (faster)
    use_kan: bool = False  # Standard MLP (faster)
    kan_type: str = "faster"  # FasterKAN with RSWAF basis
    kan_num_centers: int = 2  # Minimal centers for speed
    use_mhc: bool = False  # Disable by default
    use_full_mhc: bool = False  # Simplified mHC
    mhc_n_streams: int = 2  # Parallel streams when mHC enabled

    # FHN dynamics (Fitzhugh-Nagumo)
    fhn_threshold: float = 0.5
    fhn_tau: float = 12.5
    n_fhn_steps: int = 0  # Disabled by default for speed
    use_fhn_imex: bool = True
    use_fhn_partitioning: bool = False
    use_fhn_fused: bool = False

    # Speed optimizations
    skip_manifold_spectral: bool = False

    # Training configuration
    max_iters: int = 1000
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-3
    min_lr: float = 1e-4
    weight_decay: float = 0.1
    warmup_iters: int = 50
    lr_decay_iters: int = 1000  # Matches max_iters by default
    grad_clip: float = 1.0

    # Early stopping
    early_stopping_patience: int = 0  # Disabled

    # Evaluation and logging
    eval_interval: int = 100
    log_interval: int = 50
    eval_iters: int = 10
    sample_interval: int = 0  # Disabled

    # Output
    out_dir: str = "out-ralph"
    save_checkpoints: bool = False

    # Hardware configuration
    devices: int = 1
    precision: str = "bf16-mixed"
    compile_model: bool = False

    # Logging
    wandb_log: bool = False

    def __post_init__(self) -> None:
        """Validate configuration and set derived values."""
        # lr_decay_iters should match max_iters if not explicitly set
        if self.lr_decay_iters == 1000 and self.max_iters != 1000:
            self.lr_decay_iters = self.max_iters

        # Validate n_embd is divisible by n_head
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )

        # Validate dataset
        if self.dataset != "shakespeare_char":
            raise ValueError(
                f"RalphBaseConfig is designed for shakespeare_char dataset, got: {self.dataset}"
            )
