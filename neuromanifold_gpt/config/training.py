"""Training configuration dataclasses for NeuroManifoldGPT.

This module defines the configuration structure for training, including settings for:
- I/O and checkpointing
- Data loading and preprocessing
- Model architecture selection
- Training hyperparameters (learning rate, optimization, etc.)
- Early stopping and evaluation
- Sampling during training
- Hardware configuration (devices, precision)
- Logging and experiment tracking
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Training configuration with all hyperparameters.

    This configuration covers all aspects of the training process:
    - File I/O, checkpoint saving, and evaluation intervals
    - Dataset selection and data loading parameters
    - Model type and architecture hyperparameters
    - Optimization settings (learning rate, weight decay, etc.)
    - Early stopping and convergence criteria
    - Text sampling during training for qualitative assessment
    - Hardware settings (GPUs, mixed precision)
    - Logging and experiment tracking (WandB)

    Attributes:
        out_dir: Output directory for checkpoints and logs
        eval_interval: Steps between evaluation runs
        log_interval: Steps between logging metrics
        eval_iters: Number of iterations for evaluation
        save_checkpoints: Whether to save model checkpoints
        dataset: Dataset name (e.g., "shakespeare_char")
        batch_size: Training batch size
        block_size: Maximum sequence length (context window)
        num_workers: Number of data loading workers
        streaming: Use HuggingFace streaming for large datasets
        vocab_size: Vocabulary size (0 = auto-detect from data)
        model_type: Model architecture ("neuromanifold" or "gpt")
        n_layer: Number of transformer layers
        n_head: Number of attention heads
        n_embd: Embedding dimension (must be divisible by n_head)
        dropout: Dropout probability
        bias: Whether to use bias in linear layers
        sdr_size: Size of SDR binary vectors (NeuroManifold)
        manifold_dim: Dimension of learned manifold space (NeuroManifold)
        n_eigenvectors: Number of eigenvectors for spectral attention (NeuroManifold)
        use_sdr: Enable SDR (Sparse Distributed Representation) encoding
        use_kan: Enable KAN (Kolmogorov-Arnold Networks) for FFN
        kan_type: KAN variant ("faster", "wave", or "cheby")
        kan_wavelet: Wavelet type for WaveKAN ("dog" for efficiency)
        use_fast_wavekan: Use efficient WaveKAN implementation
        kan_num_centers: Number of basis centers for FasterKAN
        fhn_threshold: Firing threshold for FitzHugh-Nagumo dynamics
        fhn_tau: Time constant for FHN dynamics
        n_fhn_steps: Number of FHN integration steps
        use_fhn_imex: Use semi-implicit IMEX scheme for FHN
        use_fhn_partitioning: Enable energy balancing for FHN stability
        use_fhn_fused: Use fused kernel for FHN (experimental)
        use_mhc: Enable mHC (Manifold-Constrained Hyper-Connections)
        use_full_mhc: Use full multi-stream mHC vs simplified
        mhc_n_streams: Number of parallel streams for mHC
        use_kaufmann_attention: Enable Kaufmann Trifecta Attention
        skip_manifold_spectral: Skip manifold/spectral for faster training
        max_iters: Maximum training iterations
        gradient_accumulation_steps: Steps to accumulate gradients
        learning_rate: Peak learning rate for optimizer
        min_lr: Minimum learning rate after decay
        weight_decay: Weight decay for AdamW optimizer
        beta1: AdamW beta1 parameter (momentum)
        beta2: AdamW beta2 parameter (variance)
        grad_clip: Gradient clipping norm
        warmup_iters: Number of warmup iterations for learning rate
        lr_decay_iters: Total iterations for learning rate decay
        early_stopping_patience: Patience for early stopping (epochs)
        sample_interval: Steps between text generation samples
        sample_max_tokens: Maximum tokens to generate per sample
        sample_temperature: Temperature for sampling (higher = more random)
        sample_top_k: Top-k filtering for sampling
        devices: Number of GPUs/devices to use
        precision: Training precision ("bf16-mixed", "fp16-mixed", or "32")
        compile_model: Whether to compile model with torch.compile
        wandb_log: Enable Weights & Biases logging
        wandb_project: WandB project name
        wandb_run_name: WandB run name
    """

    # I/O
    out_dir: str = "out-shakespeare-char"
    eval_interval: int = 250
    log_interval: int = 10
    eval_iters: int = 200
    save_checkpoints: bool = True

    # Data
    dataset: str = "shakespeare_char"
    batch_size: int = 64
    block_size: int = 256
    num_workers: int = 4
    streaming: bool = False  # Use HuggingFace streaming for general text
    vocab_size: int = 0  # 0 = auto-detect, 50257 = GPT-2 BPE

    # Model
    model_type: str = "neuromanifold"  # "neuromanifold" or "gpt"
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
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
    skip_manifold_spectral: bool = False  # Skip manifold/spectral for faster training

    # Training
    max_iters: int = 5000
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-3
    min_lr: float = 1e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.99
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


@dataclass
class SamplingConfig:
    """Configuration for sampling from trained models.

    This configuration covers all aspects of text generation:
    - Model loading and initialization
    - Generation parameters (temperature, top-k)
    - Output settings (number of samples, length)
    - Hardware settings (device, precision)

    Attributes:
        init_from: Initialization mode ('resume' or gpt2 variant like 'gpt2-xl')
        out_dir: Directory to load checkpoint from (when init_from='resume')
        start: Initial prompt string or FILE:path.txt to load from file
        num_samples: Number of samples to generate
        max_new_tokens: Maximum number of tokens to generate per sample
        temperature: Sampling temperature (1.0=no change, <1.0=less random, >1.0=more random)
        top_k: Top-k filtering (retain only top_k most likely tokens)
        seed: Random seed for reproducibility
        device: Device to run on ('cpu', 'cuda', 'cuda:0', etc.)
        dtype: Precision for inference ('float32', 'bfloat16', or 'float16')
        compile: Whether to compile model with torch.compile (PyTorch 2.0+)
    """

    # Model loading
    init_from: str = 'resume'
    out_dir: str = 'out'

    # Generation
    start: str = "\n"
    num_samples: int = 10
    max_new_tokens: int = 500
    temperature: float = 0.8
    top_k: int = 200
    seed: int = 1337

    # Hardware
    device: str = 'cuda'
    dtype: str = 'bfloat16'
    compile: bool = False


@dataclass
class EvalConfig:
    """Configuration for zero-shot benchmark evaluation.

    This configuration covers all aspects of model evaluation:
    - Checkpoint loading and model initialization
    - Benchmark selection and evaluation parameters
    - Hardware settings (device, precision)
    - Logging and experiment tracking (WandB)

    Attributes:
        out_dir: Checkpoint directory to load model from
        benchmark: Benchmark to evaluate (lambada, hellaswag, piqa, winogrande, all)
        eval_iters: Maximum examples to evaluate (None = all examples)
        device: Device to run on ('cpu', 'cuda', 'cuda:0', etc.)
        dtype: Precision for inference ('float32', 'bfloat16', or 'float16')
        seed: Random seed for reproducibility
        compile: Whether to compile model with torch.compile (PyTorch 2.0+)
        wandb_log: Enable Weights & Biases logging
        wandb_project: WandB project name
        wandb_run_name: WandB run name (auto-generated if None)
    """

    # Checkpoint loading
    out_dir: str = 'out'

    # Benchmark
    benchmark: str = 'lambada'
    eval_iters: Optional[int] = None

    # Hardware
    device: str = 'cuda'
    dtype: str = 'bfloat16'

    # Reproducibility
    seed: int = 1337
    compile: bool = False

    # Logging
    wandb_log: bool = False
    wandb_project: str = 'neuromanifold-eval'
    wandb_run_name: Optional[str] = None
