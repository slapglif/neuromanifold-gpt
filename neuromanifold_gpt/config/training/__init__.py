"""Training configuration presets for NeuroManifoldGPT."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DistillationEvalConfig:
    """Configuration for distillation evaluation (eval_distillation.py).

    This configuration covers teacher-student model comparison:
    - Checkpoint loading for both teacher and student models
    - Benchmark selection and evaluation parameters
    - Comparison metrics (KL divergence, agreement, perplexity)
    - Hardware settings (device, precision)
    - Logging and experiment tracking (WandB)

    Attributes:
        teacher_checkpoint: Path to teacher model checkpoint
        student_checkpoint: Path to student model checkpoint
        benchmark: Benchmark to evaluate (lambada, hellaswag, piqa, winogrande, all)
        eval_iters: Maximum examples to evaluate (None = all examples)
        device: Device to run on ('cpu', 'cuda', 'cuda:0', etc.)
        dtype: Precision for inference ('float32', 'bfloat16', or 'float16')
        seed: Random seed for reproducibility
        compile: Whether to compile models with torch.compile (PyTorch 2.0+)
        wandb_log: Enable Weights & Biases logging
        wandb_project: WandB project name
        wandb_run_name: WandB run name (auto-generated if None)
    """

    # Checkpoint loading
    teacher_checkpoint: str = ''
    student_checkpoint: str = ''

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
    wandb_project: str = 'neuromanifold-distill-eval'
    wandb_run_name: Optional[str] = None


__all__ = ["DistillationEvalConfig"]
