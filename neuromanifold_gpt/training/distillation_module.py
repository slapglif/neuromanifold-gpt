# neuromanifold_gpt/training/distillation_module.py
"""
PyTorch Lightning module for knowledge distillation training.

Provides:
- DistillationLitModule: LightningModule for distilling from a teacher model
- Combines student task loss with KL divergence from teacher logits
- Temperature-scaled soft targets for improved knowledge transfer
"""
import torch
import torch.nn.functional as F
from loguru import logger

from neuromanifold_gpt.training.lightning_module import NeuroManifoldLitModule
from neuromanifold_gpt.config import NeuroManifoldConfig
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT
from neuromanifold_gpt.utils.checkpoint_loader import load_model_only


class DistillationLitModule(NeuroManifoldLitModule):
    """PyTorch Lightning module for knowledge distillation.

    Extends NeuroManifoldLitModule to support teacher-student distillation:
    - Loads frozen teacher model from checkpoint
    - Computes distillation loss (KL divergence between logits)
    - Combines student task loss with distillation loss

    Loss = alpha * student_loss + (1 - alpha) * distillation_loss

    The distillation loss uses temperature-scaled softmax to soften the probability
    distributions, allowing the student to learn from the teacher's uncertainty.

    Args:
        config: NeuroManifoldConfig for student model
        teacher_checkpoint: Path to teacher model checkpoint
        distillation_alpha: Weight for student task loss (0-1)
        distillation_temperature: Temperature for softmax in distillation
    """

    def __init__(
        self,
        config: NeuroManifoldConfig,
        teacher_checkpoint: str,
        distillation_alpha: float = 0.5,
        distillation_temperature: float = 2.0,
    ):
        super().__init__(config)

        self.teacher_checkpoint = teacher_checkpoint
        self.distillation_alpha = distillation_alpha
        self.distillation_temperature = distillation_temperature

        # Load teacher model
        logger.info(f"Loading teacher model from {teacher_checkpoint}")
        checkpoint = load_model_only(teacher_checkpoint, device='cpu', weights_only=False)

        # Create teacher model from checkpoint config
        teacher_config = checkpoint.get('config')
        if teacher_config is None:
            raise ValueError(
                f"Teacher checkpoint {teacher_checkpoint} does not contain config. "
                "Cannot instantiate teacher model."
            )

        self.teacher = NeuroManifoldGPT(teacher_config)

        # Load teacher weights
        state_dict = checkpoint['model']
        # Handle compiled model prefix
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

        self.teacher.load_state_dict(state_dict)

        # Freeze teacher model
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        logger.info(
            f"Teacher model loaded and frozen "
            f"(alpha={distillation_alpha}, temp={distillation_temperature})"
        )

        # Save hyperparameters
        self.save_hyperparameters(ignore=['config'])

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Training step with knowledge distillation.

        Computes:
        1. Student task loss (standard cross-entropy)
        2. Distillation loss (KL divergence from teacher logits)
        3. Combined loss = alpha * task_loss + (1 - alpha) * distillation_loss

        Args:
            batch: Dict with 'input_ids' and 'labels' tensors
            batch_idx: Index of current batch

        Returns:
            loss: Combined distillation loss
        """
        input_ids = batch['input_ids']
        labels = batch['labels']

        # Student forward pass
        student_logits, student_loss, _ = self(input_ids, labels)

        # Teacher forward pass (no gradients)
        with torch.no_grad():
            teacher_logits, _, _ = self.teacher(input_ids, labels)

        # Compute distillation loss (KL divergence)
        # Apply temperature scaling
        student_log_probs = F.log_softmax(
            student_logits / self.distillation_temperature, dim=-1
        )
        teacher_probs = F.softmax(
            teacher_logits / self.distillation_temperature, dim=-1
        )

        # KL divergence: sum over vocab, mean over batch and sequence
        distillation_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean',
        )

        # Scale by temperature^2 (standard in distillation)
        distillation_loss = distillation_loss * (self.distillation_temperature ** 2)

        # Combine losses
        total_loss = (
            self.distillation_alpha * student_loss +
            (1 - self.distillation_alpha) * distillation_loss
        )

        # Log metrics
        self.log('train_loss', total_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_student_loss', student_loss, on_step=True, on_epoch=True)
        self.log('train_distillation_loss', distillation_loss, on_step=True, on_epoch=True)

        return total_loss
