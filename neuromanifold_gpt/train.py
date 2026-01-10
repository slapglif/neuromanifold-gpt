# neuromanifold_gpt/train.py
"""
PyTorch Lightning training module for NeuroManifoldGPT.

Provides:
- NeuroManifoldLightning: LightningModule wrapper for training
- AdamW optimizer with separate weight decay groups
- Cosine annealing LR scheduler
"""
import torch
import lightning as L
from torch.optim.lr_scheduler import CosineAnnealingLR

from neuromanifold_gpt.config import NeuroManifoldConfig
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT


class NeuroManifoldLightning(L.LightningModule):
    """PyTorch Lightning wrapper for NeuroManifoldGPT.

    Handles:
    - Training and validation steps with loss computation
    - Optimizer configuration with weight decay groups
    - LR scheduling with cosine annealing
    - Metric logging

    Args:
        config: NeuroManifoldConfig with model and training hyperparameters
    """

    def __init__(self, config: NeuroManifoldConfig):
        super().__init__()
        self.config = config
        self.model = NeuroManifoldGPT(config)

        # Save hyperparameters for logging
        self.save_hyperparameters(ignore=['config'])

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict]:
        """Forward pass through the model.

        Args:
            input_ids: (B, T) input token indices
            labels: (B, T) target token indices for loss computation

        Returns:
            logits: (B, T, vocab_size)
            loss: scalar if labels provided, else None
            info: diagnostic dict
        """
        return self.model(input_ids, labels)

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Single training step.

        Args:
            batch: Dict with 'input_ids' and 'labels' tensors
            batch_idx: Index of current batch

        Returns:
            loss: Scalar training loss
        """
        input_ids = batch['input_ids']
        labels = batch['labels']

        _, loss, _ = self(input_ids, labels)

        # Log training metrics
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Single validation step.

        Args:
            batch: Dict with 'input_ids' and 'labels' tensors
            batch_idx: Index of current batch

        Returns:
            loss: Scalar validation loss
        """
        input_ids = batch['input_ids']
        labels = batch['labels']

        _, loss, _ = self(input_ids, labels)

        # Log validation metrics
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self) -> dict:
        """Configure optimizer and LR scheduler.

        Uses AdamW with:
        - Separate parameter groups for weight decay
        - No decay for biases and LayerNorm parameters
        - Cosine annealing LR schedule

        Returns:
            Dict with 'optimizer' and 'lr_scheduler' keys
        """
        # Separate params into decay and no-decay groups
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Biases, LayerNorm weights don't get weight decay
            if param.ndim < 2 or 'ln' in name or 'norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optim_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
        )

        # Cosine annealing scheduler
        # Default T_max to 1000 steps; use trainer.max_steps if available
        try:
            max_steps = self.trainer.max_steps if self.trainer.max_steps else 1000
        except RuntimeError:
            # Trainer not attached (e.g., in tests)
            max_steps = 1000

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max_steps,
            eta_min=self.config.learning_rate * 0.1,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            },
        }

    def on_before_optimizer_step(self, optimizer) -> None:
        """Gradient clipping before optimizer step."""
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip,
            )
