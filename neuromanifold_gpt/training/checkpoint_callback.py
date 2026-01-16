"""
Custom checkpoint callback for separated model and optimizer state saving.

This callback provides the ability to save model weights and optimizer state
to separate files, enabling:
- Smaller model-only checkpoints for inference
- Flexible optimizer state management
- Easier model sharing without optimizer buffers
"""

import os
from typing import Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from loguru import logger


class SeparatedCheckpointCallback(Callback):
    """Save model weights and optimizer state to separate files.

    This callback intercepts the checkpoint saving process and creates
    two separate files:
    - model.pt: Contains only model state_dict (weights, biases)
    - optimizer.pt: Contains optimizer state and training metadata

    This separation provides:
    - 50%+ smaller model files for inference/sharing
    - Optional optimizer state for training resumption
    - Backward compatibility with unified checkpoints

    Args:
        save_dir: Directory to save separated checkpoints
        save_interval: Save every N training steps (0 = disabled)
        save_model_only: If True, only save model.pt (no optimizer state)
        filename_prefix: Prefix for checkpoint filenames (default: "checkpoint")
    """

    def __init__(
        self,
        save_dir: str,
        save_interval: int = 1000,
        save_model_only: bool = False,
        filename_prefix: str = "checkpoint",
    ):
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.save_model_only = save_model_only
        self.filename_prefix = filename_prefix

        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        """Called after each training batch completes.

        Saves separated checkpoints at regular intervals.

        Args:
            trainer: PyTorch Lightning trainer instance
            pl_module: Lightning module being trained
            outputs: Training step outputs
            batch: Current training batch
            batch_idx: Index of current batch
        """
        if self.save_interval <= 0:
            return

        global_step = trainer.global_step
        if global_step > 0 and global_step % self.save_interval == 0:
            self._save_separated_checkpoint(pl_module, trainer, global_step)

    def _save_separated_checkpoint(
        self,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        global_step: int,
    ) -> None:
        """Save model and optimizer state to separate files.

        Args:
            pl_module: Lightning module with model and optimizer
            trainer: Trainer instance for accessing optimizer state
            global_step: Current training step for filename
        """
        # Generate filenames
        model_filename = f"{self.filename_prefix}-step{global_step:06d}-model.pt"
        optimizer_filename = f"{self.filename_prefix}-step{global_step:06d}-optimizer.pt"
        model_path = os.path.join(self.save_dir, model_filename)
        optimizer_path = os.path.join(self.save_dir, optimizer_filename)

        try:
            # Save model state (weights and biases only)
            model_state = {
                "model_state_dict": pl_module.model.state_dict(),
                "global_step": global_step,
                "config": pl_module.config if hasattr(pl_module, "config") else None,
            }
            torch.save(model_state, model_path)
            logger.info(f"Saved model checkpoint: {model_filename}")

            # Save optimizer state (unless model_only mode)
            if not self.save_model_only:
                optimizer_states = []
                lr_scheduler_states = []

                # Collect optimizer and scheduler states from trainer
                if trainer.optimizers:
                    for optimizer in trainer.optimizers:
                        optimizer_states.append(optimizer.state_dict())

                if trainer.lr_scheduler_configs:
                    for lr_config in trainer.lr_scheduler_configs:
                        scheduler = lr_config.scheduler
                        lr_scheduler_states.append(scheduler.state_dict())

                optimizer_state = {
                    "optimizer_states": optimizer_states,
                    "lr_scheduler_states": lr_scheduler_states,
                    "global_step": global_step,
                    "epoch": trainer.current_epoch,
                }
                torch.save(optimizer_state, optimizer_path)
                logger.info(f"Saved optimizer checkpoint: {optimizer_filename}")
            else:
                logger.info("Model-only mode: skipping optimizer checkpoint")

        except Exception as e:
            logger.error(f"Failed to save separated checkpoint at step {global_step}: {e}")

    def on_train_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Called when training ends.

        Saves a final checkpoint with the 'final' prefix.

        Args:
            trainer: PyTorch Lightning trainer instance
            pl_module: Lightning module being trained
        """
        # Save final checkpoint
        model_filename = f"{self.filename_prefix}-final-model.pt"
        optimizer_filename = f"{self.filename_prefix}-final-optimizer.pt"
        model_path = os.path.join(self.save_dir, model_filename)
        optimizer_path = os.path.join(self.save_dir, optimizer_filename)

        try:
            # Save final model state
            model_state = {
                "model_state_dict": pl_module.model.state_dict(),
                "global_step": trainer.global_step,
                "config": pl_module.config if hasattr(pl_module, "config") else None,
            }
            torch.save(model_state, model_path)
            logger.info(f"Saved final model checkpoint: {model_filename}")

            # Save final optimizer state (unless model_only mode)
            if not self.save_model_only:
                optimizer_states = []
                lr_scheduler_states = []

                if trainer.optimizers:
                    for optimizer in trainer.optimizers:
                        optimizer_states.append(optimizer.state_dict())

                if trainer.lr_scheduler_configs:
                    for lr_config in trainer.lr_scheduler_configs:
                        scheduler = lr_config.scheduler
                        lr_scheduler_states.append(scheduler.state_dict())

                optimizer_state = {
                    "optimizer_states": optimizer_states,
                    "lr_scheduler_states": lr_scheduler_states,
                    "global_step": trainer.global_step,
                    "epoch": trainer.current_epoch,
                }
                torch.save(optimizer_state, optimizer_path)
                logger.info(f"Saved final optimizer checkpoint: {optimizer_filename}")

        except Exception as e:
            logger.error(f"Failed to save final checkpoint: {e}")
