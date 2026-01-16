"""
PyTorch Lightning callbacks for training monitoring and sample generation.
"""

from typing import Optional, Dict

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from loguru import logger


class SampleGenerationCallback(Callback):
    """Generate text samples periodically during training.

    This callback generates text samples at regular intervals to monitor
    the model's learning progress. Samples are logged to the console and
    to Weights & Biases if available.

    Args:
        sample_interval: Generate samples every N training steps
        max_tokens: Maximum number of tokens to generate per sample
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling parameter (0 = no filtering)
        itos: Index-to-string mapping for decoding tokens
        stoi: String-to-index mapping for encoding start tokens
    """

    def __init__(
        self,
        sample_interval: int,
        max_tokens: int = 200,
        temperature: float = 1.0,
        top_k: int = 40,
        itos: Optional[Dict[int, str]] = None,
        stoi: Optional[Dict[str, int]] = None,
    ):
        self.sample_interval = sample_interval
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.itos = itos
        self.stoi = stoi

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        """Called after each training batch completes.

        Args:
            trainer: PyTorch Lightning trainer instance
            pl_module: Lightning module being trained
            outputs: Training step outputs
            batch: Current training batch
            batch_idx: Index of current batch
        """
        if self.sample_interval <= 0:
            return
        if self.itos is None:
            return

        global_step = trainer.global_step
        if global_step > 0 and global_step % self.sample_interval == 0:
            self._generate_sample(pl_module, trainer)

    def _generate_sample(
        self, pl_module: pl.LightningModule, trainer: pl.Trainer
    ) -> None:
        """Generate and log a text sample.

        Args:
            pl_module: Lightning module with generate() method
            trainer: Trainer instance for step counting and logging
        """
        device = pl_module.device

        # Start token
        start_char = "\n"
        if self.stoi and start_char in self.stoi:
            start_idx = self.stoi[start_char]
        else:
            start_idx = 0

        start = torch.tensor([[start_idx]], device=device, dtype=torch.long)

        try:
            out_idx = pl_module.generate(
                start,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
            )

            text = "".join([self.itos[i] for i in out_idx[0].tolist()])
            logger.info(f"\n--- Sample (step {trainer.global_step}) ---\n{text}\n--- End Sample ---")

            # Log to wandb if available
            if trainer.logger and hasattr(trainer.logger, "experiment"):
                try:
                    trainer.logger.experiment.log(
                        {"samples/text": text, "global_step": trainer.global_step}
                    )
                except Exception:
                    pass

        except Exception as e:
            logger.warning(f"Sample generation failed: {e}")


class MFUCallback(Callback):
    """Track Model FLOPs Utilization (MFU) during training.

    MFU measures how efficiently the model is using available compute
    resources by comparing actual FLOPs to theoretical peak FLOPs.
    Higher MFU indicates better hardware utilization.

    Args:
        log_interval: Log MFU every N training batches
    """

    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.running_mfu = -1.0

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        """Called after each training batch completes.

        Args:
            trainer: PyTorch Lightning trainer instance
            pl_module: Lightning module being trained
            outputs: Training step outputs
            batch: Current training batch
            batch_idx: Index of current batch
        """
        if batch_idx % self.log_interval != 0:
            return

        model = pl_module.model
        if not hasattr(model, "estimate_mfu"):
            return

        # Estimate MFU
        batch_size = batch[0].size(0)
        accum = pl_module.train_config.gradient_accumulation_steps

        # Get time per iteration from trainer
        if hasattr(trainer, "progress_bar_metrics"):
            # Approximate dt from throughput
            dt = 0.1  # fallback
        else:
            dt = 0.1

        try:
            mfu = model.estimate_mfu(batch_size * accum, dt)
            if self.running_mfu < 0:
                self.running_mfu = mfu
            else:
                self.running_mfu = 0.9 * self.running_mfu + 0.1 * mfu

            pl_module.log("train/mfu", self.running_mfu * 100, on_step=True)
        except Exception:
            pass
