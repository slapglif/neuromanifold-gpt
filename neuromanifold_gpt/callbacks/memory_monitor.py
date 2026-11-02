"""Memory monitoring callback for tracking GPU memory usage.

This callback tracks:
- Current GPU memory allocated
- Peak GPU memory allocated

Usage:
    from neuromanifold_gpt.callbacks.memory_monitor import MemoryMonitorCallback

    callback = MemoryMonitorCallback(log_interval=100)
    trainer.fit(model, callbacks=[callback])
"""
from typing import Any

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback


class MemoryMonitorCallback(Callback):
    """Track GPU memory usage during training.

    Monitors current and peak GPU memory allocation to provide observability
    into memory usage patterns during training.

    Args:
        log_interval: Number of steps between metric logging (default: 100)
    """

    def __init__(
        self,
        log_interval: int = 100,
    ):
        self.log_interval = log_interval

        # Memory tracking
        self.peak_memory_mb = 0.0

        # Current step tracking
        self.current_step = 0

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Log memory usage metrics at specified intervals."""
        # Update current step
        self.current_step = batch_idx

        # Log memory usage (GPU only)
        if torch.cuda.is_available():
            current_memory_mb = torch.cuda.memory_allocated() / 1e6
            peak_memory_mb = torch.cuda.max_memory_allocated() / 1e6

            # Update peak
            if peak_memory_mb > self.peak_memory_mb:
                self.peak_memory_mb = peak_memory_mb

            pl_module.log('train/memory_current_mb', current_memory_mb, on_step=True, prog_bar=True)
            pl_module.log('train/memory_peak_mb', self.peak_memory_mb, on_step=True, prog_bar=False)
