"""Throughput monitoring callback for tracking training speed and ETA.

This callback tracks:
- Step times (rolling average)
- Throughput (tokens/sec)
- Training progress and estimated time to completion (ETA)

Usage:
    from neuromanifold_gpt.callbacks.throughput_monitor import ThroughputMonitorCallback

    callback = ThroughputMonitorCallback(
        log_interval=100,
        step_time_history_size=20,
        warmup_steps=10
    )
    trainer.fit(model, callbacks=[callback])
"""
import time
from collections import deque
from typing import Any, Deque, Optional, Union

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback


class ThroughputMonitorCallback(Callback):
    """Track training throughput, step times, and ETA.

    Monitors timing statistics to provide observability into training speed
    and estimated completion time.

    Args:
        log_interval: Number of steps between metric logging (default: 100)
        step_time_history_size: Number of step times to keep for rolling average (default: 20)
        warmup_steps: Number of steps before showing ETA (default: 10)
    """

    def __init__(
        self,
        log_interval: int = 100,
        step_time_history_size: int = 20,
        warmup_steps: int = 10,
    ):
        self.log_interval = log_interval
        self.step_time_history_size = step_time_history_size
        self.warmup_steps = warmup_steps

        # Timing and throughput tracking
        self.step_start_time: Optional[float] = None
        self.step_times: Deque[float] = deque(maxlen=step_time_history_size)
        self.training_start_time: Optional[float] = None
        self.max_steps: Optional[Union[int, float]] = None

        # Current step tracking
        self.current_step = 0

    def _format_eta(self, seconds: float) -> str:
        """Format ETA in human-readable format (e.g., '2h 34m', '45m 12s').

        Args:
            seconds: Number of seconds to format

        Returns:
            Human-readable string representation of time duration
        """
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"

    def on_train_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Initialize training start time and max steps for ETA calculation.

        Args:
            trainer: The PyTorch Lightning trainer
            pl_module: The LightningModule being trained
        """
        # Record training start time for ETA calculation
        self.training_start_time = time.perf_counter()

        # Get max steps from trainer
        if trainer.max_steps and trainer.max_steps > 0:
            self.max_steps = trainer.max_steps
        elif trainer.max_epochs and trainer.max_epochs > 0:
            # Estimate steps from epochs if available
            # This is an approximation
            if hasattr(trainer, 'num_training_batches'):
                self.max_steps = trainer.max_epochs * trainer.num_training_batches

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Record start time for throughput calculation.

        Args:
            trainer: The PyTorch Lightning trainer
            pl_module: The LightningModule being trained
            batch: The current training batch
            batch_idx: The index of the current batch
        """
        self.step_start_time = time.perf_counter()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Calculate and log throughput metrics.

        Args:
            trainer: The PyTorch Lightning trainer
            pl_module: The LightningModule being trained
            outputs: The outputs from the training step
            batch: The current training batch
            batch_idx: The index of the current batch
        """
        # Update current step
        self.current_step = batch_idx

        # Calculate step time and throughput
        if self.step_start_time is not None:
            step_time = time.perf_counter() - self.step_start_time
            self.step_times.append(step_time)

            # Calculate tokens/sec
            if hasattr(batch, '__len__'):
                batch_size = len(batch)
            elif isinstance(batch, (list, tuple)):
                batch_size = batch[0].size(0) if len(batch) > 0 else 0
            elif isinstance(batch, dict):
                first_key = next(iter(batch.keys()))
                batch_size = batch[first_key].size(0)
            else:
                batch_size = 0

            seq_len = batch[0].size(1) if isinstance(batch, (list, tuple)) and len(batch) > 0 else 256
            if isinstance(batch, dict) and 'input_ids' in batch:
                seq_len = batch['input_ids'].size(1)

            tokens_per_step = batch_size * seq_len
            throughput = tokens_per_step / step_time if step_time > 0 else 0

            pl_module.log('train/throughput_tokens_per_sec', throughput, on_step=True, prog_bar=False)

            # Average step time
            avg_step_time = sum(self.step_times) / len(self.step_times) if self.step_times else 0
            pl_module.log('train/step_time_ms', avg_step_time * 1000, on_step=True, prog_bar=False)

            # Calculate and log ETA if we have enough warmup steps
            if len(self.step_times) >= self.warmup_steps and self.max_steps is not None:
                remaining_steps = self.max_steps - self.current_step
                eta_seconds = avg_step_time * remaining_steps
                eta_formatted = self._format_eta(eta_seconds)

                # Log ETA as a metric (for dashboard or other callbacks to use)
                pl_module.log('train/eta_seconds', eta_seconds, on_step=True, prog_bar=False)
