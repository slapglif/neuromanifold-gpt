"""Loss monitoring callback for tracking loss history and detecting anomalies.

This callback tracks:
- Loss history (rolling window)
- Loss spike detection (anomaly detection using rolling statistics)
- Average loss over time

Usage:
    from neuromanifold_gpt.callbacks.loss_monitor import LossMonitorCallback

    callback = LossMonitorCallback(
        log_interval=100,
        loss_history_size=100,
        loss_spike_threshold=3.0
    )
    trainer.fit(model, callbacks=[callback])
"""
from collections import deque
from typing import Any, Deque

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback


class LossMonitorCallback(Callback):
    """Track loss history and detect loss spikes.

    Monitors loss values to provide observability into training stability.
    Detects anomalies like loss spikes that may indicate training issues.

    Args:
        log_interval: Number of steps between metric logging (default: 100)
        loss_history_size: Number of loss values to keep for statistics (default: 100)
        loss_spike_threshold: Number of standard deviations for spike detection (default: 3.0)
        min_loss_history_for_detection: Minimum samples before detecting anomalies (default: 20)
    """

    def __init__(
        self,
        log_interval: int = 100,
        loss_history_size: int = 100,
        loss_spike_threshold: float = 3.0,
        min_loss_history_for_detection: int = 20,
    ):
        self.log_interval = log_interval
        self.loss_history_size = loss_history_size
        self.loss_spike_threshold = loss_spike_threshold
        self.min_loss_history_for_detection = min_loss_history_for_detection

        # Loss tracking
        self.loss_history: Deque[float] = deque(maxlen=loss_history_size)

        # Current step tracking
        self.current_step = 0

    def _detect_loss_spike(self, current_loss: float, current_step: int) -> None:
        """Detect if current loss is a spike (anomalously high).

        Uses rolling statistics: flags spike if loss > mean + threshold*std.

        Args:
            current_loss: The current loss value
            current_step: The current training step
        """
        # Need enough history to calculate meaningful statistics
        if len(self.loss_history) < self.min_loss_history_for_detection:
            return

        # Calculate rolling mean and standard deviation
        loss_list = list(self.loss_history)
        mean_loss = sum(loss_list) / len(loss_list)

        # Calculate standard deviation
        variance = sum((x - mean_loss) ** 2 for x in loss_list) / len(loss_list)
        std_loss = variance ** 0.5

        # Detect spike: current loss > mean + threshold * std
        if current_loss > mean_loss + self.loss_spike_threshold * std_loss:
            # Log warning as a simple print that can be picked up by dashboard or other callbacks
            # We don't use Rich formatting to maintain single responsibility
            print(
                f"⚠ LOSS SPIKE DETECTED at step {current_step}: "
                f"Loss={current_loss:.4f} (mean={mean_loss:.4f}, std={std_loss:.4f}). "
                f"Consider reducing learning rate or checking for data issues."
            )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Log loss metrics at specified intervals."""
        # Update current step
        self.current_step = batch_idx

        # Log current loss if available in outputs
        if outputs is not None:
            if isinstance(outputs, dict) and 'loss' in outputs:
                loss = outputs['loss']
            elif isinstance(outputs, torch.Tensor):
                loss = outputs
            else:
                loss = None

            if loss is not None:
                loss_value = loss.item() if torch.is_tensor(loss) else float(loss)

                self.loss_history.append(loss_value)

                # Detect loss spikes (anomaly detection)
                self._detect_loss_spike(loss_value, self.current_step)

                # Calculate loss statistics
                if len(self.loss_history) > 1:
                    avg_loss = sum(self.loss_history) / len(self.loss_history)
                    pl_module.log('train/loss_avg', avg_loss, on_step=True, prog_bar=False)
