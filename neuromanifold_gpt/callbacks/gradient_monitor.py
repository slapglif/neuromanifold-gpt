"""Gradient monitoring callback for tracking gradient norms and clipping events.

This callback tracks:
- Gradient norms (total norm before clipping)
- Gradient clipping events and ratios
- Gradient explosion detection (anomaly detection using rolling statistics)

Usage:
    from neuromanifold_gpt.callbacks.gradient_monitor import GradientMonitorCallback

    callback = GradientMonitorCallback(
        log_interval=100,
        gradient_norm_history_size=100,
        grad_explosion_threshold=3.0
    )
    trainer.fit(model, callbacks=[callback])
"""
from collections import deque
from typing import Any, Deque, Optional

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback


class GradientMonitorCallback(Callback):
    """Track gradient norms, clipping events, and detect gradient explosions.

    Monitors gradient statistics to provide observability into gradient behavior
    during training. Detects anomalies like gradient explosions.

    Args:
        log_interval: Number of steps between metric logging (default: 100)
        gradient_norm_history_size: Number of gradient norms to keep for statistics (default: 100)
        grad_explosion_threshold: Number of standard deviations for explosion detection (default: 3.0)
        min_grad_history_for_detection: Minimum samples before detecting gradient explosions (default: 20)
    """

    def __init__(
        self,
        log_interval: int = 100,
        gradient_norm_history_size: int = 100,
        grad_explosion_threshold: float = 3.0,
        min_grad_history_for_detection: int = 20,
    ):
        self.log_interval = log_interval
        self.gradient_norm_history_size = gradient_norm_history_size
        self.grad_explosion_threshold = grad_explosion_threshold
        self.min_grad_history_for_detection = min_grad_history_for_detection

        # Gradient tracking
        self.grad_norm_history: Deque[float] = deque(maxlen=gradient_norm_history_size)
        self.grad_norm_before_clip: Optional[float] = None
        self.total_clip_events = 0
        self.total_steps = 0
        self.clip_ratios: Deque[float] = deque(maxlen=100)  # Track clip magnitude when clipping occurs

        # Current step tracking
        self.current_step = 0

    def _detect_gradient_explosion(self, current_grad_norm: float, current_step: int) -> None:
        """Detect if current gradient norm is exploding (anomalously high).

        Uses rolling statistics: flags explosion if grad_norm > mean + threshold*std.

        Args:
            current_grad_norm: The current gradient norm value
            current_step: The current training step
        """
        # Need enough history to calculate meaningful statistics
        if len(self.grad_norm_history) < self.min_grad_history_for_detection:
            return

        # Calculate rolling mean and standard deviation
        grad_norm_list = list(self.grad_norm_history)
        mean_grad_norm = sum(grad_norm_list) / len(grad_norm_list)

        # Calculate standard deviation
        variance = sum((x - mean_grad_norm) ** 2 for x in grad_norm_list) / len(grad_norm_list)
        std_grad_norm = variance ** 0.5

        # Detect explosion: current grad norm > mean + threshold * std
        if current_grad_norm > mean_grad_norm + self.grad_explosion_threshold * std_grad_norm:
            # Log warning as a metric that can be picked up by dashboard or other callbacks
            # We don't print directly to maintain single responsibility
            print(
                f"⚠ GRADIENT EXPLOSION DETECTED at step {current_step}: "
                f"GradNorm={current_grad_norm:.4f} (mean={mean_grad_norm:.4f}, std={std_grad_norm:.4f}). "
                f"Consider reducing learning rate or enabling gradient clipping."
            )

    def on_after_backward(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Track gradient norm before clipping.

        Called after loss.backward() but before optimizer.step().
        This allows us to measure the pre-clip gradient norm.
        """
        # Calculate gradient norm before clipping
        total_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        self.grad_norm_before_clip = total_norm
        self.grad_norm_history.append(total_norm)

    def on_before_optimizer_step(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        optimizer: Any,
    ) -> None:
        """Track gradient clipping events.

        Called after gradient clipping (if enabled) but before optimizer.step().
        We can detect if clipping occurred by comparing norms.
        """
        # Check if gradient clipping occurred
        if hasattr(pl_module, 'config') and pl_module.config.grad_clip > 0:
            # Calculate gradient norm after clipping
            total_norm_after = 0.0
            for p in pl_module.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm_after += param_norm.item() ** 2
            total_norm_after = total_norm_after ** 0.5

            # If gradient norm changed significantly, clipping occurred
            if self.grad_norm_before_clip is not None:
                clip_threshold = pl_module.config.grad_clip
                if self.grad_norm_before_clip > clip_threshold * 1.01:  # Small tolerance for numerical precision
                    self.total_clip_events += 1
                    # Track the ratio of clipped gradient to threshold
                    # This shows "how much" clipping occurred
                    clip_ratio = total_norm_after / self.grad_norm_before_clip
                    self.clip_ratios.append(clip_ratio)

        self.total_steps += 1

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Log gradient metrics at specified intervals."""
        # Update current step
        self.current_step = batch_idx

        # Log gradient norm statistics
        if self.grad_norm_history:
            current_grad_norm = self.grad_norm_history[-1]
            avg_grad_norm = sum(self.grad_norm_history) / len(self.grad_norm_history)

            # Detect gradient explosions (anomaly detection)
            self._detect_gradient_explosion(current_grad_norm, self.current_step)

            pl_module.log('train/grad_norm', current_grad_norm, on_step=True, prog_bar=True)
            pl_module.log('train/grad_norm_avg', avg_grad_norm, on_step=True, prog_bar=False)

        # Log gradient clipping statistics
        if self.total_steps > 0:
            clip_rate = self.total_clip_events / self.total_steps
            pl_module.log('train/clip_rate', clip_rate, on_step=True, prog_bar=False)
            pl_module.log('train/total_clip_events', float(self.total_clip_events), on_step=True, prog_bar=False)

            # Log average clip ratio if clipping has occurred
            if self.clip_ratios:
                avg_clip_ratio = sum(self.clip_ratios) / len(self.clip_ratios)
                pl_module.log('train/avg_clip_ratio', avg_clip_ratio, on_step=True, prog_bar=False)
