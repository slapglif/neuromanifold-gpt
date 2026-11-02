"""Training dashboard callback for displaying training metrics in Rich console.

This callback displays a live dashboard with training metrics read from
trainer.logged_metrics. It does not track metrics itself - it only reads
and displays metrics that have been logged by other callbacks.

Metrics displayed (when available):
- Step/Epoch information
- Loss (current and average)
- Learning rate
- Gradient norm (current and average)
- GPU memory usage (current/peak)
- MFU (Model FLOPs Utilization)
- Gradient clipping statistics
- Step time and throughput
- ETA (Estimated Time to Arrival)

Usage:
    from neuromanifold_gpt.callbacks.training_dashboard import TrainingDashboardCallback

    # Use with other monitoring callbacks that log metrics
    from neuromanifold_gpt.callbacks.gradient_monitor import GradientMonitorCallback
    from neuromanifold_gpt.callbacks.memory_monitor import MemoryMonitorCallback
    from neuromanifold_gpt.callbacks.loss_monitor import LossMonitorCallback

    callbacks = [
        GradientMonitorCallback(),
        MemoryMonitorCallback(),
        LossMonitorCallback(),
        TrainingDashboardCallback(refresh_rate=100)
    ]
    trainer.fit(model, callbacks=callbacks)
"""
import sys
import time
from typing import Any, Optional, Union

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout


class TrainingDashboardCallback(Callback):
    """Display training metrics in a Rich console dashboard.

    Reads metrics from trainer.logged_metrics and displays them in a live
    dashboard. Does not track metrics itself - relies on other callbacks
    to log metrics.

    Args:
        refresh_rate: Number of steps between dashboard updates (default: 100)
        enable_dashboard: Whether to enable the dashboard display (default: True)
    """

    def __init__(
        self,
        refresh_rate: int = 100,
        enable_dashboard: bool = True,
    ):
        self.refresh_rate = refresh_rate

        # Detect if we're in a TTY environment (headless/non-TTY environments cannot use Rich Live)
        # Disable dashboard if stdout is not a TTY or if explicitly disabled
        self.is_tty = sys.stdout.isatty()
        self.enable_dashboard = enable_dashboard and self.is_tty

        # Rich dashboard
        self.console = Console()
        self.live: Optional[Live] = None

        # Timing for ETA calculation
        self.training_start_time: Optional[float] = None
        self.max_steps: Optional[Union[int, float]] = None
        self.warmup_steps = 10  # Number of steps before showing ETA

    def _format_eta(self, seconds: float) -> str:
        """Format ETA in human-readable format (e.g., '2h 34m', '45m 12s')."""
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

    def _generate_dashboard(self, trainer: Trainer, pl_module: LightningModule) -> Union[Table, Layout]:
        """Generate the Rich dashboard table with current metrics.

        Reads all metrics from trainer.logged_metrics.

        Args:
            trainer: The PyTorch Lightning trainer
            pl_module: The PyTorch Lightning module

        Returns:
            Either a Table (if no warnings) or a Layout (if warnings exist)
        """
        table = Table(title="[bold cyan]Training Dashboard[/bold cyan]", show_header=True)

        # Add columns
        table.add_column("Metric", style="cyan", justify="left")
        table.add_column("Value", style="green", justify="right")
        table.add_column("Details", style="yellow", justify="left")

        # Get metrics from trainer
        metrics = trainer.logged_metrics if hasattr(trainer, 'logged_metrics') else {}

        # Current step
        current_step = trainer.global_step
        total_steps = self.max_steps if self.max_steps else "Unknown"
        table.add_row(
            "Step",
            f"{current_step}",
            f"Total: {total_steps}"
        )

        # Loss
        if 'loss' in metrics or 'train/loss' in metrics:
            loss_key = 'train/loss' if 'train/loss' in metrics else 'loss'
            current_loss = metrics[loss_key].item()
            details = ""
            if 'train/loss_avg' in metrics:
                avg_loss = metrics['train/loss_avg'].item()
                details = f"Avg: {avg_loss:.4f}"
            table.add_row(
                "Loss",
                f"{current_loss:.4f}",
                details
            )

        # Learning rate
        if 'train/lr' in metrics or 'lr' in metrics:
            lr_key = 'train/lr' if 'train/lr' in metrics else 'lr'
            current_lr = metrics[lr_key].item()
            table.add_row(
                "Learning Rate",
                f"{current_lr:.6f}",
                ""
            )

        # Gradient norm
        if 'train/grad_norm' in metrics:
            current_grad_norm = metrics['train/grad_norm'].item()
            details = ""
            if 'train/grad_norm_avg' in metrics:
                avg_grad_norm = metrics['train/grad_norm_avg'].item()
                details = f"Avg: {avg_grad_norm:.4f}"
            table.add_row(
                "Grad Norm",
                f"{current_grad_norm:.4f}",
                details
            )

        # Memory
        if torch.cuda.is_available() and 'train/memory_current_mb' in metrics:
            current_memory_mb = metrics['train/memory_current_mb'].item()
            details = ""
            if 'train/memory_peak_mb' in metrics:
                peak_memory_mb = metrics['train/memory_peak_mb'].item()
                details = f"Peak: {peak_memory_mb:.0f} MB"
            table.add_row(
                "GPU Memory",
                f"{current_memory_mb:.0f} MB",
                details
            )

        # MFU
        if 'train/mfu' in metrics:
            mfu = metrics['train/mfu'].item()
            table.add_row(
                "MFU",
                f"{mfu:.2f}%",
                ""
            )

        # Gradient clipping stats
        if 'train/total_clip_events' in metrics:
            total_clip_events = int(metrics['train/total_clip_events'].item())
            total_steps_count = current_step
            clip_rate = metrics['train/clip_rate'].item() * 100 if 'train/clip_rate' in metrics else 0
            details = f"Rate: {clip_rate:.1f}%"
            if 'train/avg_clip_ratio' in metrics:
                avg_clip_ratio = metrics['train/avg_clip_ratio'].item()
                details += f" | Avg ratio: {avg_clip_ratio:.3f}"
            table.add_row(
                "Grad Clipping",
                f"{total_clip_events}/{total_steps_count}",
                details
            )

        # Step time
        if 'train/step_time_ms' in metrics:
            avg_step_time_ms = metrics['train/step_time_ms'].item()
            table.add_row(
                "Step Time",
                f"{avg_step_time_ms:.1f} ms",
                ""
            )

        # Throughput
        if 'train/throughput_tokens_per_sec' in metrics:
            throughput = metrics['train/throughput_tokens_per_sec'].item()
            table.add_row(
                "Throughput",
                f"{throughput:.0f} tok/s",
                ""
            )

        # ETA (Estimated Time of Arrival)
        if (self.training_start_time is not None and
            self.max_steps is not None and
            self.max_steps > current_step and
            'train/step_time_ms' in metrics):

            # Only show ETA after warmup
            if current_step >= self.warmup_steps:
                avg_step_time_ms = metrics['train/step_time_ms'].item()
                avg_step_time = avg_step_time_ms / 1000  # Convert to seconds
                steps_remaining = self.max_steps - current_step
                eta_seconds = steps_remaining * avg_step_time
                table.add_row(
                    "ETA",
                    self._format_eta(eta_seconds),
                    f"{steps_remaining} steps left"
                )
            else:
                # Show warmup message
                table.add_row(
                    "ETA",
                    "Warming up...",
                    f"{self.warmup_steps - current_step} steps to go"
                )

        # Check for warnings in logged metrics
        # Warnings would be logged by AnomalyDetectorCallback or other monitoring callbacks
        # For now, we just return the table
        # In the future, we could check for warning metrics and display them in a layout
        return table

    def on_train_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Initialize the live dashboard."""
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

        if self.enable_dashboard:
            self.live = Live(
                self._generate_dashboard(trainer, pl_module),
                console=self.console,
                refresh_per_second=4
            )
            self.live.start()

    def on_train_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Stop the live dashboard."""
        if self.live is not None:
            self.live.stop()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Update the dashboard at specified intervals."""
        # Update live dashboard
        if self.live is not None and batch_idx % self.refresh_rate == 0:
            self.live.update(self._generate_dashboard(trainer, pl_module))
