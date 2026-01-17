"""Training health callback for comprehensive observability.

This callback tracks and logs key training health metrics including:
- Gradient norms (total and per-layer)
- Gradient clipping events
- Memory usage (current/peak)
- Learning rate
- Loss history
- Throughput (tokens/sec)
- Estimated time to completion (ETA)
- Anomaly detection (loss spikes, gradient explosions, NaN/Inf detection)

Warnings are displayed prominently in the dashboard with red styling when
anomalies are detected.

Usage:
    from neuromanifold_gpt.callbacks.training_health import TrainingHealthCallback

    callback = TrainingHealthCallback(log_interval=100)
    trainer.fit(model, callbacks=[callback])
"""
import sys
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Union

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table


class TrainingHealthCallback(Callback):
    """Track comprehensive training health metrics.

    Monitors gradient norms, memory usage, clipping events, and throughput
    to provide detailed observability during training.

    Args:
        log_interval: Number of steps between metric logging (default: 100)
        gradient_norm_history_size: Number of gradient norms to keep for statistics (default: 100)
    """

    def __init__(
        self,
        log_interval: int = 100,
        gradient_norm_history_size: int = 100,
        enable_dashboard: bool = True,
    ):
        self.log_interval = log_interval
        self.gradient_norm_history_size = gradient_norm_history_size

        # Detect if we're in a TTY environment (headless/non-TTY environments cannot use Rich Live)
        # Disable dashboard if stdout is not a TTY or if explicitly disabled
        self.is_tty = sys.stdout.isatty()
        self.enable_dashboard = enable_dashboard and self.is_tty

        # Gradient tracking
        self.grad_norm_history: Deque[float] = deque(maxlen=gradient_norm_history_size)
        self.grad_norm_before_clip: Optional[float] = None
        self.total_clip_events = 0
        self.total_steps = 0
        self.clip_ratios: Deque[float] = deque(
            maxlen=100
        )  # Track clip magnitude when clipping occurs

        # Memory tracking
        self.peak_memory_mb = 0.0

        # Timing and throughput
        self.step_start_time: Optional[float] = None
        self.step_times: Deque[float] = deque(maxlen=20)  # Rolling average
        self.training_start_time: Optional[float] = None
        self.max_steps: Optional[Union[int, float]] = None
        self.warmup_steps = 10  # Number of steps before showing ETA

        # Loss tracking
        self.loss_history: Deque[float] = deque(maxlen=100)

        # Anomaly detection
        self.warnings: List[Dict[str, Any]] = []  # List of warning messages
        self.loss_spike_threshold = 3.0  # Number of std devs for spike detection
        self.grad_explosion_threshold = (
            3.0  # Number of std devs for gradient explosion detection
        )
        self.min_loss_history_for_detection = (
            20  # Minimum samples before detecting anomalies
        )
        self.min_grad_history_for_detection = (
            20  # Minimum samples before detecting gradient explosions
        )

        # Rich dashboard
        self.console = Console()
        self.live: Optional[Live] = None
        self.current_step = 0
        self.current_loss = 0.0
        self.current_lr = 0.0
        self.current_grad_norm = 0.0
        self.current_memory_mb = 0.0
        self.mfu = 0.0

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
        std_loss = variance**0.5

        # Detect spike: current loss > mean + threshold * std
        if current_loss > mean_loss + self.loss_spike_threshold * std_loss:
            warning_msg = (
                f"[bold red]⚠ LOSS SPIKE DETECTED[/bold red] at step {current_step}: "
                f"Loss={current_loss:.4f} (mean={mean_loss:.4f}, std={std_loss:.4f}). "
                f"Consider reducing learning rate or checking for data issues."
            )
            self.warnings.append(
                {
                    "type": "loss_spike",
                    "step": current_step,
                    "message": warning_msg,
                    "loss": current_loss,
                    "mean": mean_loss,
                    "std": std_loss,
                }
            )

    def _detect_gradient_explosion(
        self, current_grad_norm: float, current_step: int
    ) -> None:
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
        variance = sum((x - mean_grad_norm) ** 2 for x in grad_norm_list) / len(
            grad_norm_list
        )
        std_grad_norm = variance**0.5

        # Detect explosion: current grad norm > mean + threshold * std
        if (
            current_grad_norm
            > mean_grad_norm + self.grad_explosion_threshold * std_grad_norm
        ):
            warning_msg = (
                f"[bold red]⚠ GRADIENT EXPLOSION DETECTED[/bold red] at step {current_step}: "
                f"GradNorm={current_grad_norm:.4f} (mean={mean_grad_norm:.4f}, std={std_grad_norm:.4f}). "
                f"Consider reducing learning rate or enabling gradient clipping."
            )
            self.warnings.append(
                {
                    "type": "gradient_explosion",
                    "step": current_step,
                    "message": warning_msg,
                    "grad_norm": current_grad_norm,
                    "mean": mean_grad_norm,
                    "std": std_grad_norm,
                }
            )

    def _detect_nan_inf_loss(self, loss_value: float, current_step: int) -> None:
        """Detect if loss contains NaN or Inf values.

        This is a critical error that requires immediate attention.

        Args:
            loss_value: The current loss value
            current_step: The current training step
        """
        has_nan = torch.isnan(torch.tensor(loss_value)).item()
        has_inf = torch.isinf(torch.tensor(loss_value)).item()

        if has_nan:
            warning_msg = (
                f"[bold red]🚨 CRITICAL: NaN DETECTED IN LOSS[/bold red] at step {current_step}. "
                f"Training is unstable. Check: (1) Learning rate too high, (2) Gradient explosion, "
                f"(3) Data preprocessing issues, (4) Numerical instability in model."
            )
            self.warnings.append(
                {
                    "type": "nan_loss",
                    "step": current_step,
                    "message": warning_msg,
                    "loss": loss_value,
                    "severity": "critical",
                }
            )

        if has_inf:
            warning_msg = (
                f"[bold red]🚨 CRITICAL: Inf DETECTED IN LOSS[/bold red] at step {current_step}. "
                f"Training is unstable. Check: (1) Learning rate too high, (2) Gradient explosion, "
                f"(3) Numerical overflow in model computations."
            )
            self.warnings.append(
                {
                    "type": "inf_loss",
                    "step": current_step,
                    "message": warning_msg,
                    "loss": loss_value,
                    "severity": "critical",
                }
            )

    def _detect_nan_inf_gradients(
        self, pl_module: LightningModule, current_step: int
    ) -> None:
        """Detect if gradients contain NaN or Inf values.

        This is a critical error that requires immediate attention.

        Args:
            pl_module: The PyTorch Lightning module
            current_step: The current training step
        """
        nan_params = []
        inf_params = []

        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    nan_params.append(name)
                if torch.isinf(param.grad).any():
                    inf_params.append(name)

        if nan_params:
            # Limit displayed params to first 5 for readability
            display_params = nan_params[:5]
            extra_count = len(nan_params) - 5
            params_str = ", ".join(display_params)
            if extra_count > 0:
                params_str += f" (and {extra_count} more)"

            warning_msg = (
                f"[bold red]🚨 CRITICAL: NaN DETECTED IN GRADIENTS[/bold red] at step {current_step}. "
                f"Affected parameters: {params_str}. "
                f"Check: (1) Learning rate too high, (2) Numerical instability, (3) Data issues."
            )
            self.warnings.append(
                {
                    "type": "nan_gradients",
                    "step": current_step,
                    "message": warning_msg,
                    "affected_params": nan_params,
                    "severity": "critical",
                }
            )

        if inf_params:
            # Limit displayed params to first 5 for readability
            display_params = inf_params[:5]
            extra_count = len(inf_params) - 5
            params_str = ", ".join(display_params)
            if extra_count > 0:
                params_str += f" (and {extra_count} more)"

            warning_msg = (
                f"[bold red]🚨 CRITICAL: Inf DETECTED IN GRADIENTS[/bold red] at step {current_step}. "
                f"Affected parameters: {params_str}. "
                f"Check: (1) Learning rate too high, (2) Gradient explosion, (3) Enable gradient clipping."
            )
            self.warnings.append(
                {
                    "type": "inf_gradients",
                    "step": current_step,
                    "message": warning_msg,
                    "affected_params": inf_params,
                    "severity": "critical",
                }
            )

    def _generate_dashboard(self):
        """Generate the Rich dashboard table with current metrics.

        Returns:
            Either a Table (if no warnings) or a Layout (if warnings exist)
        """
        table = Table(
            title="[bold cyan]Training Health Dashboard[/bold cyan]", show_header=True
        )

        # Add columns
        table.add_column("Metric", style="cyan", justify="left")
        table.add_column("Value", style="green", justify="right")
        table.add_column("Details", style="yellow", justify="left")

        # Add rows
        table.add_row("Step", f"{self.current_step}", f"Total: {self.total_steps}")

        table.add_row(
            "Loss",
            f"{self.current_loss:.4f}",
            f"Avg: {sum(self.loss_history) / len(self.loss_history):.4f}"
            if self.loss_history
            else "-",
        )

        table.add_row("Learning Rate", f"{self.current_lr:.6f}", "")

        table.add_row(
            "Grad Norm",
            f"{self.current_grad_norm:.4f}",
            f"Avg: {sum(self.grad_norm_history) / len(self.grad_norm_history):.4f}"
            if self.grad_norm_history
            else "-",
        )

        # Memory
        if torch.cuda.is_available():
            table.add_row(
                "GPU Memory",
                f"{self.current_memory_mb:.0f} MB",
                f"Peak: {self.peak_memory_mb:.0f} MB",
            )

        # MFU
        if self.mfu > 0:
            table.add_row("MFU", f"{self.mfu:.2f}%", "")

        # Gradient clipping stats
        if self.total_steps > 0:
            clip_rate = self.total_clip_events / self.total_steps * 100
            details = f"Rate: {clip_rate:.1f}%"
            if self.clip_ratios:
                avg_clip_ratio = sum(self.clip_ratios) / len(self.clip_ratios)
                details += f" | Avg ratio: {avg_clip_ratio:.3f}"
            table.add_row(
                "Grad Clipping", f"{self.total_clip_events}/{self.total_steps}", details
            )

        # Throughput
        if self.step_times:
            avg_step_time = sum(self.step_times) / len(self.step_times)
            table.add_row("Step Time", f"{avg_step_time * 1000:.1f} ms", "")

        # ETA (Estimated Time of Arrival)
        if (
            self.step_times
            and len(self.step_times) >= self.warmup_steps
            and self.max_steps is not None
            and self.max_steps > self.current_step
        ):
            avg_step_time = sum(self.step_times) / len(self.step_times)
            steps_remaining = self.max_steps - self.current_step
            eta_seconds = steps_remaining * avg_step_time
            table.add_row(
                "ETA", self._format_eta(eta_seconds), f"{steps_remaining} steps left"
            )
        elif self.max_steps is not None and len(self.step_times) < self.warmup_steps:
            # Show warmup message
            table.add_row(
                "ETA",
                "Warming up...",
                f"{self.warmup_steps - len(self.step_times)} steps to go",
            )

        # If no warnings, return just the table
        if not self.warnings:
            return table

        # If warnings exist, create a layout with warnings panel
        layout = Layout()
        layout.split_column(
            Layout(name="warnings", size=len(self.warnings) * 3 + 2),
            Layout(table, name="metrics"),
        )

        # Create warning panels - show last 3 warnings
        recent_warnings = self.warnings[-3:]
        warning_text = "\n\n".join([w["message"] for w in recent_warnings])
        warning_panel = Panel(
            warning_text,
            title="[bold red]⚠ Training Warnings[/bold red]",
            border_style="red",
            expand=False,
        )
        layout["warnings"].update(warning_panel)

        return layout

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
            if hasattr(trainer, "num_training_batches"):
                self.max_steps = trainer.max_epochs * trainer.num_training_batches

        if self.enable_dashboard:
            self.live = Live(
                self._generate_dashboard(), console=self.console, refresh_per_second=4
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

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Record start time for throughput calculation."""
        self.step_start_time = time.perf_counter()

    def on_after_backward(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Track gradient norm before clipping.

        Called after loss.backward() but before optimizer.step().
        This allows us to measure the pre-clip gradient norm.
        """
        # Detect NaN/Inf in gradients immediately (critical check)
        self._detect_nan_inf_gradients(pl_module, self.current_step)

        # Calculate gradient norm before clipping
        total_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5

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
        if hasattr(pl_module, "config") and pl_module.config.grad_clip > 0:
            # Calculate gradient norm after clipping
            total_norm_after = 0.0
            for p in pl_module.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm_after += param_norm.item() ** 2
            total_norm_after = total_norm_after**0.5

            # If gradient norm changed significantly, clipping occurred
            if self.grad_norm_before_clip is not None:
                clip_threshold = pl_module.config.grad_clip
                if (
                    self.grad_norm_before_clip > clip_threshold * 1.01
                ):  # Small tolerance for numerical precision
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
        """Log training health metrics at specified intervals."""
        # Update current step
        self.current_step = batch_idx

        # Calculate step time and throughput
        if self.step_start_time is not None:
            step_time = time.perf_counter() - self.step_start_time
            self.step_times.append(step_time)

            # Calculate tokens/sec
            if hasattr(batch, "__len__"):
                batch_size = len(batch)
            elif isinstance(batch, (list, tuple)):
                batch_size = batch[0].size(0) if len(batch) > 0 else 0
            elif isinstance(batch, dict):
                first_key = next(iter(batch.keys()))
                batch_size = batch[first_key].size(0)
            else:
                batch_size = 0

            seq_len = (
                batch[0].size(1)
                if isinstance(batch, (list, tuple)) and len(batch) > 0
                else 256
            )
            if isinstance(batch, dict) and "input_ids" in batch:
                seq_len = batch["input_ids"].size(1)

            tokens_per_step = batch_size * seq_len
            throughput = tokens_per_step / step_time if step_time > 0 else 0

            pl_module.log(
                "train/throughput_tokens_per_sec",
                throughput,
                on_step=True,
                prog_bar=False,
            )

            # Average step time
            avg_step_time = (
                sum(self.step_times) / len(self.step_times) if self.step_times else 0
            )
            pl_module.log(
                "train/step_time_ms", avg_step_time * 1000, on_step=True, prog_bar=False
            )

        # Log gradient norm statistics
        if self.grad_norm_history:
            current_grad_norm = self.grad_norm_history[-1]
            avg_grad_norm = sum(self.grad_norm_history) / len(self.grad_norm_history)

            # Update current grad norm for dashboard
            self.current_grad_norm = current_grad_norm

            # Detect gradient explosions (anomaly detection)
            self._detect_gradient_explosion(current_grad_norm, self.current_step)

            pl_module.log(
                "train/grad_norm", current_grad_norm, on_step=True, prog_bar=True
            )
            pl_module.log(
                "train/grad_norm_avg", avg_grad_norm, on_step=True, prog_bar=False
            )

        # Log gradient clipping statistics
        if self.total_steps > 0:
            clip_rate = self.total_clip_events / self.total_steps
            pl_module.log("train/clip_rate", clip_rate, on_step=True, prog_bar=False)
            pl_module.log(
                "train/total_clip_events",
                float(self.total_clip_events),
                on_step=True,
                prog_bar=False,
            )

            # Log average clip ratio if clipping has occurred
            if self.clip_ratios:
                avg_clip_ratio = sum(self.clip_ratios) / len(self.clip_ratios)
                pl_module.log(
                    "train/avg_clip_ratio", avg_clip_ratio, on_step=True, prog_bar=False
                )

        # Log memory usage (GPU only)
        if torch.cuda.is_available():
            current_memory_mb = torch.cuda.memory_allocated() / 1e6
            peak_memory_mb = torch.cuda.max_memory_allocated() / 1e6

            # Update peak
            if peak_memory_mb > self.peak_memory_mb:
                self.peak_memory_mb = peak_memory_mb

            # Update current memory for dashboard
            self.current_memory_mb = current_memory_mb

            pl_module.log(
                "train/memory_current_mb",
                current_memory_mb,
                on_step=True,
                prog_bar=True,
            )
            pl_module.log(
                "train/memory_peak_mb",
                self.peak_memory_mb,
                on_step=True,
                prog_bar=False,
            )

        # Log current loss if available in outputs
        if outputs is not None:
            if isinstance(outputs, dict) and "loss" in outputs:
                loss = outputs["loss"]
            elif isinstance(outputs, torch.Tensor):
                loss = outputs
            else:
                loss = None

            if loss is not None:
                loss_value = loss.item() if torch.is_tensor(loss) else float(loss)

                # Detect NaN/Inf in loss immediately (critical check)
                self._detect_nan_inf_loss(loss_value, self.current_step)

                self.loss_history.append(loss_value)

                # Update current loss for dashboard
                self.current_loss = loss_value

                # Detect loss spikes (anomaly detection)
                self._detect_loss_spike(loss_value, self.current_step)

                # Calculate loss statistics
                if len(self.loss_history) > 1:
                    avg_loss = sum(self.loss_history) / len(self.loss_history)
                    pl_module.log(
                        "train/loss_avg", avg_loss, on_step=True, prog_bar=False
                    )

        # Log learning rate
        if trainer.optimizers:
            optimizer = trainer.optimizers[0]
            lr = optimizer.param_groups[0]["lr"]

            # Update current LR for dashboard
            self.current_lr = lr

            pl_module.log("train/lr", lr, on_step=True, prog_bar=True)

        # Try to get MFU from logged metrics
        if hasattr(pl_module, "trainer") and hasattr(
            pl_module.trainer, "logged_metrics"
        ):
            if "train/mfu" in pl_module.trainer.logged_metrics:
                self.mfu = pl_module.trainer.logged_metrics["train/mfu"].item()

        # Update live dashboard
        if self.live is not None and batch_idx % self.log_interval == 0:
            self.live.update(self._generate_dashboard())
