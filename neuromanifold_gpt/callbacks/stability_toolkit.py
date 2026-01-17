"""Training stability toolkit callbacks.

This module provides callbacks for detecting and monitoring common training
stability issues:
- SDR collapse detection (when SDR representations degenerate)
- Loss spike detection
- Checkpoint rollback on divergence
- Attention pattern visualization

Usage:
    from neuromanifold_gpt.callbacks.stability_toolkit import SDRCollapseMonitor

    callback = SDRCollapseMonitor(check_interval=100)
    trainer.fit(model, callbacks=[callback])
"""
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, Deque, List, Optional, Tuple

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from rich.console import Console


class SDRCollapseMonitor(Callback):
    """Monitor SDR representations for mode collapse.

    Tracks SDR-specific metrics to detect when representations degenerate:
    - Active bits per SDR (should match expected sparsity)
    - Bit usage diversity (should use many bits, not concentrate on few)
    - Unique SDR patterns (should have high diversity, not repeat)
    - Temperature (should remain in reasonable range)
    - Duty cycle statistics (should be relatively uniform)

    Warnings are logged when collapse is detected, allowing early intervention.

    Args:
        check_interval: Number of steps between SDR health checks (default: 100)
        history_size: Number of samples to keep for statistics (default: 100)
        enable_warnings: Whether to print warnings to console (default: True)
        collapse_threshold: Threshold for unique pattern ratio to trigger warning (default: 0.3)
        bit_usage_threshold: Minimum fraction of bits that should be used (default: 0.5)
    """

    def __init__(
        self,
        check_interval: int = 100,
        history_size: int = 100,
        enable_warnings: bool = True,
        collapse_threshold: float = 0.3,
        bit_usage_threshold: float = 0.5,
    ):
        self.check_interval = check_interval
        self.history_size = history_size
        self.enable_warnings = enable_warnings
        self.collapse_threshold = collapse_threshold
        self.bit_usage_threshold = bit_usage_threshold

        # Detect if we're in a TTY environment
        self.is_tty = sys.stdout.isatty()

        # Tracking metrics
        self.active_bits_history: Deque[float] = deque(maxlen=history_size)
        self.unique_patterns_history: Deque[float] = deque(maxlen=history_size)
        self.bit_usage_history: Deque[float] = deque(maxlen=history_size)
        self.temperature_history: Deque[float] = deque(maxlen=history_size)
        self.duty_cycle_std_history: Deque[float] = deque(maxlen=history_size)

        # Warning tracking
        self.warnings: List[Dict[str, Any]] = []
        self.collapse_detected = False

        # Console for rich output
        self.console = Console()

        # Current step
        self.current_step = 0

    def _detect_low_diversity(
        self,
        unique_pattern_ratio: float,
        bit_usage_ratio: float,
        current_step: int,
    ) -> None:
        """Detect if SDR diversity has collapsed.

        Args:
            unique_pattern_ratio: Ratio of unique SDR patterns to total patterns
            bit_usage_ratio: Ratio of bits used to total bits available
            current_step: Current training step
        """
        if unique_pattern_ratio < self.collapse_threshold:
            warning_msg = (
                f"[bold red]⚠ SDR COLLAPSE DETECTED[/bold red] at step {current_step}: "
                f"Only {unique_pattern_ratio:.1%} unique patterns. "
                f"Consider: (1) Increasing temperature, (2) Checking token discrimination loss, "
                f"(3) Reducing learning rate for encoder."
            )
            self.warnings.append({
                'type': 'sdr_collapse',
                'step': current_step,
                'message': warning_msg,
                'unique_ratio': unique_pattern_ratio,
                'severity': 'high'
            })
            self.collapse_detected = True

        if bit_usage_ratio < self.bit_usage_threshold:
            warning_msg = (
                f"[bold yellow]⚠ LOW BIT USAGE[/bold yellow] at step {current_step}: "
                f"Only {bit_usage_ratio:.1%} of SDR bits are being used. "
                f"Consider: (1) Checking boosting mechanism, (2) Reviewing duty cycle updates."
            )
            self.warnings.append({
                'type': 'low_bit_usage',
                'step': current_step,
                'message': warning_msg,
                'bit_usage_ratio': bit_usage_ratio,
                'severity': 'medium'
            })

    def _detect_temperature_issues(
        self,
        temperature: float,
        current_step: int,
    ) -> None:
        """Detect if temperature has diverged.

        Args:
            temperature: Current SDR temperature value
            current_step: Current training step
        """
        # Temperature should typically be in range [0.01, 10.0]
        if temperature < 0.001:
            warning_msg = (
                f"[bold yellow]⚠ TEMPERATURE TOO LOW[/bold yellow] at step {current_step}: "
                f"Temperature={temperature:.6f}. This may cause hard discretization. "
                f"Consider: (1) Clipping temperature gradient, (2) Setting minimum temperature bound."
            )
            self.warnings.append({
                'type': 'low_temperature',
                'step': current_step,
                'message': warning_msg,
                'temperature': temperature,
                'severity': 'medium'
            })
        elif temperature > 100.0:
            warning_msg = (
                f"[bold yellow]⚠ TEMPERATURE TOO HIGH[/bold yellow] at step {current_step}: "
                f"Temperature={temperature:.2f}. This may cause loss of sparsity. "
                f"Consider: (1) Clipping temperature gradient, (2) Setting maximum temperature bound."
            )
            self.warnings.append({
                'type': 'high_temperature',
                'step': current_step,
                'message': warning_msg,
                'temperature': temperature,
                'severity': 'medium'
            })

    def _detect_duty_cycle_imbalance(
        self,
        duty_cycle_std: float,
        duty_cycle_max: float,
        current_step: int,
    ) -> None:
        """Detect if duty cycle is highly imbalanced.

        Args:
            duty_cycle_std: Standard deviation of duty cycle across bits
            duty_cycle_max: Maximum duty cycle value
            current_step: Current training step
        """
        # High std or max indicates some bits are overused
        # Expected duty cycle for uniform usage: 1/sdr_size
        # With boosting, should remain relatively balanced
        if duty_cycle_std > 0.01:  # Threshold may need tuning
            warning_msg = (
                f"[bold yellow]⚠ DUTY CYCLE IMBALANCE[/bold yellow] at step {current_step}: "
                f"Std={duty_cycle_std:.6f}, Max={duty_cycle_max:.6f}. "
                f"Some bits are being overused. Check boosting mechanism."
            )
            self.warnings.append({
                'type': 'duty_cycle_imbalance',
                'step': current_step,
                'message': warning_msg,
                'duty_std': duty_cycle_std,
                'duty_max': duty_cycle_max,
                'severity': 'low'
            })

    def _analyze_sdr_health(
        self,
        pl_module: LightningModule,
        batch: Any,
    ) -> Tuple[float, float, float, float, float, float]:
        """Analyze SDR health metrics from a batch.

        Args:
            pl_module: The PyTorch Lightning module
            batch: Current training batch

        Returns:
            Tuple of (avg_active_bits, unique_pattern_ratio, bit_usage_ratio,
                     temperature, duty_cycle_std, duty_cycle_max)
        """
        # Check if model has SDR encoder
        if not hasattr(pl_module, 'encoder'):
            return 0.0, 1.0, 1.0, 1.0, 0.0, 0.0

        # Get tokens from batch
        if isinstance(batch, (list, tuple)) and len(batch) >= 1:
            tokens = batch[0]
        elif isinstance(batch, dict) and 'input_ids' in batch:
            tokens = batch['input_ids']
        else:
            return 0.0, 1.0, 1.0, 1.0, 0.0, 0.0

        # Get SDRs from encoder
        with torch.no_grad():
            encoder = pl_module.encoder
            sdr, scores, _, _, _ = encoder(tokens)

            # 1. Active bits per SDR
            active_per_sdr = sdr.sum(dim=-1)  # (B, T)
            avg_active_bits = active_per_sdr.float().mean().item()

            # 2. Unique SDR patterns
            B, T, sdr_size = sdr.shape
            sdr_flat = sdr.view(-1, sdr_size)  # (B*T, sdr_size)

            # Convert to hashable tuples for uniqueness check
            # Only check a sample to avoid memory issues
            sample_size = min(1000, sdr_flat.size(0))
            if sample_size > 0:
                indices = torch.randperm(sdr_flat.size(0))[:sample_size]
                sdr_sample = sdr_flat[indices]

                # Convert to tuples of active indices
                sdr_hashes = set()
                for i in range(sdr_sample.size(0)):
                    active_indices = tuple(sdr_sample[i].nonzero().squeeze(-1).tolist())
                    sdr_hashes.add(active_indices)

                unique_count = len(sdr_hashes)
                unique_pattern_ratio = unique_count / sample_size
            else:
                unique_pattern_ratio = 1.0

            # 3. Bit usage across all SDRs
            bit_usage = sdr.sum(dim=(0, 1))  # (sdr_size,)
            bits_used = (bit_usage > 0).sum().item()
            bit_usage_ratio = bits_used / sdr_size

            # 4. Temperature
            temperature = encoder.temperature.abs().item()

            # 5. Duty cycle statistics
            duty_cycle = encoder.bit_duty_cycle
            duty_cycle_std = duty_cycle.std().item()
            duty_cycle_max = duty_cycle.max().item()

        return (
            avg_active_bits,
            unique_pattern_ratio,
            bit_usage_ratio,
            temperature,
            duty_cycle_std,
            duty_cycle_max,
        )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Check SDR health at specified intervals."""
        self.current_step = batch_idx

        # Only check at specified intervals
        if batch_idx % self.check_interval != 0:
            return

        # Skip if model doesn't use SDR
        if not hasattr(pl_module, 'encoder'):
            return

        # Analyze SDR health
        (
            avg_active_bits,
            unique_pattern_ratio,
            bit_usage_ratio,
            temperature,
            duty_cycle_std,
            duty_cycle_max,
        ) = self._analyze_sdr_health(pl_module, batch)

        # Track history
        self.active_bits_history.append(avg_active_bits)
        self.unique_patterns_history.append(unique_pattern_ratio)
        self.bit_usage_history.append(bit_usage_ratio)
        self.temperature_history.append(temperature)
        self.duty_cycle_std_history.append(duty_cycle_std)

        # Log metrics
        pl_module.log('sdr/active_bits_avg', avg_active_bits, on_step=True, prog_bar=False)
        pl_module.log('sdr/unique_pattern_ratio', unique_pattern_ratio, on_step=True, prog_bar=False)
        pl_module.log('sdr/bit_usage_ratio', bit_usage_ratio, on_step=True, prog_bar=True)
        pl_module.log('sdr/temperature', temperature, on_step=True, prog_bar=False)
        pl_module.log('sdr/duty_cycle_std', duty_cycle_std, on_step=True, prog_bar=False)
        pl_module.log('sdr/duty_cycle_max', duty_cycle_max, on_step=True, prog_bar=False)

        # Detect anomalies
        self._detect_low_diversity(
            unique_pattern_ratio,
            bit_usage_ratio,
            self.current_step,
        )
        self._detect_temperature_issues(temperature, self.current_step)
        self._detect_duty_cycle_imbalance(
            duty_cycle_std,
            duty_cycle_max,
            self.current_step,
        )

        # Print warnings if enabled
        if self.enable_warnings and self.warnings and self.is_tty:
            # Print only new warnings (last one added)
            warning = self.warnings[-1]
            self.console.print(warning['message'])

    def on_train_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Print summary of SDR health at end of training."""
        if not self.active_bits_history:
            return

        self.console.print("\n[bold cyan]SDR Health Summary[/bold cyan]")
        self.console.print("=" * 60)

        # Average metrics
        avg_active = sum(self.active_bits_history) / len(self.active_bits_history)
        avg_unique = sum(self.unique_patterns_history) / len(self.unique_patterns_history)
        avg_bit_usage = sum(self.bit_usage_history) / len(self.bit_usage_history)
        avg_temp = sum(self.temperature_history) / len(self.temperature_history)

        self.console.print(f"Average Active Bits: {avg_active:.1f}")
        self.console.print(f"Average Unique Pattern Ratio: {avg_unique:.1%}")
        self.console.print(f"Average Bit Usage Ratio: {avg_bit_usage:.1%}")
        self.console.print(f"Average Temperature: {avg_temp:.6f}")

        # Collapse status
        if self.collapse_detected:
            self.console.print("\n[bold red]⚠ SDR collapse was detected during training[/bold red]")
        else:
            self.console.print("\n[bold green]✓ No SDR collapse detected[/bold green]")

        # Warning summary
        if self.warnings:
            self.console.print(f"\nTotal warnings: {len(self.warnings)}")
            warning_types = {}
            for w in self.warnings:
                wtype = w['type']
                warning_types[wtype] = warning_types.get(wtype, 0) + 1
            for wtype, count in warning_types.items():
                self.console.print(f"  - {wtype}: {count}")


class DivergenceRollbackCallback(Callback):
    """Automatically rollback to previous checkpoint when training diverges.

    Monitors training loss and triggers automatic rollback when divergence is detected:
    - Tracks rolling loss statistics
    - Saves periodic checkpoints for rollback candidates
    - Detects when loss > 2x recent average for N consecutive steps
    - Loads best recent checkpoint and resumes training

    This callback helps recover from training instabilities without manual intervention.

    Args:
        checkpoint_dir: Directory to save rollback checkpoints (default: "out/rollback_checkpoints")
        loss_history_size: Number of loss values to track for statistics (default: 100)
        divergence_threshold: Multiplier of average loss to trigger divergence (default: 2.0)
        consecutive_divergence_steps: Number of consecutive divergent steps before rollback (default: 3)
        checkpoint_interval: Number of steps between checkpoint saves (default: 500)
        max_checkpoints: Maximum number of checkpoints to keep (default: 10)
        enable_warnings: Whether to print warnings to console (default: True)
    """

    def __init__(
        self,
        checkpoint_dir: str = "out/rollback_checkpoints",
        loss_history_size: int = 100,
        divergence_threshold: float = 2.0,
        consecutive_divergence_steps: int = 3,
        checkpoint_interval: int = 500,
        max_checkpoints: int = 10,
        enable_warnings: bool = True,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.loss_history_size = loss_history_size
        self.divergence_threshold = divergence_threshold
        self.consecutive_divergence_steps = consecutive_divergence_steps
        self.checkpoint_interval = checkpoint_interval
        self.max_checkpoints = max_checkpoints
        self.enable_warnings = enable_warnings

        # Detect if we're in a TTY environment
        self.is_tty = sys.stdout.isatty()

        # Loss tracking
        self.loss_history: Deque[float] = deque(maxlen=loss_history_size)
        self.consecutive_divergent_count = 0
        self.min_loss_history_for_detection = 20  # Minimum samples before detecting divergence

        # Checkpoint tracking
        self.checkpoint_paths: Deque[Tuple[int, str, float]] = deque(maxlen=max_checkpoints)
        # Each entry: (step, checkpoint_path, loss_value)

        # Rollback tracking
        self.total_rollbacks = 0
        self.last_rollback_step = -1

        # Console for rich output
        self.console = Console()

        # Current step
        self.current_step = 0

    def setup(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        stage: Optional[str] = None,
    ) -> None:
        """Create checkpoint directory on setup."""
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def _get_current_loss(self, trainer: Trainer) -> Optional[float]:
        """Extract current loss from trainer's logged metrics.

        Args:
            trainer: PyTorch Lightning trainer

        Returns:
            Current loss value, or None if not available
        """
        # Try to get from callback_metrics (most recent)
        if 'loss' in trainer.callback_metrics:
            loss_tensor = trainer.callback_metrics['loss']
            return loss_tensor.item() if torch.is_tensor(loss_tensor) else float(loss_tensor)

        # Try to get from logged_metrics
        if hasattr(trainer, 'logged_metrics') and 'loss' in trainer.logged_metrics:
            loss_tensor = trainer.logged_metrics['loss']
            return loss_tensor.item() if torch.is_tensor(loss_tensor) else float(loss_tensor)

        # Try train_loss
        if 'train_loss' in trainer.callback_metrics:
            loss_tensor = trainer.callback_metrics['train_loss']
            return loss_tensor.item() if torch.is_tensor(loss_tensor) else float(loss_tensor)

        return None

    def _detect_divergence(self, current_loss: float, current_step: int) -> bool:
        """Detect if training is diverging based on loss statistics.

        Args:
            current_loss: Current loss value
            current_step: Current training step

        Returns:
            True if divergence detected, False otherwise
        """
        # Need enough history to calculate meaningful statistics
        if len(self.loss_history) < self.min_loss_history_for_detection:
            return False

        # Calculate rolling average loss
        loss_list = list(self.loss_history)
        avg_loss = sum(loss_list) / len(loss_list)

        # Check if current loss exceeds threshold
        if current_loss > self.divergence_threshold * avg_loss:
            self.consecutive_divergent_count += 1

            if self.enable_warnings and self.is_tty:
                self.console.print(
                    f"[yellow]⚠ Divergence warning ({self.consecutive_divergent_count}/"
                    f"{self.consecutive_divergence_steps})[/yellow] at step {current_step}: "
                    f"Loss={current_loss:.4f} > {self.divergence_threshold}x avg "
                    f"({self.divergence_threshold * avg_loss:.4f})"
                )

            # Trigger rollback if consecutive divergence exceeds threshold
            if self.consecutive_divergent_count >= self.consecutive_divergence_steps:
                return True
        else:
            # Reset counter if loss is within normal range
            self.consecutive_divergent_count = 0

        return False

    def _save_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        step: int,
        loss: float,
    ) -> None:
        """Save checkpoint for potential rollback.

        Args:
            trainer: PyTorch Lightning trainer
            pl_module: PyTorch Lightning module
            step: Current training step
            loss: Current loss value
        """
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"rollback_ckpt-{step:06d}-{loss:.4f}.ckpt"
        )

        # Save checkpoint
        trainer.save_checkpoint(checkpoint_path)

        # Track checkpoint
        self.checkpoint_paths.append((step, checkpoint_path, loss))

        if self.enable_warnings and self.is_tty:
            self.console.print(
                f"[dim]Saved rollback checkpoint at step {step} "
                f"(loss={loss:.4f})[/dim]"
            )

    def _find_best_checkpoint(self) -> Optional[Tuple[int, str, float]]:
        """Find best checkpoint for rollback (lowest loss).

        Returns:
            Tuple of (step, checkpoint_path, loss), or None if no checkpoints available
        """
        if not self.checkpoint_paths:
            return None

        # Find checkpoint with lowest loss
        best_checkpoint = min(self.checkpoint_paths, key=lambda x: x[2])
        return best_checkpoint

    def _rollback_to_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint_info: Tuple[int, str, float],
        current_step: int,
    ) -> None:
        """Rollback training to a previous checkpoint.

        Args:
            trainer: PyTorch Lightning trainer
            pl_module: PyTorch Lightning module
            checkpoint_info: Tuple of (step, checkpoint_path, loss)
            current_step: Current (divergent) training step
        """
        step, checkpoint_path, loss = checkpoint_info

        if self.enable_warnings and self.is_tty:
            self.console.print(
                f"\n[bold red]🔄 ROLLBACK TRIGGERED[/bold red] at step {current_step}\n"
                f"Rolling back to checkpoint from step {step} (loss={loss:.4f})\n"
            )

        # Load checkpoint
        try:
            checkpoint = torch.load(checkpoint_path)

            # Restore model state
            pl_module.load_state_dict(checkpoint['state_dict'])

            # Restore optimizer states
            if 'optimizer_states' in checkpoint:
                for idx, optimizer in enumerate(trainer.optimizers):
                    if idx < len(checkpoint['optimizer_states']):
                        optimizer.load_state_dict(checkpoint['optimizer_states'][idx])

            # Restore learning rate scheduler states (if present)
            if 'lr_schedulers' in checkpoint:
                for idx, scheduler_config in enumerate(trainer.lr_scheduler_configs):
                    if idx < len(checkpoint['lr_schedulers']):
                        scheduler_config.scheduler.load_state_dict(checkpoint['lr_schedulers'][idx])

            # Clear loss history after rollback to start fresh
            self.loss_history.clear()
            self.consecutive_divergent_count = 0
            self.total_rollbacks += 1
            self.last_rollback_step = current_step

            if self.enable_warnings and self.is_tty:
                self.console.print(
                    f"[bold green]✓ Rollback successful[/bold green]\n"
                    f"Restored model and optimizer from step {step}\n"
                )

        except Exception as e:
            if self.enable_warnings and self.is_tty:
                self.console.print(
                    f"[bold red]✗ Rollback failed: {str(e)}[/bold red]\n"
                )
            raise

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Monitor loss and trigger rollback if divergence detected."""
        self.current_step = trainer.global_step

        # Get current loss
        current_loss = self._get_current_loss(trainer)
        if current_loss is None:
            return

        # Check for NaN/Inf - immediate rollback
        if torch.isnan(torch.tensor(current_loss)).item() or torch.isinf(torch.tensor(current_loss)).item():
            if self.enable_warnings and self.is_tty:
                self.console.print(
                    f"[bold red]⚠ NaN/Inf DETECTED[/bold red] at step {self.current_step}: "
                    f"Loss={current_loss}"
                )

            # Try to rollback if we have checkpoints
            best_checkpoint = self._find_best_checkpoint()
            if best_checkpoint:
                self._rollback_to_checkpoint(trainer, pl_module, best_checkpoint, self.current_step)
                return
            else:
                if self.enable_warnings and self.is_tty:
                    self.console.print(
                        "[bold red]No checkpoints available for rollback. Continuing...[/bold red]"
                    )
                return

        # Add to loss history
        self.loss_history.append(current_loss)

        # Save checkpoint periodically
        if self.current_step % self.checkpoint_interval == 0 and self.current_step > 0:
            self._save_checkpoint(trainer, pl_module, self.current_step, current_loss)

        # Detect divergence
        if self._detect_divergence(current_loss, self.current_step):
            # Find best checkpoint and rollback
            best_checkpoint = self._find_best_checkpoint()
            if best_checkpoint:
                self._rollback_to_checkpoint(trainer, pl_module, best_checkpoint, self.current_step)
            else:
                if self.enable_warnings and self.is_tty:
                    self.console.print(
                        "[bold yellow]Divergence detected but no checkpoints available for rollback[/bold yellow]"
                    )

    def on_train_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Print rollback summary at end of training."""
        self.console.print("\n[bold cyan]Divergence Rollback Summary[/bold cyan]")
        self.console.print("=" * 60)

        if self.total_rollbacks > 0:
            self.console.print(f"Total rollbacks: {self.total_rollbacks}")
            self.console.print(f"Last rollback at step: {self.last_rollback_step}")
            self.console.print(
                "[yellow]⚠ Training experienced divergence and was rolled back[/yellow]"
            )
        else:
            self.console.print("[bold green]✓ No rollbacks needed - training was stable[/bold green]")

        # Checkpoint summary
        if self.checkpoint_paths:
            self.console.print(f"\nCheckpoints saved: {len(self.checkpoint_paths)}")
            best_checkpoint = self._find_best_checkpoint()
            if best_checkpoint:
                step, path, loss = best_checkpoint
                self.console.print(f"Best checkpoint: step {step} (loss={loss:.4f})")


class AttentionVisualizationCallback(Callback):
    """Visualize and save attention patterns periodically during training.

    Captures attention weights from the model at regular intervals and saves
    them as PNG heatmap visualizations. Useful for debugging attention mechanisms,
    monitoring attention collapse, and understanding how attention patterns evolve
    during training.

    The callback uses forward hooks to capture attention weights without modifying
    the model's forward pass. Visualizations are saved to the specified output
    directory with filenames indicating the step number.

    Args:
        output_dir: Directory to save attention visualizations (default: "out/attention_viz")
        save_interval: Number of steps between saving visualizations (default: 500)
        max_seq_len: Maximum sequence length to visualize (default: 64)
            Long sequences are truncated to this length for visualization
        layer_indices: List of layer indices to visualize (default: None, saves all layers)
            Example: [0, -1] to save first and last layer only
        save_multihead: Whether to save individual attention heads (default: False)
            If True, saves each head separately. If False, saves averaged attention.
        enable_console_output: Whether to print progress messages (default: True)
    """

    def __init__(
        self,
        output_dir: str = "out/attention_viz",
        save_interval: int = 500,
        max_seq_len: int = 64,
        layer_indices: Optional[List[int]] = None,
        save_multihead: bool = False,
        enable_console_output: bool = True,
    ):
        self.output_dir = output_dir
        self.save_interval = save_interval
        self.max_seq_len = max_seq_len
        self.layer_indices = layer_indices
        self.save_multihead = save_multihead
        self.enable_console_output = enable_console_output

        # Detect if we're in a TTY environment
        self.is_tty = sys.stdout.isatty()

        # Storage for captured attention patterns
        self.attention_weights: Dict[str, torch.Tensor] = {}
        self.hooks: List[Any] = []

        # Console for rich output
        self.console = Console()

        # Current step
        self.current_step = 0

        # Track number of visualizations saved
        self.num_saved = 0

    def setup(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        stage: Optional[str] = None,
    ) -> None:
        """Create output directory on setup."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def _attention_hook(self, layer_name: str):
        """Create a forward hook to capture attention weights.

        Args:
            layer_name: Name identifier for this layer

        Returns:
            Hook function that captures attention weights
        """
        def hook(module, input, output):
            # For standard attention, we need to recompute attention weights
            # Since the forward pass doesn't return them by default
            # We'll capture them during the forward pass by storing q, k after computation
            pass

        return hook

    def _register_hooks(self, pl_module: LightningModule) -> None:
        """Register forward hooks on attention layers to capture attention weights.

        Args:
            pl_module: PyTorch Lightning module
        """
        # Find all attention layers in the model
        # Standard GPT model structure: model.transformer.h[i].attn
        if not hasattr(pl_module, 'transformer'):
            return

        transformer = pl_module.transformer
        if not hasattr(transformer, 'h'):
            return

        # Determine which layers to hook
        num_layers = len(transformer.h)
        if self.layer_indices is None:
            # Hook all layers
            indices = list(range(num_layers))
        else:
            # Convert negative indices to positive
            indices = [idx if idx >= 0 else num_layers + idx for idx in self.layer_indices]

        # Register hooks on selected layers
        for idx in indices:
            if idx < 0 or idx >= num_layers:
                continue

            layer = transformer.h[idx]
            if not hasattr(layer, 'attn'):
                continue

            attn_layer = layer.attn

            # Create a hook that captures attention weights
            def make_hook(layer_idx):
                def hook(module, input, output):
                    # We need to capture attention weights during forward pass
                    # For Flash Attention, this is not directly available
                    # For manual attention, we can capture the 'att' variable
                    # Since we can't modify the forward pass here, we'll use a different approach
                    pass
                return hook

            hook_handle = attn_layer.register_forward_hook(make_hook(idx))
            self.hooks.append(hook_handle)

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def _extract_attention_weights(
        self,
        pl_module: LightningModule,
        batch: Any,
    ) -> Dict[str, torch.Tensor]:
        """Extract attention weights from model by running a forward pass.

        Since attention weights are not returned by default, we temporarily
        modify the attention computation to capture them.

        Args:
            pl_module: PyTorch Lightning module
            batch: Current batch of data

        Returns:
            Dictionary mapping layer names to attention weight tensors
        """
        attention_weights = {}

        # Check if model has transformer structure
        if not hasattr(pl_module, 'transformer'):
            return attention_weights

        transformer = pl_module.transformer
        if not hasattr(transformer, 'h'):
            return attention_weights

        # Get input tokens from batch
        if isinstance(batch, (list, tuple)) and len(batch) >= 1:
            tokens = batch[0]
        elif isinstance(batch, dict) and 'input_ids' in batch:
            tokens = batch['input_ids']
        else:
            return attention_weights

        # Limit sequence length for visualization
        seq_len = min(tokens.size(1), self.max_seq_len)
        tokens = tokens[:, :seq_len]

        # Only use first sample from batch for visualization
        tokens = tokens[:1]

        # Run forward pass with attention capturing
        # We need to temporarily patch the attention forward method
        num_layers = len(transformer.h)
        if self.layer_indices is None:
            indices = list(range(num_layers))
        else:
            indices = [idx if idx >= 0 else num_layers + idx for idx in self.layer_indices]

        with torch.no_grad():
            # For each attention layer, we'll manually compute attention weights
            for idx in indices:
                if idx < 0 or idx >= num_layers:
                    continue

                layer = transformer.h[idx]
                if not hasattr(layer, 'attn'):
                    continue

                attn_layer = layer.attn

                # Get intermediate activations
                # We need to access the attention computation
                # This requires knowing the model structure
                # For standard CausalSelfAttention, we can recompute attention weights

                # Get embeddings up to this layer
                x = tokens
                if hasattr(pl_module, 'get_layer_input'):
                    x = pl_module.get_layer_input(tokens, idx)
                else:
                    # Manually compute layer input
                    x = transformer.wte(tokens)
                    if hasattr(transformer, 'wpe'):
                        pos = torch.arange(0, seq_len, dtype=torch.long, device=tokens.device).unsqueeze(0)
                        x = x + transformer.wpe(pos)
                    x = transformer.drop(x)

                    # Pass through layers up to target layer
                    for layer_idx in range(idx):
                        x = transformer.h[layer_idx](x)

                # Now compute attention for this layer
                # Apply layer norm before attention
                if hasattr(layer, 'ln_1'):
                    attn_input = layer.ln_1(x)
                else:
                    attn_input = x

                # Manually compute attention weights
                B, T, C = attn_input.size()

                # Get Q, K, V
                if hasattr(attn_layer, 'c_attn'):
                    qkv = attn_layer.c_attn(attn_input)
                    q, k, v = qkv.split(attn_layer.n_embd, dim=2)

                    # Reshape for multi-head attention
                    k = k.view(B, T, attn_layer.n_head, C // attn_layer.n_head).transpose(1, 2)
                    q = q.view(B, T, attn_layer.n_head, C // attn_layer.n_head).transpose(1, 2)
                    v = v.view(B, T, attn_layer.n_head, C // attn_layer.n_head).transpose(1, 2)

                    # Apply RoPE if present
                    if attn_layer.rope is not None:
                        q, k = attn_layer.rope(q, k)

                    # Compute attention weights (without using flash attention)
                    att = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(k.size(-1), dtype=torch.float32)))

                    # Add ALiBi bias if present
                    if attn_layer.alibi is not None:
                        alibi_bias = attn_layer.alibi(T)
                        att = att + alibi_bias.squeeze(0)

                    # Apply causal mask
                    if hasattr(attn_layer, 'bias'):
                        att = att.masked_fill(attn_layer.bias[:, :, :T, :T] == 0, float('-inf'))

                    # Softmax to get attention weights
                    att = torch.nn.functional.softmax(att, dim=-1)

                    # Store attention weights
                    # Shape: (B, num_heads, T, T)
                    attention_weights[f'layer_{idx}'] = att.cpu()

        return attention_weights

    def _save_attention_visualizations(
        self,
        attention_weights: Dict[str, torch.Tensor],
        step: int,
    ) -> None:
        """Save attention weight visualizations to PNG files.

        Args:
            attention_weights: Dictionary mapping layer names to attention tensors
            step: Current training step
        """
        # Import visualization utility
        try:
            from neuromanifold_gpt.utils.attention_viz import visualize_attention_pattern
        except ImportError:
            if self.enable_console_output and self.is_tty:
                self.console.print(
                    "[yellow]Warning: Could not import attention visualization utility[/yellow]"
                )
            return

        for layer_name, att_weights in attention_weights.items():
            # att_weights shape: (B, num_heads, T, T)
            # Take first batch item
            if att_weights.dim() == 4:
                att_weights = att_weights[0]  # Now (num_heads, T, T)

            # Save visualization
            if self.save_multihead:
                # Save multi-head visualization
                output_path = os.path.join(
                    self.output_dir,
                    f"attention_{layer_name}_step_{step:06d}_multihead.png"
                )
                visualize_attention_pattern(
                    att_weights,
                    output_path=output_path,
                    title=f"Attention Pattern - {layer_name.replace('_', ' ').title()} - Step {step}",
                    dpi=150,  # Lower DPI for faster saving
                )
            else:
                # Average across heads and save single visualization
                att_avg = att_weights.mean(dim=0)  # (T, T)
                output_path = os.path.join(
                    self.output_dir,
                    f"attention_{layer_name}_step_{step:06d}.png"
                )
                visualize_attention_pattern(
                    att_avg,
                    output_path=output_path,
                    title=f"Attention Pattern (Averaged) - {layer_name.replace('_', ' ').title()} - Step {step}",
                    dpi=150,
                )

        self.num_saved += 1

        if self.enable_console_output and self.is_tty:
            self.console.print(
                f"[dim]Saved attention visualizations for step {step} "
                f"({len(attention_weights)} layers)[/dim]"
            )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Save attention visualizations at specified intervals."""
        self.current_step = trainer.global_step

        # Only save at specified intervals
        if self.current_step % self.save_interval != 0:
            return

        # Skip if step 0 (initialization)
        if self.current_step == 0:
            return

        try:
            # Extract attention weights
            attention_weights = self._extract_attention_weights(pl_module, batch)

            if not attention_weights:
                if self.enable_console_output and self.is_tty and self.num_saved == 0:
                    self.console.print(
                        "[yellow]Warning: Could not extract attention weights from model[/yellow]"
                    )
                return

            # Save visualizations
            self._save_attention_visualizations(attention_weights, self.current_step)

        except Exception as e:
            if self.enable_console_output and self.is_tty:
                self.console.print(
                    f"[yellow]Warning: Failed to save attention visualization at step {self.current_step}: {str(e)}[/yellow]"
                )

    def on_train_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Print summary at end of training."""
        # Remove any registered hooks
        self._remove_hooks()

        if self.enable_console_output and self.is_tty:
            self.console.print("\n[bold cyan]Attention Visualization Summary[/bold cyan]")
            self.console.print("=" * 60)
            self.console.print(f"Total visualizations saved: {self.num_saved}")
            self.console.print(f"Output directory: {self.output_dir}")
            if self.num_saved > 0:
                self.console.print("[bold green]✓ Attention patterns saved successfully[/bold green]")
            else:
                self.console.print("[yellow]⚠ No attention patterns were saved[/yellow]")
