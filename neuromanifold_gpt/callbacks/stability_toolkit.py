"""Training stability toolkit callbacks.

This module provides callbacks for detecting and monitoring common training
stability issues:
- SDR collapse detection (when SDR representations degenerate)
- Loss spike detection
- Checkpoint rollback on divergence

Usage:
    from neuromanifold_gpt.callbacks.stability_toolkit import SDRCollapseMonitor

    callback = SDRCollapseMonitor(check_interval=100)
    trainer.fit(model, callbacks=[callback])
"""
import sys
from collections import deque
from typing import Any, Dict, Deque, List, Optional, Tuple

import torch
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
