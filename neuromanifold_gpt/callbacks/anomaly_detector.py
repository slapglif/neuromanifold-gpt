"""Anomaly detection callback for critical training issues.

This callback detects critical anomalies during training:
- NaN values in loss
- Inf values in loss
- NaN values in gradients
- Inf values in gradients

These are critical errors that indicate training instability and require
immediate attention. Warnings are logged prominently when detected.

Usage:
    from neuromanifold_gpt.callbacks.anomaly_detector import AnomalyDetectorCallback

    callback = AnomalyDetectorCallback()
    trainer.fit(model, callbacks=[callback])
"""
from typing import Any, Dict, List

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback


class AnomalyDetectorCallback(Callback):
    """Detect critical training anomalies (NaN/Inf in gradients and loss).

    Monitors for NaN and Inf values in both gradients and loss values,
    which indicate critical training instability requiring immediate attention.

    Args:
        None - this callback has no configurable parameters
    """

    def __init__(self):
        # Anomaly tracking
        self.warnings: List[Dict[str, Any]] = []
        self.current_step = 0

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
                f"🚨 CRITICAL: NaN DETECTED IN LOSS at step {current_step}. "
                f"Training is unstable. Check: (1) Learning rate too high, (2) Gradient explosion, "
                f"(3) Data preprocessing issues, (4) Numerical instability in model."
            )
            self.warnings.append({
                'type': 'nan_loss',
                'step': current_step,
                'message': warning_msg,
                'loss': loss_value,
                'severity': 'critical'
            })
            # Print immediately for visibility
            print(f"\n{'=' * 80}")
            print(warning_msg)
            print(f"{'=' * 80}\n")

        if has_inf:
            warning_msg = (
                f"🚨 CRITICAL: Inf DETECTED IN LOSS at step {current_step}. "
                f"Training is unstable. Check: (1) Learning rate too high, (2) Gradient explosion, "
                f"(3) Numerical overflow in model computations."
            )
            self.warnings.append({
                'type': 'inf_loss',
                'step': current_step,
                'message': warning_msg,
                'loss': loss_value,
                'severity': 'critical'
            })
            # Print immediately for visibility
            print(f"\n{'=' * 80}")
            print(warning_msg)
            print(f"{'=' * 80}\n")

    def _detect_nan_inf_gradients(self, pl_module: LightningModule, current_step: int) -> None:
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
                f"🚨 CRITICAL: NaN DETECTED IN GRADIENTS at step {current_step}. "
                f"Affected parameters: {params_str}. "
                f"Check: (1) Learning rate too high, (2) Numerical instability, (3) Data issues."
            )
            self.warnings.append({
                'type': 'nan_gradients',
                'step': current_step,
                'message': warning_msg,
                'affected_params': nan_params,
                'severity': 'critical'
            })
            # Print immediately for visibility
            print(f"\n{'=' * 80}")
            print(warning_msg)
            print(f"{'=' * 80}\n")

        if inf_params:
            # Limit displayed params to first 5 for readability
            display_params = inf_params[:5]
            extra_count = len(inf_params) - 5
            params_str = ", ".join(display_params)
            if extra_count > 0:
                params_str += f" (and {extra_count} more)"

            warning_msg = (
                f"🚨 CRITICAL: Inf DETECTED IN GRADIENTS at step {current_step}. "
                f"Affected parameters: {params_str}. "
                f"Check: (1) Learning rate too high, (2) Gradient explosion, (3) Enable gradient clipping."
            )
            self.warnings.append({
                'type': 'inf_gradients',
                'step': current_step,
                'message': warning_msg,
                'affected_params': inf_params,
                'severity': 'critical'
            })
            # Print immediately for visibility
            print(f"\n{'=' * 80}")
            print(warning_msg)
            print(f"{'=' * 80}\n")

    def on_after_backward(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Detect NaN/Inf in gradients immediately after backward pass.

        Called after loss.backward() but before optimizer.step().
        This is the critical point to detect gradient anomalies.
        """
        # Detect NaN/Inf in gradients immediately (critical check)
        self._detect_nan_inf_gradients(pl_module, self.current_step)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Detect NaN/Inf in loss values at the end of each batch.

        Args:
            trainer: The PyTorch Lightning trainer
            pl_module: The PyTorch Lightning module
            outputs: The outputs from training_step
            batch: The current batch
            batch_idx: The current batch index
        """
        # Update current step
        self.current_step = batch_idx

        # Extract loss from outputs if available
        if outputs is not None:
            if isinstance(outputs, dict) and 'loss' in outputs:
                loss = outputs['loss']
            elif isinstance(outputs, torch.Tensor):
                loss = outputs
            else:
                loss = None

            if loss is not None:
                loss_value = loss.item() if torch.is_tensor(loss) else float(loss)

                # Detect NaN/Inf in loss immediately (critical check)
                self._detect_nan_inf_loss(loss_value, self.current_step)
