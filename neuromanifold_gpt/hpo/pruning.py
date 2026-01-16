"""
Optuna pruning callback for PyTorch Lightning.

This module provides a callback that integrates Optuna's trial pruning
with PyTorch Lightning training. It monitors validation loss and allows
Optuna to stop unpromising trials early, saving computational resources.
"""

from typing import Optional

import optuna
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from loguru import logger


class OptunaPruningCallback(Callback):
    """PyTorch Lightning callback for Optuna trial pruning.

    This callback integrates with Optuna's pruning mechanism to stop
    unpromising trials early during training. It reports validation
    loss to the trial at regular intervals and checks if the trial
    should be pruned based on the configured pruning algorithm.

    Pruning is beneficial when:
    - Running many trials with limited compute budget
    - Early validation loss strongly correlates with final performance
    - Some hyperparameter configurations are clearly inferior

    The callback hooks into Lightning's validation cycle to:
    1. Report validation loss to Optuna after each validation
    2. Check if the trial should be pruned based on intermediate results
    3. Raise TrialPruned exception to stop training if pruning is triggered

    Args:
        trial: Optuna trial instance for reporting and pruning decisions
        monitor: Metric name to monitor for pruning (default: "val/loss")

    Example:
        >>> import optuna
        >>> trial = study.ask()
        >>> callback = OptunaPruningCallback(trial, monitor="val/loss")
        >>> trainer = pl.Trainer(callbacks=[callback])
        >>> trainer.fit(model, datamodule)
    """

    def __init__(
        self,
        trial: optuna.Trial,
        monitor: str = "val/loss",
    ):
        """Initialize the pruning callback.

        Args:
            trial: Optuna trial instance
            monitor: Name of the metric to monitor (must be logged by model)
        """
        self.trial = trial
        self.monitor = monitor
        self._current_step = 0

    def on_validation_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Called after validation completes.

        Reports the monitored metric to Optuna and checks if the trial
        should be pruned. If pruning is triggered, raises TrialPruned
        exception to stop training.

        Args:
            trainer: PyTorch Lightning trainer instance
            pl_module: Lightning module being trained

        Raises:
            optuna.TrialPruned: If Optuna decides to prune this trial
        """
        # Get current metric value
        current_score = trainer.callback_metrics.get(self.monitor)

        if current_score is None:
            logger.warning(
                f"Metric '{self.monitor}' not found in callback_metrics. "
                f"Available metrics: {list(trainer.callback_metrics.keys())}"
            )
            return

        # Convert tensor to float if needed
        if hasattr(current_score, "item"):
            current_score = current_score.item()

        # Use global_step as the step for reporting
        step = trainer.global_step

        # Report to Optuna
        self.trial.report(current_score, step)

        # Check if trial should be pruned
        if self.trial.should_prune():
            message = f"Trial {self.trial.number} pruned at step {step} with {self.monitor}={current_score:.4f}"
            logger.info(message)
            raise optuna.TrialPruned(message)
