"""Unit tests for Optuna pruning callback."""

from unittest.mock import Mock

import optuna
import pytest

from neuromanifold_gpt.hpo.pruning import OptunaPruningCallback


def test_pruning_callback_init():
    """Test callback initialization."""
    trial = Mock()
    callback = OptunaPruningCallback(trial, monitor="val/loss")

    assert callback.trial == trial
    assert callback.monitor == "val/loss"


def test_pruning_callback_default_monitor():
    """Test default monitor metric."""
    trial = Mock()
    callback = OptunaPruningCallback(trial)

    assert callback.monitor == "val/loss"


def test_on_validation_end_reports_to_trial():
    """Test that callback reports metrics to Optuna trial."""
    trial = Mock()
    trial.should_prune = Mock(return_value=False)
    trial.report = Mock()

    callback = OptunaPruningCallback(trial, monitor="val/loss")

    # Mock trainer and pl_module
    trainer = Mock()
    trainer.global_step = 5
    trainer.callback_metrics = {"val/loss": 2.5}

    pl_module = Mock()

    # Call the callback
    callback.on_validation_end(trainer, pl_module)

    # Verify report was called with correct values
    trial.report.assert_called_once_with(2.5, 5)


def test_should_prune_raises_exception():
    """Test that pruning raises TrialPruned exception."""
    trial = Mock()
    trial.should_prune = Mock(return_value=True)
    trial.report = Mock()
    trial.number = 1

    callback = OptunaPruningCallback(trial, monitor="val/loss")

    trainer = Mock()
    trainer.global_step = 3
    trainer.callback_metrics = {"val/loss": 5.0}

    pl_module = Mock()

    # Should raise TrialPruned
    with pytest.raises(optuna.TrialPruned):
        callback.on_validation_end(trainer, pl_module)


def test_missing_metric_logs_warning():
    """Test that missing metric is handled gracefully."""
    trial = Mock()
    trial.should_prune = Mock(return_value=False)
    trial.report = Mock()

    callback = OptunaPruningCallback(trial, monitor="val/loss")

    trainer = Mock()
    trainer.global_step = 2
    trainer.callback_metrics = {}  # Missing val/loss

    pl_module = Mock()

    # Should not raise, just log warning (we can't easily test logging here)
    # This should not crash
    callback.on_validation_end(trainer, pl_module)

    # Report should not be called when metric is missing
    trial.report.assert_not_called()
