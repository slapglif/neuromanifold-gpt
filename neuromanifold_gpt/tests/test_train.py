# neuromanifold_gpt/tests/test_train.py
"""Tests for Lightning training module."""
import pytest
import torch


def test_lightning_module_init():
    """Lightning module should initialize."""
    from neuromanifold_gpt.config import NeuroManifoldConfigNano
    from neuromanifold_gpt.train import NeuroManifoldLightning

    config = NeuroManifoldConfigNano()
    module = NeuroManifoldLightning(config)
    assert module.model is not None


def test_training_step():
    """Training step should return loss."""
    from neuromanifold_gpt.config import NeuroManifoldConfigNano
    from neuromanifold_gpt.train import NeuroManifoldLightning

    config = NeuroManifoldConfigNano()
    module = NeuroManifoldLightning(config)
    batch = {
        'input_ids': torch.randint(0, config.vocab_size, (2, 50)),
        'labels': torch.randint(0, config.vocab_size, (2, 50))
    }
    loss = module.training_step(batch, 0)
    assert loss.ndim == 0
    assert not torch.isnan(loss)


def test_configure_optimizers():
    """Should return optimizer and scheduler."""
    from neuromanifold_gpt.config import NeuroManifoldConfigNano
    from neuromanifold_gpt.train import NeuroManifoldLightning

    config = NeuroManifoldConfigNano()
    module = NeuroManifoldLightning(config)
    result = module.configure_optimizers()
    assert 'optimizer' in result


def test_validation_step():
    """Validation step should return loss."""
    from neuromanifold_gpt.config import NeuroManifoldConfigNano
    from neuromanifold_gpt.train import NeuroManifoldLightning

    config = NeuroManifoldConfigNano()
    module = NeuroManifoldLightning(config)
    batch = {
        'input_ids': torch.randint(0, config.vocab_size, (2, 50)),
        'labels': torch.randint(0, config.vocab_size, (2, 50))
    }
    loss = module.validation_step(batch, 0)
    assert loss.ndim == 0
    assert not torch.isnan(loss)


def test_optimizer_has_weight_decay_groups():
    """Optimizer should have separate weight decay groups."""
    from neuromanifold_gpt.config import NeuroManifoldConfigNano
    from neuromanifold_gpt.train import NeuroManifoldLightning

    config = NeuroManifoldConfigNano()
    module = NeuroManifoldLightning(config)
    result = module.configure_optimizers()
    optimizer = result['optimizer']

    # Should have 2 param groups: decay and no_decay
    assert len(optimizer.param_groups) == 2

    # Check weight decay values
    weight_decays = [g['weight_decay'] for g in optimizer.param_groups]
    assert config.weight_decay in weight_decays
    assert 0.0 in weight_decays


def test_scheduler_is_cosine():
    """Should return cosine annealing scheduler."""
    from neuromanifold_gpt.config import NeuroManifoldConfigNano
    from neuromanifold_gpt.train import NeuroManifoldLightning

    config = NeuroManifoldConfigNano()
    module = NeuroManifoldLightning(config)
    result = module.configure_optimizers()

    assert 'lr_scheduler' in result
    scheduler_config = result['lr_scheduler']
    assert 'scheduler' in scheduler_config


def test_training_step_logs_metrics():
    """Training step should log train_loss."""
    from neuromanifold_gpt.config import NeuroManifoldConfigNano
    from neuromanifold_gpt.train import NeuroManifoldLightning
    from unittest.mock import MagicMock

    config = NeuroManifoldConfigNano()
    module = NeuroManifoldLightning(config)
    module.log = MagicMock()

    batch = {
        'input_ids': torch.randint(0, config.vocab_size, (2, 50)),
        'labels': torch.randint(0, config.vocab_size, (2, 50))
    }
    module.training_step(batch, 0)

    # Check that log was called with train_loss
    module.log.assert_called()
    call_args = [call[0] for call in module.log.call_args_list]
    logged_names = [args[0] for args in call_args]
    assert 'train_loss' in logged_names


def test_config_has_training_params():
    """Config should have all training parameters."""
    from neuromanifold_gpt.config import NeuroManifoldConfig

    config = NeuroManifoldConfig()

    assert hasattr(config, 'learning_rate')
    assert hasattr(config, 'weight_decay')
    assert hasattr(config, 'beta1')
    assert hasattr(config, 'beta2')
    assert hasattr(config, 'grad_clip')

    # Check default values
    assert config.learning_rate == 3e-4
    assert config.weight_decay == 0.1
    assert config.beta1 == 0.9
    assert config.beta2 == 0.95
    assert config.grad_clip == 1.0
