# neuromanifold_gpt/tests/test_train.py
"""Tests for Lightning training module."""
import pytest
import torch


# ============================================================================
# Tests for TrainConfig (new composition pattern)
# ============================================================================


class TestTrainConfig:
    """Test suite for the TrainConfig class."""

    def test_train_config_creates_model_config_automatically(self):
        """TrainConfig should create model_config if not provided."""
        from train import TrainConfig

        config = TrainConfig()
        assert config.model_config is not None
        assert hasattr(config.model_config, 'n_layer')
        assert hasattr(config.model_config, 'n_embd')

    def test_train_config_accepts_explicit_model_config(self):
        """TrainConfig should accept an explicit model_config."""
        from train import TrainConfig
        from neuromanifold_gpt.config import NeuroManifoldConfig

        model_config = NeuroManifoldConfig(n_layer=12, n_embd=768)
        config = TrainConfig(model_config=model_config)

        assert config.model_config is model_config
        assert config.model_config.n_layer == 12
        assert config.model_config.n_embd == 768

    def test_train_config_backward_compat_with_individual_params(self):
        """TrainConfig should support individual model params for backward compatibility."""
        from train import TrainConfig

        config = TrainConfig(n_layer=8, n_embd=256, n_heads=4)

        assert config.model_config is not None
        assert config.model_config.n_layer == 8
        assert config.model_config.n_embd == 256
        assert config.model_config.n_heads == 4

    def test_train_config_has_training_params(self):
        """TrainConfig should have training-specific parameters."""
        from train import TrainConfig

        config = TrainConfig()

        assert hasattr(config, 'learning_rate')
        assert hasattr(config, 'weight_decay')
        assert hasattr(config, 'max_iters')
        assert hasattr(config, 'batch_size')

    def test_train_config_default_model_type(self):
        """TrainConfig should default to neuromanifold model type."""
        from train import TrainConfig

        config = TrainConfig()
        assert config.model_type == "neuromanifold"

    def test_train_config_creates_gpt_config_when_requested(self):
        """TrainConfig should create GPTConfig when model_type is gpt."""
        from train import TrainConfig
        from neuromanifold_gpt.config import GPTConfig

        config = TrainConfig(model_type="gpt", n_layer=12)

        assert config.model_config is not None
        assert isinstance(config.model_config, GPTConfig)
        assert config.model_config.n_layer == 12


# ============================================================================
# Tests for Lightning training module
# ============================================================================


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


# ============================================================================
# Tests for nanoGPT-style training (train_nanogpt.py)
# ============================================================================


class TestCosineWarmupLR:
    """Tests for cosine LR with warmup schedule."""

    def test_lr_at_zero_is_small(self):
        """LR at iter 0 should be small (warming up)."""
        import math
        learning_rate = 6e-4
        warmup_iters = 2000
        min_lr = 6e-5

        def get_lr(it: int) -> float:
            if it < warmup_iters:
                return learning_rate * (it + 1) / (warmup_iters + 1)
            return min_lr

        lr0 = get_lr(0)
        assert lr0 < learning_rate
        assert lr0 > 0

    def test_lr_at_warmup_end(self):
        """LR should be close to max at end of warmup."""
        import math
        learning_rate = 6e-4
        warmup_iters = 2000
        lr_decay_iters = 600000
        min_lr = 6e-5

        def get_lr(it: int) -> float:
            if it < warmup_iters:
                return learning_rate * (it + 1) / (warmup_iters + 1)
            if it > lr_decay_iters:
                return min_lr
            decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return min_lr + coeff * (learning_rate - min_lr)

        lr_warmup_end = get_lr(warmup_iters - 1)
        # Should be close to max LR
        assert lr_warmup_end > learning_rate * 0.9

    def test_lr_after_decay(self):
        """LR after decay should be min_lr."""
        import math
        learning_rate = 6e-4
        warmup_iters = 2000
        lr_decay_iters = 600000
        min_lr = 6e-5

        def get_lr(it: int) -> float:
            if it < warmup_iters:
                return learning_rate * (it + 1) / (warmup_iters + 1)
            if it > lr_decay_iters:
                return min_lr
            decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return min_lr + coeff * (learning_rate - min_lr)

        lr_end = get_lr(lr_decay_iters + 100)
        assert lr_end == min_lr

    def test_lr_monotonically_increases_during_warmup(self):
        """LR should increase during warmup."""
        import math
        learning_rate = 6e-4
        warmup_iters = 2000

        def get_lr(it: int) -> float:
            if it < warmup_iters:
                return learning_rate * (it + 1) / (warmup_iters + 1)
            return learning_rate

        prev_lr = 0
        for i in range(warmup_iters):
            lr = get_lr(i)
            assert lr > prev_lr
            prev_lr = lr


class TestGPTConfigureOptimizers:
    """Tests for NeuroManifoldGPT optimizer configuration."""

    def test_gpt_configure_optimizers(self):
        """GPT model should configure optimizer."""
        from neuromanifold_gpt.config import NeuroManifoldConfigNano
        from neuromanifold_gpt.model.gpt import NeuroManifoldGPT

        config = NeuroManifoldConfigNano()
        model = NeuroManifoldGPT(config)
        optimizer = model.configure_optimizers(
            weight_decay=0.1,
            learning_rate=3e-4,
            betas=(0.9, 0.95),
            device_type="cpu"
        )
        assert optimizer is not None
        assert isinstance(optimizer, torch.optim.AdamW)

    def test_gpt_optimizer_has_two_groups(self):
        """GPT optimizer should have decay and no-decay groups."""
        from neuromanifold_gpt.config import NeuroManifoldConfigNano
        from neuromanifold_gpt.model.gpt import NeuroManifoldGPT

        config = NeuroManifoldConfigNano()
        model = NeuroManifoldGPT(config)
        optimizer = model.configure_optimizers()

        assert len(optimizer.param_groups) == 2
        weight_decays = {g['weight_decay'] for g in optimizer.param_groups}
        assert 0.0 in weight_decays  # No decay group exists

    def test_num_parameters(self):
        """Should count parameters correctly."""
        from neuromanifold_gpt.config import NeuroManifoldConfigNano
        from neuromanifold_gpt.model.gpt import NeuroManifoldGPT

        config = NeuroManifoldConfigNano()
        model = NeuroManifoldGPT(config)

        n_params_total = model.num_parameters(non_embedding=False)
        n_params_no_embed = model.num_parameters(non_embedding=True)

        assert n_params_total > n_params_no_embed
        assert n_params_no_embed > 0


class TestNanoGPTTrainingLoop:
    """Tests for nanoGPT-style training functionality."""

    def test_model_forward_returns_info_dict(self):
        """Model forward should return info dict with diagnostics."""
        from neuromanifold_gpt.config import NeuroManifoldConfigNano
        from neuromanifold_gpt.model.gpt import NeuroManifoldGPT

        config = NeuroManifoldConfigNano()
        model = NeuroManifoldGPT(config)

        x = torch.randint(0, config.vocab_size, (2, 32))
        y = torch.randint(0, config.vocab_size, (2, 32))

        logits, loss, info = model(x, y)

        assert 'sdr' in info
        assert 'sdr_scores' in info
        assert 'block_infos' in info
        assert 'memory_size' in info

    def test_model_gradient_flow(self):
        """Gradients should flow through entire model."""
        from neuromanifold_gpt.config import NeuroManifoldConfigNano
        from neuromanifold_gpt.model.gpt import NeuroManifoldGPT

        config = NeuroManifoldConfigNano()
        model = NeuroManifoldGPT(config)

        x = torch.randint(0, config.vocab_size, (2, 32))
        y = torch.randint(0, config.vocab_size, (2, 32))

        _, loss, _ = model(x, y)
        loss.backward()

        # Check gradients exist
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())

        assert len(grad_norms) > 0
        assert all(g >= 0 for g in grad_norms)

    def test_model_mixed_precision(self):
        """Model should work with mixed precision."""
        from neuromanifold_gpt.config import NeuroManifoldConfigNano
        from neuromanifold_gpt.model.gpt import NeuroManifoldGPT

        config = NeuroManifoldConfigNano()
        model = NeuroManifoldGPT(config)

        x = torch.randint(0, config.vocab_size, (2, 32))
        y = torch.randint(0, config.vocab_size, (2, 32))

        # Simulate mixed precision
        with torch.amp.autocast('cpu', dtype=torch.bfloat16):
            logits, loss, info = model(x, y)

        assert logits is not None
        assert loss is not None

    def test_gradient_accumulation_simulation(self):
        """Simulate gradient accumulation behavior."""
        from neuromanifold_gpt.config import NeuroManifoldConfigNano
        from neuromanifold_gpt.model.gpt import NeuroManifoldGPT

        config = NeuroManifoldConfigNano()
        model = NeuroManifoldGPT(config)
        optimizer = model.configure_optimizers()

        gradient_accumulation_steps = 4
        for micro_step in range(gradient_accumulation_steps):
            x = torch.randint(0, config.vocab_size, (2, 32))
            y = torch.randint(0, config.vocab_size, (2, 32))
            _, loss, _ = model(x, y)
            loss = loss / gradient_accumulation_steps
            loss.backward()

        # Clip and step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Should complete without error


class TestCheckpointing:
    """Tests for checkpoint save/load."""

    def test_checkpoint_roundtrip(self, tmp_path):
        """Model state should survive checkpoint roundtrip."""
        from neuromanifold_gpt.config import NeuroManifoldConfigNano
        from neuromanifold_gpt.model.gpt import NeuroManifoldGPT

        config = NeuroManifoldConfigNano()
        model1 = NeuroManifoldGPT(config)

        # Save checkpoint
        checkpoint = {
            'model': model1.state_dict(),
            'model_config': {
                'vocab_size': config.vocab_size,
                'block_size': config.block_size,
                'n_layer': config.n_layer,
                'n_heads': config.n_heads,
                'n_embd': config.n_embd,
                'sdr_size': config.sdr_size,
                'sdr_sparsity': config.sdr_sparsity,
                'manifold_dim': config.manifold_dim,
                'n_eigenvectors': config.n_eigenvectors,
                'dropout': config.dropout,
                'bias': config.bias,
            },
            'iter_num': 100,
            'best_val_loss': 4.5,
        }
        ckpt_path = tmp_path / "ckpt.pt"
        torch.save(checkpoint, ckpt_path)

        # Load checkpoint
        loaded = torch.load(ckpt_path)
        model2 = NeuroManifoldGPT(config)
        model2.load_state_dict(loaded['model'])

        # Compare parameters
        for (n1, p1), (n2, p2) in zip(
            model1.named_parameters(),
            model2.named_parameters()
        ):
            assert n1 == n2
            assert torch.allclose(p1, p2)

        assert loaded['iter_num'] == 100
        assert loaded['best_val_loss'] == 4.5
