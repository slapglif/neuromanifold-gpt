# neuromanifold_gpt/tests/test_training_modules.py
"""Integration tests for refactored training modules."""
import pytest
import torch
import tempfile
import os
import pickle
import numpy as np
from unittest.mock import MagicMock


def test_train_config_initialization():
    """TrainConfig should initialize with defaults."""
    from neuromanifold_gpt.training.config import TrainConfig

    config = TrainConfig()
    assert config.out_dir == "out-lightning"
    assert config.batch_size == 64
    assert config.block_size == 256
    assert config.learning_rate == 3e-4
    assert config.weight_decay == 0.1
    assert config.model_type == "neuromanifold"


def test_train_config_custom_values():
    """TrainConfig should accept custom values."""
    from neuromanifold_gpt.training.config import TrainConfig

    config = TrainConfig(
        batch_size=32,
        learning_rate=1e-3,
        n_layer=12,
        use_kan=False,
    )
    assert config.batch_size == 32
    assert config.learning_rate == 1e-3
    assert config.n_layer == 12
    assert config.use_kan is False


def test_memmap_dataset_initialization():
    """MemmapDataset should initialize with data file."""
    from neuromanifold_gpt.training.data_modules import MemmapDataset

    # Create temporary memmap file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        temp_path = f.name
        data = np.random.randint(0, 100, size=1000, dtype=np.uint16)
        data.tofile(f)

    try:
        dataset = MemmapDataset(temp_path, block_size=128)
        assert len(dataset) > 0
        x, y = dataset[0]
        assert x.shape == (128,)
        assert y.shape == (128,)
    finally:
        os.unlink(temp_path)


def test_text_data_module_initialization():
    """TextDataModule should initialize."""
    from neuromanifold_gpt.training.data_modules import TextDataModule

    data_module = TextDataModule(
        data_dir="data/test",
        block_size=256,
        batch_size=32,
        num_workers=0,
    )
    assert data_module.block_size == 256
    assert data_module.batch_size == 32


def test_text_data_module_setup():
    """TextDataModule should setup with memmap files."""
    from neuromanifold_gpt.training.data_modules import TextDataModule

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock train and val files
        train_path = os.path.join(temp_dir, "train.bin")
        val_path = os.path.join(temp_dir, "val.bin")
        meta_path = os.path.join(temp_dir, "meta.pkl")

        train_data = np.random.randint(0, 100, size=1000, dtype=np.uint16)
        val_data = np.random.randint(0, 100, size=500, dtype=np.uint16)

        train_data.tofile(train_path)
        val_data.tofile(val_path)

        # Create meta file
        meta = {"vocab_size": 100, "itos": None, "stoi": None}
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)

        data_module = TextDataModule(
            data_dir=temp_dir,
            block_size=64,
            batch_size=4,
            num_workers=0,
        )
        data_module.setup()

        assert data_module.vocab_size == 100
        assert data_module.train_ds is not None
        assert data_module.val_ds is not None


def test_streaming_data_module_initialization():
    """StreamingDataModule should initialize."""
    from neuromanifold_gpt.training.data_modules import StreamingDataModule

    data_module = StreamingDataModule(
        dataset_name="test/dataset",
        block_size=256,
        batch_size=32,
        num_workers=0,
    )
    assert data_module.block_size == 256
    assert data_module.batch_size == 32
    assert data_module.vocab_size == 50257  # GPT-2 BPE


def test_lightning_module_initialization():
    """NeuroManifoldLitModule should initialize."""
    from neuromanifold_gpt.training.lightning_module import NeuroManifoldLitModule
    from neuromanifold_gpt.config import NeuroManifoldConfigNano

    config = NeuroManifoldConfigNano()
    module = NeuroManifoldLitModule(config)
    assert module.model is not None
    assert module.config == config


def test_lightning_module_training_step():
    """NeuroManifoldLitModule training step should work."""
    from neuromanifold_gpt.training.lightning_module import NeuroManifoldLitModule
    from neuromanifold_gpt.config import NeuroManifoldConfigNano

    config = NeuroManifoldConfigNano()
    module = NeuroManifoldLitModule(config)

    batch = {
        'input_ids': torch.randint(0, config.vocab_size, (2, 50)),
        'labels': torch.randint(0, config.vocab_size, (2, 50))
    }

    loss = module.training_step(batch, 0)
    assert loss.ndim == 0
    assert not torch.isnan(loss)


def test_lightning_module_validation_step():
    """NeuroManifoldLitModule validation step should work."""
    from neuromanifold_gpt.training.lightning_module import NeuroManifoldLitModule
    from neuromanifold_gpt.config import NeuroManifoldConfigNano

    config = NeuroManifoldConfigNano()
    module = NeuroManifoldLitModule(config)

    batch = {
        'input_ids': torch.randint(0, config.vocab_size, (2, 50)),
        'labels': torch.randint(0, config.vocab_size, (2, 50))
    }

    loss = module.validation_step(batch, 0)
    assert loss.ndim == 0
    assert not torch.isnan(loss)


def test_lightning_module_configure_optimizers():
    """NeuroManifoldLitModule should configure optimizer and scheduler."""
    from neuromanifold_gpt.training.lightning_module import NeuroManifoldLitModule
    from neuromanifold_gpt.config import NeuroManifoldConfigNano

    config = NeuroManifoldConfigNano()
    module = NeuroManifoldLitModule(config)

    result = module.configure_optimizers()
    assert 'optimizer' in result
    assert 'lr_scheduler' in result

    optimizer = result['optimizer']
    assert len(optimizer.param_groups) == 2  # decay and no_decay

    # Check weight decay values
    weight_decays = [g['weight_decay'] for g in optimizer.param_groups]
    assert config.weight_decay in weight_decays
    assert 0.0 in weight_decays


def test_sample_generation_callback_initialization():
    """SampleGenerationCallback should initialize."""
    from neuromanifold_gpt.training.callbacks import SampleGenerationCallback

    callback = SampleGenerationCallback(
        sample_interval=100,
        max_tokens=50,
        temperature=0.8,
        top_k=20,
    )
    assert callback.sample_interval == 100
    assert callback.max_tokens == 50
    assert callback.temperature == 0.8
    assert callback.top_k == 20


def test_mfu_callback_initialization():
    """MFUCallback should initialize."""
    from neuromanifold_gpt.training.callbacks import MFUCallback

    callback = MFUCallback(log_interval=50)
    assert callback.log_interval == 50
    assert callback.running_mfu == -1.0


def test_integration_imports():
    """All training components should be importable from training submodules."""
    from neuromanifold_gpt.training.config import TrainConfig
    from neuromanifold_gpt.training.data_modules import (
        MemmapDataset,
        TextDataModule,
        StreamingDataModule,
    )
    from neuromanifold_gpt.training.lightning_module import NeuroManifoldLitModule
    from neuromanifold_gpt.training.callbacks import (
        SampleGenerationCallback,
        MFUCallback,
    )
    from neuromanifold_gpt.training.trainer import train

    # Check that all expected components are importable
    assert TrainConfig is not None
    assert MemmapDataset is not None
    assert TextDataModule is not None
    assert StreamingDataModule is not None
    assert NeuroManifoldLitModule is not None
    assert SampleGenerationCallback is not None
    assert MFUCallback is not None
    assert train is not None


def test_integration_data_to_module():
    """Integration: Data module should work with Lightning module."""
    from neuromanifold_gpt.training.data_modules import TextDataModule
    from neuromanifold_gpt.training.lightning_module import NeuroManifoldLitModule
    from neuromanifold_gpt.config import NeuroManifoldConfigNano

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock data
        train_path = os.path.join(temp_dir, "train.bin")
        val_path = os.path.join(temp_dir, "val.bin")
        meta_path = os.path.join(temp_dir, "meta.pkl")

        train_data = np.random.randint(0, 100, size=1000, dtype=np.uint16)
        val_data = np.random.randint(0, 100, size=500, dtype=np.uint16)

        train_data.tofile(train_path)
        val_data.tofile(val_path)

        meta = {"vocab_size": 100, "itos": None, "stoi": None}
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)

        # Setup data module
        data_module = TextDataModule(
            data_dir=temp_dir,
            block_size=64,
            batch_size=4,
            num_workers=0,
        )
        data_module.setup()

        # Create Lightning module with matching vocab size
        config = NeuroManifoldConfigNano()
        config.vocab_size = data_module.vocab_size
        module = NeuroManifoldLitModule(config)

        # Get a batch from data loader
        train_loader = data_module.train_dataloader()
        batch_x, batch_y = next(iter(train_loader))

        # Should be able to run training step
        batch_dict = {'input_ids': batch_x, 'labels': batch_y}
        loss = module.training_step(batch_dict, 0)
        assert not torch.isnan(loss)


def test_integration_config_to_module():
    """Integration: TrainConfig should configure Lightning module."""
    from neuromanifold_gpt.training.config import TrainConfig
    from neuromanifold_gpt.training.lightning_module import NeuroManifoldLitModule
    from neuromanifold_gpt.config import NeuroManifoldConfig

    train_config = TrainConfig(
        learning_rate=1e-3,
        weight_decay=0.05,
        beta1=0.85,
        beta2=0.90,
        grad_clip=0.5,
    )

    model_config = NeuroManifoldConfig(
        vocab_size=100,
        n_layer=4,
        n_heads=4,
        n_embd=128,
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        beta1=train_config.beta1,
        beta2=train_config.beta2,
        grad_clip=train_config.grad_clip,
    )

    module = NeuroManifoldLitModule(model_config)
    optimizer_config = module.configure_optimizers()

    # Check optimizer uses config values
    optimizer = optimizer_config['optimizer']
    assert optimizer.param_groups[0]['weight_decay'] == 0.05
    # Learning rate should match
    for param_group in optimizer.param_groups:
        assert param_group['lr'] == 1e-3


def test_callbacks_work_with_lightning_module():
    """Integration: Callbacks should work with Lightning module."""
    from neuromanifold_gpt.training.lightning_module import NeuroManifoldLitModule
    from neuromanifold_gpt.training.callbacks import (
        SampleGenerationCallback,
        MFUCallback,
    )
    from neuromanifold_gpt.training.config import TrainConfig
    from neuromanifold_gpt.config import NeuroManifoldConfigNano

    config = NeuroManifoldConfigNano()
    module = NeuroManifoldLitModule(config)

    # MFUCallback expects train_config on the module
    module.train_config = TrainConfig()

    # Create callbacks
    sample_callback = SampleGenerationCallback(
        sample_interval=10,
        max_tokens=20,
    )
    mfu_callback = MFUCallback(log_interval=5)

    # Callbacks should be instantiable
    assert sample_callback is not None
    assert mfu_callback is not None

    # Mock trainer
    mock_trainer = MagicMock()
    mock_trainer.global_step = 10

    # Sample callback should handle batch end
    sample_callback.on_train_batch_end(
        mock_trainer, module, None, None, 0
    )

    # MFU callback should handle batch end (need a proper batch tuple)
    batch = (torch.zeros(2, 50), torch.zeros(2, 50))
    mfu_callback.on_train_batch_end(
        mock_trainer, module, None, batch, 0
    )


def test_lightning_module_logs_metrics():
    """Lightning module should log metrics during training."""
    from neuromanifold_gpt.training.lightning_module import NeuroManifoldLitModule
    from neuromanifold_gpt.config import NeuroManifoldConfigNano
    from unittest.mock import MagicMock

    config = NeuroManifoldConfigNano()
    module = NeuroManifoldLitModule(config)
    module.log = MagicMock()

    batch = {
        'input_ids': torch.randint(0, config.vocab_size, (2, 50)),
        'labels': torch.randint(0, config.vocab_size, (2, 50))
    }

    # Training step should log train_loss
    module.training_step(batch, 0)
    module.log.assert_called()
    call_args = [call[0] for call in module.log.call_args_list]
    logged_names = [args[0] for args in call_args]
    assert 'train_loss' in logged_names

    # Reset mock
    module.log.reset_mock()

    # Validation step should log val_loss and val_perplexity
    module.validation_step(batch, 0)
    module.log.assert_called()
    call_args = [call[0] for call in module.log.call_args_list]
    logged_names = [args[0] for args in call_args]
    assert 'val_loss' in logged_names
    assert 'val_perplexity' in logged_names


def test_module_has_all_config_attributes():
    """Lightning module should preserve config attributes."""
    from neuromanifold_gpt.training.lightning_module import NeuroManifoldLitModule
    from neuromanifold_gpt.config import NeuroManifoldConfig

    config = NeuroManifoldConfig(
        vocab_size=100,
        n_layer=6,
        n_heads=6,
        n_embd=384,
        learning_rate=3e-4,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        grad_clip=1.0,
    )

    module = NeuroManifoldLitModule(config)

    assert module.config.vocab_size == 100
    assert module.config.n_layer == 6
    assert module.config.learning_rate == 3e-4
    assert module.config.weight_decay == 0.1
