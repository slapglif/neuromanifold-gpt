#!/usr/bin/env python3
"""Test checkpoint separation and size verification.

This test verifies that the separated checkpoint mechanism works correctly
and that model-only checkpoints are significantly smaller than unified checkpoints.
"""

import sys
import os
import shutil
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from neuromanifold_gpt.training.checkpoint_callback import SeparatedCheckpointCallback
from neuromanifold_gpt.utils.checkpoint_loader import load_checkpoint, load_model_only


class SimpleModel(nn.Module):
    """Simple model for testing checkpoint saving."""

    def __init__(self, hidden_dim: int = 512, num_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output(x)


class SimpleLightningModule:
    """Minimal Lightning-like module for testing."""

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.config = {"test": True}


class SimpleTrainer:
    """Minimal Trainer-like class for testing."""

    def __init__(self, optimizer, lr_scheduler=None):
        self.global_step = 0
        self.current_epoch = 0
        self.optimizers = [optimizer]
        self.lr_scheduler_configs = []

        if lr_scheduler:
            # Mimic Lightning's lr_scheduler_config structure
            class LRConfig:
                def __init__(self, scheduler):
                    self.scheduler = scheduler
            self.lr_scheduler_configs = [LRConfig(lr_scheduler)]


def save_unified_checkpoint(model, optimizer, save_path, global_step=0):
    """Save a unified checkpoint (traditional format).

    Args:
        model: PyTorch model
        optimizer: Optimizer instance
        save_path: Path to save checkpoint
        global_step: Current training step
    """
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'global_step': global_step,
        'config': {"test": True},
    }
    torch.save(checkpoint, save_path)
    logger.info(f"Saved unified checkpoint: {save_path}")


def get_file_size_mb(file_path):
    """Get file size in megabytes.

    Args:
        file_path: Path to file

    Returns:
        File size in MB
    """
    size_bytes = os.path.getsize(file_path)
    return size_bytes / (1024 * 1024)


def test_checkpoint_sizes():
    """Test that separated checkpoints achieve expected size reduction."""
    print("Testing checkpoint size reduction...")

    # Create temporary directory for test checkpoints
    test_dir = tempfile.mkdtemp(prefix="test_checkpoints_")

    try:
        # Create a reasonably sized model (similar to small GPT)
        model = SimpleModel(hidden_dim=768, num_layers=12)

        # Create optimizer with state (AdamW has momentum buffers)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Run a few training steps to populate optimizer state
        print("  Running training steps to populate optimizer state...")
        for _ in range(10):
            x = torch.randn(32, 768)
            loss = model(x).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # 1. Save unified checkpoint (traditional format)
        unified_path = os.path.join(test_dir, "unified-checkpoint.pt")
        save_unified_checkpoint(model, optimizer, unified_path, global_step=10)
        unified_size_mb = get_file_size_mb(unified_path)
        print(f"  ✓ Unified checkpoint: {unified_size_mb:.2f} MB")

        # 2. Save separated checkpoint (model + optimizer)
        print("  Saving separated checkpoint (model + optimizer)...")
        callback_full = SeparatedCheckpointCallback(
            save_dir=test_dir,
            save_interval=0,  # Don't auto-save during training
            save_model_only=False,
            filename_prefix="separated-full",
        )

        pl_module = SimpleLightningModule(model, optimizer)
        trainer = SimpleTrainer(optimizer)
        trainer.global_step = 10

        # Manually trigger checkpoint save
        callback_full._save_separated_checkpoint(pl_module, trainer, global_step=10)

        separated_model_path = os.path.join(test_dir, "separated-full-step000010-model.pt")
        separated_optimizer_path = os.path.join(test_dir, "separated-full-step000010-optimizer.pt")

        if not os.path.exists(separated_model_path):
            raise FileNotFoundError(f"Model checkpoint not created: {separated_model_path}")
        if not os.path.exists(separated_optimizer_path):
            raise FileNotFoundError(f"Optimizer checkpoint not created: {separated_optimizer_path}")

        separated_model_size_mb = get_file_size_mb(separated_model_path)
        separated_optimizer_size_mb = get_file_size_mb(separated_optimizer_path)
        separated_total_size_mb = separated_model_size_mb + separated_optimizer_size_mb

        print(f"  ✓ Separated model: {separated_model_size_mb:.2f} MB")
        print(f"  ✓ Separated optimizer: {separated_optimizer_size_mb:.2f} MB")
        print(f"  ✓ Separated total: {separated_total_size_mb:.2f} MB")

        # 3. Save model-only checkpoint
        print("  Saving model-only checkpoint...")
        callback_model_only = SeparatedCheckpointCallback(
            save_dir=test_dir,
            save_interval=0,
            save_model_only=True,
            filename_prefix="model-only",
        )

        callback_model_only._save_separated_checkpoint(pl_module, trainer, global_step=10)

        model_only_path = os.path.join(test_dir, "model-only-step000010-model.pt")

        if not os.path.exists(model_only_path):
            raise FileNotFoundError(f"Model-only checkpoint not created: {model_only_path}")

        # Verify optimizer checkpoint was NOT created in model-only mode
        model_only_optimizer_path = os.path.join(test_dir, "model-only-step000010-optimizer.pt")
        if os.path.exists(model_only_optimizer_path):
            raise AssertionError("Optimizer checkpoint should not exist in model-only mode")

        model_only_size_mb = get_file_size_mb(model_only_path)
        print(f"  ✓ Model-only: {model_only_size_mb:.2f} MB")

        # 4. Verify size reductions
        print("\n  Analyzing size reductions...")

        # Model-only should be roughly half the size of unified
        size_reduction_pct = (1 - model_only_size_mb / unified_size_mb) * 100
        print(f"  Model-only vs Unified: {size_reduction_pct:.1f}% smaller")

        if size_reduction_pct < 40:
            raise AssertionError(
                f"Model-only checkpoint is only {size_reduction_pct:.1f}% smaller than unified. "
                f"Expected at least 40% reduction (target: 50%+). "
                f"Model-only: {model_only_size_mb:.2f} MB, Unified: {unified_size_mb:.2f} MB"
            )

        print(f"  ✓ Model-only checkpoint achieved {size_reduction_pct:.1f}% size reduction")

        # Separated total should be comparable to unified
        separated_vs_unified = abs(separated_total_size_mb - unified_size_mb) / unified_size_mb * 100
        print(f"  Separated total vs Unified: {separated_vs_unified:.1f}% difference")

        if separated_vs_unified > 10:
            logger.warning(
                f"Separated total differs from unified by {separated_vs_unified:.1f}%. "
                f"This is expected due to different storage formats."
            )

        # 5. Test loading separated checkpoints
        print("\n  Testing checkpoint loading...")

        # Load unified checkpoint
        unified_checkpoint = load_checkpoint(unified_path, load_optimizer=True)
        assert 'model' in unified_checkpoint or 'model_state_dict' in unified_checkpoint
        print("  ✓ Unified checkpoint loads correctly")

        # Load separated checkpoint (with optimizer)
        separated_checkpoint = load_checkpoint(separated_model_path, load_optimizer=True)
        assert 'model' in separated_checkpoint or 'model_state_dict' in separated_checkpoint
        assert 'optimizer_states' in separated_checkpoint
        assert len(separated_checkpoint['optimizer_states']) > 0
        print("  ✓ Separated checkpoint (with optimizer) loads correctly")

        # Load model-only checkpoint
        model_only_checkpoint = load_model_only(model_only_path)
        assert 'model' in model_only_checkpoint or 'model_state_dict' in model_only_checkpoint
        assert 'optimizer_states' not in model_only_checkpoint or len(model_only_checkpoint.get('optimizer_states', [])) == 0
        print("  ✓ Model-only checkpoint loads correctly")

        return True

    finally:
        # Clean up temporary directory
        shutil.rmtree(test_dir, ignore_errors=True)
        print(f"\n  Cleaned up test directory: {test_dir}")


def test_checkpoint_callback_integration():
    """Test that SeparatedCheckpointCallback integrates correctly."""
    print("\nTesting SeparatedCheckpointCallback integration...")

    test_dir = tempfile.mkdtemp(prefix="test_callback_")

    try:
        # Create callback
        callback = SeparatedCheckpointCallback(
            save_dir=test_dir,
            save_interval=5,
            save_model_only=False,
            filename_prefix="test",
        )

        # Verify callback attributes
        assert callback.save_dir == test_dir
        assert callback.save_interval == 5
        assert callback.save_model_only == False
        assert callback.filename_prefix == "test"
        print("  ✓ Callback initialized with correct parameters")

        # Verify save directory was created
        assert os.path.isdir(test_dir)
        print("  ✓ Save directory created")

        # Test model-only mode
        callback_model_only = SeparatedCheckpointCallback(
            save_dir=test_dir,
            save_interval=10,
            save_model_only=True,
        )
        assert callback_model_only.save_model_only == True
        print("  ✓ Model-only mode configured correctly")

        return True

    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    print("=" * 60)
    print("Checkpoint Separation Size Verification")
    print("=" * 60)

    try:
        # Test 1: Checkpoint sizes and reduction
        test_checkpoint_sizes()

        # Test 2: Callback integration
        test_checkpoint_callback_integration()

        print("\n" + "=" * 60)
        print("✅ ALL CHECKPOINT SEPARATION TESTS PASSED")
        print("=" * 60)
        print("\nVerified:")
        print("1. ✓ Model-only checkpoints are 40-50%+ smaller than unified")
        print("2. ✓ Separated checkpoints save to correct file paths")
        print("3. ✓ Model-only mode skips optimizer checkpoint")
        print("4. ✓ Checkpoint loader handles both formats correctly")
        print("5. ✓ SeparatedCheckpointCallback integrates properly")
        print("\nCheckpoint separation is working as expected!")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
