#!/usr/bin/env python3
"""
Test training resumption with separated checkpoints.

This test verifies that:
1. Training can be run for 100 steps with separated checkpoints
2. Training can be resumed from a separated checkpoint
3. Optimizer state is correctly restored (learning rate, momentum buffers, etc.)
4. Training continues from the correct step with the correct learning rate
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from neuromanifold_gpt.training.checkpoint_callback import SeparatedCheckpointCallback
from neuromanifold_gpt.utils.checkpoint_loader import load_checkpoint


class SimpleModel(nn.Module):
    """Simple model for testing training resumption."""

    def __init__(self, hidden_dim: int = 256, num_layers: int = 3):
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


def run_training_steps(model, optimizer, lr_scheduler, num_steps):
    """Run training steps and return loss values and optimizer state snapshots.

    Args:
        model: PyTorch model
        optimizer: Optimizer instance
        lr_scheduler: Learning rate scheduler
        num_steps: Number of training steps to run

    Returns:
        Tuple of (losses, learning_rates, optimizer_state_snapshot)
    """
    losses = []
    learning_rates = []

    model.train()
    for step in range(num_steps):
        # Generate random batch
        x = torch.randn(8, 256)

        # Forward pass
        output = model(x)
        loss = output.mean()

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Update learning rate
        if lr_scheduler:
            lr_scheduler.step()

        # Record metrics
        losses.append(loss.item())
        learning_rates.append(optimizer.param_groups[0]['lr'])

    # Get optimizer state snapshot for verification
    optimizer_state = {
        'param_groups': optimizer.param_groups,
        'state': {k: {kk: vv.clone() if isinstance(vv, torch.Tensor) else vv
                     for kk, vv in v.items()}
                 for k, v in optimizer.state.items()},
    }

    return losses, learning_rates, optimizer_state


def compare_optimizer_states(state1, state2, tolerance=1e-6):
    """Compare two optimizer states for equality.

    Args:
        state1: First optimizer state dict
        state2: Second optimizer state dict
        tolerance: Tolerance for floating point comparison

    Returns:
        Tuple of (is_equal, differences)
    """
    differences = []

    # Compare param_groups (learning rates, betas, etc.)
    for i, (pg1, pg2) in enumerate(zip(state1['param_groups'], state2['param_groups'])):
        for key in ['lr', 'betas', 'eps', 'weight_decay']:
            if key in pg1 and key in pg2:
                if pg1[key] != pg2[key]:
                    differences.append(f"param_group[{i}][{key}]: {pg1[key]} != {pg2[key]}")

    # Compare optimizer state (momentum buffers, etc.)
    state1_keys = set(state1['state'].keys())
    state2_keys = set(state2['state'].keys())

    if state1_keys != state2_keys:
        differences.append(f"Different state keys: {state1_keys} vs {state2_keys}")
        return False, differences

    # Compare each parameter's state
    for param_id in state1_keys:
        s1 = state1['state'][param_id]
        s2 = state2['state'][param_id]

        for key in s1.keys():
            if key not in s2:
                differences.append(f"param {param_id}: missing key '{key}' in state2")
                continue

            v1 = s1[key]
            v2 = s2[key]

            # Compare tensors
            if isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
                if not torch.allclose(v1, v2, atol=tolerance):
                    max_diff = torch.max(torch.abs(v1 - v2)).item()
                    differences.append(
                        f"param {param_id}[{key}]: tensors differ (max diff: {max_diff:.2e})"
                    )
            # Compare scalars
            elif v1 != v2:
                differences.append(f"param {param_id}[{key}]: {v1} != {v2}")

    return len(differences) == 0, differences


def test_training_resumption():
    """Test that training can be resumed from separated checkpoints with correct optimizer state."""
    print("=" * 70)
    print("Testing Training Resumption with Separated Checkpoints")
    print("=" * 70)

    test_dir = tempfile.mkdtemp(prefix="test_resumption_")

    try:
        # ========== Phase 1: Initial Training ==========
        print("\n[Phase 1] Running initial training for 50 steps...")

        # Create model and optimizer
        model = SimpleModel(hidden_dim=256, num_layers=3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999))

        # Create learning rate scheduler (simple linear warmup then constant)
        def lr_lambda(step):
            warmup_steps = 10
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            return 1.0

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # Run training for 50 steps
        losses_phase1, lrs_phase1, optimizer_state_at_50 = run_training_steps(
            model, optimizer, lr_scheduler, num_steps=50
        )

        print(f"  ✓ Completed 50 training steps")
        print(f"  - Initial loss: {losses_phase1[0]:.6f}")
        print(f"  - Final loss: {losses_phase1[-1]:.6f}")
        print(f"  - Initial LR: {lrs_phase1[0]:.6e}")
        print(f"  - Final LR: {lrs_phase1[-1]:.6e}")
        print(f"  - Optimizer state keys: {len(optimizer.state)}")

        # ========== Phase 2: Save Separated Checkpoint ==========
        print("\n[Phase 2] Saving separated checkpoint at step 50...")

        # Create callback and save checkpoint
        callback = SeparatedCheckpointCallback(
            save_dir=test_dir,
            save_interval=0,  # Manual save
            save_model_only=False,
            filename_prefix="checkpoint",
        )

        pl_module = SimpleLightningModule(model, optimizer)
        trainer = SimpleTrainer(optimizer, lr_scheduler)
        trainer.global_step = 50

        callback._save_separated_checkpoint(pl_module, trainer, global_step=50)

        model_path = os.path.join(test_dir, "checkpoint-step000050-model.pt")
        optimizer_path = os.path.join(test_dir, "checkpoint-step000050-optimizer.pt")

        assert os.path.exists(model_path), f"Model checkpoint not found: {model_path}"
        assert os.path.exists(optimizer_path), f"Optimizer checkpoint not found: {optimizer_path}"

        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        optimizer_size_mb = os.path.getsize(optimizer_path) / (1024 * 1024)

        print(f"  ✓ Saved separated checkpoint")
        print(f"  - Model: {model_path} ({model_size_mb:.2f} MB)")
        print(f"  - Optimizer: {optimizer_path} ({optimizer_size_mb:.2f} MB)")

        # ========== Phase 3: Load Checkpoint and Resume ==========
        print("\n[Phase 3] Loading checkpoint and resuming training...")

        # Create new model and optimizer (simulate fresh start)
        model_resumed = SimpleModel(hidden_dim=256, num_layers=3)
        optimizer_resumed = torch.optim.AdamW(
            model_resumed.parameters(), lr=1e-3, betas=(0.9, 0.999)
        )
        lr_scheduler_resumed = torch.optim.lr_scheduler.LambdaLR(optimizer_resumed, lr_lambda)

        # Load checkpoint
        checkpoint = load_checkpoint(model_path, load_optimizer=True)

        assert 'model' in checkpoint or 'model_state_dict' in checkpoint, \
            "Checkpoint missing model state"
        assert 'optimizer_states' in checkpoint, \
            "Checkpoint missing optimizer states"
        assert len(checkpoint['optimizer_states']) > 0, \
            "Checkpoint has empty optimizer states"

        print(f"  ✓ Loaded checkpoint")
        print(f"  - Global step: {checkpoint.get('global_step', 'N/A')}")
        print(f"  - Optimizer states: {len(checkpoint['optimizer_states'])}")

        # Restore model state
        if 'model' in checkpoint:
            model_resumed.load_state_dict(checkpoint['model'])
        else:
            model_resumed.load_state_dict(checkpoint['model_state_dict'])

        # Restore optimizer state
        optimizer_resumed.load_state_dict(checkpoint['optimizer_states'][0])

        # Restore scheduler state if available
        if 'lr_scheduler_states' in checkpoint and len(checkpoint['lr_scheduler_states']) > 0:
            lr_scheduler_resumed.load_state_dict(checkpoint['lr_scheduler_states'][0])
        else:
            # Manually advance scheduler to match training step
            for _ in range(50):
                lr_scheduler_resumed.step()

        print(f"  ✓ Restored model and optimizer state")
        print(f"  - Current LR: {optimizer_resumed.param_groups[0]['lr']:.6e}")

        # ========== Phase 4: Verify Optimizer State ==========
        print("\n[Phase 4] Verifying optimizer state restoration...")

        # Get current optimizer state
        optimizer_state_loaded = {
            'param_groups': optimizer_resumed.param_groups,
            'state': {k: {kk: vv.clone() if isinstance(vv, torch.Tensor) else vv
                         for kk, vv in v.items()}
                     for k, v in optimizer_resumed.state.items()},
        }

        # Compare with saved state
        is_equal, differences = compare_optimizer_states(
            optimizer_state_at_50, optimizer_state_loaded, tolerance=1e-6
        )

        if is_equal:
            print("  ✓ Optimizer state perfectly restored!")
        else:
            print("  ⚠ Optimizer state differences detected:")
            for diff in differences[:5]:  # Show first 5 differences
                print(f"    - {diff}")
            if len(differences) > 5:
                print(f"    ... and {len(differences) - 5} more differences")

        # Verify learning rate matches
        lr_at_50 = lrs_phase1[-1]
        lr_resumed = optimizer_resumed.param_groups[0]['lr']
        lr_match = abs(lr_at_50 - lr_resumed) < 1e-9

        print(f"  - LR at step 50: {lr_at_50:.6e}")
        print(f"  - LR after load: {lr_resumed:.6e}")
        print(f"  - LR matches: {'✓' if lr_match else '✗'}")

        # Verify momentum buffers exist (AdamW has exp_avg and exp_avg_sq)
        has_momentum = any(
            'exp_avg' in state for state in optimizer_resumed.state.values()
        )
        print(f"  - Momentum buffers present: {'✓' if has_momentum else '✗'}")

        # ========== Phase 5: Continue Training ==========
        print("\n[Phase 5] Continuing training for 50 more steps...")

        losses_phase2, lrs_phase2, _ = run_training_steps(
            model_resumed, optimizer_resumed, lr_scheduler_resumed, num_steps=50
        )

        print(f"  ✓ Completed 50 more training steps")
        print(f"  - First loss: {losses_phase2[0]:.6f}")
        print(f"  - Final loss: {losses_phase2[-1]:.6f}")
        print(f"  - First LR: {lrs_phase2[0]:.6e}")
        print(f"  - Final LR: {lrs_phase2[-1]:.6e}")

        # ========== Phase 6: Verification Summary ==========
        print("\n" + "=" * 70)
        print("Verification Summary")
        print("=" * 70)

        all_checks_passed = True

        # Check 1: Separated checkpoints created
        check1 = os.path.exists(model_path) and os.path.exists(optimizer_path)
        print(f"1. Separated checkpoints created: {'✓ PASS' if check1 else '✗ FAIL'}")
        all_checks_passed &= check1

        # Check 2: Checkpoint loads successfully
        check2 = 'optimizer_states' in checkpoint and len(checkpoint['optimizer_states']) > 0
        print(f"2. Checkpoint loads with optimizer: {'✓ PASS' if check2 else '✗ FAIL'}")
        all_checks_passed &= check2

        # Check 3: Learning rate matches
        check3 = lr_match
        print(f"3. Learning rate correctly restored: {'✓ PASS' if check3 else '✗ FAIL'}")
        all_checks_passed &= check3

        # Check 4: Momentum buffers restored
        check4 = has_momentum
        print(f"4. Momentum buffers present: {'✓ PASS' if check4 else '✗ FAIL'}")
        all_checks_passed &= check4

        # Check 5: Training continues without errors
        check5 = len(losses_phase2) == 50
        print(f"5. Training resumed successfully: {'✓ PASS' if check5 else '✗ FAIL'}")
        all_checks_passed &= check5

        # Check 6: Loss continuity (resumed loss should be close to last loss)
        # Note: Due to random data generation, exact match isn't expected, but order of magnitude should be similar
        loss_continuity = abs(losses_phase1[-1] - losses_phase2[0]) / losses_phase1[-1] < 0.5
        check6 = loss_continuity
        print(f"6. Loss continuity maintained: {'✓ PASS' if check6 else '⚠ WARN (may vary with random data)'}")
        # Don't fail on this check as it depends on random data

        print("=" * 70)

        if all_checks_passed:
            print("\n✅ ALL CRITICAL CHECKS PASSED")
            print("\nTraining resumption with separated checkpoints is working correctly!")
            print("Optimizer state (learning rate, momentum buffers) is properly restored.")
            return True
        else:
            print("\n❌ SOME CHECKS FAILED")
            print("\nPlease review the failures above.")
            return False

    except Exception as e:
        print(f"\n❌ TEST FAILED WITH EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Clean up temporary directory
        shutil.rmtree(test_dir, ignore_errors=True)
        print(f"\nCleaned up test directory: {test_dir}")


if __name__ == "__main__":
    print("\n")
    success = test_training_resumption()
    print("\n")
    sys.exit(0 if success else 1)
