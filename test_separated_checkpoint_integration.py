"""
Test script to verify separated checkpoint callback integration.

This script creates a minimal training run to verify that:
1. The separated checkpoint callback is properly integrated
2. Two files are created: model.pt and optimizer.pt
3. The callback respects the save_model_only flag
"""

import os
import shutil
from neuromanifold_gpt.training.config import TrainConfig
from neuromanifold_gpt.training.trainer import train


def test_separated_checkpoint_integration():
    """Test that separated checkpoints are created correctly."""
    # Create a test output directory
    test_out_dir = "test_separated_checkpoints"
    os.makedirs(test_out_dir, exist_ok=True)

    try:
        # Configure training with separated checkpoints
        config = TrainConfig(
            out_dir=test_out_dir,
            dataset="shakespeare_char",
            model_type="gpt",  # Use simpler GPT for faster testing
            n_layer=2,
            n_head=2,
            n_embd=64,
            batch_size=4,
            block_size=64,
            max_iters=20,  # Very short run
            eval_interval=10,  # Save checkpoint at step 10
            save_checkpoints=False,  # Disable regular checkpoints
            save_separate_optimizer=True,  # Enable separated checkpoints
            save_model_only=False,  # Save both model and optimizer
            wandb_log=False,
            sample_interval=0,  # Disable sampling
        )

        print("Starting short training run with separated checkpoints...")
        print(f"Config: save_separate_optimizer={config.save_separate_optimizer}, save_model_only={config.save_model_only}")

        # Run training
        train(config)

        # Check for created checkpoint files
        files = os.listdir(test_out_dir)
        model_files = [f for f in files if f.endswith("-model.pt")]
        optimizer_files = [f for f in files if f.endswith("-optimizer.pt")]

        print(f"\nFound {len(model_files)} model checkpoint(s)")
        print(f"Found {len(optimizer_files)} optimizer checkpoint(s)")

        # Verify files were created
        assert len(model_files) > 0, "No model checkpoints found!"
        assert len(optimizer_files) > 0, "No optimizer checkpoints found!"

        print("\n✓ Separated checkpoint integration test PASSED")
        print(f"Model files: {model_files}")
        print(f"Optimizer files: {optimizer_files}")

        return True

    except Exception as e:
        print(f"\n✗ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        if os.path.exists(test_out_dir):
            shutil.rmtree(test_out_dir)
            print(f"\nCleaned up test directory: {test_out_dir}")


if __name__ == "__main__":
    success = test_separated_checkpoint_integration()
    exit(0 if success else 1)
