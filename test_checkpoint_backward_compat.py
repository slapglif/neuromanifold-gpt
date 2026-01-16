#!/usr/bin/env python3
"""Test backward compatibility of checkpoint loading."""
import os
import tempfile
import shutil
import torch
import torch.nn as nn
from neuromanifold_gpt.utils.checkpoint_loader import load_checkpoint, load_model_only


class SimpleModel(nn.Module):
    """Simple test model."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


# Create temporary directory for test
test_dir = tempfile.mkdtemp(prefix="test_backward_compat_")

try:
    # Create model and optimizer
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters())

    # Save old-style unified checkpoint (model + optimizer in single file)
    unified_checkpoint_path = os.path.join(test_dir, "old_unified.pt")
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'global_step': 100,
        'config': {'test': True},
    }
    torch.save(checkpoint, unified_checkpoint_path)

    # Test 1: Load unified checkpoint with new loader (with optimizer)
    loaded_checkpoint = load_checkpoint(unified_checkpoint_path, load_optimizer=True)
    assert 'model' in loaded_checkpoint or 'model_state_dict' in loaded_checkpoint
    assert 'global_step' in loaded_checkpoint

    # Test 2: Load unified checkpoint as model-only
    model_only_checkpoint = load_model_only(unified_checkpoint_path)
    assert 'model' in model_only_checkpoint or 'model_state_dict' in model_only_checkpoint

    # Test 3: Load model state into new model instance
    new_model = SimpleModel()
    if 'model' in loaded_checkpoint:
        new_model.load_state_dict(loaded_checkpoint['model'])
    else:
        new_model.load_state_dict(loaded_checkpoint['model_state_dict'])

    print('OK')

finally:
    # Clean up
    shutil.rmtree(test_dir, ignore_errors=True)
