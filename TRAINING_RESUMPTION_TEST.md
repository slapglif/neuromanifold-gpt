# Training Resumption Test

This document describes the training resumption test for separated checkpoints.

## Overview

The `test_training_resumption.py` script verifies that training can be correctly resumed from separated checkpoints with full optimizer state restoration.

## What It Tests

### 1. Checkpoint Creation
- Trains a model for 50 steps with AdamW optimizer
- Saves separated checkpoint (model.pt + optimizer.pt) at step 50
- Verifies both files are created

### 2. Checkpoint Loading
- Creates fresh model and optimizer instances
- Loads from separated checkpoint files
- Restores model weights, optimizer state, and LR scheduler state

### 3. Optimizer State Verification
The test performs detailed verification of optimizer state restoration:

- **Learning Rate**: Verifies exact match after loading
- **Momentum Buffers**: Checks that AdamW's `exp_avg` and `exp_avg_sq` buffers are present
- **State Tensors**: Compares all optimizer state tensors with tolerance of 1e-6
- **Parameter Groups**: Verifies learning rate, betas, eps, and weight_decay match

### 4. Training Continuation
- Resumes training for 50 more steps
- Verifies training continues without errors
- Checks loss continuity (should be similar order of magnitude)

## Test Phases

The test executes in 6 phases:

1. **Phase 1**: Initial training for 50 steps
2. **Phase 2**: Save separated checkpoint at step 50
3. **Phase 3**: Load checkpoint and create fresh instances
4. **Phase 4**: Verify optimizer state restoration
5. **Phase 5**: Continue training for 50 more steps
6. **Phase 6**: Display verification summary

## Running the Test

### Prerequisites
- PyTorch installed
- neuromanifold_gpt package available

### Execution

```bash
python test_training_resumption.py
```

The test will output detailed progress for each phase and a final verification summary.

### Expected Output

```
======================================================================
Testing Training Resumption with Separated Checkpoints
======================================================================

[Phase 1] Running initial training for 50 steps...
  ✓ Completed 50 training steps
  - Initial loss: 0.123456
  - Final loss: 0.098765
  - Initial LR: 1.00e-04
  - Final LR: 1.00e-03
  - Optimizer state keys: 123

[Phase 2] Saving separated checkpoint at step 50...
  ✓ Saved separated checkpoint
  - Model: /tmp/test_resumption_xyz/checkpoint-step000050-model.pt (2.45 MB)
  - Optimizer: /tmp/test_resumption_xyz/checkpoint-step000050-optimizer.pt (2.67 MB)

[Phase 3] Loading checkpoint and resuming training...
  ✓ Loaded checkpoint
  - Global step: 50
  - Optimizer states: 1
  ✓ Restored model and optimizer state
  - Current LR: 1.00e-03

[Phase 4] Verifying optimizer state restoration...
  ✓ Optimizer state perfectly restored!
  - LR at step 50: 1.00e-03
  - LR after load: 1.00e-03
  - LR matches: ✓
  - Momentum buffers present: ✓

[Phase 5] Continuing training for 50 more steps...
  ✓ Completed 50 more training steps
  - First loss: 0.098123
  - Final loss: 0.087654
  - First LR: 1.00e-03
  - Final LR: 1.00e-03

======================================================================
Verification Summary
======================================================================
1. Separated checkpoints created: ✓ PASS
2. Checkpoint loads with optimizer: ✓ PASS
3. Learning rate correctly restored: ✓ PASS
4. Momentum buffers present: ✓ PASS
5. Training resumed successfully: ✓ PASS
6. Loss continuity maintained: ✓ PASS
======================================================================

✅ ALL CRITICAL CHECKS PASSED

Training resumption with separated checkpoints is working correctly!
Optimizer state (learning rate, momentum buffers) is properly restored.

Cleaned up test directory: /tmp/test_resumption_xyz
```

## What Gets Verified

The test verifies all acceptance criteria for training resumption:

- ✅ Separated checkpoints can be saved (model.pt + optimizer.pt)
- ✅ Checkpoints can be loaded with full optimizer state
- ✅ Learning rate is correctly restored
- ✅ Momentum buffers (AdamW state) are present and correct
- ✅ Training continues from correct step without errors
- ✅ Loss continuity is maintained across resumption

## Implementation Details

### Model
- Simple 3-layer neural network (256 hidden dim)
- ~200K parameters

### Optimizer
- AdamW with lr=1e-3, betas=(0.9, 0.999)
- Maintains exp_avg and exp_avg_sq momentum buffers

### Learning Rate Schedule
- Linear warmup for first 10 steps
- Constant learning rate thereafter

### Data
- Random tensors (8 batch size, 256 features)
- Ensures reproducible test behavior

## Files Created

- `test_training_resumption.py`: Main test script
- `TRAINING_RESUMPTION_TEST.md`: This documentation

## Related Tests

- `test_checkpoint_separation.py`: Tests checkpoint size reduction
- `test_checkpoint_backward_compat.py`: Tests backward compatibility with unified checkpoints
- `test_separated_checkpoint_integration.py`: Tests callback integration with trainer

## Manual Verification

For real-world validation, you can also manually test with:

```bash
# 1. Train for 100 steps with separated checkpoints
python train.py --save_separate_optimizer=True --max_iters=100 --out_dir=test_checkpoints

# 2. Check that checkpoint files exist
ls test_checkpoints/checkpoint-*-model.pt
ls test_checkpoints/checkpoint-*-optimizer.pt

# 3. Resume training (Lightning will auto-resume from checkpoint)
python train.py --save_separate_optimizer=True --max_iters=200 --out_dir=test_checkpoints

# 4. Verify training continues from step 100 with correct learning rate
```

## Troubleshooting

If the test fails:

1. **Check PyTorch installation**: Ensure PyTorch is properly installed
2. **Check imports**: Verify neuromanifold_gpt package is in PYTHONPATH
3. **Check disk space**: Ensure sufficient space for checkpoint files (~5 MB)
4. **Review error messages**: The test provides detailed error output

## Maintenance

When updating checkpoint logic:

1. Run this test to ensure training resumption still works
2. Verify all 6 checks pass
3. Update this documentation if test behavior changes
