# Gradient Checkpointing Memory Reduction Verification

## Overview

This document describes the verification process for testing that gradient checkpointing successfully reduces memory usage during training.

## What is Gradient Checkpointing?

Gradient checkpointing is a memory optimization technique that trades computation for memory:
- During forward pass, intermediate activations are not saved (except at checkpoints)
- During backward pass, activations are recomputed from checkpoints as needed
- This reduces peak memory usage by ~20-40% at the cost of ~20% more computation

## Verification Methods

### Method 1: Automated Verification Script (Recommended)

Run the automated verification script:

```bash
python3 verify_gradient_checkpointing.py
```

This script will:
1. Run training WITHOUT gradient checkpointing (baseline)
2. Run training WITH gradient checkpointing (optimized)
3. Extract peak memory usage from both runs
4. Compare and verify ~20-40% reduction

**Expected Output:**
```
RESULTS
================================================================================

  Baseline peak memory:    X.XXX GB
  Optimized peak memory:   Y.YYY GB
  Memory reduction:        Z.ZZZ GB (NN.N%)

  ✓ SUCCESS: Memory reduction (NN.N%) is within expected range (20-40%)

VERIFICATION PASSED ✓
```

### Method 2: Manual Verification

If you prefer to run the tests manually:

#### Step 1: Run baseline training (no gradient checkpointing)

```bash
python3 train.py --config config/train_shakespeare_char.py --max_iters 10 --gradient_checkpointing false
```

**What to look for:**
- Memory statistics logged during training
- Note the peak memory value: `memory/gpu_max_allocated_gb: X.XXX`

#### Step 2: Run optimized training (with gradient checkpointing)

```bash
python3 train.py --config config/train_shakespeare_char.py --max_iters 10 --gradient_checkpointing true
```

**What to look for:**
- Memory statistics logged during training
- Note the peak memory value: `memory/gpu_max_allocated_gb: Y.YYY`
- Startup report should show "✓ Gradient checkpointing: ENABLED"

#### Step 3: Compare Results

Calculate memory reduction:
```
Reduction (%) = ((Baseline - Optimized) / Baseline) * 100
```

**Expected:** 20-40% reduction in peak memory usage

## Implementation Details

### Components Involved

1. **Config Flag** (`neuromanifold_gpt/config/base.py`)
   - `gradient_checkpointing: bool = False`
   - Controls whether gradient checkpointing is enabled

2. **Model Implementation** (`neuromanifold_gpt/model/gpt.py`, `neuromanifold_gpt/model/block.py`)
   - Uses `torch.utils.checkpoint.checkpoint()` to wrap forward passes
   - Only active during training when flag is enabled

3. **Training Integration** (`train.py`)
   - `TrainConfig.gradient_checkpointing` flag
   - Passed to model configs during initialization
   - Displayed in startup memory report

4. **Memory Monitoring** (`train.py`)
   - `MemoryMonitorCallback` tracks GPU memory every `log_interval` batches
   - Logs: `memory/gpu_allocated_gb`, `memory/gpu_reserved_gb`, `memory/gpu_max_allocated_gb`

### How It Works

When `gradient_checkpointing=True`:

1. During forward pass in `NeuroManifoldBlock`:
   ```python
   if self.gradient_checkpointing and self.training:
       output = checkpoint(self.attention, x, use_reentrant=False)
   ```

2. PyTorch's checkpoint function:
   - Doesn't save intermediate activations
   - Recomputes them during backward pass
   - Reduces memory at cost of extra computation

3. Memory savings are most pronounced with:
   - Larger models (more layers/parameters)
   - Longer sequences (more activations to store)
   - Higher batch sizes (more samples)

## Expected Results

### Small Model (Shakespeare char, ~10M params)
- **Baseline peak memory:** ~1-2 GB
- **With checkpointing:** ~0.7-1.5 GB
- **Reduction:** ~20-30%

### Medium Model (GPT-2 124M params)
- **Baseline peak memory:** ~4-6 GB
- **With checkpointing:** ~3-4 GB
- **Reduction:** ~25-35%

### Large Model (GPT-2 355M params)
- **Baseline peak memory:** ~10-15 GB
- **With checkpointing:** ~6-10 GB
- **Reduction:** ~30-40%

## Troubleshooting

### "CUDA is not available"
- Gradient checkpointing can only be tested on GPU hardware
- Memory reduction is not applicable on CPU (no memory pressure)

### Memory reduction less than 20%
- Small models have less activation memory to save
- Try with larger model or longer sequences
- Still counts as passing if reduction > 0%

### Training fails with OOM
- Even with checkpointing, very large models may OOM
- Try reducing batch size or sequence length
- Check recommended batch size from startup report

### No memory statistics in logs
- Ensure `MemoryMonitorCallback` is registered in train.py
- Check that CUDA is available
- Verify `log_interval` is not too high

## Verification Checklist

- [ ] Gradient checkpointing flag exists in `NeuroManifoldConfig`
- [ ] Gradient checkpointing flag exists in `TrainConfig`
- [ ] Flag is passed from TrainConfig to model config
- [ ] Model uses `torch.utils.checkpoint.checkpoint()` when flag is True
- [ ] Memory monitoring callback logs peak memory
- [ ] Startup report shows gradient checkpointing status
- [ ] Training with checkpointing completes without errors
- [ ] Peak memory is reduced by >0% (ideally 20-40%)

## References

- PyTorch Checkpoint API: https://pytorch.org/docs/stable/checkpoint.html
- Gradient Checkpointing Paper: "Training Deep Nets with Sublinear Memory Cost"
- Related Implementation: HuggingFace Transformers, LitGPT
