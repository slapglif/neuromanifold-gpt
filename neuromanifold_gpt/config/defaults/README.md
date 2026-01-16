# Default Configuration Guide

This directory contains tested, production-ready configurations for common GPT training scenarios. These configs are designed to work out-of-the-box without modification, providing sensible defaults for different hardware setups.

## Available Configurations

### 1. Single GPU Training - `gpt2_124m_single_gpu.py`

**When to use:**
- You have access to a single GPU with 16-24GB VRAM
- You want to train GPT-2 124M from scratch on your own dataset
- You're experimenting and iterating on model architecture or training hyperparameters
- Cost efficiency is a priority and you're willing to trade training time for lower infrastructure costs

**Hardware requirements:**
- **GPU:** 1x GPU with 16-24GB VRAM (e.g., RTX 3090, RTX 4090, V100, A100, A6000)
- **RAM:** 32GB+ system RAM recommended
- **Storage:** ~50GB for checkpoints and training data
- **CUDA:** Version 11.8+ with PyTorch 2.0+

**Configuration details:**
- **Batch size:** 4 samples per batch
- **Context length:** 1024 tokens
- **Gradient accumulation:** 12 steps
- **Effective batch size:** ~49K tokens per iteration (4 × 1024 × 12)
- **Total training tokens:** 300B tokens (same dataset coverage as multi-GPU setup)
- **Training iterations:** 600,000 iterations

**Expected training time:**
- **Full 300B tokens:** ~25-30 days on RTX 3090, ~15-20 days on A100
- **First checkpoint (1000 iters):** ~2-3 hours
- **Validation loss plateau:** Typically after 100K-200K iterations (~3-7 days)
- **Production-quality model:** 300K-400K iterations (~10-15 days)

**Launch command:**
```bash
python train.py neuromanifold_gpt/config/defaults/gpt2_124m_single_gpu.py
```

**Cost estimate:**
- Cloud GPU (A100): ~$2-3/hour × 480 hours = ~$960-1440 for full training
- Consumer GPU (electricity): ~$0.30/day × 20 days = ~$6 for full training

---

### 2. Multi-GPU Training - `gpt2_124m_multi_gpu.py`

**When to use:**
- You have access to multiple high-end GPUs (8x A100 or similar)
- You need to train a production-quality model quickly
- You're reproducing GPT-2 124M results or training on large corpora like OpenWebText
- Training time is more critical than infrastructure cost

**Hardware requirements:**
- **GPUs:** 8x A100 40GB or 8x A100 80GB (can scale to fewer GPUs by adjusting gradient_accumulation_steps)
- **RAM:** 256GB+ system RAM recommended
- **Storage:** 500GB+ for checkpoints, training data, and intermediate files
- **Network:** High-bandwidth interconnect (NVLink, InfiniBand) for optimal multi-GPU performance
- **CUDA:** Version 11.8+ with PyTorch 2.0+ compiled with NCCL support

**Configuration details:**
- **Batch size:** 12 samples per GPU
- **Context length:** 1024 tokens
- **Gradient accumulation:** 40 steps (5 × 8 GPUs)
- **Effective batch size:** ~491K tokens per iteration (12 × 1024 × 5 × 8)
- **Total training tokens:** 300B tokens
- **Training iterations:** 600,000 iterations

**Expected training time:**
- **Full 300B tokens:** ~4 days on 8x A100 40GB
- **First checkpoint (1000 iters):** ~15-20 minutes
- **Validation loss plateau:** Typically after 100K-200K iterations (~16-32 hours)
- **Production-quality model:** 300K-400K iterations (~2-3 days)

**Launch command:**
```bash
torchrun --standalone --nproc_per_node=8 train.py neuromanifold_gpt/config/defaults/gpt2_124m_multi_gpu.py
```

**Scaling to different GPU counts:**
To adapt this config for 4 GPUs instead of 8, adjust gradient accumulation:
```python
gradient_accumulation_steps = 5 * 4  # Changed from 5 * 8
```
This maintains the same effective batch size and training dynamics.

**Cost estimate:**
- Cloud (8x A100): ~$20-30/hour × 96 hours = ~$1920-2880 for full training
- On-prem cluster: Electricity and depreciation typically ~$500-800 for full training

---

### 3. Finetuning - `finetune_gpt2_medium.py`

**When to use:**
- You want to adapt a pretrained GPT-2 model to your specific domain or task
- You have a smaller dataset (1M-100M tokens) and want to leverage transfer learning
- You need quick turnaround for domain adaptation (hours instead of days)
- You're experimenting with prompt formats, writing styles, or specialized knowledge

**Hardware requirements:**
- **GPU:** 1x GPU with 24GB+ VRAM (e.g., RTX 3090, RTX 4090, A100, A6000)
  - GPT-2 Medium (345M params) requires more memory than 124M
  - Can work with 16GB VRAM by reducing batch_size to 1
- **RAM:** 32GB+ system RAM recommended
- **Storage:** ~10GB for checkpoint and finetuning data

**Configuration details:**
- **Starting point:** `gpt2-medium` (345M parameters)
- **Batch size:** 2 samples per batch
- **Gradient accumulation:** 16 steps
- **Effective batch size:** ~32K tokens per iteration (2 × 16 × 1024)
- **Learning rate:** 3e-5 (constant, no decay)
- **Training iterations:** 20 (this is a minimal example - adjust based on your dataset size)
- **Checkpoint saving:** Only saves when validation loss improves (space-efficient)

**Expected training time:**
- **20 iterations (example):** ~5-10 minutes on A100
- **Typical finetuning (1000 iters):** ~2-4 hours on A100
- **Domain adaptation (5000 iters):** ~10-20 hours on A100
- **Significant style transfer (20K iters):** ~2-4 days on A100

**Recommended iteration counts by use case:**
- **Quick style adaptation:** 100-500 iterations (~10-60 minutes)
- **Domain specialization:** 1000-5000 iterations (~2-20 hours)
- **Full finetuning:** 10K-50K iterations (~1-10 days)

**Launch command:**
```bash
python train.py neuromanifold_gpt/config/defaults/finetune_gpt2_medium.py
```

**Finetuning other model sizes:**
To finetune GPT-2 124M (smaller, faster):
```python
init_from = 'gpt2'  # 124M parameters
batch_size = 4      # Can use larger batch size with smaller model
```

To finetune GPT-2 Large (774M parameters):
```python
init_from = 'gpt2-large'  # 774M parameters
batch_size = 1            # Requires reduction for memory constraints
```

**Cost estimate:**
- Cloud GPU (A100): ~$2-3/hour × typical 10-20 hours = ~$20-60 for domain adaptation
- Consumer GPU (electricity): Negligible (~$1) for typical finetuning runs

---

## General Tips

### Monitoring Training Progress

All configurations enable weights & biases logging by default (except finetuning). To monitor:

1. **Console output:** Watch for decreasing loss values
   ```
   iter 0: loss 4.2302
   iter 100: loss 2.6451
   iter 1000: loss 1.6234
   ```

2. **W&B dashboard:** Check `wandb_project` for detailed metrics, learning rate schedules, and gradient norms

3. **Checkpoint validation:** Periodically sample from saved checkpoints to qualitatively assess generation quality

### When to Stop Training

- **Loss plateaus:** If validation loss stops decreasing for 50K-100K iterations
- **Overfitting:** If training loss continues decreasing but validation loss increases
- **Time/budget constraints:** Use the checkpoint with lowest validation loss
- **Quality milestones:** Sample periodically and stop when generation quality meets your needs

### Customizing These Configs

These configs are starting points. Common modifications:

**Adjust learning rate:**
```python
learning_rate = 6e-4  # Default for 124M
learning_rate = 3e-4  # Often better for larger models
learning_rate = 3e-5  # Recommended for finetuning
```

**Change dataset:**
```python
dataset = 'openwebtext'     # Large web corpus
dataset = 'shakespeare_char' # Character-level Shakespeare
dataset = 'your_custom_dataset'  # Your prepared dataset
```

**Modify model size:**
```python
n_layer = 12  # Increase depth
n_head = 12   # Increase attention heads
n_embd = 768  # Increase embedding dimension
```

**Enable/disable logging:**
```python
wandb_log = False  # Disable W&B logging
wandb_log = True   # Enable W&B logging
```

### Memory Optimization

If you encounter OOM (Out of Memory) errors:

1. **Reduce batch size:**
   ```python
   batch_size = 2  # Reduce from 4
   ```

2. **Increase gradient accumulation** (maintains effective batch size):
   ```python
   gradient_accumulation_steps = 24  # Increase from 12
   ```

3. **Reduce context length:**
   ```python
   block_size = 512  # Reduce from 1024
   ```

4. **Enable gradient checkpointing** (in model.py):
   ```python
   torch.utils.checkpoint.checkpoint(...)  # Trade compute for memory
   ```

### Troubleshooting

**Loss is NaN or exploding:**
- Reduce learning rate by 2-5x
- Check for data quality issues (corrupted samples, extreme outliers)
- Verify gradient clipping is enabled: `grad_clip = 1.0`

**Training is too slow:**
- Verify CUDA is being used: Check for "Using device: cuda" in logs
- Enable `compile = True` in train.py (requires PyTorch 2.0+)
- Check GPU utilization with `nvidia-smi` - should be 95%+

**Validation loss not decreasing:**
- Train longer (especially for single GPU, give it 10K+ iterations)
- Increase learning rate slightly
- Verify dataset is prepared correctly and train/val split is proper
- Check that model architecture matches pretrained checkpoint (if finetuning)

---

## Quick Reference Table

| Configuration | GPUs | VRAM | Training Time | Cost | Best For |
|--------------|------|------|---------------|------|----------|
| **gpt2_124m_single_gpu** | 1 | 16-24GB | 15-30 days | ~$1K cloud / $6 local | Experimentation, learning, cost-efficiency |
| **gpt2_124m_multi_gpu** | 8 | 40GB each | 3-4 days | ~$2-3K cloud / $500 cluster | Production training, reproducing GPT-2 |
| **finetune_gpt2_medium** | 1 | 24GB+ | 2-20 hours | ~$20-60 cloud / $1 local | Domain adaptation, quick iteration |

---

## Additional Resources

- **Main documentation:** See [README.md](../../../README.md) for installation and general usage
- **Custom datasets:** See [data/](../../../data/) for examples of dataset preparation
- **Model architecture:** See [model.py](../../../model.py) for GPT implementation details
- **Training script:** See [train.py](../../../train.py) for training loop implementation

## Contributing

Found an issue with these configs or have suggestions for additional scenarios? Please open an issue or submit a pull request. We especially appreciate:

- Reports of actual training times on different hardware
- Successful finetuning recipes for specific domains
- Cost optimizations or memory-saving techniques
- Additional tested configurations for other model sizes or hardware setups
