# Finetuning Quick-Start Guide

**Get your GPT-2 model finetuned on custom data in under 30 minutes**

This guide walks you through the complete process of finetuning pretrained GPT-2 models on your own datasets. Whether you're adapting to a specific domain, teaching your model a writing style, or instruction tuning, this guide provides battle-tested configurations and practical advice to get you started quickly.

## Table of Contents

1. [What is Finetuning?](#what-is-finetuning)
2. [Quick Start: Finetune in 3 Steps](#quick-start-finetune-in-3-steps)
3. [Dataset Preparation](#dataset-preparation)
4. [Choosing Your Model Size](#choosing-your-model-size)
5. [Hyperparameter Recommendations](#hyperparameter-recommendations)
6. [Step-by-Step Finetuning Walkthrough](#step-by-step-finetuning-walkthrough)
7. [Expected Results and Benchmarks](#expected-results-and-benchmarks)
8. [Troubleshooting Common Issues](#troubleshooting-common-issues)
9. [Advanced Topics](#advanced-topics)

---

## What is Finetuning?

Finetuning is the process of taking a pretrained model (like GPT-2) and continuing training on your specific dataset. Unlike training from scratch, finetuning:

- **Starts from pretrained weights** - You inherit the model's existing language understanding
- **Requires less data** - Can work with datasets as small as 100K tokens
- **Trains faster** - Typically completes in minutes to hours, not days
- **Uses lower learning rates** - Prevents "catastrophic forgetting" of pretrained knowledge

**Common finetuning scenarios:**
- **Domain adaptation** - Adapt a general model to legal, medical, scientific, or code domains
- **Style transfer** - Train on Shakespeare, technical writing, or specific author styles
- **Instruction tuning** - Teach the model to follow instructions or perform specific tasks
- **Task-specific optimization** - Improve performance on Q&A, summarization, etc.

---

## Quick Start: Finetune in 3 Steps

Want to jump right in? Here's the fastest path to a finetuned model:

### Step 1: Prepare Your Data

```sh
# Example: Use the Shakespeare dataset
python data/shakespeare/prepare.py
```

This downloads Shakespeare's works (~1MB) and creates `train.bin` and `val.bin` files. For your own data, see [Dataset Preparation](#dataset-preparation) below.

### Step 2: Choose a Model and Start Training

```sh
# Finetune GPT-2 Small (124M params) - good for experimentation
python train.py config/finetune/gpt2_small.py

# Or try a larger model for better quality
python train.py config/finetune/gpt2_medium.py  # 350M params
```

**What to expect:**
- Training will automatically detect your GPU and run
- Progress updates every 10 iterations
- Takes ~5-30 minutes depending on model size and GPU
- Checkpoints saved to `out-finetune-gpt2-small/`

### Step 3: Sample from Your Finetuned Model

```sh
python sample.py --out_dir=out-finetune-gpt2-small
```

That's it! You now have a model finetuned on your dataset.

---

## Dataset Preparation

Your dataset quality directly impacts finetuning results. Here's how to prepare your data:

### Data Format

nanoGPT expects data in binary format (`train.bin` and `val.bin`). You have two options:

#### Option 1: Character-Level (Simple, Good for Exploration)

Best for: Small datasets, experimentation, character-level modeling

```python
# See data/shakespeare_char/prepare.py for a complete example
import numpy as np
import pickle

# Read your text data
with open('your_data.txt', 'r') as f:
    data = f.read()

# Create character vocabulary
chars = sorted(list(set(data)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Encode and split
def encode(s):
    return [stoi[c] for c in s]

train_data = data[:int(len(data) * 0.9)]
val_data = data[int(len(data) * 0.9):]

train_ids = np.array(encode(train_data), dtype=np.uint16)
val_ids = np.array(encode(val_data), dtype=np.uint16)

train_ids.tofile('data/your_dataset/train.bin')
val_ids.tofile('data/your_dataset/val.bin')

# Save vocabulary metadata
meta = {
    'vocab_size': len(chars),
    'itos': itos,
    'stoi': stoi,
}
with open('data/your_dataset/meta.pkl', 'wb') as f:
    pickle.dump(meta, f)
```

#### Option 2: BPE Tokenization (Production, Better Quality)

Best for: Production use, larger datasets, leveraging GPT-2 pretrained knowledge

```python
# See data/shakespeare/prepare.py for a complete example
import numpy as np
import tiktoken

# Use GPT-2's tokenizer (matches pretrained model)
enc = tiktoken.get_encoding("gpt2")

# Read and tokenize your data
with open('your_data.txt', 'r') as f:
    data = f.read()

train_data = data[:int(len(data) * 0.9)]
val_data = data[int(len(data) * 0.9):]

train_ids = np.array(enc.encode_ordinary(train_data), dtype=np.uint16)
val_ids = np.array(enc.encode_ordinary(val_data), dtype=np.uint16)

train_ids.tofile('data/your_dataset/train.bin')
val_ids.tofile('data/your_dataset/val.bin')
```

### Dataset Size Guidelines

| Dataset Size | Recommended Model | Expected Quality | Training Time |
|--------------|-------------------|------------------|---------------|
| **Small** (<1M tokens) | GPT-2 Small (124M) | Good for style/domain | 5-15 minutes |
| **Medium** (1-10M tokens) | GPT-2 Medium (350M) | High quality | 15-60 minutes |
| **Large** (>10M tokens) | GPT-2 Large/XL (774M-1.5B) | Best quality | 1-4 hours |

**Pro Tips:**
- **Minimum dataset size:** ~100K tokens can work but may overfit
- **Quality over quantity:** Clean, relevant data beats noisy large datasets
- **Train/val split:** Use 90/10 or 95/5 split. Keep validation set representative.
- **Data cleaning:** Remove duplicates, fix encoding issues, ensure consistent formatting

---

## Choosing Your Model Size

nanoGPT provides four pretrained GPT-2 model sizes. Choose based on your needs:

### GPT-2 Small (124M parameters)
**Config:** `config/finetune/gpt2_small.py`

**Best for:**
- Quick experimentation and iteration
- Small to medium datasets (<5M tokens)
- Limited GPU memory (8GB+ VRAM)
- Fast inference requirements

**Stats:**
- Architecture: 12 layers, 12 heads, 768 dimensions
- Memory: ~500MB model weights, ~2-4GB VRAM during training
- Training speed: ~200-400 tokens/sec on A100
- Quality: Good for most use cases

**When to use:** Default choice for most finetuning tasks. Start here unless you have specific needs.

### GPT-2 Medium (350M parameters)
**Config:** `config/finetune/gpt2_medium.py`

**Best for:**
- Higher quality outputs
- Medium to large datasets (1-20M tokens)
- Moderate GPU memory (16GB+ VRAM)

**Stats:**
- Architecture: 24 layers, 16 heads, 1024 dimensions
- Memory: ~1.4GB model weights, ~6-8GB VRAM during training
- Training speed: ~100-200 tokens/sec on A100
- Quality: Noticeably better than Small

**When to use:** When quality matters more than speed, and you have adequate GPU memory.

### GPT-2 Large (774M parameters)
**Config:** `config/finetune/gpt2_large.py`

**Best for:**
- High-quality outputs
- Large datasets (>5M tokens)
- High-end GPUs (24GB+ VRAM)

**Stats:**
- Architecture: 36 layers, 20 heads, 1280 dimensions
- Memory: ~3GB model weights, ~12-16GB VRAM during training
- Training speed: ~50-100 tokens/sec on A100
- Quality: Professional-grade outputs

**When to use:** Production deployments where quality is critical and you have GPU resources.

### GPT-2 XL (1.5B parameters)
**Config:** `config/finetune/gpt2_xl.py`

**Best for:**
- Maximum quality
- Very large datasets (>10M tokens)
- High-end GPUs (40GB+ VRAM) or multi-GPU setups

**Stats:**
- Architecture: 48 layers, 25 heads, 1600 dimensions
- Memory: ~6GB model weights, ~20-32GB VRAM during training
- Training speed: ~25-50 tokens/sec on A100
- Quality: State-of-the-art for GPT-2 family

**When to use:** When you need the absolute best quality and have the hardware to support it.

### Quick Decision Matrix

```
Do you have GPU with 8GB+ VRAM?
├─ Yes: Start with GPT-2 Small
│   ├─ Quality not good enough? → Try Medium (needs 16GB)
│   └─ Still not good enough? → Try Large (needs 24GB)
│
└─ No: Use CPU training with Small model
    (Will be slow: ~30-60 minutes for Shakespeare)
```

---

## Hyperparameter Recommendations

Our finetuning configs provide tested defaults, but you may want to adjust based on your dataset. Here's what each parameter does and when to tune it:

### Learning Rate

**Default:** `3e-5` (all models)

The learning rate is the **most important** hyperparameter for finetuning.

**Guidelines by dataset size:**
- **Small datasets (<1M tokens):** `3e-5` to `5e-5` - Start with default
- **Medium datasets (1-10M tokens):** `3e-5` to `1e-4` - Can use slightly higher
- **Large datasets (>10M tokens):** `1e-5` to `3e-5` - Lower to avoid catastrophic forgetting

**Signs you need to adjust:**
- Loss increasing or unstable? → **Decrease** learning rate by 10x
- Training too slow / loss barely decreasing? → **Increase** by 2-3x
- Model outputs gibberish? → Learning rate likely too high

**Pro tip:** If unsure, start with `3e-5`. It works well across most scenarios.

### Batch Size and Gradient Accumulation

**Defaults:**
- Small: `batch_size=8`, `gradient_accumulation_steps=8` → 65,536 tokens/iter
- Medium: `batch_size=4`, `gradient_accumulation_steps=8` → 32,768 tokens/iter
- Large: `batch_size=2`, `gradient_accumulation_steps=8` → 16,384 tokens/iter
- XL: `batch_size=1`, `gradient_accumulation_steps=8` → 8,192 tokens/iter

**When to adjust:**
- **Out of memory?** → Decrease `batch_size`, increase `gradient_accumulation_steps` to compensate
- **Training too slow?** → Increase `batch_size` if you have VRAM headroom
- **Want larger effective batch size?** → Increase `gradient_accumulation_steps`

**Formula:**
```
Tokens per iteration = batch_size × gradient_accumulation_steps × block_size (1024)
```

**Pro tip:** Keep tokens/iter between 8K-64K for stable training. Higher can work but may need learning rate adjustment.

### Training Iterations

**Default:** `max_iters=5000`

**Guidelines by dataset size:**
- **Small datasets (<1M tokens):** 5000-10000 iters - More iterations help learning
- **Medium datasets (1-10M tokens):** 3000-5000 iters - Default is good
- **Large datasets (>10M tokens):** 2000-5000 iters - Can finish sooner

**How to determine:** Watch validation loss. Stop when it plateaus or starts increasing (overfitting).

**Pro tip:** Set `always_save_checkpoint=False` (default) to only save when validation loss improves. This gives you the best model automatically.

### Learning Rate Schedule

**Defaults:**
```python
decay_lr = True
warmup_iters = 100
lr_decay_iters = 5000  # Match max_iters
min_lr = 3e-6  # learning_rate / 10
```

**When to adjust:**
- **Short training (<2000 iters)?** → Reduce `warmup_iters` to 50
- **Long training (>10000 iters)?** → Increase `warmup_iters` to 200-500
- **Want constant LR?** → Set `decay_lr=False`

**Pro tip:** The defaults work well. Only adjust if you have specific reasons.

### Dataset-Specific Recipes

#### Recipe 1: Small Domain-Specific Dataset (100K-1M tokens)
```python
# Examples: Company docs, niche domain text, small book
model = "gpt2_small"
learning_rate = 5e-5        # Slightly higher for small data
max_iters = 10000           # More iterations
warmup_iters = 200
batch_size = 8              # Keep default
gradient_accumulation_steps = 8
```

**Expected:** Model adapts to domain, may slightly overfit (acceptable for domain adaptation)

#### Recipe 2: Medium General Dataset (1-10M tokens)
```python
# Examples: Multi-book corpus, scraped articles, code repository
model = "gpt2_medium"
learning_rate = 3e-5        # Use defaults
max_iters = 5000
warmup_iters = 100
batch_size = 4
gradient_accumulation_steps = 8
```

**Expected:** High-quality finetuned model with good generalization

#### Recipe 3: Large Diverse Dataset (>10M tokens)
```python
# Examples: Full book series, large code repos, web scrapes
model = "gpt2_large"
learning_rate = 1e-5        # Lower to preserve pretrained knowledge
max_iters = 3000
warmup_iters = 100
batch_size = 2
gradient_accumulation_steps = 8
```

**Expected:** Professional-quality model that maintains broad capabilities

#### Recipe 4: Style Transfer (Shakespeare, specific author)
```python
# Goal: Learn specific writing style
model = "gpt2_small" or "gpt2_medium"
learning_rate = 3e-5
max_iters = 5000
warmup_iters = 100
# Defaults work well
```

**Expected:** Model outputs mimic training data style while maintaining coherence

---

## Step-by-Step Finetuning Walkthrough

Let's walk through a complete finetuning example, from raw data to working model.

### Example: Finetuning on Shakespeare

We'll finetune GPT-2 Medium on Shakespeare's works to create a model that writes in Shakespearean style.

#### Step 1: Prepare the Dataset

```sh
cd data/shakespeare
python prepare.py
```

**What this does:**
- Downloads Shakespeare's complete works (~1MB)
- Tokenizes using GPT-2 BPE tokenizer (matching pretrained model)
- Creates `train.bin` (90% of data) and `val.bin` (10% of data)
- Saves tokenizer metadata in `meta.pkl`

**Output you'll see:**
```
Dataset has 1115394 characters, 338025 tokens
train has 304221 tokens
val has 33804 tokens
```

#### Step 2: Review the Training Configuration

Let's look at what `config/finetune/gpt2_medium.py` contains:

```python
# Model initialization
init_from = 'gpt2-medium'  # Start from pretrained checkpoint

# Output directory
out_dir = 'out-finetune-gpt2-medium'

# Batch configuration (32K tokens per iteration)
batch_size = 4
gradient_accumulation_steps = 8
max_iters = 5000

# Learning rate schedule (conservative for finetuning)
learning_rate = 3e-5
decay_lr = True
warmup_iters = 100
lr_decay_iters = 5000
min_lr = 3e-6

# Evaluation
eval_interval = 250        # Validate every 250 iterations
eval_iters = 200          # Use 200 batches for validation

# Checkpointing
always_save_checkpoint = False  # Only save when val loss improves
```

**Optional:** Create a copy and modify it for your needs:
```sh
cp config/finetune/gpt2_medium.py config/my_shakespeare_config.py
# Edit config/my_shakespeare_config.py with your changes
```

#### Step 3: Start Training

```sh
python train.py config/finetune/gpt2_medium.py
```

**What happens:**
1. **Model initialization** - Downloads GPT-2 Medium checkpoint (~1.4GB) on first run
2. **Dataset loading** - Loads your `train.bin` and `val.bin`
3. **Training loop** - Begins finetuning with progress updates

**Console output you'll see:**
```
Initializing from OpenAI GPT-2 weights: gpt2-medium
number of parameters: 354.82M
num decayed parameter tensors: 194, with 354,747,392 parameters
num non-decayed parameter tensors: 98, with 50,176 parameters
=> using fused AdamW: True

iter 0: loss 3.2844, time 1234.56ms, mfu -100.00%
iter 10: loss 2.9234, time 156.78ms, mfu 15.23%
iter 20: loss 2.7123, time 155.43ms, mfu 15.45%
...
iter 250: loss 1.8234, time 154.32ms, mfu 15.67%
step 250: train loss 1.8234, val loss 1.8456
saving checkpoint to out-finetune-gpt2-medium
iter 260: loss 1.8012, time 155.21ms, mfu 15.66%
...
```

**Key metrics explained:**
- **iter**: Current iteration number
- **loss**: Training loss (lower is better, expect 1.0-2.0 for Shakespeare)
- **time**: Milliseconds per iteration
- **mfu**: Model FLOPs Utilization (GPU efficiency, 15-25% is typical)
- **val loss**: Validation loss (evaluated every `eval_interval` steps)

#### Step 4: Monitor Training Progress

**Watching for convergence:**
- Loss should decrease steadily for first 1000-2000 iterations
- Validation loss should track training loss (slightly higher)
- Training typically converges in 2000-5000 iterations

**Warning signs:**
- **Loss increases:** Learning rate too high → Kill training, reduce LR by 10x
- **Loss stuck/not decreasing:** Learning rate too low → Increase LR by 2-3x
- **Val loss >> train loss:** Overfitting → Stop training early or increase dataset size
- **Both losses plateau:** Training complete → Can stop early

**Optional: Enable WandB logging:**
```python
# In your config file
wandb_log = True
wandb_project = 'shakespeare-finetune'
wandb_run_name = 'gpt2-medium-shakespeare'
```

Then view real-time metrics at wandb.ai

#### Step 5: Evaluate the Finetuned Model

Once training completes (or you stop it early), your best checkpoint is saved:

```sh
ls out-finetune-gpt2-medium/
# ckpt.pt  - Best model checkpoint
# config.json - Training configuration
# iter_xxxxx.pt - Intermediate checkpoints (if enabled)
```

**Generate samples:**
```sh
python sample.py --out_dir=out-finetune-gpt2-medium
```

**Example outputs (Shakespeare-style):**
```
THEODORE:
Thou shalt sell me to the highest bidder: if I die,
I sell thee to the first; if I go mad,
I sell thee to the second; if I lie,
I sell thee to the third; if I slay,
I sell thee to the fourth: so buy or sell,
I tell thee again, thou shalt not sell my possession.

JULIET:
And if thou steal, thou shalt not sell thyself.

THEODORE:
I do not steal; I sell the stolen goods.
```

**Customize generation:**
```sh
# Generate with a specific prompt
python sample.py \
    --out_dir=out-finetune-gpt2-medium \
    --start="To be or not to be" \
    --num_samples=3 \
    --max_new_tokens=200

# Control randomness
python sample.py \
    --out_dir=out-finetune-gpt2-medium \
    --temperature=0.8 \    # Lower = more conservative (default: 1.0)
    --top_k=40             # Top-k sampling (default: 200)
```

#### Step 6: Iterate and Improve

Not happy with results? Try these improvements:

**If output quality is poor:**
1. Train longer (increase `max_iters`)
2. Try a larger model (Medium → Large)
3. Check if validation loss is still decreasing (if yes, keep training)
4. Ensure dataset quality (clean, consistent formatting)

**If model outputs are too generic:**
1. Increase learning rate slightly (3e-5 → 5e-5)
2. Train longer to fully adapt to your domain
3. Check dataset size (might need more data)

**If model outputs are gibberish:**
1. Learning rate too high → Reduce by 10x
2. Training may have diverged → Restart with lower LR
3. Check dataset (encoding issues, corruption)

---

## Expected Results and Benchmarks

Here's what you should expect from finetuning on common datasets:

### Shakespeare Dataset (~1M tokens)

**Dataset stats:**
- Training tokens: ~1M
- Vocabulary: GPT-2 BPE (50,257 tokens)
- Domain: Early Modern English, verse and prose

**Performance benchmarks:**

| Model | Training Time | Final Val Loss | Sample Quality |
|-------|---------------|----------------|----------------|
| **GPT-2 Small** | ~5 min (A100) | 1.00-1.10 | Good Shakespearean style |
| **GPT-2 Medium** | ~15 min (A100) | 0.85-0.95 | Excellent style, coherent |
| **GPT-2 Large** | ~30 min (A100) | 0.80-0.90 | Professional quality |
| **GPT-2 XL** | ~60 min (A100) | 0.75-0.85 | Best quality |

**Hardware timing:**
- **A100 (40GB):** Times above
- **V100 (32GB):** ~2x longer
- **RTX 4090 (24GB):** ~1.5x longer
- **RTX 3090 (24GB):** ~2-3x longer
- **CPU only:** 30-120 minutes (Small model only practical)

### Comparison to Baseline

**Baseline (pretrained GPT-2, no finetuning):**
- Validation loss on Shakespeare: ~3.0-3.5
- Output style: Modern English, generic content

**After finetuning:**
- Validation loss: 0.8-1.1 (66-73% improvement)
- Output style: Shakespearean vocabulary, verse structure, period grammar

**Example quality comparison:**

**Pretrained GPT-2 (no finetuning):**
```
To be or not to be, that is the question everyone asks themselves
at some point in their lives. It's one of the most famous lines
from Shakespeare's Hamlet, and it continues to resonate today...
```
*(Modern English, explanatory style)*

**Finetuned GPT-2 Small:**
```
To be or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles,
And by opposing end them?
```
*(Shakespearean style, iambic pentameter)*

**Finetuned GPT-2 Medium/Large:**
```
To be or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles,
And by opposing end them? To die: to sleep;
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to.
```
*(High-quality Shakespearean verse, coherent themes)*

### Loss Trajectory Examples

**Typical training loss curve (GPT-2 Medium on Shakespeare):**
```
Iter     Train Loss    Val Loss
0        3.28          3.30      (starting from pretrained)
500      2.15          2.18
1000     1.45          1.50
2000     1.05          1.12
3000     0.92          1.00
4000     0.87          0.96
5000     0.84          0.93      (converged)
```

**Healthy training:**
- Steady decrease for first 1000-2000 iters
- Gradual improvement afterward
- Val loss stays close to train loss (within 0.05-0.10)

**Overfitting (warning):**
```
Iter     Train Loss    Val Loss
0        3.28          3.30
2000     1.05          1.15
4000     0.65          1.25      ⚠️ Val loss increasing
6000     0.45          1.35      ⚠️ Severe overfitting
```
→ Stop training early when val loss starts increasing

### Domain-Specific Expectations

**Code datasets:**
- Expected loss: 0.5-1.5 depending on language complexity
- Quality: Model learns syntax, common patterns, API usage
- Recommendation: Use larger models (Medium/Large) for better code structure

**Scientific/Medical text:**
- Expected loss: 1.0-2.0 (technical vocabulary)
- Quality: Model learns domain terminology, writing patterns
- Recommendation: Need larger datasets (5M+ tokens) for good coverage

**Creative writing:**
- Expected loss: 0.8-1.5
- Quality: Model learns style, tone, narrative patterns
- Recommendation: Even small datasets (500K tokens) can work well

---

## Troubleshooting Common Issues

### Issue 1: Out of Memory (CUDA OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions (try in order):**

1. **Reduce batch size:**
```python
batch_size = 1  # Down from 4 or 8
gradient_accumulation_steps = 32  # Increase to maintain effective batch size
```

2. **Use a smaller model:**
```sh
# Instead of gpt2_large, use:
python train.py config/finetune/gpt2_medium.py
```

3. **Reduce block size (context length):**
```python
block_size = 512  # Down from 1024
# Note: This changes model architecture, may impact quality
```

4. **Enable gradient checkpointing (advanced):**
```python
# In train.py, uncomment or add:
# model.gradient_checkpointing_enable()
```

5. **Use CPU (slow but works):**
```python
device = 'cpu'
compile = False
```

### Issue 2: Loss Not Decreasing / Training Stuck

**Symptoms:**
- Loss stays constant or decreases very slowly
- After 1000+ iterations, still near starting loss

**Possible causes and fixes:**

**Cause 1: Learning rate too low**
```python
learning_rate = 1e-4  # Increase from 3e-5
```

**Cause 2: Dataset too small / repetitive**
- Check dataset size: `ls -lh data/your_dataset/*.bin`
- If <100K tokens, consider getting more data
- Verify data quality: Check `train.bin` isn't corrupted

**Cause 3: Bad initialization**
```python
# Verify init_from is correct
init_from = 'gpt2'  # Not 'resume' or 'scratch'
```

**Debugging steps:**
1. Verify dataset loads correctly: Check first iteration loss (~3.0-3.5 for GPT-2)
2. Try default config first: `python train.py config/finetune/gpt2_small.py`
3. Test on Shakespeare dataset to rule out config issues

### Issue 3: Model Outputs Gibberish

**Symptoms:**
- Training appears to work but samples are nonsense
- Random characters or repetitive tokens

**Possible causes:**

**Cause 1: Learning rate too high (training diverged)**
```python
learning_rate = 3e-6  # Reduce by 10x from 3e-5
max_iters = 10000     # Train longer with lower LR
```

**Cause 2: Trained too long (overfitting)**
- Check if val loss started increasing
- Use checkpoint from earlier in training (before divergence)

**Cause 3: Wrong tokenizer/vocabulary**
- Ensure you used same tokenizer for data prep and training
- For GPT-2 finetuning, must use GPT-2 BPE tokenizer (tiktoken)

**Cause 4: Corrupted checkpoint**
```sh
# Remove checkpoints and restart
rm -rf out-finetune-gpt2-medium/*.pt
python train.py config/finetune/gpt2_medium.py
```

### Issue 4: Training is Very Slow

**Symptoms:**
- <100 tokens/second on GPU
- Hours to complete small dataset

**Solutions:**

**For GPU:**
1. **Enable compilation (PyTorch 2.0+):**
```python
compile = True  # Should be default
```

2. **Check GPU utilization:**
```sh
nvidia-smi -l 1
# Should see >80% GPU utilization during training
```

3. **Increase batch size:**
```python
batch_size = 8  # Or 16 if memory allows
gradient_accumulation_steps = 4  # Reduce if increasing batch_size
```

4. **Verify CUDA is working:**
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))
```

**For CPU:**
- CPU training is inherently slow (expected)
- Consider cloud GPU: Google Colab (free), AWS, Lambda Labs

### Issue 5: Val Loss Much Higher Than Train Loss

**Symptoms:**
```
iter 5000: train loss 0.65, val loss 1.35
```
*(Gap >0.3 indicates overfitting)*

**Solutions:**

1. **Stop training early:**
   - Use checkpoint from when val loss was lowest
   - Set `max_iters` to that iteration for next run

2. **Get more data:**
   - Dataset too small for model size
   - Try smaller model or larger dataset

3. **Add regularization (advanced):**
```python
dropout = 0.1     # Add dropout (default 0.0)
weight_decay = 0.1  # Increase weight decay
```

4. **Reduce model size:**
```sh
# Use smaller model that won't overfit
python train.py config/finetune/gpt2_small.py
```

### Issue 6: "Checkpoint Not Found" Error

**Symptoms:**
```
FileNotFoundError: Could not find checkpoint at out-xxx/ckpt.pt
```

**Solutions:**

1. **Check output directory:**
```sh
ls out-finetune-gpt2-small/
# Should contain ckpt.pt if training completed
```

2. **Training may not have finished:**
   - Check if training is still running
   - Wait for "saving checkpoint" message

3. **Specify correct out_dir:**
```sh
# Make sure --out_dir matches training config
python sample.py --out_dir=out-finetune-gpt2-small
```

4. **If no checkpoint saved:**
   - Training may have failed before first checkpoint
   - Check terminal for error messages
   - Try with `always_save_checkpoint=True` to force saving

### Issue 7: Results Not as Good as Expected

**Quality is subjective, but here are improvement strategies:**

**Strategy 1: Train longer**
- If val loss still decreasing, train more iterations
- Double `max_iters` and resume training

**Strategy 2: Use larger model**
```sh
# Small → Medium → Large → XL
python train.py config/finetune/gpt2_large.py
```

**Strategy 3: Tune temperature during sampling**
```sh
# More creative (default 1.0)
python sample.py --out_dir=out-xxx --temperature=1.2

# More conservative
python sample.py --out_dir=out-xxx --temperature=0.7
```

**Strategy 4: Increase learning rate slightly**
```python
learning_rate = 5e-5  # From 3e-5
# Can lead to faster adaptation to your domain
```

**Strategy 5: Check dataset quality**
- Remove duplicates, clean formatting
- Ensure text is representative of desired output
- More diverse data → better generalization

### Getting Help

If you're still stuck:

1. **Check configuration:** Compare your config to default recipes in `config/finetune/`
2. **Test on Shakespeare:** Verify your setup works with known-good dataset
3. **Check GitHub issues:** [nanoGPT Issues](https://github.com/karpathy/nanoGPT/issues)
4. **Reduce to minimal case:** Try smallest model, smallest dataset, default config

---

## Advanced Topics

### Multi-GPU Training

Finetune faster with multiple GPUs using DDP (Distributed Data Parallel):

```sh
# Use all available GPUs
torchrun --standalone --nproc_per_node=GPU_COUNT train.py config/finetune/gpt2_large.py

# Example: 4 GPUs
torchrun --standalone --nproc_per_node=4 train.py config/finetune/gpt2_large.py
```

**What changes:**
- Training speed scales near-linearly (4x GPUs ≈ 4x faster)
- Effective batch size multiplies by GPU count
- Consider reducing `gradient_accumulation_steps` to compensate

### Resuming Interrupted Training

Training interrupted? Resume from checkpoint:

```python
# In your config file, change:
init_from = 'resume'
# Then run training again with same config
```

Or via command line:
```sh
python train.py config/finetune/gpt2_medium.py --init_from=resume
```

### Custom Learning Rate Schedules

Want more control over learning rate decay?

**Constant learning rate (no decay):**
```python
decay_lr = False
learning_rate = 3e-5
```

**Longer warmup:**
```python
warmup_iters = 500  # Default 100
lr_decay_iters = 10000
max_iters = 10000
```

**Cosine annealing (default):**
```python
decay_lr = True
# LR follows cosine curve from learning_rate to min_lr
```

### Finetuning on Multiple Datasets

Want to finetune on multiple domains sequentially?

**Approach 1: Concatenate datasets**
```python
# Combine train.bin files
cat data/dataset1/train.bin data/dataset2/train.bin > data/combined/train.bin
```

**Approach 2: Sequential finetuning**
```sh
# Finetune on dataset 1
python train.py config/finetune/gpt2_medium.py

# Then finetune the result on dataset 2
python train.py my_dataset2_config.py --init_from=resume --out_dir=out-finetune-gpt2-medium
```

### Evaluation Metrics

Beyond validation loss, you can evaluate:

**Perplexity:**
```python
perplexity = math.exp(val_loss)
# Lower is better. <20 is excellent for domain-specific models
```

**Custom evaluation:**
```python
# In train.py, add custom evaluation logic
# Example: Task-specific accuracy, BLEU score, etc.
```

### Exporting for Production

After finetuning, export your model:

**Checkpoint contains:**
- Model weights (`model` key in `ckpt.pt`)
- Optimizer state (not needed for inference)
- Training configuration

**For inference-only:**
```python
# Load checkpoint
checkpoint = torch.load('out-xxx/ckpt.pt')
model_weights = checkpoint['model']

# Save just weights
torch.save(model_weights, 'model_weights_only.pt')
# Much smaller file size
```

### Converting to HuggingFace Format

Want to use your model with HuggingFace transformers?

```python
# See: https://github.com/karpathy/nanoGPT#huggingface-transformers-compatibility
# nanoGPT checkpoints can be loaded into HuggingFace GPT-2 models
# (More detailed conversion script TBD)
```

---

## Quick Reference Card

**Starting a finetune:**
```sh
python train.py config/finetune/gpt2_small.py
```

**Sampling from finetuned model:**
```sh
python sample.py --out_dir=out-finetune-gpt2-small
```

**Common config overrides:**
```sh
# Lower learning rate
python train.py config/finetune/gpt2_small.py --learning_rate=1e-5

# Train longer
python train.py config/finetune/gpt2_small.py --max_iters=10000

# Reduce batch size (OOM)
python train.py config/finetune/gpt2_small.py --batch_size=4

# Enable wandb logging
python train.py config/finetune/gpt2_small.py --wandb_log=True
```

**Model size decision:**
- Quick experimentation → **Small** (124M)
- Production quality → **Medium** (350M)
- Best results → **Large** (774M) or **XL** (1.5B)

**Learning rate rules:**
- Start with: **3e-5**
- Loss exploding → **Divide by 10**
- Training too slow → **Multiply by 3**

**Dataset size guidelines:**
- <1M tokens → Small model, more iterations
- 1-10M tokens → Medium/Large model
- >10M tokens → Can use XL effectively

---

## Next Steps

Now that you've successfully finetuned your first model, explore:

1. **Try different model sizes** - Compare Small vs Medium vs Large
2. **Experiment with your own data** - Domain adaptation to your specific use case
3. **Tune hyperparameters** - Find optimal settings for your dataset
4. **Enable WandB logging** - Track experiments and compare runs
5. **Multi-GPU training** - Scale up with DDP for faster iteration

**Additional resources:**
- **Configuration reference:** `docs/configuration-reference.md` - Complete parameter documentation
- **Finetuning configs:** `config/finetune/README.md` - Detailed config explanations
- **Original README:** `README.md` - Training from scratch, reproducing GPT-2

**Happy finetuning! 🚀**
