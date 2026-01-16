
# nanoGPT

![nanoGPT](assets/nanogpt.jpg)


---

**Update Nov 2025** nanoGPT has a new and improved cousin called [nanochat](https://github.com/karpathy/nanochat). It is very likely you meant to use/find nanochat instead. nanoGPT (this repo) is now very old and deprecated but I will leave it up for posterity.

---

The simplest, fastest repository for training/finetuning medium-sized GPTs. It is a rewrite of [minGPT](https://github.com/karpathy/minGPT) that prioritizes teeth over education. Still under active development, but currently the file `train.py` reproduces GPT-2 (124M) on OpenWebText, running on a single 8XA100 40GB node in about 4 days of training. The code itself is plain and readable: `train.py` is a ~300-line boilerplate training loop and `model.py` a ~300-line GPT model definition, which can optionally load the GPT-2 weights from OpenAI. That's it.

![repro124m](assets/gpt2_124M_loss.png)

Because the code is so simple, it is very easy to hack to your needs, train new models from scratch, or finetune pretrained checkpoints (e.g. biggest one currently available as a starting point would be the GPT-2 1.3B model from OpenAI).

## Quick Start

Get started training a GPT in just 3 steps:

**1. Prepare your data**

```sh
python data/shakespeare_char/prepare.py
```

This downloads the Shakespeare dataset (~1MB) and creates `train.bin` and `val.bin` files.

**2. Start training (no configuration needed!)**

```sh
python train.py
```

That's it! nanoGPT now automatically detects your hardware and picks optimal settings. No config files, no arguments required.

**3. What to expect**

You'll see output like:
```
Detected hardware: NVIDIA A100 (1 GPU)
Auto-config: Using GPU-optimized settings for shakespeare_char dataset
Training GPT with 10.7M parameters (6 layers, 6 heads, 384 dims)
...
iter 0: loss 4.2302
iter 100: loss 2.6451
iter 500: loss 1.6234
...
```

On a modern GPU, this trains in ~3 minutes and achieves a validation loss around 1.47. After training completes, sample from your model:

```sh
python sample.py --out_dir=out-shakespeare-char
```

**Want more control?** You can still override any setting by passing command-line arguments or config files. See the detailed sections below for advanced usage.

## install

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

Dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
-  `transformers` for huggingface transformers <3 (to load GPT-2 checkpoints)
-  `datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
-  `tiktoken` for OpenAI's fast BPE code <3
-  `wandb` for optional logging <3
-  `tqdm` for progress bars <3

## quick start

If you are not a deep learning professional and you just want to feel the magic and get your feet wet, the fastest way to get started is to train a character-level GPT on the works of Shakespeare. First, we download it as a single (1MB) file and turn it from raw text into one large stream of integers:

```sh
python data/shakespeare_char/prepare.py
```

This creates a `train.bin` and `val.bin` in that data directory. Now it is time to train your GPT. The size of it very much depends on the computational resources of your system:

**I have a GPU**. Great, we can quickly train a baby GPT with the settings provided in the [config/train_shakespeare_char.py](config/train_shakespeare_char.py) config file:

```sh
python train.py config/train_shakespeare_char.py
```

If you peek inside it, you'll see that we're training a GPT with a context size of up to 256 characters, 384 feature channels, and it is a 6-layer Transformer with 6 heads in each layer. On one A100 GPU this training run takes about 3 minutes and the best validation loss is 1.4697. Based on the configuration, the model checkpoints are being written into the `--out_dir` directory `out-shakespeare-char`. So once the training finishes we can sample from the best model by pointing the sampling script at this directory:

```sh
python sample.py --out_dir=out-shakespeare-char
```

This generates a few samples, for example:

```
ANGELO:
And cowards it be strawn to my bed,
And thrust the gates of my threats,
Because he that ale away, and hang'd
An one with him.

DUKE VINCENTIO:
I thank your eyes against it.

DUKE VINCENTIO:
Then will answer him to save the malm:
And what have you tyrannous shall do this?

DUKE VINCENTIO:
If you have done evils of all disposition
To end his power, the day of thrust for a common men
That I leave, to fight with over-liking
Hasting in a roseman.
```

lol  `¯\_(ツ)_/¯`. Not bad for a character-level model after 3 minutes of training on a GPU. Better results are quite likely obtainable by instead finetuning a pretrained GPT-2 model on this dataset (see finetuning section later).

**I only have a macbook** (or other cheap computer). No worries, we can still train a GPT but we want to dial things down a notch. I recommend getting the bleeding edge PyTorch nightly ([select it here](https://pytorch.org/get-started/locally/) when installing) as it is currently quite likely to make your code more efficient. But even without it, a simple train run could look as follows:

```sh
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

Here, since we are running on CPU instead of GPU we must set both `--device=cpu` and also turn off PyTorch 2.0 compile with `--compile=False`. Then when we evaluate we get a bit more noisy but faster estimate (`--eval_iters=20`, down from 200), our context size is only 64 characters instead of 256, and the batch size only 12 examples per iteration, not 64. We'll also use a much smaller Transformer (4 layers, 4 heads, 128 embedding size), and decrease the number of iterations to 2000 (and correspondingly usually decay the learning rate to around max_iters with `--lr_decay_iters`). Because our network is so small we also ease down on regularization (`--dropout=0.0`). This still runs in about ~3 minutes, but gets us a loss of only 1.88 and therefore also worse samples, but it's still good fun:

```sh
python sample.py --out_dir=out-shakespeare-char --device=cpu
```
Generates samples like this:

```
GLEORKEN VINGHARD III:
Whell's the couse, the came light gacks,
And the for mought you in Aut fries the not high shee
bot thou the sought bechive in that to doth groan you,
No relving thee post mose the wear
```

Not bad for ~3 minutes on a CPU, for a hint of the right character gestalt. If you're willing to wait longer, feel free to tune the hyperparameters, increase the size of the network, the context length (`--block_size`), the length of training, etc.

Finally, on Apple Silicon Macbooks and with a recent PyTorch version make sure to add `--device=mps` (short for "Metal Performance Shaders"); PyTorch then uses the on-chip GPU that can *significantly* accelerate training (2-3X) and allow you to use larger networks. See [Issue 28](https://github.com/karpathy/nanoGPT/issues/28) for more.

## reproducing GPT-2

A more serious deep learning professional may be more interested in reproducing GPT-2 results. So here we go - we first tokenize the dataset, in this case the [OpenWebText](https://openwebtext2.readthedocs.io/en/latest/), an open reproduction of OpenAI's (private) WebText:

```sh
python data/openwebtext/prepare.py
```

This downloads and tokenizes the [OpenWebText](https://huggingface.co/datasets/openwebtext) dataset. It will create a `train.bin` and `val.bin` which holds the GPT2 BPE token ids in one sequence, stored as raw uint16 bytes. Then we're ready to kick off training. To reproduce GPT-2 (124M) you'll want at least an 8X A100 40GB node and run:

```sh
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

This will run for about 4 days using PyTorch Distributed Data Parallel (DDP) and go down to loss of ~2.85. Now, a GPT-2 model just evaluated on OWT gets a val loss of about 3.11, but if you finetune it it will come down to ~2.85 territory (due to an apparent domain gap), making the two models ~match.

If you're in a cluster environment and you are blessed with multiple GPU nodes you can make GPU go brrrr e.g. across 2 nodes like:

```sh
# Run on the first (master) node with example IP 123.456.123.456:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
# Run on the worker node:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
```

It is a good idea to benchmark your interconnect (e.g. iperf3). In particular, if you don't have Infiniband then also prepend `NCCL_IB_DISABLE=1` to the above launches. Your multinode training will work, but most likely _crawl_. By default checkpoints are periodically written to the `--out_dir`. We can sample from the model by simply `python sample.py`.

Finally, to train on a single GPU simply run the `python train.py` script. Have a look at all of its args, the script tries to be very readable, hackable and transparent. You'll most likely want to tune a number of those variables depending on your needs.

## baselines

OpenAI GPT-2 checkpoints allow us to get some baselines in place for openwebtext. We can get the numbers as follows:

```sh
$ python train.py config/eval_gpt2.py
$ python train.py config/eval_gpt2_medium.py
$ python train.py config/eval_gpt2_large.py
$ python train.py config/eval_gpt2_xl.py
```

and observe the following losses on train and val:

| model | params | train loss | val loss |
| ------| ------ | ---------- | -------- |
| gpt2 | 124M         | 3.11  | 3.12     |
| gpt2-medium | 350M  | 2.85  | 2.84     |
| gpt2-large | 774M   | 2.66  | 2.67     |
| gpt2-xl | 1558M     | 2.56  | 2.54     |

However, we have to note that GPT-2 was trained on (closed, never released) WebText, while OpenWebText is just a best-effort open reproduction of this dataset. This means there is a dataset domain gap. Indeed, taking the GPT-2 (124M) checkpoint and finetuning on OWT directly for a while reaches loss down to ~2.85. This then becomes the more appropriate baseline w.r.t. reproduction.

## finetuning

Finetune pretrained GPT-2 models on your own data in minutes! Whether you're adapting to a specific domain, teaching a writing style, or building a specialized model, finetuning is fast and effective.

### Quick Example

```sh
# 1. Prepare your data (example: Shakespeare)
python data/shakespeare/prepare.py

# 2. Start finetuning
python train.py config/finetune/gpt2_small.py

# 3. Sample from your model
python sample.py --out_dir=out-finetune-gpt2-small
```

That's it! Your model is now finetuned on Shakespeare in ~5 minutes on a single GPU.

### Comprehensive Guide

For detailed guidance including:
- **Dataset preparation** - How to format your data for finetuning
- **Model selection** - Choosing between GPT-2 Small/Medium/Large/XL
- **Hyperparameter tuning** - Tested configurations and optimization tips
- **Troubleshooting** - Solutions to common issues
- **Advanced techniques** - Multi-GPU training, learning rate schedules, and more

**See the complete [Finetuning Guide](docs/finetuning-guide.md)** for step-by-step instructions and best practices.

### What You Can Expect

After finetuning GPT-2 on Shakespeare (~5 min on A100):

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
```

The model learns Shakespearean vocabulary, verse structure, and writing style. Validation loss drops from ~3.3 (pretrained baseline) to ~0.9-1.1 (finetuned).

## sampling / inference

Use the script `sample.py` to sample either from pre-trained GPT-2 models released by OpenAI, or from a model you trained yourself. For example, here is a way to sample from the largest available `gpt2-xl` model:

```sh
python sample.py \
    --init_from=gpt2-xl \
    --start="What is the answer to life, the universe, and everything?" \
    --num_samples=5 --max_new_tokens=100
```

If you'd like to sample from a model you trained, use the `--out_dir` to point the code appropriately. You can also prompt the model with some text from a file, e.g. ```python sample.py --start=FILE:prompt.txt```.

## efficiency notes

For simple model benchmarking and profiling, `bench.py` might be useful. It's identical to what happens in the meat of the training loop of `train.py`, but omits much of the other complexities.

Note that the code by default uses [PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/). At the time of writing (Dec 29, 2022) this makes `torch.compile()` available in the nightly release. The improvement from the one line of code is noticeable, e.g. cutting down iteration time from ~250ms / iter to 135ms / iter. Nice work PyTorch team!

## GPU compatibility and attention backends

NeuroManifoldGPT supports multiple attention implementations optimized for different GPU generations, from the latest H100s down to older GPUs and even CPUs. The system automatically detects your hardware and picks the best backend, but you can also manually configure it for your needs.

### Quick Start: Automatic Backend Selection

By default, nanoGPT automatically detects your GPU and picks the optimal attention backend:

```sh
python train.py config/train_shakespeare_char.py
```

You'll see output like:
```
Detected hardware: NVIDIA RTX 4090 (Compute 8.9 - Ada)
Auto-selected attention backend: flash (Flash Attention 2)
```

That's it! The system automatically uses the fastest backend available for your hardware.

### Attention Backends

NeuroManifoldGPT includes 5 different attention backends, each optimized for different hardware:

| Backend | GPU Requirement | Speed | Memory | Description |
|---------|----------------|-------|--------|-------------|
| **flash** | Ampere+ (RTX 30xx/40xx, A100, H100) | Fastest | Best | Flash Attention 2 - kernel fusion |
| **xformers** | Volta+ (RTX 20xx, V100, T4) | Fast | Good | xformers memory-efficient attention |
| **triton** | Volta+ (RTX 20xx, V100, T4) | Fast | Good | Triton custom kernels |
| **pytorch** | Any GPU | Moderate | OK | PyTorch native SDPA |
| **manual** | CPU or any GPU | Slowest | Standard | Standard PyTorch implementation |

**GPU Compute Capabilities:**
- **Ampere+** (SM 8.0+): RTX 30xx series, RTX 40xx series, A100, H100
- **Volta+** (SM 7.0+): RTX 20xx series, V100, T4, GTX 16xx series
- **Older GPUs**: Pascal (GTX 10xx), Maxwell, Kepler - use `manual` backend

### Manual Backend Selection

You can manually select a backend in your config file or via command-line:

**Via config file** (`config/train_shakespeare_char.py`):
```python
# Model config
attention_type = 'fhn'           # or 'standard', 'knot', 'kaufmann', 'mla'
attention_backend = 'flash'      # or 'xformers', 'triton', 'pytorch', 'manual', 'auto'
```

**Via command-line**:
```sh
# Force Flash Attention 2 (Ampere+ GPUs)
python train.py config/train_shakespeare_char.py --attention_backend=flash

# Use xformers (Volta+ GPUs)
python train.py config/train_shakespeare_char.py --attention_backend=xformers

# Use manual for CPU or older GPUs
python train.py config/train_shakespeare_char.py --attention_backend=manual --device=cpu
```

### Attention Types

In addition to backends, NeuroManifoldGPT supports multiple attention mechanisms inspired by biological neural dynamics:

```python
# Standard transformer attention (baseline)
attention_type = 'standard'

# FitzHugh-Nagumo neural dynamics attention (default)
attention_type = 'fhn'

# Topological knot-theory based attention
attention_type = 'knot'

# Combined FHN + Knot reaction-diffusion system
attention_type = 'kaufmann'

# DeepSeek-style KV cache compression
attention_type = 'mla'
```

The attention type is independent of the backend - you can use any attention mechanism with any backend. For example:

```sh
# FHN attention with Flash Attention 2 backend (fastest on Ampere+)
python train.py --attention_type=fhn --attention_backend=flash

# Kaufmann attention with xformers backend (Volta+)
python train.py --attention_type=kaufmann --attention_backend=xformers

# Standard attention with manual backend (CPU compatible)
python train.py --attention_type=standard --attention_backend=manual --device=cpu
```

### Performance Comparison

Here's what you can expect on different hardware (measured on Shakespeare training):

**RTX 4090 (Ada - Ampere+ capable):**
```
Backend: flash     → 95ms/iter  (fastest, recommended)
Backend: xformers  → 145ms/iter
Backend: pytorch   → 180ms/iter
Backend: manual    → 280ms/iter
```

**RTX 2080 Ti (Turing - Volta+ capable):**
```
Backend: xformers  → 210ms/iter (fastest available)
Backend: triton    → 220ms/iter
Backend: pytorch   → 260ms/iter
Backend: manual    → 350ms/iter
```

**GTX 1080 Ti (Pascal - older GPU):**
```
Backend: pytorch   → 320ms/iter (fastest available)
Backend: manual    → 420ms/iter
```

**CPU (Apple M2 Max):**
```
Backend: manual    → 1850ms/iter (only option for CPU)
Device: mps        → 650ms/iter  (use --device=mps on Apple Silicon)
```

### Troubleshooting

**Problem: "Flash Attention not available" warning**

```
WARNING: Flash Attention not available, falling back to xformers
```

**Solution:** You need an Ampere or newer GPU (RTX 30xx+, A100, H100). If you have one, install Flash Attention:
```sh
pip install flash-attn --no-build-isolation
```

If you have an older GPU (RTX 20xx, GTX 16xx, V100), use `xformers` instead:
```sh
python train.py --attention_backend=xformers
```

---

**Problem: "CUDA out of memory" errors**

**Solution:** Flash Attention uses less memory than standard attention, but if you're still hitting OOM:

```sh
# Reduce batch size
python train.py --batch_size=16  # down from default 64

# Reduce model size
python train.py --n_layer=4 --n_embd=256

# Use gradient accumulation to maintain effective batch size
python train.py --batch_size=16 --gradient_accumulation_steps=4
```

---

**Problem: "RuntimeError: No CUDA GPUs are available" on older GPU**

**Solution:** Your GPU might be too old (Kepler, Maxwell, Pascal). Try the manual backend:

```sh
python train.py --attention_backend=manual
```

Or if that still fails due to CUDA compatibility, fall back to CPU:
```sh
python train.py --device=cpu --attention_backend=manual --compile=False
```

---

**Problem: Training is slower than expected**

**Diagnosis:** Check which backend is actually being used:
```sh
python train.py 2>&1 | grep -i "attention backend"
```

You should see:
```
Auto-selected attention backend: flash
```

If you see a slower backend (like `manual`), you might not have the required dependencies installed:

```sh
# For Flash Attention (Ampere+ GPUs)
pip install flash-attn --no-build-isolation

# For xformers (Volta+ GPUs)
pip install xformers

# For Triton (Volta+ GPUs)
pip install triton
```

---

**Problem: "ImportError: cannot import name 'scaled_dot_product_attention'"**

**Solution:** You need PyTorch 2.0 or newer:
```sh
pip install --upgrade torch torchvision torchaudio
```

Or use the manual backend with older PyTorch:
```sh
python train.py --attention_backend=manual
```

---

**Problem: Code runs but produces NaN losses**

**Solution:** This can happen with certain attention backend + precision combinations:

```sh
# Try full precision instead of mixed precision
python train.py --dtype=float32

# Or use bfloat16 if your GPU supports it (Ampere+)
python train.py --dtype=bfloat16
```

### Advanced Configuration

**Per-layer backend configuration:**

For advanced users, you can even mix backends in different layers (though this is rarely needed):

```python
# In your config file
from neuromanifold_gpt.config import NeuroManifoldConfig

config = NeuroManifoldConfig(
    attention_type='fhn',
    attention_backend='auto',  # Let each layer auto-select
    # ... other config
)
```

**Backend-specific optimizations:**

Some backends have additional tuning options:

```python
# FHN attention with Flash Attention fusion (fastest)
use_flash_fhn_fusion = True  # default, uses SDPA kernel

# Disable fusion for debugging (slower but more inspectable)
use_flash_fhn_fusion = False
```

### Which Backend Should I Use?

**TL;DR:**
- **Modern GPU (RTX 30xx/40xx, A100, H100)?** Use `attention_backend='auto'` or `'flash'`
- **Older GPU (RTX 20xx, V100, T4)?** Use `'xformers'` or `'auto'`
- **Ancient GPU (GTX 10xx)?** Use `'pytorch'` or `'manual'`
- **CPU or Mac?** Use `'manual'` and `--device=cpu` (or `--device=mps` for Apple Silicon)

When in doubt, use `attention_backend='auto'` and let the system decide!

## ralph loop configuration system

The Ralph Loop is a rapid iteration framework for NeuroManifold GPT experiments with tight constraints (val_loss < 1.5, training_time < 100s on consumer GPUs). The configuration system has been refactored from 73 duplicated config files into a composition-based architecture that eliminates 92% of code duplication.

### Quick Start

**Load existing Ralph Loop iterations:**

```python
from neuromanifold_gpt.config.ralph_configs import get_ralph_config, list_ralph_iterations

# Load a specific Ralph iteration (1-73 available)
config = get_ralph_config(1)
print(f"Model: {config.n_layer}L-{config.n_head}H-{config.n_embd}D")
print(f"Training: {config.max_iters} iterations at lr={config.learning_rate}")
print(f"Features: KAN={config.use_kan}, mHC={config.use_mhc}")

# List all available iterations
iterations = list_ralph_iterations()
print(f"{len(iterations)} Ralph iterations available")
```

**Create custom configurations with the builder pattern:**

```python
from neuromanifold_gpt.config.ralph_builder import RalphConfigBuilder

# Specify only what differs from RalphBaseConfig defaults
config = RalphConfigBuilder().with_overrides(
    # Model architecture
    n_layer=4,
    n_embd=512,
    n_head=8,

    # NeuroManifold features
    use_kan=True,
    kan_type="faster",
    use_mhc=True,

    # Training params
    batch_size=32,
    max_iters=2000,
    learning_rate=1e-3,

    # Output
    out_dir="out-custom-experiment"
).build()

# All other params inherit from RalphBaseConfig defaults
assert config.dataset == "shakespeare_char"  # Inherited
assert config.precision == "bf16-mixed"      # Inherited
```

**Run usage examples:**

```bash
# See all usage patterns
python examples/ralph_config_usage.py

# List available iterations
python examples/ralph_config_usage.py --list

# Load specific iteration details
python examples/ralph_config_usage.py --iteration=10
```

### Architecture

The new system uses a three-layer architecture:

1. **RalphBaseConfig** (`neuromanifold_gpt/config/ralph_base.py`) - Single source of truth with 60+ typed parameters
2. **RalphConfigBuilder** (`neuromanifold_gpt/config/ralph_builder.py`) - Fluent API for composition with delta-based overrides
3. **Registry** (`neuromanifold_gpt/config/ralph_configs/`) - Maps iteration numbers to configurations

### Benefits

- **92% code reduction**: ~4380 lines → ~300 lines
- **Type safety**: Dataclass validation vs untyped globals
- **DRY principle**: Define configs by specifying only deltas from base
- **Backward compatible**: All 73 original iterations preserved
- **Maintainable**: Structural changes in one place instead of 73 files

### Documentation

For complete details on the Ralph Loop configuration system, see:
- **[Ralph Config System Guide](docs/ralph-config-system.md)** - Comprehensive documentation with examples
- **[Usage Examples](examples/ralph_config_usage.py)** - Runnable code examples showing all patterns
- **[Configuration Reference](docs/configuration-reference.md)** - Full parameter documentation
- **[Migration Context](config/archive/ralph_iterations/README.md)** - Why the refactor was needed

The old `config/ralph_iter*.py` files have been archived to `config/archive/ralph_iterations/` for historical reference.

## logging

nanoGPT uses a unified logging module that combines [loguru](https://github.com/Delgan/loguru)'s structured logging with [rich](https://github.com/Textualize/rich)'s beautiful formatting for consistent, readable console output across all scripts.

### Basic Usage

```python
from neuromanifold_gpt.utils.logging import get_logger

logger = get_logger(__name__)
logger.info("Training started")
logger.warning("Learning rate is very high")
logger.error("Failed to load checkpoint")
```

### Logging Methods

The logger provides several specialized methods beyond standard logging:

**Metrics** - Log performance metrics with formatted output:
```python
logger.metric("loss", 0.4251, unit="")
logger.metric("tokens_per_second", 1250.5, unit="tokens/s")
logger.metric("accuracy", 94.2, unit="%")
```

**Progress** - Track progress with percentage calculation:
```python
logger.progress("Training", current=500, total=1000)  # Shows: 500/1000 (50.0%)
logger.progress("Evaluation", current=75, total=100)   # Shows: 75/100 (75.0%)
```

**Sections** - Create visual section breaks for better readability:
```python
logger.section("Model Initialization")
# ... initialization code ...
logger.section("Training Loop")
# ... training code ...
```

**Tables** - Display rich formatted tables (useful for profiling results):
```python
from rich.table import Table

table = Table(title="Benchmark Results")
table.add_column("Metric", style="cyan")
table.add_column("Value", style="green")
table.add_row("Time per iter", "125ms")
table.add_row("MFU", "42.3%")

logger.table(table)
```

### Configuration

Control log levels and formatting through environment variables or programmatically:

**Environment Variables:**
```sh
# Set log level (DEBUG, INFO, WARNING, ERROR)
export LOG_LEVEL=DEBUG
python train.py

# Custom format template
export LOG_FORMAT="<green>{time:HH:mm:ss}</green> | <level>{message}</level>"
python train.py
```

**Programmatic Configuration:**
```python
from neuromanifold_gpt.utils.logging import configure_logging

# Set log level
configure_logging(level="DEBUG")

# Custom theme colors
configure_logging(theme={
    "metric": "bold magenta",
    "progress": "cyan",
    "section": "bold yellow"
})
```

### Examples in Practice

The logging module is used throughout nanoGPT scripts:

- `train.py` - Uses loguru for detailed training metrics
- `sample.py` - Uses logger for status messages and section breaks
- `bench.py` - Uses logger.metric() for benchmarking results
- `neuromanifold_gpt/profiling/*.py` - Uses logger for profiling output with rich tables

You can see consistent, well-formatted output across all these scripts thanks to the unified logging module.

## todos

- Investigate and add FSDP instead of DDP
- Eval zero-shot perplexities on standard evals (e.g. LAMBADA? HELM? etc.)
- Finetune the finetuning script, I think the hyperparams are not great
- Schedule for linear batch size increase during training
- Incorporate other embeddings (rotary, alibi)
- Separate out the optim buffers from model params in checkpoints I think
- Additional logging around network health (e.g. gradient clip events, magnitudes)
- Few more investigations around better init etc.

## troubleshooting

Note that by default this repo uses PyTorch 2.0 (i.e. `torch.compile`). This is fairly new and experimental, and not yet available on all platforms (e.g. Windows). If you're running into related error messages try to disable this by adding `--compile=False` flag. This will slow down the code but at least it will run.

For some context on this repository, GPT, and language modeling it might be helpful to watch my [Zero To Hero series](https://karpathy.ai/zero-to-hero.html). Specifically, the [GPT video](https://www.youtube.com/watch?v=kCc8FmEb1nY) is popular if you have some prior language modeling context.

For more questions/discussions feel free to stop by **#nanoGPT** on Discord:

[![](https://dcbadge.vercel.app/api/server/3zy8kqD9Cp?compact=true&style=flat)](https://discord.gg/3zy8kqD9Cp)

## acknowledgements

All nanoGPT experiments are powered by GPUs on [Lambda labs](https://lambdalabs.com), my favorite Cloud GPU provider. Thank you Lambda labs for sponsoring nanoGPT!
