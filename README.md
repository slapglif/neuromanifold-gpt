
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

That's it! nanoGPT uses sensible defaults with no config files or arguments required. The configuration system is type-safe and provides helpful validation.

**3. What to expect**

You'll see output like:
```
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

**Want more control?** You can override any setting via command-line arguments or use preset configurations:

```sh
# Override specific settings
python train.py --batch_size=32 --learning_rate=1e-4

# Use a preset configuration
python train.py neuromanifold_gpt.config.presets.train_gpt2

# Combine preset + overrides
python train.py neuromanifold_gpt.config.presets.train_gpt2 --batch_size=32
```

See [Configuration Guide](docs/configuration.md) for detailed usage, or run `python train.py --help` to see all available options.

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

**I have a GPU**. Great, we can quickly train a baby GPT using the default configuration (or the Shakespeare preset):

```sh
# Option 1: Use defaults (optimized for Shakespeare)
python train.py

# Option 2: Use the Shakespeare preset explicitly
python train.py neuromanifold_gpt.config.presets.train_shakespeare_char
```

The default configuration trains a GPT with a context size of up to 256 characters, 384 feature channels, and it is a 6-layer Transformer with 6 heads in each layer. On one A100 GPU this training run takes about 3 minutes and the best validation loss is 1.4697. Model checkpoints are written to the `out-shakespeare-char` directory. Once training finishes, we can sample from the best model:

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
python train.py --devices=1 --precision=32 --compile_model=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

Here we turn off PyTorch 2.0 compile with `--compile_model=False` and use full precision (`--precision=32`). We get a faster but noisier evaluation estimate (`--eval_iters=20`, down from 200), our context size is only 64 characters instead of 256, and the batch size only 12 examples per iteration, not 64. We'll also use a much smaller Transformer (4 layers, 4 heads, 128 embedding size), and decrease the number of iterations to 2000 (and correspondingly usually decay the learning rate to around max_iters with `--lr_decay_iters`). Because our network is so small we also ease down on regularization (`--dropout=0.0`). This still runs in about ~3 minutes, but gets us a loss of only 1.88 and therefore also worse samples, but it's still good fun:

```sh
python sample.py --out_dir=out-shakespeare-char
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

Finally, on Apple Silicon Macbooks with a recent PyTorch version, the system can automatically use Metal Performance Shaders (MPS) for GPU acceleration that can *significantly* accelerate training (2-3X) and allow you to use larger networks. See [Issue 28](https://github.com/karpathy/nanoGPT/issues/28) for more.

## reproducing GPT-2

A more serious deep learning professional may be more interested in reproducing GPT-2 results. So here we go - we first tokenize the dataset, in this case the [OpenWebText](https://openwebtext2.readthedocs.io/en/latest/), an open reproduction of OpenAI's (private) WebText:

```sh
python data/openwebtext/prepare.py
```

This downloads and tokenizes the [OpenWebText](https://huggingface.co/datasets/openwebtext) dataset. It will create a `train.bin` and `val.bin` which holds the GPT2 BPE token ids in one sequence, stored as raw uint16 bytes. Then we're ready to kick off training. To reproduce GPT-2 (124M) you'll want at least an 8X A100 40GB node and run:

```sh
# Use the GPT-2 preset configuration
python train.py neuromanifold_gpt.config.presets.train_gpt2 --devices=8
```

This will run for about 4 days using PyTorch Lightning with Distributed Data Parallel (DDP) and go down to loss of ~2.85. Now, a GPT-2 model just evaluated on OWT gets a val loss of about 3.11, but if you finetune it it will come down to ~2.85 territory (due to an apparent domain gap), making the two models ~match.

Multi-node training is supported via PyTorch Lightning. It is a good idea to benchmark your interconnect (e.g. iperf3). By default checkpoints are periodically written to the `--out_dir`. We can sample from the model by simply running `python sample.py`.

To train on a single GPU, simply run `python train.py`. To see all available configuration options:

```sh
python train.py --help
```

This shows every parameter with its type and default value. You can override any setting via command-line arguments. The configuration system is type-safe and will validate your inputs with helpful error messages.

## baselines

OpenAI GPT-2 checkpoints allow us to get some baselines in place for openwebtext. We can get the numbers using the evaluation presets:

```sh
$ python train.py neuromanifold_gpt.config.presets.eval_gpt2
$ python train.py neuromanifold_gpt.config.presets.eval_gpt2_medium
$ python train.py neuromanifold_gpt.config.presets.eval_gpt2_large
$ python train.py neuromanifold_gpt.config.presets.eval_gpt2_xl
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
python train.py neuromanifold_gpt.config.presets.finetune.gpt2_small

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

## checkpoint management

nanoGPT supports separated checkpoints that split model weights and optimizer state into separate files. This reduces checkpoint sizes by 50%+ and makes it easy to share models without optimizer buffers.

### Quick Example

```sh
# Save separated checkpoints (model + optimizer in separate files)
python train.py --save_separate_optimizer=True

# Save model-only checkpoints (inference-ready, 50%+ smaller)
python train.py --save_model_only=True
```

**Benefits:**
- **Smaller files** - Model-only checkpoints are 50-60% smaller (e.g., GPT-2 124M: 240MB unified → 100MB model-only)
- **Easy sharing** - Share inference-ready models without optimizer state
- **Flexible loading** - Load just the model or both model + optimizer as needed
- **Backward compatible** - Automatically detects and loads both old unified and new separated checkpoint formats

### Comprehensive Guide

For detailed information including:
- **Checkpoint formats** - Understanding unified, separated, and model-only checkpoints
- **Training resumption** - How optimizer state is preserved and restored
- **File structure** - Naming conventions and directory organization
- **Best practices** - When to use each checkpoint format
- **Migration guide** - Converting from unified to separated checkpoints
- **Troubleshooting** - Solutions to common checkpoint issues

**See the complete [Checkpoint Management Guide](docs/checkpoint-management.md)** for step-by-step instructions and best practices.

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

## configuration

nanoGPT uses a type-safe configuration system built on Python dataclasses. This provides IDE autocomplete, type checking, and validation while remaining easy to use.

**Quick usage:**

```sh
# Use defaults
python train.py

# Override specific settings
python train.py --batch_size=32 --learning_rate=1e-4

# Use a preset configuration
python train.py neuromanifold_gpt.config.presets.train_gpt2

# Combine preset + overrides
python train.py neuromanifold_gpt.config.presets.train_gpt2 --max_iters=50000

# See all available options
python train.py --help
```

**See the complete [Configuration Guide](docs/configuration.md)** for detailed documentation, examples, and best practices.

## troubleshooting

Note that by default this repo uses PyTorch 2.0 (i.e. `torch.compile`). This is fairly new and experimental, and not yet available on all platforms (e.g. Windows). If you're running into related error messages try to disable this by adding `--compile_model=False` flag. This will slow down the code but at least it will run.

For some context on this repository, GPT, and language modeling it might be helpful to watch my [Zero To Hero series](https://karpathy.ai/zero-to-hero.html). Specifically, the [GPT video](https://www.youtube.com/watch?v=kCc8FmEb1nY) is popular if you have some prior language modeling context.

For more questions/discussions feel free to stop by **#nanoGPT** on Discord:

[![](https://dcbadge.vercel.app/api/server/3zy8kqD9Cp?compact=true&style=flat)](https://discord.gg/3zy8kqD9Cp)

## acknowledgements

All nanoGPT experiments are powered by GPUs on [Lambda labs](https://lambdalabs.com), my favorite Cloud GPU provider. Thank you Lambda labs for sponsoring nanoGPT!
