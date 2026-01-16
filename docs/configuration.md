# Configuration System

nanoGPT uses a type-safe configuration system built on Python dataclasses. This replaces the legacy `exec(open())` pattern with a secure, IDE-friendly approach that provides autocomplete, type checking, and validation.

## Table of Contents

- [Quick Start](#quick-start)
- [Configuration Methods](#configuration-methods)
- [Available Configs](#available-configs)
- [CLI Overrides](#cli-overrides)
- [Creating Custom Configs](#creating-custom-configs)
- [Migration from Legacy Config](#migration-from-legacy-config)

## Quick Start

### Using Default Configuration

The simplest way to train is with no configuration - just use the defaults:

```bash
python train.py
```

This uses the default `TrainingConfig` settings optimized for the Shakespeare character-level dataset.

### Using a Preset Configuration

Load a preset configuration module:

```bash
python train.py neuromanifold_gpt.config.presets.train_gpt2
```

### Overriding Individual Settings

Override specific settings via command-line arguments:

```bash
python train.py --batch_size=32 --learning_rate=1e-4
```

### Combining Preset + Overrides

Load a preset and override specific values:

```bash
python train.py neuromanifold_gpt.config.presets.train_gpt2 --batch_size=32 --max_iters=10000
```

## Configuration Methods

### 1. Default Configuration

Every script has sensible defaults defined in its config dataclass. For example, `train.py` uses `TrainingConfig`:

```python
from neuromanifold_gpt.config.training import TrainingConfig
from neuromanifold_gpt.config.loader import load_config

config = load_config(TrainingConfig)  # Uses defaults
```

### 2. Preset Configurations

Preset configurations are Python modules that define configuration values:

```python
# neuromanifold_gpt/config/presets/my_preset.py

# Training data
dataset = 'shakespeare_char'
batch_size = 128
block_size = 256

# Model architecture
n_layer = 12
n_head = 12
n_embd = 768

# Training
max_iters = 10000
learning_rate = 6e-4
```

Load it with:

```bash
python train.py neuromanifold_gpt.config.presets.my_preset
```

### 3. Command-Line Overrides

Override any configuration value from the command line:

```bash
python train.py --batch_size=64 --learning_rate=3e-4 --max_iters=5000
```

**Syntax:**
- Use `--key=value` format
- Boolean values: `--compile=True` or `--compile=False`
- Strings: `--dataset=shakespeare_char`
- Numbers: `--batch_size=32` or `--learning_rate=1e-4`
- Lists/tuples: Use Python literal syntax

**Type Safety:**
The loader validates types automatically. If you try `--batch_size=hello`, you'll get a clear error message:

```
ValidationError: Cannot override 'batch_size': expected int, got str
```

## Available Configs

### TrainingConfig

Used by `train.py` for training models. Key parameters:

**Data:**
- `dataset`: Dataset name (e.g., "shakespeare_char")
- `batch_size`: Training batch size (default: 64)
- `block_size`: Sequence length/context window (default: 256)

**Model:**
- `model_type`: "neuromanifold" or "gpt"
- `n_layer`: Number of transformer layers (default: 6)
- `n_head`: Number of attention heads (default: 6)
- `n_embd`: Embedding dimension (default: 384)
- `dropout`: Dropout probability (default: 0.2)

**Training:**
- `max_iters`: Maximum training iterations (default: 5000)
- `learning_rate`: Peak learning rate (default: 1e-3)
- `gradient_accumulation_steps`: Steps to accumulate gradients (default: 1)

**Hardware:**
- `devices`: Number of GPUs (default: 1)
- `precision`: "bf16-mixed", "fp16-mixed", or "32" (default: "bf16-mixed")
- `compile_model`: Use torch.compile (default: False)

**Logging:**
- `wandb_log`: Enable Weights & Biases logging (default: False)
- `wandb_project`: WandB project name
- `out_dir`: Output directory for checkpoints

[See full list with `python train.py --help`]

### SamplingConfig

Used by `sample.py` for generating text. Key parameters:

**Model Loading:**
- `init_from`: "resume" (from checkpoint) or GPT-2 variant ("gpt2", "gpt2-medium", etc.)
- `out_dir`: Checkpoint directory (when init_from="resume")

**Generation:**
- `start`: Initial prompt string (default: "\n")
- `num_samples`: Number of samples to generate (default: 10)
- `max_new_tokens`: Max tokens per sample (default: 500)
- `temperature`: Sampling temperature (default: 0.8)
- `top_k`: Top-k filtering (default: 200)

**Hardware:**
- `device`: "cpu", "cuda", "cuda:0", etc.
- `dtype`: "float32", "bfloat16", or "float16"
- `compile`: Use torch.compile (default: False)

Example:
```bash
python sample.py --out_dir=out-shakespeare-char --temperature=1.0 --num_samples=5
```

### EvalConfig

Used by `eval.py` for zero-shot benchmark evaluation:

**Benchmark:**
- `benchmark`: "lambada", "hellaswag", "piqa", "winogrande", or "all"
- `eval_iters`: Max examples to evaluate (None = all)

**Model:**
- `out_dir`: Checkpoint directory to load from

**Hardware:**
- `device`, `dtype`, `compile`: Same as SamplingConfig

Example:
```bash
python eval.py --out_dir=out-gpt2 --benchmark=lambada
```

### BenchConfig

Used by `bench.py` for performance benchmarking:

**Benchmarking:**
- `profile`: Use PyTorch profiler (True) or simple timing (False)
- `burnin_steps`: Burn-in steps for simple benchmarking (default: 10)
- `benchmark_steps`: Measurement steps (default: 20)

**Data & Model:**
- `batch_size`, `block_size`, `n_layer`, `n_head`, `n_embd`
- `real_data`: Use real data vs synthetic (default: True)

Example:
```bash
python bench.py --batch_size=12 --profile=True
```

## CLI Overrides

### Viewing Available Options

See all available configuration options for any script:

```bash
python train.py --help
python sample.py --help
python eval.py --help
python bench.py --help
```

This shows every configurable parameter with its type and default value.

### Type Validation

The configuration loader validates types automatically:

```bash
# ✓ Valid - integer
python train.py --batch_size=32

# ✗ Invalid - string instead of integer
python train.py --batch_size=hello
# Error: Cannot override 'batch_size': expected int, got str

# ✓ Valid - float in scientific notation
python train.py --learning_rate=1e-4

# ✗ Invalid - typo in parameter name
python train.py --batch_szie=32
# Error: Unknown config key: batch_szie
# Available: batch_size, learning_rate, ...
```

### Boolean Values

Use Python boolean literals (case-sensitive):

```bash
python train.py --compile_model=True
python train.py --wandb_log=False
```

### String Values

String values are automatically detected:

```bash
python train.py --dataset=openwebtext
python train.py --model_type=gpt
```

### Numeric Values

Integers and floats are parsed automatically:

```bash
python train.py --batch_size=64           # int
python train.py --learning_rate=3e-4      # float
python train.py --dropout=0.1             # float
```

## Creating Custom Configs

### Method 1: Create a Preset Module

Create a new Python file in `neuromanifold_gpt/config/presets/`:

```python
# neuromanifold_gpt/config/presets/my_experiment.py
"""
Custom configuration for my experiment.
"""

# Data
dataset = 'openwebtext'
batch_size = 128
block_size = 1024

# Model - Large GPT
model_type = 'gpt'
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0

# Training - Match GPT-2 training
max_iters = 600000
learning_rate = 6e-4
min_lr = 6e-5
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Hardware
devices = 8
precision = 'bf16-mixed'
compile_model = True

# Logging
wandb_log = True
wandb_project = 'nanogpt-gpt2-repro'
wandb_run_name = 'gpt2-124m-repro'
```

Use it with:

```bash
python train.py neuromanifold_gpt.config.presets.my_experiment
```

### Method 2: Command-Line Overrides Only

For quick experiments, just override defaults:

```bash
python train.py \
    --dataset=openwebtext \
    --batch_size=128 \
    --n_layer=12 \
    --n_head=12 \
    --n_embd=768 \
    --max_iters=100000 \
    --learning_rate=6e-4
```

### Method 3: Preset + Overrides (Recommended)

Start with a preset and tweak specific values:

```bash
# Use GPT-2 preset but with smaller batch size and shorter training
python train.py neuromanifold_gpt.config.presets.train_gpt2 \
    --batch_size=32 \
    --max_iters=50000
```

## Migration from Legacy Config

### Legacy System (exec-based)

The old system used `exec(open())` to execute Python config files:

```python
# OLD - DEPRECATED AND REMOVED
exec(open('configurator.py').read())  # Security risk!

# Config values were injected into global namespace
print(batch_size)  # Where did this come from?
```

**Problems:**
- Security vulnerability (arbitrary code execution)
- No type checking or autocomplete
- Hard to trace where variables come from
- Runtime errors instead of early validation

### New System (type-safe)

The new system uses dataclasses and safe imports:

```python
# NEW - Type-safe and secure
from neuromanifold_gpt.config.training import TrainingConfig
from neuromanifold_gpt.config.loader import load_config

config = load_config(TrainingConfig, sys.argv[1:])

# Type-safe access with autocomplete
print(config.batch_size)  # IDE knows this exists and is an int
```

**Benefits:**
- No security vulnerabilities
- IDE autocomplete and type checking
- Early validation with helpful error messages
- Clear data flow and dependencies

### Converting Old Config Files

If you have old config files in the `config/` directory, they've been moved to `config/legacy/` for reference.

To convert an old config to the new system:

**Before (config/train_my_model.py):**
```python
# old exec-based config
batch_size = 128
learning_rate = 3e-4
max_iters = 10000
n_layer = 8
# ... more variables
```

**After (neuromanifold_gpt/config/presets/train_my_model.py):**
```python
# new type-safe preset
"""Configuration for my model."""

# Data
batch_size = 128

# Training
learning_rate = 3e-4
max_iters = 10000

# Model
n_layer = 8
# ... more variables
```

The structure is the same - just move it to the presets directory!

**Using it:**

```bash
# Old way (no longer works)
python train.py config/train_my_model.py

# New way
python train.py neuromanifold_gpt.config.presets.train_my_model
```

## Examples

### Example 1: Quick Shakespeare Training

Train on Shakespeare with default settings:

```bash
python data/shakespeare_char/prepare.py
python train.py
```

### Example 2: Larger Model with Custom Settings

Train a larger model with specific hyperparameters:

```bash
python train.py \
    --n_layer=12 \
    --n_head=12 \
    --n_embd=768 \
    --batch_size=32 \
    --max_iters=10000 \
    --learning_rate=6e-4
```

### Example 3: GPT-2 Reproduction

Use the GPT-2 preset with multi-GPU:

```bash
python train.py neuromanifold_gpt.config.presets.train_gpt2 --devices=8
```

### Example 4: Resume Training with Modified LR

Resume training from a checkpoint with different learning rate:

```bash
python train.py \
    --out_dir=out-shakespeare-char \
    --learning_rate=1e-4 \
    --max_iters=10000
```

### Example 5: Generate Samples

Generate creative samples with high temperature:

```bash
python sample.py \
    --out_dir=out-shakespeare-char \
    --temperature=1.2 \
    --num_samples=5 \
    --max_new_tokens=1000
```

### Example 6: Evaluate on Benchmarks

Evaluate a trained model on zero-shot benchmarks:

```bash
python eval.py \
    --out_dir=out-gpt2 \
    --benchmark=all
```

## Best Practices

### 1. Use Presets for Reproducibility

Create preset configs for experiments you want to reproduce:

```bash
# Easy to share and reproduce
python train.py neuromanifold_gpt.config.presets.paper_experiment_1
```

### 2. Document Your Presets

Add docstrings to your preset files:

```python
"""
Configuration for Shakespeare character-level model.

Training time: ~3 minutes on A100
Expected val loss: ~1.47

Based on: config/train_shakespeare_char.py
"""
```

### 3. Use Descriptive Run Names

When logging to WandB, use clear run names:

```bash
python train.py \
    --wandb_log=True \
    --wandb_run_name=gpt-shakespeare-6layer-$(date +%Y%m%d)
```

### 4. Version Control Your Presets

Commit your preset configs to git for reproducibility:

```bash
git add neuromanifold_gpt/config/presets/my_experiment.py
git commit -m "Add config for experiment XYZ"
```

### 5. Check Help Before Experimenting

Always check available options first:

```bash
python train.py --help
```

## Troubleshooting

### "Unknown config key" Error

```
ValidationError: Unknown config key: batch_szie
Use one of: batch_size, learning_rate, ...
```

**Fix:** Check spelling. Use `--help` to see available keys.

### Type Mismatch Error

```
ValidationError: Cannot override 'batch_size': expected int, got str
```

**Fix:** Don't quote numeric values: `--batch_size=32` not `--batch_size="32"`

### Module Not Found

```
ValidationError: Cannot load config module: my.preset
```

**Fix:** Ensure the module exists in `neuromanifold_gpt/config/presets/` and use the full module path.

### Boolean Not Working

```bash
# Wrong
python train.py --compile_model=true  # lowercase won't work

# Right
python train.py --compile_model=True  # Python literal, case-sensitive
```

## Advanced Usage

### Programmatic Config Loading

You can load configs programmatically in your own scripts:

```python
from neuromanifold_gpt.config.training import TrainingConfig
from neuromanifold_gpt.config.loader import load_config

# Load with custom args
config = load_config(
    TrainingConfig,
    ['neuromanifold_gpt.config.presets.nano', '--batch_size=32'],
    show_help=False
)

print(f"Batch size: {config.batch_size}")
print(f"Learning rate: {config.learning_rate}")
```

### Creating New Config Classes

Define your own config dataclasses:

```python
from dataclasses import dataclass
from neuromanifold_gpt.config.loader import load_config

@dataclass
class MyCustomConfig:
    """My custom configuration."""
    experiment_name: str = "my_experiment"
    num_trials: int = 100
    use_feature_x: bool = True
    threshold: float = 0.95

# Load it
config = load_config(MyCustomConfig, sys.argv[1:])
```

### Nested Configurations

For complex configurations, you can use nested dataclasses:

```python
from dataclasses import dataclass

@dataclass
class ModelConfig:
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384

@dataclass
class TrainingConfig:
    model: ModelConfig = ModelConfig()
    batch_size: int = 64
    learning_rate: float = 1e-3
```

## Summary

The new type-safe configuration system provides:

- **Security:** No arbitrary code execution
- **Type Safety:** Validation and IDE support
- **Flexibility:** Defaults, presets, and CLI overrides
- **Clarity:** Explicit data flow and dependencies
- **Maintainability:** Easy to understand and modify

Start with defaults, use presets for common experiments, and override specific values as needed. The system will validate your inputs and provide helpful error messages.

For more examples, see the preset configurations in `neuromanifold_gpt/config/presets/`.
