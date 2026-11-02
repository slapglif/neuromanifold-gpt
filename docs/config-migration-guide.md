# Configuration System Migration Guide

This guide helps you migrate from the legacy `exec()`-based configuration system to the new type-safe configuration system using dataclasses.

## Table of Contents

1. [Overview](#overview)
2. [Before/After Examples](#beforeafter-examples)
3. [Migration Steps](#migration-steps)
4. [Breaking Changes](#breaking-changes)
5. [FAQ](#faq)

---

## Overview

### Why Migrate?

The legacy configuration system used `exec(open('configurator.py').read())` which had several problems:

- **Security Risk**: Executes arbitrary Python code from config files
- **No Type Safety**: No IDE autocomplete or type checking
- **Hidden Dependencies**: Config values injected into global namespace
- **Poor Error Messages**: Runtime errors instead of validation errors

The new system uses Python dataclasses with:

- **Type Safety**: Full IDE support with autocomplete and type checking
- **Validation**: Clear error messages with recovery suggestions
- **Security**: No code execution, just data imports
- **Documentation**: Self-documenting with field types and defaults

### Architecture Changes

**Old System:**
```
configurator.py (exec'd) → globals() → manual parsing
```

**New System:**
```
Dataclass Defaults → Preset Module (imported) → CLI Overrides → Validated Instance
```

---

## Before/After Examples

### Example 1: Training Script

**BEFORE (Old Pattern - DEPRECATED):**

```python
# train.py
import sys
from ast import literal_eval

# Default configuration
out_dir = 'out'
eval_interval = 250
batch_size = 64
learning_rate = 1e-3

# Load config file using exec() (SECURITY RISK!)
if len(sys.argv) > 1:
    exec(open(sys.argv[1]).read())

# Apply CLI overrides manually
for arg in sys.argv[2:]:
    if '=' in arg:
        k, v = arg.split('=')
        k = k.lstrip('-')
        try:
            v = literal_eval(v)
        except:
            pass
        globals()[k] = v

# Use global variables
print(f"Training with batch_size={batch_size}, lr={learning_rate}")
```

**AFTER (New Pattern - RECOMMENDED):**

```python
# train.py
import sys
from neuromanifold_gpt.config.loader import load_config
from neuromanifold_gpt.config.training import TrainingConfig

# Load configuration with type-safe validation
config = load_config(TrainingConfig, sys.argv[1:])

# Use typed config object with autocomplete
print(f"Training with batch_size={config.batch_size}, lr={config.learning_rate}")
```

### Example 2: Custom Configuration File

**BEFORE (Old Pattern - DEPRECATED):**

```python
# config/my_experiment.py
# This file is exec()'d - can contain arbitrary code!

out_dir = 'out-my-experiment'
batch_size = 128
learning_rate = 5e-4
n_layer = 12
n_head = 8
dropout = 0.1

# Could even run code during config load (BAD!)
import os
os.system('rm -rf /')  # DANGEROUS!
```

**AFTER (New Pattern - RECOMMENDED):**

```python
# neuromanifold_gpt/config/presets/my_experiment.py
"""Custom experiment preset configuration.

This module provides configuration values for my custom experiment.
It's imported (not exec'd), so it's safe and type-checked.
"""

# Configuration values (module-level variables)
out_dir = 'out-my-experiment'
batch_size = 128
learning_rate = 5e-4
n_layer = 12
n_head = 8
dropout = 0.1

# No arbitrary code execution allowed - just data!
```

### Example 3: CLI Override Usage

**BEFORE:**

```bash
# Using old exec-based system
python train.py config/train_shakespeare_char.py --batch_size=32 --learning_rate=1e-4
```

**AFTER:**

```bash
# Using new type-safe system
python train.py neuromanifold_gpt.config.presets.shakespeare_char --batch_size=32 --learning_rate=1e-4

# Or with defaults only
python train.py --batch_size=32 --learning_rate=1e-4

# Get help on available options
python train.py --help
```

### Example 4: Programmatic Config Creation

**BEFORE:**

```python
# Create config dict manually
config = {
    'batch_size': 64,
    'learning_rate': 1e-3,
    'n_layer': 6,
    'n_head': 6,
    'dropout': 0.2
}

# Pass dict around (no type checking)
def train(config_dict):
    lr = config_dict['learning_rate']  # No autocomplete, typos possible
    # ...
```

**AFTER:**

```python
from neuromanifold_gpt.config.training import TrainingConfig

# Create typed config object
config = TrainingConfig(
    batch_size=64,
    learning_rate=1e-3,
    n_layer=6,
    n_head=6,
    dropout=0.2
)

# Pass typed object (full IDE support)
def train(config: TrainingConfig):
    lr = config.learning_rate  # Autocomplete works!
    # ...
```

---

## Migration Steps

Follow these steps to migrate your code from the old system to the new system.

### Step 1: Identify Configuration Usage

Find all places where you use the old pattern:

```bash
# Search for exec(open()) patterns
grep -r "exec(open" .

# Search for configurator.py files
find . -name "configurator*.py"
```

### Step 2: Choose Your Configuration Class

Identify which configuration dataclass you need:

| Use Case | Configuration Class | Location |
|----------|-------------------|----------|
| Training models | `TrainingConfig` | `neuromanifold_gpt.config.training` |
| Sampling/generation | `SamplingConfig` | `neuromanifold_gpt.config.training` |
| Evaluation/benchmarks | `EvalConfig` | `neuromanifold_gpt.config.training` |
| Performance benchmarking | `BenchConfig` | `neuromanifold_gpt.config.training` |

### Step 3: Update Script Imports

Replace old pattern with new imports:

```python
# Remove this:
import sys
from ast import literal_eval

# Add this:
from neuromanifold_gpt.config.loader import load_config
from neuromanifold_gpt.config.training import TrainingConfig  # or other config class
```

### Step 4: Replace Configuration Loading

Replace the old loading pattern:

```python
# REMOVE THIS (old pattern):
# ❌ Old way
config_dict = {}
exec(open('configurator.py').read())

# REPLACE WITH (new pattern):
# ✅ New way
config = load_config(TrainingConfig, sys.argv[1:])
```

### Step 5: Migrate Configuration Files to Presets

If you have custom configuration files:

1. Create a new preset module:
   ```
   neuromanifold_gpt/config/presets/my_preset.py
   ```

2. Copy your configuration values (data only, no code):
   ```python
   # neuromanifold_gpt/config/presets/my_preset.py
   """My custom preset description."""

   # Configuration values
   out_dir = 'out-my-experiment'
   batch_size = 128
   learning_rate = 5e-4
   # ... other values
   ```

3. Use the preset:
   ```bash
   python train.py neuromanifold_gpt.config.presets.my_preset
   ```

### Step 6: Update Global Variable Access

Replace global variable access with config object access:

```python
# BEFORE:
def train():
    print(f"Batch size: {batch_size}")  # Global variable
    print(f"Learning rate: {learning_rate}")

# AFTER:
def train(config: TrainingConfig):
    print(f"Batch size: {config.batch_size}")  # Typed attribute
    print(f"Learning rate: {config.learning_rate}")
```

### Step 7: Update CLI Override Handling

The new system handles CLI overrides automatically:

```python
# BEFORE:
for arg in sys.argv[1:]:
    if '=' in arg:
        k, v = arg.split('=')
        # ... manual parsing

# AFTER:
# No code needed! load_config() handles it automatically
config = load_config(TrainingConfig, sys.argv[1:])
```

### Step 8: Test Your Migration

1. Run with defaults:
   ```bash
   python train.py
   ```

2. Run with a preset:
   ```bash
   python train.py neuromanifold_gpt.config.presets.nano
   ```

3. Run with CLI overrides:
   ```bash
   python train.py --batch_size=32 --learning_rate=1e-4
   ```

4. Test error handling:
   ```bash
   python train.py --invalid_key=123  # Should show helpful error
   python train.py --batch_size=invalid  # Should show type error
   ```

---

## Breaking Changes

### 1. Configuration File Format

**Change:** Configuration files are now Python modules (imported), not scripts (exec'd).

**Impact:** Config files can only contain data assignments, not arbitrary code.

**Migration:**
- Remove any `import` statements that aren't used for type hints
- Remove any function calls or code execution
- Keep only variable assignments

**Example:**
```python
# ❌ OLD - This will fail
import torch
out_dir = f'out-{torch.cuda.device_count()}'

# ✅ NEW - Just data
out_dir = 'out-multi-gpu'
```

### 2. Configuration Access Pattern

**Change:** Configuration is now a typed object, not global variables.

**Impact:** Must access config via `config.field_name` instead of global `field_name`.

**Migration:**
```python
# ❌ OLD
def train():
    print(batch_size)  # Global variable

# ✅ NEW
def train(config: TrainingConfig):
    print(config.batch_size)  # Object attribute
```

### 3. CLI Override Syntax

**Change:** CLI overrides now require `--` prefix and use Python literal syntax.

**Impact:** Some override formats may need adjustment.

**Migration:**
```bash
# OLD - May have worked
python train.py batch_size=32

# NEW - Requires --prefix
python train.py --batch_size=32

# Boolean values need proper capitalization
python train.py --compile_model=True  # Not 'true' or '1'
```

### 4. Type Validation

**Change:** Configuration values are now type-checked.

**Impact:** Invalid types will raise errors immediately instead of causing runtime failures.

**Migration:**
```bash
# ❌ This will fail with clear error
python train.py --batch_size=abc

# ✅ This works
python train.py --batch_size=32
```

### 5. Unknown Configuration Keys

**Change:** Unknown configuration keys now raise errors immediately.

**Impact:** Typos in config keys are caught early.

**Migration:**
```bash
# ❌ Typo caught immediately
python train.py --btch_size=32
# Error: Unknown config key: btch_size
# Use one of: batch_size, block_size, ...

# ✅ Correct spelling
python train.py --batch_size=32
```

### 6. Configuration File Location

**Change:** Presets now live in `neuromanifold_gpt/config/presets/` instead of `config/`.

**Impact:** Update paths when loading presets.

**Migration:**
```bash
# OLD
python train.py config/train_shakespeare_char.py

# NEW
python train.py neuromanifold_gpt.config.presets.shakespeare_char
```

### 7. No Dynamic Config Modification

**Change:** Configuration files can't modify themselves based on runtime conditions.

**Impact:** Complex config logic needs to move to application code.

**Migration:**
```python
# ❌ OLD - Config file with logic
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# ✅ NEW - Logic in application
config = load_config(TrainingConfig)
device = 'cuda' if torch.cuda.is_available() else config.device
```

---

## FAQ

### General Questions

#### Q: Do I have to migrate immediately?

**A:** No, but it's recommended. The old `exec()` pattern is deprecated and may be removed in future versions. A compatibility layer exists in `neuromanifold_gpt.config.compat` for gradual migration.

#### Q: What if I have a lot of custom config files?

**A:** You can migrate them incrementally:
1. Use the compatibility layer (`apply_config_overrides()`) as a temporary bridge
2. Migrate high-priority configs first
3. Create presets for commonly-used configurations
4. Gradually update remaining configs

#### Q: Can I still use file-based configs?

**A:** Yes, but they must be valid Python modules (not scripts). Place them in `neuromanifold_gpt/config/presets/` and import them by module path.

### Technical Questions

#### Q: How do I see all available configuration options?

**A:** Use the `--help` flag:
```bash
python train.py --help
```

Or inspect the dataclass:
```python
from neuromanifold_gpt.config.training import TrainingConfig
from dataclasses import fields

for field in fields(TrainingConfig):
    print(f"{field.name}: {field.type} = {field.default}")
```

#### Q: How do I override nested configuration?

**A:** The current system uses flat configuration. For nested configs, use dot notation in the future or set them programmatically:

```python
config = load_config(TrainingConfig)
# Modify after loading if needed
config.out_dir = f'out-{config.dataset}'
```

#### Q: Can I validate my own custom constraints?

**A:** Yes, use a custom config class with a `__post_init__` method:

```python
from dataclasses import dataclass
from neuromanifold_gpt.config.training import TrainingConfig

@dataclass
class MyTrainingConfig(TrainingConfig):
    def __post_init__(self):
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )
```

#### Q: How do I pass configuration between functions?

**A:** Pass the config object as a typed parameter:

```python
def train(config: TrainingConfig) -> None:
    model = create_model(config)
    optimizer = create_optimizer(config)
    # ...

def create_model(config: TrainingConfig) -> nn.Module:
    return Model(
        n_layer=config.n_layer,
        n_head=config.n_head,
        # ...
    )
```

#### Q: How do I serialize configuration for logging?

**A:** Use `dataclasses.asdict()`:

```python
from dataclasses import asdict
import json

config = load_config(TrainingConfig)

# Convert to dict
config_dict = asdict(config)

# Save as JSON
with open('config.json', 'w') as f:
    json.dump(config_dict, f, indent=2)

# Log to wandb
import wandb
wandb.config.update(config_dict)
```

### Migration Questions

#### Q: What if I get "Unknown config key" errors?

**A:** This means you're trying to set a configuration parameter that doesn't exist. Check the error message for available keys:

```bash
$ python train.py --my_custom_key=123
Error: Unknown config key: my_custom_key
Use one of: batch_size, learning_rate, n_layer, ...
```

Either:
1. Fix the typo if it's a known parameter
2. Add the field to your config dataclass if it's a new parameter

#### Q: What if I get type mismatch errors?

**A:** The new system validates types. Check the error message:

```bash
$ python train.py --batch_size=abc
Error: Cannot override 'batch_size': expected int, got str
```

Provide the correct type:
```bash
python train.py --batch_size=32  # Integer
python train.py --learning_rate=1e-4  # Float
python train.py --compile_model=True  # Boolean (capital T)
```

#### Q: How do I migrate complex config files with logic?

**A:** Move the logic to your application code:

**OLD:**
```python
# config/complex.py
import torch

# Dynamic configuration
if torch.cuda.device_count() > 1:
    batch_size = 128
    devices = torch.cuda.device_count()
else:
    batch_size = 64
    devices = 1
```

**NEW:**
```python
# neuromanifold_gpt/config/presets/complex.py
# Static preset (multi-GPU assumption)
batch_size = 128
devices = 4

# Application code handles runtime logic
import torch
from neuromanifold_gpt.config.loader import load_config
from neuromanifold_gpt.config.training import TrainingConfig

config = load_config(TrainingConfig, ['neuromanifold_gpt.config.presets.complex'])

# Override based on actual hardware
if torch.cuda.device_count() < config.devices:
    config.devices = torch.cuda.device_count()
    config.batch_size = 64  # Reduce for fewer GPUs
```

#### Q: Can I still use environment variables?

**A:** Yes, but handle them in application code:

```python
import os
from neuromanifold_gpt.config.loader import load_config
from neuromanifold_gpt.config.training import TrainingConfig

config = load_config(TrainingConfig)

# Override from environment
if 'BATCH_SIZE' in os.environ:
    config.batch_size = int(os.environ['BATCH_SIZE'])
if 'LEARNING_RATE' in os.environ:
    config.learning_rate = float(os.environ['LEARNING_RATE'])
```

### Best Practices

#### Q: Where should I put my custom presets?

**A:** For project-specific presets, use:
```
neuromanifold_gpt/config/presets/your_preset.py
```

For experiment-specific configs, consider:
```
experiments/your_experiment/config.py
```

Then load with:
```python
config = load_config(TrainingConfig, ['experiments.your_experiment.config'])
```

#### Q: How do I organize multiple related configurations?

**A:** Create a preset module with variants:

```python
# neuromanifold_gpt/config/presets/my_experiments.py
"""Configurations for my research experiments."""

# Base configuration
base_batch_size = 64
base_learning_rate = 1e-3

# Variant A: Higher learning rate
variant_a_learning_rate = 5e-3

# Variant B: Larger batch
variant_b_batch_size = 128
```

Then load variants with overrides:
```bash
# Variant A
python train.py neuromanifold_gpt.config.presets.my_experiments \
    --learning_rate=5e-3

# Variant B
python train.py neuromanifold_gpt.config.presets.my_experiments \
    --batch_size=128
```

#### Q: How do I document my configurations?

**A:** Use docstrings and comments:

```python
# neuromanifold_gpt/config/presets/my_preset.py
"""Configuration for reproducing Paper X results.

This preset replicates the hyperparameters from:
    Smith et al., "Amazing Research Paper", NeurIPS 2024
    https://arxiv.org/abs/2024.12345

Key differences from default:
- Higher learning rate for faster convergence
- Reduced dropout for better memorization
- Larger batch size for stable gradients
"""

# Model architecture (matches paper Table 1)
n_layer = 12
n_head = 12
n_embd = 768

# Training hyperparameters (matches paper Section 4.2)
learning_rate = 5e-4  # Higher than default for faster convergence
batch_size = 128      # Larger for gradient stability
dropout = 0.1         # Reduced from 0.2 for better memorization
```

---

## Additional Resources

- **Configuration Reference**: See `docs/configuration-reference.md` for complete field documentation
- **Example Presets**: Browse `neuromanifold_gpt/config/presets/` for preset examples
- **Loader Implementation**: See `neuromanifold_gpt/config/loader.py` for technical details
- **Compatibility Layer**: See `neuromanifold_gpt/config/compat.py` for backward compatibility
- **Type Definitions**: See `neuromanifold_gpt/config/training.py` for all configuration classes

---

## Getting Help

If you encounter issues during migration:

1. **Check Error Messages**: The new system provides detailed error messages with recovery suggestions
2. **Use `--help`**: Run scripts with `--help` to see available options
3. **Review Examples**: Check `neuromanifold_gpt/config/presets/` for working examples
4. **Test Incrementally**: Migrate one script at a time and test thoroughly

---

*Last Updated: 2024-01-16*
*Configuration System Version: 2.0*
