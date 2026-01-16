# Ralph Loop Configuration System

**A Composition-Based Approach to Experimental Hyperparameter Management**

This guide documents the Ralph Loop configuration system, which provides a DRY (Don't Repeat Yourself) approach to managing experimental configurations through composition patterns rather than file duplication.

## Overview

The Ralph Loop is a rapid iteration framework for NeuroManifold GPT experiments with tight constraints:
- **Validation loss target**: < 1.5
- **Training time target**: < 100 seconds
- **Hardware**: Single consumer GPU (6-12GB VRAM)
- **Dataset**: Shakespeare character-level

The configuration system evolved from 73 duplicated files (`ralph_iter1.py` through `ralph_iter116.py`) to a composition-based architecture that:
- **Eliminates duplication**: 92% code reduction (~4380 lines → ~300 lines)
- **Improves maintainability**: Single source of truth for common parameters
- **Enables type safety**: Dataclass-based configuration with validation
- **Preserves history**: All 73 iterations accessible via registry

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Core Components](#core-components)
4. [Usage Patterns](#usage-patterns)
5. [Creating New Configurations](#creating-new-configurations)
6. [Migration Guide](#migration-guide)
7. [API Reference](#api-reference)
8. [Best Practices](#best-practices)

---

## Quick Start

### Loading an Existing Iteration

```python
from neuromanifold_gpt.config.ralph_configs import get_ralph_config

# Load Ralph iteration 1 configuration
config = get_ralph_config(1)

# Access configuration attributes (type-safe)
print(f"Batch size: {config.batch_size}")
print(f"Layers: {config.n_layer}")
print(f"Embedding dim: {config.n_embd}")
```

### Creating a Custom Configuration

```python
from neuromanifold_gpt.config.ralph_builder import RalphConfigBuilder

# Build a custom config with delta overrides
config = RalphConfigBuilder().with_overrides(
    batch_size=32,
    n_layer=4,
    n_embd=512,
    use_kan=True,
    learning_rate=1e-3
).build()

# Configuration is validated on build()
print(f"Custom config: {config.batch_size} batch, {config.n_layer} layers")
```

### Listing Available Iterations

```python
from neuromanifold_gpt.config.ralph_configs import list_ralph_iterations

# Get all available iteration numbers
iterations = list_ralph_iterations()
print(f"Available iterations: {iterations}")
# Output: [1, 2, 3, ..., 116]
```

---

## Architecture Overview

The Ralph config system uses a three-layer architecture:

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: Registry (ralph_configs/registry.py)              │
│ - Maps iteration numbers to config builders                │
│ - Provides get_ralph_config(iteration) API                 │
│ - Stores only deltas from base (DRY principle)             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: Builder (ralph_builder.py)                        │
│ - Fluent API for config composition                        │
│ - Method chaining: with_overrides(**kwargs).build()        │
│ - Validates configuration on build()                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: Base Config (ralph_base.py)                       │
│ - Single source of truth for all parameters                │
│ - Dataclass with type annotations and defaults             │
│ - ~60 configuration fields covering all Ralph features     │
└─────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Composition over Duplication**: Define configs by specifying only what differs from base
2. **Single Source of Truth**: `RalphBaseConfig` defines all common parameters once
3. **Type Safety**: Dataclass validation ensures configuration correctness
4. **Explicitness**: Deltas make it clear what changed in each iteration
5. **Backward Compatibility**: All 73 original iterations preserved via registry

---

## Core Components

### 1. RalphBaseConfig

**Location**: `neuromanifold_gpt/config/ralph_base.py`

The foundational dataclass that defines all configuration parameters with sensible defaults optimized for Ralph Loop experiments.

**Key Features**:
- 60+ typed configuration fields
- Default values optimized for fast iteration
- Automatic validation via `__post_init__`
- Comprehensive docstrings for all parameters

**Categories**:
- **Data**: `dataset`, `batch_size`, `block_size`, `num_workers`
- **Model Architecture**: `n_layer`, `n_head`, `n_embd`, `dropout`
- **NeuroManifold Features**: `use_sdr`, `use_kan`, `use_mhc`, `use_mla`
- **FHN Dynamics**: `fhn_threshold`, `fhn_tau`, `n_fhn_steps`
- **Training**: `max_iters`, `learning_rate`, `weight_decay`, `grad_clip`
- **Evaluation**: `eval_interval`, `eval_iters`, `log_interval`
- **Output**: `out_dir`, `save_checkpoints`, `wandb_log`
- **Hardware**: `devices`, `precision`, `compile_model`

**Example**:
```python
from neuromanifold_gpt.config.ralph_base import RalphBaseConfig

# Use default configuration
config = RalphBaseConfig()

# Override specific parameters
config = RalphBaseConfig(
    batch_size=32,
    n_layer=4,
    learning_rate=1e-3
)
```

### 2. RalphConfigBuilder

**Location**: `neuromanifold_gpt/config/ralph_builder.py`

A builder pattern implementation for fluent configuration composition with delta-based overrides.

**Key Features**:
- Method chaining for readable config construction
- Delta specification (only list what changes)
- Validation on `build()` call
- Type-safe overrides

**API**:
- `with_overrides(**kwargs)`: Add configuration deltas
- `build()`: Construct validated `RalphBaseConfig` instance

**Example**:
```python
from neuromanifold_gpt.config.ralph_builder import RalphConfigBuilder

# Chain multiple override sets
config = (
    RalphConfigBuilder()
    .with_overrides(
        # Model size
        n_layer=8,
        n_embd=512,
    )
    .with_overrides(
        # Training settings
        batch_size=64,
        learning_rate=2e-3,
        max_iters=2000,
    )
    .with_overrides(
        # NeuroManifold features
        use_kan=True,
        use_mhc=True,
        use_mla=True,
    )
    .build()
)

print(f"Built config: {config.n_layer} layers, {config.n_embd} dims")
```

### 3. Ralph Config Registry

**Location**: `neuromanifold_gpt/config/ralph_configs/`

A centralized registry mapping iteration numbers to configuration builders.

**Structure**:
```python
# registry.py contains:
_RALPH_ITERATIONS = {
    1: ralph_iter1,
    2: ralph_iter2,
    ...
    116: ralph_iter116,
}

def get_ralph_config(iteration: int) -> RalphBaseConfig:
    """Get configuration for a Ralph Loop iteration."""
    ...

def list_ralph_iterations() -> list[int]:
    """List all available iteration numbers."""
    ...
```

**Iteration Functions**:
Each iteration is defined as a function that uses the builder pattern:

```python
def ralph_iter1() -> RalphBaseConfig:
    """Ralph Loop Iteration 1 - Tiny config for 6GB GPU, sub-100s training.

    GOALS: val_loss < 1.5, training_time < 100s

    Key changes from base:
    - Reduced model size (2 layers, 256 embd)
    - Minimal batch/block size for memory
    - Short training run (500 iters)
    - mHC enabled for stability
    """
    return RalphConfigBuilder().with_overrides(
        batch_size=32,
        block_size=128,
        n_layer=2,
        n_embd=256,
        use_mhc=True,
        max_iters=500,
        out_dir="out-ralph-iter1",
    ).build()
```

---

## Usage Patterns

### Pattern 1: Load Existing Iteration

Use this when you want to reproduce or build upon a previous experiment.

```python
from neuromanifold_gpt.config.ralph_configs import get_ralph_config

# Load iteration 50
config = get_ralph_config(50)

# Use with training script
from train import train
train(config)
```

### Pattern 2: Customize Existing Iteration

Modify a specific iteration's configuration:

```python
from neuromanifold_gpt.config.ralph_configs import get_ralph_config
from dataclasses import replace

# Load base iteration
config = get_ralph_config(10)

# Override specific parameters
custom_config = replace(
    config,
    batch_size=128,      # Double batch size
    learning_rate=5e-4,  # Lower learning rate
    out_dir="out-custom-iter10"
)
```

### Pattern 3: Build New Iteration from Scratch

Create a completely new configuration:

```python
from neuromanifold_gpt.config.ralph_builder import RalphConfigBuilder

# Define a new experimental configuration
config = RalphConfigBuilder().with_overrides(
    # Experiment: "Large model with all NeuroManifold features"

    # Model scaling
    n_layer=12,
    n_embd=768,
    n_head=12,

    # Enable all advanced features
    use_sdr=True,
    use_kan=True,
    kan_type="faster",
    use_mhc=True,
    use_full_mhc=True,
    use_mla=True,
    use_mtp=True,

    # FHN dynamics
    n_fhn_steps=10,
    use_fhn_imex=True,
    use_fhn_partitioning=True,

    # Training (longer run for larger model)
    batch_size=32,
    max_iters=5000,
    learning_rate=1e-3,
    warmup_iters=500,

    # Output
    out_dir="out-large-all-features",
    save_checkpoints=True,
    wandb_log=True,
).build()
```

### Pattern 4: Configuration Families

Create variations systematically:

```python
from neuromanifold_gpt.config.ralph_builder import RalphConfigBuilder

def create_layer_sweep_configs(base_config_id=1):
    """Create configs sweeping layer count."""
    configs = {}
    for n_layers in [2, 4, 6, 8, 12]:
        config = RalphConfigBuilder().with_overrides(
            n_layer=n_layers,
            n_embd=n_layers * 64,  # Scale embedding with depth
            out_dir=f"out-layer-sweep-{n_layers}",
        ).build()
        configs[n_layers] = config
    return configs

# Generate experiment family
layer_configs = create_layer_sweep_configs()

# Run all experiments
for n_layers, config in layer_configs.items():
    print(f"Training {n_layers}-layer model...")
    train(config)
```

### Pattern 5: Batch Loading for Comparison

Load multiple iterations for analysis:

```python
from neuromanifold_gpt.config.ralph_configs import get_ralph_config, list_ralph_iterations

# Load all iterations for comparison
all_configs = {
    i: get_ralph_config(i)
    for i in list_ralph_iterations()
}

# Analyze configuration trends
import pandas as pd

data = {
    'iteration': [],
    'n_layer': [],
    'n_embd': [],
    'batch_size': [],
    'learning_rate': [],
}

for i, config in all_configs.items():
    data['iteration'].append(i)
    data['n_layer'].append(config.n_layer)
    data['n_embd'].append(config.n_embd)
    data['batch_size'].append(config.batch_size)
    data['learning_rate'].append(config.learning_rate)

df = pd.DataFrame(data)
print(df.describe())

# Plot hyperparameter evolution
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(df['iteration'], df['n_layer'])
plt.title('Layer Count Evolution')
plt.subplot(2, 2, 2)
plt.plot(df['iteration'], df['n_embd'])
plt.title('Embedding Dimension Evolution')
plt.subplot(2, 2, 3)
plt.plot(df['iteration'], df['batch_size'])
plt.title('Batch Size Evolution')
plt.subplot(2, 2, 4)
plt.plot(df['iteration'], df['learning_rate'])
plt.title('Learning Rate Evolution')
plt.tight_layout()
plt.savefig('ralph_hyperparameter_evolution.png')
```

---

## Creating New Configurations

### Step 1: Understand the Base

Review `RalphBaseConfig` to see default values:

```python
from neuromanifold_gpt.config.ralph_base import RalphBaseConfig
import inspect

# Print all defaults
base = RalphBaseConfig()
for field in base.__dataclass_fields__:
    value = getattr(base, field)
    print(f"{field}: {value}")
```

### Step 2: Identify Your Deltas

Determine what needs to change from defaults:

```python
# Example: Experiment with KAN layers
deltas = {
    'use_kan': True,           # Enable KAN
    'kan_type': 'faster',      # Use FasterKAN variant
    'kan_num_centers': 8,      # RSWAF centers
    'n_layer': 4,              # Smaller model for speed
    'learning_rate': 5e-4,     # Lower LR for stability
    'out_dir': 'out-kan-experiment',
}
```

### Step 3: Build and Validate

```python
from neuromanifold_gpt.config.ralph_builder import RalphConfigBuilder

# Build configuration
config = RalphConfigBuilder().with_overrides(**deltas).build()

# Validation happens automatically on build()
# Check critical constraints
assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
assert config.batch_size > 0, "batch_size must be positive"

print(f"✓ Configuration validated: KAN experiment ready")
```

### Step 4: Document Your Configuration

Add docstring and comments:

```python
def ralph_iter_custom_kan() -> RalphBaseConfig:
    """Custom KAN experiment - exploring adaptive basis functions.

    GOALS:
    - Validate KAN integration with NeuroManifold
    - Compare FasterKAN vs baseline performance
    - Target val_loss < 1.5

    Hypothesis: KAN's adaptive basis functions will improve
    convergence on character-level modeling.

    Changes from base:
    - KAN enabled with FasterKAN variant
    - Smaller model (4 layers) for fair comparison
    - Lower learning rate for KAN stability
    """
    return RalphConfigBuilder().with_overrides(
        # KAN configuration
        use_kan=True,
        kan_type='faster',
        kan_num_centers=8,

        # Model adjustments
        n_layer=4,
        learning_rate=5e-4,

        # Output
        out_dir='out-kan-experiment',
    ).build()
```

### Step 5: Add to Registry (Optional)

If this becomes a numbered iteration:

```python
# In neuromanifold_gpt/config/ralph_configs/registry.py

def ralph_iter117() -> RalphBaseConfig:
    """Your new iteration docstring..."""
    return RalphConfigBuilder().with_overrides(
        # Your deltas
    ).build()

# Register it
_RALPH_ITERATIONS[117] = ralph_iter117
```

---

## Migration Guide

### From Old System (ralph_iter*.py)

**Old way** (deprecated):
```python
# config/ralph_iter10.py
dataset = "shakespeare_char"
batch_size = 64
block_size = 128
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1
# ... 50+ more lines of redeclared variables
```

**New way** (recommended):
```python
# neuromanifold_gpt/config/ralph_configs/iterations.py
def ralph_iter10() -> RalphBaseConfig:
    """Ralph iteration 10 - balanced config."""
    return RalphConfigBuilder().with_overrides(
        # Only specify deltas from base
        batch_size=64,
        n_layer=6,
        out_dir="out-ralph-iter10",
    ).build()
```

### Migration Benefits

1. **Less Code**: ~60 lines → ~10 lines per iteration
2. **Type Safety**: Dataclass validation vs global variables
3. **Clear Deltas**: Obvious what changed from defaults
4. **No Import Pollution**: Clean namespace vs `from ralph_iter10 import *`
5. **Tooling Support**: IDE autocomplete, type checking, refactoring

### Automated Migration

The migration script converted all 73 iterations automatically:

```bash
# How the migration was performed
python scripts/migrate_ralph_configs.py \
    --output neuromanifold_gpt/config/ralph_configs/iterations.py

# Equivalence verification
pytest neuromanifold_gpt/tests/test_ralph_config_migration.py -v
```

### Manual Migration Steps

If you have custom ralph-style configs:

1. **Extract parameters** from your config file
2. **Identify deltas** by comparing to `RalphBaseConfig` defaults
3. **Create builder function** with only deltas specified
4. **Verify equivalence** by comparing outputs
5. **Archive old file** to `config/archive/`

Example:
```python
# Your old config/my_custom_ralph.py
dataset = "shakespeare_char"
batch_size = 128  # CHANGED from default 64
n_layer = 8       # CHANGED from default 6
n_embd = 512      # CHANGED from default 384
# ... etc

# New equivalent
def my_custom_config() -> RalphBaseConfig:
    return RalphConfigBuilder().with_overrides(
        batch_size=128,  # Only list changes
        n_layer=8,
        n_embd=512,
    ).build()
```

---

## API Reference

### get_ralph_config()

```python
def get_ralph_config(iteration: int) -> RalphBaseConfig:
    """Get configuration for a Ralph Loop iteration.

    Args:
        iteration: The iteration number (1-116).

    Returns:
        RalphBaseConfig instance for the specified iteration.

    Raises:
        ValueError: If iteration number is not in registry.

    Example:
        >>> config = get_ralph_config(1)
        >>> config.batch_size
        32
    """
```

### list_ralph_iterations()

```python
def list_ralph_iterations() -> list[int]:
    """List all available Ralph iteration numbers.

    Returns:
        Sorted list of iteration numbers in the registry.

    Example:
        >>> iterations = list_ralph_iterations()
        >>> len(iterations)
        73
        >>> iterations[:5]
        [1, 2, 3, 4, 5]
    """
```

### RalphConfigBuilder

```python
class RalphConfigBuilder:
    """Builder for RalphBaseConfig with delta-based overrides."""

    def __init__(self) -> None:
        """Initialize builder with empty overrides."""

    def with_overrides(self, **kwargs: Any) -> RalphConfigBuilder:
        """Add configuration overrides.

        Args:
            **kwargs: Configuration fields to override.

        Returns:
            Self for method chaining.
        """

    def build(self) -> RalphBaseConfig:
        """Build the final configuration.

        Returns:
            RalphBaseConfig instance with overrides applied.

        Raises:
            ValueError: If overrides result in invalid configuration.
        """
```

### RalphBaseConfig

See `neuromanifold_gpt/config/ralph_base.py` for the complete field reference.

Key fields:
- **dataset**: `str` - Dataset name (default: "shakespeare_char")
- **batch_size**: `int` - Training batch size (default: 64)
- **block_size**: `int` - Maximum sequence length (default: 128)
- **n_layer**: `int` - Number of transformer layers (default: 6)
- **n_head**: `int` - Number of attention heads (default: 6)
- **n_embd**: `int` - Embedding dimension (default: 384)
- **use_mhc**: `bool` - Enable mHC connections (default: False)
- **use_kan**: `bool` - Enable KAN layers (default: False)
- **use_mla**: `bool` - Enable multi-head latent attention (default: False)
- **max_iters**: `int` - Maximum training iterations (default: 1000)
- **learning_rate**: `float` - Peak learning rate (default: 2e-3)
- **out_dir**: `str` - Output directory (default: "out-ralph")

---

## Best Practices

### 1. Document Your Experiments

Always include a docstring explaining:
- **Goals**: What are you trying to achieve?
- **Hypothesis**: What do you expect to happen?
- **Changes**: What differs from the base configuration?

```python
def my_experiment() -> RalphBaseConfig:
    """Experiment: Impact of dropout on small models.

    GOALS: Determine if dropout helps or hurts tiny models
    HYPOTHESIS: Dropout may be harmful for 2-layer models

    Changes from base:
    - Minimal model (2 layers, 128 embd)
    - Sweep dropout: 0.0, 0.1, 0.2
    - Extended training (2000 iters)
    """
    return RalphConfigBuilder().with_overrides(
        n_layer=2,
        n_embd=128,
        dropout=0.1,  # Vary this in experiments
        max_iters=2000,
    ).build()
```

### 2. Use Descriptive Output Directories

Make it easy to identify experiments:

```python
# Good
out_dir="out-mhc-ablation-8layers"
out_dir="out-kan-faster-vs-wave"
out_dir="out-lr-sweep-1e3"

# Bad
out_dir="out-test1"
out_dir="out-experiment"
out_dir="out"
```

### 3. Validate Assumptions

Check that your configuration makes sense:

```python
config = RalphConfigBuilder().with_overrides(
    n_layer=12,
    n_embd=768,
    n_head=12,
).build()

# Validate head dimension
head_dim = config.n_embd // config.n_head
assert head_dim >= 32, f"Head dim {head_dim} may be too small"

# Validate memory requirements (rough estimate)
params = config.n_layer * config.n_embd ** 2 * 12
print(f"Estimated parameters: {params / 1e6:.1f}M")
```

### 4. Start from Known Good Configs

Base new experiments on validated iterations:

```python
from neuromanifold_gpt.config.ralph_configs import get_ralph_config
from dataclasses import replace

# Start from a known-good baseline
baseline = get_ralph_config(10)

# Make surgical changes
experiment = replace(
    baseline,
    use_kan=True,  # Only change we're testing
    out_dir="out-kan-on-iter10-baseline",
)
```

### 5. Version Your Experiments

Add iterations to the registry for reproducibility:

```python
# Instead of ad-hoc configs scattered everywhere,
# add them to the registry with sequential numbers

def ralph_iter117() -> RalphBaseConfig:
    """First KAN experiment (2024-01-15)"""
    return RalphConfigBuilder().with_overrides(
        use_kan=True,
        out_dir="out-ralph-iter117",
    ).build()

_RALPH_ITERATIONS[117] = ralph_iter117
```

### 6. Use Type Hints

Leverage Python's type system:

```python
from neuromanifold_gpt.config.ralph_base import RalphBaseConfig

def create_experimental_config(
    n_layers: int,
    embedding_dim: int,
    enable_mhc: bool = False
) -> RalphBaseConfig:
    """Create config with specified architecture."""
    return RalphConfigBuilder().with_overrides(
        n_layer=n_layers,
        n_embd=embedding_dim,
        use_mhc=enable_mhc,
    ).build()

# Type checker catches errors
config = create_experimental_config(
    n_layers="six",  # Type error! Should be int
    embedding_dim=384,
)
```

---

## Examples Gallery

### Example 1: Feature Ablation Study

```python
"""Ablation study: Which NeuroManifold features matter most?"""

from neuromanifold_gpt.config.ralph_builder import RalphConfigBuilder

# Baseline: No NeuroManifold features
baseline = RalphConfigBuilder().with_overrides(
    out_dir="out-ablation-baseline",
).build()

# + SDR only
with_sdr = RalphConfigBuilder().with_overrides(
    use_sdr=True,
    out_dir="out-ablation-sdr",
).build()

# + KAN only
with_kan = RalphConfigBuilder().with_overrides(
    use_kan=True,
    kan_type="faster",
    out_dir="out-ablation-kan",
).build()

# + mHC only
with_mhc = RalphConfigBuilder().with_overrides(
    use_mhc=True,
    out_dir="out-ablation-mhc",
).build()

# All features
with_all = RalphConfigBuilder().with_overrides(
    use_sdr=True,
    use_kan=True,
    use_mhc=True,
    out_dir="out-ablation-all",
).build()

ablation_configs = {
    "baseline": baseline,
    "sdr": with_sdr,
    "kan": with_kan,
    "mhc": with_mhc,
    "all": with_all,
}
```

### Example 2: Hyperparameter Grid Search

```python
"""Grid search over batch size and learning rate."""

from itertools import product
from neuromanifold_gpt.config.ralph_builder import RalphConfigBuilder

batch_sizes = [32, 64, 128]
learning_rates = [1e-4, 5e-4, 1e-3, 2e-3]

grid_configs = {}
for bs, lr in product(batch_sizes, learning_rates):
    key = f"bs{bs}_lr{lr:.0e}"
    config = RalphConfigBuilder().with_overrides(
        batch_size=bs,
        learning_rate=lr,
        out_dir=f"out-grid-{key}",
    ).build()
    grid_configs[key] = config

print(f"Created {len(grid_configs)} grid search configurations")
```

### Example 3: Model Scaling Study

```python
"""Study how model size affects performance."""

from neuromanifold_gpt.config.ralph_builder import RalphConfigBuilder

# Define scaling levels
scales = {
    "tiny": {"n_layer": 2, "n_embd": 128, "n_head": 4},
    "small": {"n_layer": 4, "n_embd": 256, "n_head": 4},
    "medium": {"n_layer": 6, "n_embd": 384, "n_head": 6},
    "large": {"n_layer": 8, "n_embd": 512, "n_head": 8},
    "xlarge": {"n_layer": 12, "n_embd": 768, "n_head": 12},
}

scaling_configs = {}
for name, params in scales.items():
    config = RalphConfigBuilder().with_overrides(
        **params,
        out_dir=f"out-scaling-{name}",
        # Adjust training time for larger models
        max_iters=1000 if name in ["tiny", "small"] else 2000,
    ).build()
    scaling_configs[name] = config
```

---

## Troubleshooting

### Issue: "ValueError: n_embd must be divisible by n_head"

**Cause**: Invalid architecture configuration.

**Solution**:
```python
# Bad
config = RalphConfigBuilder().with_overrides(
    n_embd=100,
    n_head=6,  # 100 / 6 = 16.67 (not integer)
).build()

# Good
config = RalphConfigBuilder().with_overrides(
    n_embd=384,
    n_head=6,  # 384 / 6 = 64 ✓
).build()
```

### Issue: "KeyError: Iteration X not found"

**Cause**: Requesting an iteration that doesn't exist in the registry.

**Solution**:
```python
from neuromanifold_gpt.config.ralph_configs import list_ralph_iterations

# Check available iterations
available = list_ralph_iterations()
print(f"Available: {available}")

# Use valid iteration number
if 42 in available:
    config = get_ralph_config(42)
```

### Issue: Configuration loads but training crashes with OOM

**Cause**: Config specifies more memory than available.

**Solution**:
```python
# Reduce memory footprint
config = RalphConfigBuilder().with_overrides(
    batch_size=16,      # Smaller batches
    block_size=64,      # Shorter sequences
    n_layer=4,          # Fewer layers
    n_embd=256,         # Smaller model
    gradient_accumulation_steps=4,  # Simulate larger batch
).build()
```

---

## Additional Resources

- **RalphBaseConfig Source**: `neuromanifold_gpt/config/ralph_base.py`
- **Builder Implementation**: `neuromanifold_gpt/config/ralph_builder.py`
- **Registry Source**: `neuromanifold_gpt/config/ralph_configs/registry.py`
- **Migration Script**: `scripts/migrate_ralph_configs.py`
- **Equivalence Tests**: `neuromanifold_gpt/tests/test_ralph_config_migration.py`
- **Archived Configs**: `config/archive/ralph_iterations/`
- **Configuration Reference**: `docs/configuration-reference.md`

---

## Summary

The Ralph Loop configuration system demonstrates how composition patterns can dramatically reduce code duplication while improving maintainability and type safety:

- **92% code reduction**: ~4380 lines → ~300 lines
- **Type-safe**: Dataclass validation vs untyped globals
- **DRY principle**: Single source of truth for common parameters
- **Clear deltas**: Explicit about what changed in each iteration
- **Backward compatible**: All 73 iterations preserved
- **Extensible**: Easy to add new iterations or configuration families

Whether you're reproducing past experiments, creating new configurations, or analyzing hyperparameter trends, the new system provides a clean, maintainable interface for managing Ralph Loop experimental configurations.
