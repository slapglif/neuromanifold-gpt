# Ralph Iterations Archive

## Overview

This directory contains 73 archived Ralph Loop iteration configurations (`ralph_iter1.py` through `ralph_iter116.py`) from the original experimental config system.

These files have been **archived** (not deleted) to preserve the historical record of experimental hyperparameter configurations while transitioning to the new composition-based config system.

## What Were These Files?

Each `ralph_iter*.py` file represented a single experimental configuration with:
- Specific hyperparameter settings (batch_size, n_layer, n_head, etc.)
- Training objectives and goals (e.g., "val_loss < 1.5, training_time < 100s")
- Variations on model architecture, NeuroManifold features, and FHN dynamics
- Comments documenting the experiment's purpose

The files followed a nearly identical structure with ~60+ configuration variables each, resulting in massive duplication (~4380 lines of config code).

## Why Were They Archived?

The original system violated DRY (Don't Repeat Yourself) principles:
- **Code Duplication**: Each file redeclared all 60+ config variables
- **Maintenance Burden**: Structural changes required updating dozens of files
- **Inconsistency Risk**: Easy to introduce subtle differences between configs
- **Clutter**: 73 near-identical files in the config directory

## New System

All Ralph iteration configurations have been migrated to the new composition-based system:

```python
# New location
from neuromanifold_gpt.config.ralph_configs import get_ralph_config

# Load any iteration by number
config = get_ralph_config(1)  # Equivalent to old ralph_iter1.py
config = get_ralph_config(50)  # Equivalent to old ralph_iter50.py
```

The new system:
- **Base Config**: Single source of truth in `RalphBaseConfig` dataclass
- **Builder Pattern**: Create variations using `RalphConfigBuilder` with delta overrides
- **Registry**: All 73 iterations accessible via `get_ralph_config(iteration_number)`
- **Composition**: Define configs by specifying only what differs from base

### Code Reduction

- **Before**: 73 files × ~60 lines = ~4380 lines of duplication
- **After**: ~300 lines (base + builder + registry + iterations)
- **Reduction**: ~92% less code

## Migration Details

The migration was performed by `scripts/migrate_ralph_configs.py`:
1. Parsed each `ralph_iter*.py` file using Python AST
2. Extracted configuration parameters
3. Identified deltas by comparing against base defaults
4. Generated builder-based code in `neuromanifold_gpt/config/ralph_configs/iterations.py`
5. Verified equivalence with comprehensive test suite

## Equivalence Guarantee

All 73 configurations were verified equivalent between old and new systems via:
- `neuromanifold_gpt/tests/test_ralph_config_migration.py`
- Parametrized tests comparing every config attribute
- Spot checks for representative iterations

## Usage

**Old Way** (deprecated):
```python
import sys
sys.path.append('config')
from ralph_iter1 import *  # Imports all globals
```

**New Way** (recommended):
```python
from neuromanifold_gpt.config.ralph_configs import get_ralph_config

config = get_ralph_config(1)
print(config.batch_size)  # Type-safe attribute access
```

## Documentation

For complete documentation on the new config system, see:
- `docs/ralph-config-system.md` - System overview and migration guide
- `examples/ralph_config_usage.py` - Usage examples
- `neuromanifold_gpt/config/ralph_base.py` - Base configuration dataclass
- `neuromanifold_gpt/config/ralph_builder.py` - Builder pattern implementation
- `neuromanifold_gpt/config/ralph_configs/` - Registry and iterations

## Preservation Note

These files are preserved for historical reference and reproducibility:
- They document the experimental evolution of the Ralph Loop
- They serve as a reference for understanding past experimental results
- They can be compared against the new system if verification is needed

**Do not delete these files.** They represent valuable historical data about the experimental process.

---

*Archived as part of config composition refactor (Task 059)*
*Migration Date: 2026-01-16*
