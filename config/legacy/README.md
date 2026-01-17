# ⚠️ DEPRECATED: Legacy Configuration Files

**STATUS:** These configuration files are **DEPRECATED** and should not be used for new projects.

---

## 🚨 Security Warning

**The files in this directory were designed for use with `exec()`-based configuration loading, which poses serious security risks:**

- ❌ **Code Execution Risk**: Using `exec()` to load these files executes arbitrary Python code
- ❌ **No Validation**: No type checking or validation of configuration values
- ❌ **Hard to Debug**: Runtime errors instead of clear validation messages
- ❌ **No IDE Support**: No autocomplete or type hints for configuration fields

**DO NOT** use `exec(open('config.py').read())` or similar patterns to load these files.

---

## ✅ Recommended Approach: Type-Safe Configuration

The new type-safe configuration system provides:

- ✅ **Security**: No code execution, just data imports
- ✅ **Type Safety**: Full IDE autocomplete and type checking
- ✅ **Validation**: Clear error messages with helpful suggestions
- ✅ **Documentation**: Self-documenting with field types and defaults

### Quick Start with New System

**Instead of this (DEPRECATED):**
```python
# OLD WAY - DO NOT USE
import sys
exec(open('config/legacy/train_shakespeare_char.py').read())
```

**Use this (RECOMMENDED):**
```python
# NEW WAY - Type-safe and secure
from neuromanifold_gpt.config.loader import load_config
from neuromanifold_gpt.config.presets import get_shakespeare_char_config

# Option 1: Use built-in preset
config = get_shakespeare_char_config()

# Option 2: Load with CLI overrides
config = load_config(TrainingConfig, ['--batch_size=64', '--learning_rate=1e-3'])

# Option 3: Override specific fields
config = get_shakespeare_char_config()
config.batch_size = 64
config.learning_rate = 1e-3
```

---

## 📚 Migration Guide

For detailed migration instructions, see:

**[Configuration System Migration Guide](../../docs/config-migration-guide.md)**

The migration guide includes:

1. **Before/After Examples** - See how to convert your config files
2. **Step-by-Step Migration** - Detailed process for updating your code
3. **Breaking Changes** - What changed and why
4. **FAQ** - Common questions and troubleshooting

---

## 🔄 Available Presets

The new system includes built-in presets that replace these legacy files:

| Legacy File | New Preset Function | Description |
|------------|-------------------|-------------|
| `train_shakespeare_char.py` | `get_shakespeare_char_config()` | Character-level Shakespeare training |
| `config/presets/nano.py` | `get_nano_config()` | Tiny model for debugging |
| `config/presets/small.py` | `get_small_config()` | Small GPT-2 equivalent |
| `config/presets/medium.py` | `get_medium_config()` | Medium GPT-2 equivalent |
| `config/presets/reasoning.py` | `get_reasoning_config()` | Reasoning-focused configuration |

### Using Presets

```python
from neuromanifold_gpt.config.presets import (
    get_nano_config,
    get_shakespeare_char_config,
    get_small_config,
    get_medium_config,
    get_reasoning_config
)

# Get a preset configuration
config = get_nano_config()

# Override specific fields as needed
config.batch_size = 32
config.learning_rate = 5e-4
config.max_iters = 1000
```

---

## 🔧 Command-Line Usage

**Instead of this (DEPRECATED):**
```bash
# OLD WAY - DO NOT USE
python train.py config/legacy/train_shakespeare_char.py --batch_size=64
```

**Use this (RECOMMENDED):**
```bash
# NEW WAY - Using preset with overrides
python train.py \
    --preset=shakespeare_char \
    --batch_size=64 \
    --learning_rate=1e-3 \
    --max_iters=5000
```

Or create a new config file using dataclasses:

```python
# my_config.py - New style config
from neuromanifold_gpt.config.base import NeuroManifoldConfig

def get_my_config() -> NeuroManifoldConfig:
    """Custom configuration for my experiment."""
    return NeuroManifoldConfig(
        # Model architecture
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.2,

        # Training
        batch_size=64,
        learning_rate=1e-3,
        max_iters=5000,

        # Data
        dataset='shakespeare_char',
        block_size=256,

        # Output
        out_dir='out-my-experiment',
        wandb_log=False,
    )
```

Then use it:
```python
from my_config import get_my_config

config = get_my_config()
# Start training with config...
```

---

## 🗓️ Deprecation Timeline

- **Current Status**: Legacy configs are deprecated but still present
- **Migration Period**: Use the migration guide to update your code
- **Future Removal**: These files may be removed in a future major version

**Action Required**: Please migrate your code to the new type-safe configuration system.

---

## ❓ Need Help?

1. **Read the Migration Guide**: [docs/config-migration-guide.md](../../docs/config-migration-guide.md)
2. **Check Examples**: See `eval.py` and `examples/train_with_component_metrics.py` for updated patterns
3. **API Documentation**: Check docstrings in `neuromanifold_gpt/config/` modules
4. **Compatibility Shim**: For gradual migration, see `neuromanifold_gpt/config/compat.py`

---

## 📋 Summary

| Feature | Legacy System ❌ | New System ✅ |
|---------|----------------|--------------|
| Loading Method | `exec()` code execution | Import dataclasses |
| Type Safety | None | Full type hints |
| IDE Support | No autocomplete | Full autocomplete |
| Validation | Runtime errors | Clear validation errors |
| Security | Executes code | Data only |
| Documentation | Comments only | Self-documenting types |

**Please migrate to the new type-safe configuration system for better security, reliability, and developer experience.**

---

*For more information, see the [Configuration System Migration Guide](../../docs/config-migration-guide.md)*
