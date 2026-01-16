# Configuration Files - Legacy Directory

## ⚠️ Notice: Configuration System Migration

This directory contains **legacy configuration files** that used the old `exec(open())` pattern. These files have been replaced with a modern, type-safe configuration system.

## Migration Status

**All configuration files in this directory are archived and no longer used by the codebase.**

### What Changed

The old configuration system:
- Used `exec(open('config/file.py').read())` to load configs
- Had security vulnerabilities (arbitrary code execution)
- Lacked type safety and IDE support
- Made debugging difficult

The new configuration system:
- Uses Python dataclasses with type annotations
- Provides type-safe configuration loading
- Supports CLI overrides with validation
- Located in `neuromanifold_gpt/config/`

## Where to Find Current Configs

### Training Configuration Presets

The actively maintained configuration presets have been migrated to:

```
neuromanifold_gpt/config/presets/
```

Available presets include:
- `train_gpt2.py` - GPT-2 (124M) training configuration
- `train_shakespeare_char.py` - Character-level Shakespeare model
- `nano.py` - Nano model preset
- `small.py` - Small model preset
- `medium.py` - Medium model preset
- `reasoning.py` - Reasoning model preset
- `shakespeare_char.py` - Shakespeare character model preset

### Configuration Dataclasses

The type-safe configuration definitions are in:

```
neuromanifold_gpt/config/training.py
```

Available configurations:
- `TrainingConfig` - For `train.py`
- `SamplingConfig` - For `sample.py`
- `EvalConfig` - For `eval.py`
- `BenchConfig` - For `bench.py`

### Configuration Loader

The type-safe configuration loader is available at:

```
neuromanifold_gpt/config/loader.py
```

## Usage Examples

### Old Way (Deprecated - DO NOT USE)

```python
# ❌ Old, insecure pattern
exec(open('configurator.py').read())
for k, v in config.items():
    globals()[k] = v
```

### New Way (Current)

```python
# ✅ New, type-safe pattern
from neuromanifold_gpt.config.loader import load_config
from neuromanifold_gpt.config.training import TrainingConfig

config = load_config(TrainingConfig)
print(config.learning_rate)
print(config.batch_size)
```

## Legacy Files

All files in `config/legacy/` are preserved for historical reference only:
- `legacy/*.py` - Various training configurations (train_*, eval_*, ralph_iter*)
- `legacy/finetune/` - Fine-tuning configurations

**These files are not loaded by any active code and should not be modified.**

## For More Information

- See the implementation plan: `.auto-claude/specs/070-replace-exec-open-config-loading-with-type-safe-co/`
- Configuration module documentation: `neuromanifold_gpt/config/`
- Type-safe loader: `neuromanifold_gpt/config/loader.py`

---

*This directory was archived as part of the security and type-safety refactoring initiative to remove all `exec(open())` patterns from the codebase.*
