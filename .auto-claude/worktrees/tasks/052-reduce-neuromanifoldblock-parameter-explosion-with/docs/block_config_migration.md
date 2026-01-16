# NeuroManifoldBlock Configuration Migration Guide

**Migrating from 24+ Individual Parameters to Structured Config Objects**

This guide explains the refactoring of `NeuroManifoldBlock` from a parameter-heavy initialization API to a clean, structured configuration object pattern. This migration reduces code smell, improves maintainability, and provides better documentation of related parameters.

## Table of Contents

1. [Overview](#overview)
2. [The Problem](#the-problem)
3. [The Solution](#the-solution)
4. [Migration Examples](#migration-examples)
5. [Configuration Structure](#configuration-structure)
6. [Creating Block Configs](#creating-block-configs)
7. [Sub-Configuration Reference](#sub-configuration-reference)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

---

## Overview

**What Changed:**
- **Before:** `NeuroManifoldBlock.__init__` accepted 24+ individual parameters
- **After:** `NeuroManifoldBlock.__init__` accepts a single `NeuroManifoldBlockConfig` object

**Benefits:**
- ✅ Reduced parameter explosion (24+ params → 2 params: `self`, `config`)
- ✅ Named parameter groups (FHN, KAN, mHC, MLA, MoE)
- ✅ Better defaults documentation through dataclass docstrings
- ✅ Easier testing through object construction
- ✅ Type safety with dataclass validation
- ✅ No parameter ordering bugs
- ✅ Follows modern architecture patterns (DeepSeek, Qwen)

---

## The Problem

### Before: Parameter Explosion

The original `NeuroManifoldBlock.__init__` signature violated the "too many parameters" code smell threshold (4-5 parameters):

```python
# OLD WAY - Don't use this anymore!
block = NeuroManifoldBlock(
    sdr_size=2048,
    embed_dim=384,
    manifold_dim=64,
    n_eigenvectors=32,
    n_heads=8,
    mlp_ratio=4.0,
    dropout=0.0,
    bias=False,
    # FHN parameters (9 params)
    fhn_threshold=0.5,
    fhn_tau=12.5,
    fhn_velocity=1.0,
    pulse_width_base=4,
    n_fhn_steps=2,
    use_fhn_imex=True,
    use_fhn_partitioning=True,
    use_fhn_fused=False,
    use_fhn_parallel=True,
    # KAN parameters (7 params)
    use_kan=True,
    kan_type="faster",
    kan_degree=4,
    kan_wavelet="dog",
    use_fast_wavekan=True,
    kan_num_centers=3,
    use_kan_everywhere=False,
    # mHC parameters (6 params)
    use_mhc=True,
    use_full_mhc=True,
    mhc_n_streams=2,
    mhc_residual_weight=0.9,
    mhc_sinkhorn_iters=5,
    mhc_sinkhorn_tau=0.05,
    # MLA parameters (3 params)
    use_mla=False,
    mla_latent_dim=64,
    mla_rope_dim=32,
    # MoE parameters (5 params)
    use_moe=False,
    moe_n_experts=8,
    moe_n_active=2,
    use_shared_expert=True,
    use_e7_routing=False,
    # ... and more
)
```

**Issues:**
- 🔴 Hard to read and understand
- 🔴 Easy to make parameter ordering mistakes
- 🔴 No clear grouping of related parameters
- 🔴 Difficult to provide defaults and documentation
- 🔴 Hard to test different configurations

---

## The Solution

### After: Structured Configuration

```python
# NEW WAY - Use this!
from neuromanifold_gpt.config.block_config import NeuroManifoldBlockConfig

# Use defaults for everything
config = NeuroManifoldBlockConfig(
    sdr_size=2048,
    embed_dim=384,
)
block = NeuroManifoldBlock(config)

# Or customize specific sub-configs
from neuromanifold_gpt.config.block_config import FHNConfig, KANConfig

config = NeuroManifoldBlockConfig(
    sdr_size=2048,
    embed_dim=384,
    fhn=FHNConfig(fhn_threshold=0.3, fhn_tau=15.0),
    kan=KANConfig(kan_type="wave", use_kan_everywhere=True),
)
block = NeuroManifoldBlock(config)
```

---

## Migration Examples

### Example 1: Basic Block Creation

**Before:**
```python
block = NeuroManifoldBlock(
    sdr_size=2048,
    embed_dim=384,
    n_heads=8,
    # ... 20+ more parameters
)
```

**After:**
```python
from neuromanifold_gpt.config.block_config import NeuroManifoldBlockConfig

config = NeuroManifoldBlockConfig(
    sdr_size=2048,
    embed_dim=384,
    n_heads=8,
)
block = NeuroManifoldBlock(config)
```

### Example 2: Customizing FHN Dynamics

**Before:**
```python
block = NeuroManifoldBlock(
    sdr_size=2048,
    embed_dim=384,
    fhn_threshold=0.3,
    fhn_tau=15.0,
    use_fhn_parallel=False,
    # ... all other parameters
)
```

**After:**
```python
from neuromanifold_gpt.config.block_config import NeuroManifoldBlockConfig, FHNConfig

config = NeuroManifoldBlockConfig(
    sdr_size=2048,
    embed_dim=384,
    fhn=FHNConfig(
        fhn_threshold=0.3,
        fhn_tau=15.0,
        use_fhn_parallel=False,
    ),
)
block = NeuroManifoldBlock(config)
```

### Example 3: Creating from Model Config

**Before:**
```python
# In NeuroManifoldGPT.make_block()
block = NeuroManifoldBlock(
    sdr_size=self.config.sdr_size,
    embed_dim=self.config.n_embd,
    manifold_dim=self.config.manifold_dim,
    n_eigenvectors=self.config.n_eigenvectors,
    n_heads=self.config.n_heads,
    dropout=self.config.dropout,
    bias=self.config.bias,
    fhn_threshold=self.config.fhn_threshold,
    fhn_tau=self.config.fhn_tau,
    # ... 20+ more parameter mappings
)
```

**After:**
```python
# In NeuroManifoldGPT.make_block()
from neuromanifold_gpt.config.block_config import NeuroManifoldBlockConfig

block_config = NeuroManifoldBlockConfig.from_model_config(self.config, layer_idx)
block = NeuroManifoldBlock(block_config)
```

### Example 4: Testing Different Configurations

**Before:**
```python
# Test with MoE enabled
block_moe = NeuroManifoldBlock(
    sdr_size=2048,
    embed_dim=384,
    # ... 20+ parameters
    use_moe=True,
    moe_n_experts=16,
    moe_n_active=4,
)

# Test with KAN disabled
block_no_kan = NeuroManifoldBlock(
    sdr_size=2048,
    embed_dim=384,
    # ... 20+ parameters (all repeated)
    use_kan=False,
)
```

**After:**
```python
from neuromanifold_gpt.config.block_config import (
    NeuroManifoldBlockConfig,
    MoEConfig,
    KANConfig,
)

# Test with MoE enabled
config_moe = NeuroManifoldBlockConfig(
    sdr_size=2048,
    embed_dim=384,
    moe=MoEConfig(use_moe=True, moe_n_experts=16, moe_n_active=4),
)
block_moe = NeuroManifoldBlock(config_moe)

# Test with KAN disabled
config_no_kan = NeuroManifoldBlockConfig(
    sdr_size=2048,
    embed_dim=384,
    kan=KANConfig(use_kan=False),
)
block_no_kan = NeuroManifoldBlock(config_no_kan)
```

---

## Configuration Structure

The `NeuroManifoldBlockConfig` composes five specialized sub-configurations:

```
NeuroManifoldBlockConfig
├── Core Parameters (sdr_size, embed_dim, n_heads, etc.)
├── FHNConfig       → FitzHugh-Nagumo dynamics (9 parameters)
├── KANConfig       → Kolmogorov-Arnold Networks (7 parameters)
├── MHCConfig       → Manifold-constrained hyper-connections (6 parameters)
├── MLAConfig       → Multi-head latent attention (3 parameters)
└── MoEConfig       → Mixture of Experts (5 parameters)
```

### Core Parameters

Located directly in `NeuroManifoldBlockConfig`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sdr_size` | int | 2048 | Size of SDR binary vectors |
| `embed_dim` | int | 384 | Embedding dimension (same as n_embd) |
| `manifold_dim` | int | 64 | Dimension of learned manifold space |
| `n_eigenvectors` | int | 32 | Number of eigenvectors for spectral attention |
| `n_heads` | int | 8 | Number of attention heads |
| `mlp_ratio` | float | 4.0 | MLP hidden dimension ratio |
| `dropout` | float | 0.0 | Dropout probability |
| `bias` | bool | False | Whether to use bias in linear layers |
| `skip_manifold_spectral` | bool | False | Skip manifold/spectral for speed |
| `use_knot_attention` | bool | False | Enable knot-theoretic attention |
| `use_kaufmann_attention` | bool | False | Enable Kaufmann trifecta attention |

---

## Creating Block Configs

### Method 1: Direct Construction with Defaults

```python
from neuromanifold_gpt.config.block_config import NeuroManifoldBlockConfig

# All sub-configs use their defaults
config = NeuroManifoldBlockConfig(
    sdr_size=2048,
    embed_dim=384,
    n_heads=8,
)
```

### Method 2: Customizing Sub-Configs

```python
from neuromanifold_gpt.config.block_config import (
    NeuroManifoldBlockConfig,
    FHNConfig,
    KANConfig,
)

config = NeuroManifoldBlockConfig(
    sdr_size=2048,
    embed_dim=384,
    n_heads=8,
    fhn=FHNConfig(
        fhn_threshold=0.3,
        fhn_tau=15.0,
    ),
    kan=KANConfig(
        kan_type="wave",
        kan_wavelet="mexican_hat",
    ),
)
```

### Method 3: From Model Config (Recommended)

```python
from neuromanifold_gpt.config import NeuroManifoldConfig
from neuromanifold_gpt.config.block_config import NeuroManifoldBlockConfig

# Create model-level config
model_config = NeuroManifoldConfig(
    vocab_size=50304,
    n_layer=12,
    n_heads=8,
    n_embd=768,
)

# Extract block config for a specific layer
block_config = NeuroManifoldBlockConfig.from_model_config(
    model_config,
    layer_idx=0
)
```

---

## Sub-Configuration Reference

### FHNConfig - FitzHugh-Nagumo Dynamics

Controls soliton wave propagation in attention mechanism.

```python
from neuromanifold_gpt.config.block_config import FHNConfig

fhn = FHNConfig(
    fhn_threshold=0.5,       # Firing threshold for activation
    fhn_tau=12.5,            # Time constant for recovery dynamics
    fhn_velocity=1.0,        # Propagation velocity of waves
    pulse_width_base=4,      # Base width of soliton pulses
    n_fhn_steps=2,           # Integration steps (IMEX scheme)
    use_fhn_imex=True,       # Use semi-implicit IMEX scheme
    use_fhn_partitioning=True,  # Enable energy balancing
    use_fhn_fused=False,     # Use fused kernel (disabled)
    use_fhn_parallel=True,   # Use FFT-based parallel scan
)
```

**Key Parameters:**
- **fhn_threshold**: Lower values → more neurons fire → more attention propagation
- **fhn_tau**: Higher values → slower recovery → longer memory of past states
- **use_fhn_parallel**: Enable for maximum speed via FFT (recommended: True)

### KANConfig - Kolmogorov-Arnold Networks

Controls basis functions for FFN layers.

```python
from neuromanifold_gpt.config.block_config import KANConfig

kan = KANConfig(
    use_kan=True,            # Enable KAN for FFN
    kan_type="faster",       # "faster" (RSWAF), "wave", or "cheby"
    kan_degree=4,            # Polynomial degree (ChebyKAN only)
    kan_wavelet="dog",       # Wavelet type (WaveKAN only)
    use_fast_wavekan=True,   # Efficient WaveKAN variant
    kan_num_centers=3,       # RSWAF basis centers
    use_kan_everywhere=False,  # Replace ALL Linear with KAN
)
```

**Key Parameters:**
- **kan_type**:
  - `"faster"` (RSWAF) → Best speed/stability (recommended)
  - `"wave"` → Wavelet basis for frequency decomposition
  - `"cheby"` → Chebyshev polynomials for smooth functions
- **use_kan_everywhere**: ⚠️ WARNING: Causes parameter explosion. Only enable for experiments.

### MHCConfig - Manifold-Constrained Hyper-Connections

Controls DeepSeek-style residual routing for training stability.

```python
from neuromanifold_gpt.config.block_config import MHCConfig

mhc = MHCConfig(
    use_mhc=True,              # Enable mHC
    use_full_mhc=True,         # Use full multi-stream variant
    mhc_n_streams=2,           # Number of parallel streams
    mhc_residual_weight=0.9,   # Identity mapping bias
    mhc_sinkhorn_iters=5,      # Sinkhorn iterations
    mhc_sinkhorn_tau=0.05,     # Sinkhorn temperature
)
```

**Key Parameters:**
- **use_full_mhc**: Full version has parallel streams with learned routing (recommended: True)
- **mhc_residual_weight**: Higher values preserve identity mapping (stable initialization)
- **mhc_sinkhorn_iters**: 3-5 iterations sufficient for doubly-stochastic convergence

### MLAConfig - Multi-Head Latent Attention

Controls KV cache compression for memory efficiency (optional).

```python
from neuromanifold_gpt.config.block_config import MLAConfig

mla = MLAConfig(
    use_mla=False,          # Enable MLA (off by default)
    mla_latent_dim=64,      # KV compression dimension
    mla_rope_dim=32,        # Decoupled RoPE dimension
)
```

**Key Parameters:**
- **use_mla**: Enable for ~8x KV cache memory reduction (useful for long context)
- **mla_latent_dim**: Lower values → more compression → less capacity (typical: embed_dim / 4-8)

### MoEConfig - Mixture of Experts

Controls conditional computation with expert routing (optional).

```python
from neuromanifold_gpt.config.block_config import MoEConfig

moe = MoEConfig(
    use_moe=False,            # Enable MoE (off by default)
    moe_n_experts=8,          # Total number of experts
    moe_n_active=2,           # Active experts per token
    use_shared_expert=True,   # Always-active shared expert
    use_e7_routing=False,     # E7 curriculum routing
)
```

**Key Parameters:**
- **use_moe**: Enable for model scaling with sparse activation (⚠️ increases parameters)
- **moe_n_active**: Lower values → less compute per token (typical: 2 for efficiency)
- **use_shared_expert**: Recommended for stability (prevents expert collapse)

---

## Best Practices

### ✅ DO: Use `from_model_config()` for Production

```python
# Recommended: Ensures consistency with model config
block_config = NeuroManifoldBlockConfig.from_model_config(model_config, layer_idx)
block = NeuroManifoldBlock(block_config)
```

### ✅ DO: Use Named Sub-Configs for Clarity

```python
# Clear which parameters belong to which component
config = NeuroManifoldBlockConfig(
    sdr_size=2048,
    embed_dim=384,
    fhn=FHNConfig(fhn_threshold=0.3),
    kan=KANConfig(kan_type="wave"),
)
```

### ✅ DO: Leverage Dataclass Defaults

```python
# Only specify what you need to change
config = NeuroManifoldBlockConfig(
    sdr_size=2048,
    embed_dim=384,
    # All sub-configs use sensible defaults
)
```

### ❌ DON'T: Pass Individual Parameters Anymore

```python
# OLD - This no longer works!
block = NeuroManifoldBlock(
    sdr_size=2048,
    embed_dim=384,
    fhn_threshold=0.5,
    # ...
)
```

### ❌ DON'T: Enable `use_kan_everywhere` Without Good Reason

```python
# This causes parameter explosion - only for experiments
kan = KANConfig(use_kan_everywhere=True)  # ⚠️ WARNING
```

### ❌ DON'T: Enable Both MLA and MoE Without Testing

```python
# Both add significant complexity - test independently first
config = NeuroManifoldBlockConfig(
    sdr_size=2048,
    embed_dim=384,
    mla=MLAConfig(use_mla=True),  # Memory optimization
    moe=MoEConfig(use_moe=True),  # Conditional computation
    # ⚠️ Test individually before combining
)
```

---

## Troubleshooting

### Issue: `TypeError: __init__() got an unexpected keyword argument`

**Cause:** Trying to pass individual parameters instead of config object.

**Solution:**
```python
# Wrong
block = NeuroManifoldBlock(sdr_size=2048, embed_dim=384)

# Correct
config = NeuroManifoldBlockConfig(sdr_size=2048, embed_dim=384)
block = NeuroManifoldBlock(config)
```

### Issue: `AttributeError: 'NeuroManifoldBlockConfig' object has no attribute 'xxx'`

**Cause:** Trying to access a parameter that doesn't exist in the config.

**Solution:** Check if the parameter belongs to a sub-config:
```python
# Wrong
threshold = config.fhn_threshold

# Correct
threshold = config.fhn.fhn_threshold
```

### Issue: How do I override a single FHN parameter?

**Solution:** Create a custom FHNConfig and pass it to the block config:
```python
from neuromanifold_gpt.config.block_config import NeuroManifoldBlockConfig, FHNConfig

config = NeuroManifoldBlockConfig(
    sdr_size=2048,
    embed_dim=384,
    fhn=FHNConfig(fhn_threshold=0.3),  # Override just this parameter
)
```

### Issue: How do I replicate old behavior exactly?

**Solution:** Use `from_model_config()` which preserves the exact mapping:
```python
# This creates the same config as the old parameter-passing approach
block_config = NeuroManifoldBlockConfig.from_model_config(model_config, layer_idx)
```

---

## Summary

The migration to structured configuration objects provides:

1. **Reduced Complexity**: 24+ parameters → 2 parameters (`self`, `config`)
2. **Better Organization**: Related parameters grouped into semantic sub-configs
3. **Improved Documentation**: Dataclass docstrings document defaults and relationships
4. **Type Safety**: Dataclass validation catches configuration errors early
5. **Easier Testing**: Object-based configs are easier to construct and modify
6. **Modern Patterns**: Follows DeepSeek/Qwen configuration architecture

For questions or issues, please refer to:
- **Config Source**: `neuromanifold_gpt/config/block_config.py`
- **Block Source**: `neuromanifold_gpt/model/block.py`
- **Model Integration**: `neuromanifold_gpt/model/gpt.py` (see `make_block()`)
