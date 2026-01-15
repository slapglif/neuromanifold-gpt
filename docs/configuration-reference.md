# NeuroManifold Configuration Reference

**A Comprehensive Guide to Model Configuration**

This guide provides detailed documentation for all 100+ parameters in the `NeuroManifoldConfig` dataclass, explaining their purpose, valid ranges, interdependencies, and tuning recommendations.

## Overview

NeuroManifold is a novel transformer architecture that combines:

- **SDR (Sparse Distributed Representation)** - Biological plausibility through sparse encoding
- **Manifold Projection** - Learned Riemannian geometry with E7 subgroup chains
- **FHN Dynamics** - Soliton-based attention via FitzHugh-Nagumo neural dynamics
- **mHC (Manifold-Constrained Hyper-Connections)** - DeepSeek-style stability improvements
- **System 2 Reasoning** - DAG planning, hierarchical memory, and imagination modules
- **Advanced Architectures** - MLA (latent attention), MTP (multi-token prediction), MoE (mixture of experts)

Configuration is the primary interface for experimentation. This guide helps you:

1. **Understand** what each parameter controls
2. **Validate** parameter combinations and interdependencies
3. **Tune** settings for your specific use case
4. **Optimize** performance vs accuracy tradeoffs
5. **Debug** common configuration issues

## Table of Contents

1. [Core Architecture Parameters](#1-core-architecture-parameters)
   - Vocabulary and Sequence
   - Model Dimensions
   - Layer Configuration

2. [SDR Encoding Parameters](#2-sdr-encoding-parameters)
   - Sparse Distributed Representations
   - Biological Plausibility
   - Memory Requirements

3. [Manifold and Spectral Parameters](#3-manifold-and-spectral-parameters)
   - Manifold Projection
   - Spectral Decomposition
   - Multi-scale E7 Subgroup Chains

4. [FHN Dynamics Parameters](#4-fhn-dynamics-parameters)
   - FitzHugh-Nagumo Equations
   - Soliton Propagation
   - Numerical Integration Methods
   - Optimization Techniques

5. [Attention Variants](#5-attention-variants)
   - Standard Attention
   - Knot-Theoretic Attention
   - Kaufmann Trifecta Attention
   - QK Normalization

6. [mHC (Manifold-Constrained Hyper-Connections)](#6-mhc-manifold-constrained-hyper-connections)
   - DeepSeek Architecture
   - Sinkhorn-Knopp Normalization
   - Multi-stream Configuration

7. [KAN (Kolmogorov-Arnold Networks)](#7-kan-kolmogorov-arnold-networks)
   - KAN Types: FasterKAN, WaveKAN, ChebyKAN
   - RSWAF Basis Functions
   - Parameter Bloat Considerations

8. [Advanced Architectures](#8-advanced-architectures)
   - MLA (Multi-Head Latent Attention)
   - MTP (Multi-Token Prediction)
   - MoE (Mixture of Experts)

9. [System 2 Reasoning Components](#9-system-2-reasoning-components)
   - Hybrid Reasoning Modes
   - DAG Planning
   - Hierarchical Memory
   - Imagination Module

10. [Training and Optimization](#10-training-and-optimization)
    - Learning Rate Configuration
    - Optimizer Settings (AdamW)
    - LR Schedules: WSD vs Cosine
    - Gradient Clipping
    - Numerical Stability

11. [Fast Mode Optimizations](#11-fast-mode-optimizations)
    - Performance vs Accuracy Tradeoffs
    - Skip Flags
    - Memory Optimizations

12. [Parameter Interdependencies](#12-parameter-interdependencies)
    - Critical Constraints
    - Validation Rules
    - Common Pitfalls

13. [Common Presets](#13-common-presets)
    - Nano (Testing/Development)
    - Small (Experimentation)
    - Medium (Production)
    - Custom Configurations

14. [Tuning Guide and Best Practices](#14-tuning-guide-and-best-practices)
    - Performance Optimization
    - Accuracy Tuning
    - Memory Management
    - Debugging Common Issues

---

## Quick Start

For impatient experimenters, here are the most common configuration patterns:

```python
# Default configuration (balanced)
from neuromanifold_gpt.config import NeuroManifoldConfig
config = NeuroManifoldConfig()

# Nano preset (fast iteration)
from neuromanifold_gpt.config import NeuroManifoldConfigNano
config = NeuroManifoldConfigNano()

# Custom configuration
config = NeuroManifoldConfig(
    n_layer=12,           # More layers
    n_embd=768,           # Larger model
    use_mhc=True,         # Stability
    use_mtp=True,         # Better representations
    fast_mode=False       # Full features
)
```

---

## 1. Core Architecture Parameters

**Category:** Foundation parameters that define model size and capacity.

### vocab_size
- **Type:** `int`
- **Default:** `50304`
- **Range:** Typically 32000-151936 (Qwen3)
- **Description:** Size of the vocabulary (number of unique tokens)
- **Details:**
  - Default 50304 is GPT-2 vocabulary (50257) padded to multiple of 64 for efficiency
  - Must match your tokenizer's vocabulary
  - Larger vocabularies (151K+) require `label_smoothing > 0` and `lm_head_fp32=True`
- **Tuning Tips:**
  - Use multiples of 64/128 for optimal GPU memory alignment
  - Large vocabularies (100K+) increase memory and may require label smoothing

### block_size
- **Type:** `int`
- **Default:** `1024`
- **Range:** 128-32768 (limited by memory)
- **Description:** Maximum sequence length (context window)
- **Details:**
  - Determines the maximum number of tokens the model can process at once
  - Memory usage grows quadratically with block_size in standard attention
  - Affects positional encoding and attention patterns
- **Interdependencies:**
  - `sdr_context_size` should be ≤ `block_size`
  - Larger `block_size` requires more VRAM
- **Tuning Tips:**
  - Start with 256-512 for experimentation
  - Use 1024-2048 for production
  - Enable `use_mla=True` for efficient long-context handling

### n_layer
- **Type:** `int`
- **Default:** `6`
- **Range:** 2-48 (typical: 4-24)
- **Description:** Number of transformer layers (depth)
- **Details:**
  - Each layer adds a residual block of: Attention → LayerNorm → FFN → LayerNorm
  - More layers = more capacity but slower training
  - DeepSeek uses 60+ layers, GPT-3 uses 96
- **Interdependencies:**
  - With `use_mhc=True`, mHC connections route across all layers
  - `n_thinking_layers` adds extra layers for reasoning mode
- **Tuning Tips:**
  - Nano: 4-6 layers
  - Small: 8-12 layers
  - Medium: 12-24 layers
  - Diminishing returns beyond 24 layers without architectural changes

### n_embd
- **Type:** `int`
- **Default:** `384`
- **Range:** 64-4096 (must be divisible by `n_heads`)
- **Description:** Embedding dimension (model width)
- **Details:**
  - Dimension of token embeddings and hidden states
  - Must be divisible by `n_heads` for multi-head attention
  - Affects parameter count quadratically (weight matrices are n_embd × n_embd)
- **Interdependencies:**
  - **CRITICAL:** `n_embd % n_heads == 0` (validated in `__post_init__`)
  - `head_dim = n_embd // n_heads`
  - Larger `n_embd` requires proportionally larger `manifold_dim` and `sdr_embed_dim`
- **Tuning Tips:**
  - Common values: 128 (nano), 384 (small), 768 (medium), 1024 (large)
  - Use multiples of 128 for optimal performance
  - Balance with `n_layer`: wide shallow vs narrow deep

### n_heads
- **Type:** `int`
- **Default:** `8`
- **Range:** 1-32 (typical: 4-16)
- **Description:** Number of attention heads
- **Details:**
  - Multi-head attention splits queries/keys/values across parallel heads
  - Each head has dimension `head_dim = n_embd // n_heads`
  - More heads = more diverse attention patterns but higher overhead
- **Interdependencies:**
  - **CRITICAL:** `n_embd` must be divisible by `n_heads`
  - `mhc_n_streams` is independent (default 2)
- **Tuning Tips:**
  - Keep `head_dim` in range 32-128 for best performance
  - Typical: 4 heads (nano), 8 heads (small/medium), 12-16 heads (large)
  - More heads ≠ always better; diminishing returns beyond 16

### dropout
- **Type:** `float`
- **Default:** `0.0`
- **Range:** 0.0-0.5
- **Description:** Dropout probability for regularization
- **Details:**
  - Randomly zeros activations during training to prevent overfitting
  - Modern best practices often use 0.0 dropout (rely on weight decay instead)
  - Can be selectively applied to attention, residual, or embedding layers
- **Tuning Tips:**
  - Start with 0.0 (current best practice)
  - Use 0.1-0.2 if severe overfitting occurs
  - DeepSeek/MiniMax use 0.0 dropout

### bias
- **Type:** `bool`
- **Default:** `False`
- **Description:** Whether to use bias terms in linear layers
- **Details:**
  - Adds learnable bias vectors to all linear transformations
  - Slightly increases parameter count
  - Modern architectures (GPT-NeoX, PaLM) often disable bias
- **Tuning Tips:**
  - `False` is standard for modern models (saves parameters)
  - `True` may help if training instability occurs
  - Minimal impact on performance either way

---

## 2. SDR Encoding Parameters

**Category:** Sparse Distributed Representation for biological plausibility and memory efficiency.

*(This section will be expanded in subsequent subtasks)*

---

## 3. Manifold and Spectral Parameters

**Category:** Learned Riemannian geometry and spectral decomposition.

*(This section will be expanded in subsequent subtasks)*

---

## 4. FHN Dynamics Parameters

**Category:** FitzHugh-Nagumo neural dynamics for soliton-based attention.

*(This section will be expanded in subsequent subtasks)*

---

## 5. Attention Variants

**Category:** Different attention mechanisms and normalizations.

*(This section will be expanded in subsequent subtasks)*

---

## 6. mHC (Manifold-Constrained Hyper-Connections)

**Category:** DeepSeek-style architecture for training stability.

*(This section will be expanded in subsequent subtasks)*

---

## 7. KAN (Kolmogorov-Arnold Networks)

**Category:** Learnable activation functions replacing standard MLPs.

*(This section will be expanded in subsequent subtasks)*

---

## 8. Advanced Architectures

**Category:** MLA, MTP, and MoE configurations.

*(This section will be expanded in subsequent subtasks)*

---

## 9. System 2 Reasoning Components

**Category:** Deliberative reasoning, planning, and imagination.

*(This section will be expanded in subsequent subtasks)*

---

## 10. Training and Optimization

**Category:** Learning rate, optimizer, and training configuration.

*(This section will be expanded in subsequent subtasks)*

---

## 11. Fast Mode Optimizations

**Category:** Performance vs accuracy tradeoffs.

*(This section will be expanded in subsequent subtasks)*

---

## 12. Parameter Interdependencies

**Category:** Critical constraints and validation rules.

*(This section will be expanded in subsequent subtasks)*

---

## 13. Common Presets

**Category:** Pre-configured setups for different use cases.

*(This section will be expanded in subsequent subtasks)*

---

## 14. Tuning Guide and Best Practices

**Category:** Practical recommendations for experimentation.

*(This section will be expanded in subsequent subtasks)*

---

## Appendix: Mathematical Foundations

### The Kaufmann Trifecta

The NeuroManifold architecture is grounded in the "Kaufmann Trifecta" - a unified theory combining:

1. **Konrad Kaufmann (Thermodynamics):** Soliton propagation in FHN dynamics
2. **Stuart Kauffman (Complexity):** Fitness landscapes and the Adjacent Possible
3. **Louis Kauffman (Topology):** Knot-theoretic semantic relationships

See `neuromanifold_gpt/research/kaufmann_attention.md` for detailed theoretical background.

---

**Last Updated:** 2026-01-15
**Version:** 1.0
**Maintainer:** NeuroManifold Team
