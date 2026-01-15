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

SDR (Sparse Distributed Representation) encoding is inspired by biological neural systems where information is represented by sparse activation patterns. NeuroManifold uses **Semantic Folding** to transform tokens into high-dimensional binary vectors with ~2% active bits, similar to cortical columns in the mammalian brain.

**Key Benefits:**
- **Biological Plausibility:** Mimics sparse coding in cortex (~2% neuron activation)
- **Semantic Similarity:** Overlap between SDR vectors measures semantic distance
- **Robustness:** Noise-tolerant through distributed redundancy
- **Memory Efficiency:** Enables infinite-context memory via SDR engram storage

**Tradeoff:** SDR encoding adds computational overhead. For maximum speed, use `use_sdr=False` (dense embeddings).

### use_sdr
- **Type:** `bool`
- **Default:** `False`
- **Description:** Enable/disable Sparse Distributed Representation encoding
- **Details:**
  - When `True`: Tokens are encoded as sparse binary vectors via Semantic Folding
  - When `False`: Standard dense embeddings (faster, less memory, standard transformer behavior)
  - SDR mode enables biological plausibility and infinite-context memory via engrams
  - Dense mode is recommended for faster iteration and baseline comparisons
- **Interdependencies:**
  - When `True`, all `sdr_*` parameters become active
  - Affects memory module behavior (SDR engrams vs dense embeddings)
  - SDR mode requires more compute for encoding/decoding
- **Tuning Tips:**
  - Start with `False` for fast prototyping
  - Enable `True` when experimenting with biological plausibility or infinite context
  - SDR provides semantic similarity matching for memory retrieval

### sdr_size
- **Type:** `int`
- **Default:** `2048`
- **Range:** 512-16384 (typical: 1024-4096)
- **Description:** Size of SDR binary vectors (total number of bits)
- **Details:**
  - Dimensionality of the high-dimensional binary space
  - Larger size = higher capacity for distinct representations
  - Must be large enough to support sparse activation with low collision probability
  - 2048 bits is inspired by minicolumn counts in cortical regions
- **Interdependencies:**
  - `sdr_n_active = int(sdr_size * sdr_sparsity)` (computed automatically)
  - Larger `sdr_size` improves semantic resolution but increases memory
  - Should be >> `sdr_n_active` for true sparsity
- **Tuning Tips:**
  - Use 1024 for small vocabularies or experimentation
  - Use 2048-4096 for production (balances capacity and memory)
  - Increase if seeing high SDR collision rates (similar tokens get identical SDRs)

### sdr_sparsity
- **Type:** `float`
- **Default:** `0.02` (2%)
- **Range:** 0.01-0.05 (typical: 0.015-0.03)
- **Description:** Target sparsity ratio (fraction of active bits in SDR)
- **Details:**
  - Determines how many bits are "on" in each SDR vector
  - 0.02 = 2% sparsity, matching biological cortex activation levels
  - Lower sparsity = more selective representations (harder to match)
  - Higher sparsity = more distributed representations (easier overlap)
  - Affects semantic similarity matching and memory retrieval
- **Interdependencies:**
  - `sdr_n_active = int(sdr_size * sdr_sparsity)`
  - Example: `2048 * 0.02 = 40` active bits
  - Must be < 0.1 for meaningful sparse coding
- **Tuning Tips:**
  - 0.02 is biologically plausible (start here)
  - Increase to 0.03-0.04 if too few memory matches
  - Decrease to 0.015-0.02 if too many false positive matches
  - Very sparse (0.01) = high selectivity but may miss semantic neighbors

### sdr_n_active
- **Type:** `int`
- **Default:** Computed as `int(sdr_size * sdr_sparsity)` (≈41 for defaults)
- **Range:** Auto-computed (do not set manually)
- **Description:** Number of active bits in each SDR (computed field)
- **Details:**
  - Automatically calculated in `__post_init__` from `sdr_size` and `sdr_sparsity`
  - Determines the actual sparsity of SDR vectors
  - Used by semantic folding encoder to select which bits to activate
  - Affects collision probability and semantic resolution
- **Interdependencies:**
  - Read-only field (computed from `sdr_size` and `sdr_sparsity`)
  - Critical for memory overlap calculations (`engram_threshold`)
- **Tuning Tips:**
  - Tune via `sdr_sparsity`, not directly
  - Typical values: 20-80 active bits
  - Higher `sdr_n_active` = more robust but less selective

### sdr_embed_dim
- **Type:** `int`
- **Default:** `256`
- **Range:** 64-512 (should be ≤ `n_embd`)
- **Description:** Embedding dimension for SDR projection into continuous space
- **Details:**
  - SDR vectors are binary; this projects them into continuous embeddings
  - Acts as a bottleneck between sparse binary space and transformer embeddings
  - Smaller values = stronger compression (may lose semantic information)
  - Larger values = better preservation but higher memory
- **Interdependencies:**
  - Should be ≤ `n_embd` (typically `n_embd // 2` or `n_embd // 1.5`)
  - Affects the SDR encoder/decoder module size
  - Larger `sdr_embed_dim` closer to `n_embd` reduces information loss
- **Tuning Tips:**
  - Use `n_embd // 2` as a starting point (e.g., 192 for n_embd=384)
  - Use `n_embd` for maximum information preservation (no bottleneck)
  - Reduce to `n_embd // 4` if memory is tight

### sdr_context_size
- **Type:** `int`
- **Default:** `5`
- **Range:** 1-32 (typical: 3-10)
- **Description:** Context window size for semantic folding encoder
- **Details:**
  - Number of neighboring tokens to consider when encoding SDR
  - Semantic Folding uses local context to disambiguate word meanings
  - Larger context = better disambiguation but more computation
  - Similar to n-gram context in NLP (but with semantic overlap)
- **Interdependencies:**
  - Must be ≤ `block_size` (sequence length)
  - Larger values increase SDR encoding time
  - Affects the local attention mechanism in SDR encoder
- **Tuning Tips:**
  - Use 3-5 for fast encoding (bigram/trigram context)
  - Use 7-10 for better semantic disambiguation
  - Diminishing returns beyond 10
  - Set to 1 to disable context (pure word2SDR mapping)

### SDR Memory and Overlap Threshold

When using SDR encoding, the following memory parameters become relevant:

- **`engram_threshold`**: Minimum SDR overlap (0.0-1.0) for memory retrieval
  - Default: 0.3 (30% bit overlap)
  - Higher = stricter matching (only very similar SDRs retrieved)
  - Lower = looser matching (more memory neighbors retrieved)
  - With `sdr_sparsity=0.02` and `sdr_n_active=41`, 30% overlap = ~12 matching bits

**Example Configuration:**

```python
# Enable SDR mode with biological parameters
config = NeuroManifoldConfig(
    use_sdr=True,           # Enable sparse encoding
    sdr_size=2048,          # Cortical minicolumn count
    sdr_sparsity=0.02,      # 2% activation (biological)
    sdr_embed_dim=384,      # Match n_embd for no bottleneck
    sdr_context_size=7,     # 7-token semantic context
    engram_threshold=0.3,   # 30% overlap for memory retrieval
)
```

**Fast Dense Mode (No SDR):**

```python
# Disable SDR for maximum speed (standard transformer)
config = NeuroManifoldConfig(
    use_sdr=False,          # Dense embeddings (default)
    # All sdr_* parameters ignored
)
```

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
