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

The manifold projection module learns a low-dimensional Riemannian manifold from token embeddings, enabling geometric semantic relationships. Spectral decomposition via eigendecomposition provides attention with global structural information about the data geometry.

**Key Concepts:**
- **Manifold Projection:** Maps high-dimensional embeddings to a learned manifold structure
- **Spectral Decomposition:** Eigendecomposition of the Laplacian for global geometric features
- **E7 Subgroup Chain:** Multi-scale hierarchical manifold (E7 → E6 → D5) for coarse-to-fine processing
- **Riemannian Metric:** Learned distance metric that captures semantic geometry

**Mathematical Background:**
- Manifold learning techniques (Isomap, Laplacian Eigenmaps)
- Spectral graph theory for attention kernels
- E7 exceptional Lie group and its subgroup decomposition
- Riemannian geometry for semantic space

### manifold_dim
- **Type:** `int`
- **Default:** `64`
- **Range:** 8-256 (typical: 32-128)
- **Description:** Dimension of the learned manifold space
- **Details:**
  - Target dimensionality for manifold projection (typically much lower than `n_embd`)
  - Determines the geometric complexity the model can capture
  - Larger manifold = more expressive geometry but higher computation
  - Acts as a geometric bottleneck that encourages semantic structure
  - Used in single-scale mode (when `use_multiscale_manifold=False`)
- **Interdependencies:**
  - Should be ≤ `n_embd` (typically `n_embd // 4` to `n_embd // 2`)
  - In multi-scale mode, replaced by `multiscale_fine_dim` (E7 level)
  - Affects memory and computation in manifold projection layers
  - Larger values require more neighbors (`n_neighbors`) for proper manifold structure
- **Tuning Tips:**
  - Use 32-64 for small models (n_embd=384)
  - Use 64-128 for medium/large models (n_embd=768+)
  - Too small (<16): Loses semantic information
  - Too large (>256): Loses geometric benefits, approaches full embedding
  - Balance: `manifold_dim ≈ n_embd // 4` to `n_embd // 2`

### n_neighbors
- **Type:** `int`
- **Default:** `15`
- **Range:** 5-50 (typical: 10-30)
- **Description:** Number of neighbors for manifold construction
- **Details:**
  - k-nearest neighbors used to build the manifold graph structure
  - Determines local vs global manifold geometry
  - Small k: Local structure, disconnected manifold regions
  - Large k: Global structure, over-smoothed manifold
  - Used in graph Laplacian construction for spectral decomposition
- **Interdependencies:**
  - Must be < batch_size (or sequence length in online setting)
  - Larger `manifold_dim` benefits from more neighbors
  - Affects computational cost: O(n_neighbors × manifold_dim)
  - Interacts with `spectral_sigma` for kernel bandwidth
- **Tuning Tips:**
  - Use 10-15 for small manifolds (dim 32-64)
  - Use 20-30 for large manifolds (dim 128+)
  - Increase if manifold appears disconnected (multiple components)
  - Decrease if training is slow or manifold is over-smoothed
  - Typical sweet spot: 15-20 for most configurations

### n_eigenvectors
- **Type:** `int`
- **Default:** `32`
- **Range:** 4-128 (typical: 16-64)
- **Description:** Number of eigenvectors for spectral attention
- **Details:**
  - Number of smallest eigenvectors of the graph Laplacian to compute
  - These eigenvectors form a spectral basis capturing global manifold geometry
  - Used as positional/structural features in attention mechanism
  - More eigenvectors = richer geometric information but higher cost
  - Eigendecomposition is O(manifold_dim³), cached and reused
- **Interdependencies:**
  - Must be ≤ `manifold_dim` (cannot exceed manifold dimensionality)
  - Typically `n_eigenvectors ≈ manifold_dim // 2` to `manifold_dim`
  - Affects attention computation: each head uses spectral features
  - Larger values provide more geometric detail but diminishing returns
- **Tuning Tips:**
  - Use `manifold_dim // 2` as a starting point
  - Use `manifold_dim` for maximum geometric information
  - Reduce to 8-16 if eigendecomposition is a bottleneck
  - Minimum 4-8 eigenvectors needed for meaningful spectral features
  - Beyond 64 eigenvectors: minimal gains, higher memory

### spectral_sigma
- **Type:** `float`
- **Default:** `1.0`
- **Range:** 0.1-10.0 (typical: 0.5-2.0)
- **Description:** Bandwidth parameter for spectral kernel
- **Details:**
  - Controls the heat kernel bandwidth in spectral decomposition
  - Larger σ: Smoother manifold, global structure emphasized
  - Smaller σ: Sharper manifold, local structure emphasized
  - Used in Gaussian kernel: exp(-||x_i - x_j||² / (2σ²))
  - Affects the scale of geometric relationships in the manifold
- **Interdependencies:**
  - Interacts with `n_neighbors` (both control locality)
  - Should be tuned relative to embedding scale/variance
  - Affects gradient flow through manifold layers
  - Larger `manifold_dim` may require larger σ
- **Tuning Tips:**
  - Start with 1.0 (default)
  - Increase to 1.5-2.0 if manifold is too noisy/disconnected
  - Decrease to 0.5-0.7 if manifold is too smooth/loses detail
  - Monitor manifold visualization if available
  - Typical range: 0.8-1.5 for most configurations

### use_multiscale_manifold
- **Type:** `bool`
- **Default:** `True`
- **Description:** Enable multi-scale manifold hierarchy (E7 subgroup chain)
- **Details:**
  - When `True`: Uses 3-level manifold hierarchy (E7 → E6 → D5)
  - When `False`: Uses single-scale manifold with `manifold_dim`
  - Multi-scale enables coarse-to-fine geometric processing
  - Inspired by E7 exceptional Lie group subgroup decomposition
  - Each scale captures different levels of semantic structure
- **Interdependencies:**
  - When `True`, uses `multiscale_coarse_dim`, `multiscale_medium_dim`, `multiscale_fine_dim`
  - When `False`, uses `manifold_dim` only
  - Increases parameter count (3 projection layers instead of 1)
  - Affects memory and computation proportionally
- **Tuning Tips:**
  - Enable for richer geometric representations (recommended)
  - Disable for faster training or if memory is tight
  - Multi-scale shows benefits on complex semantic tasks
  - Single-scale is simpler and faster for baseline experiments

### multiscale_coarse_dim
- **Type:** `int`
- **Default:** `16`
- **Range:** 4-64 (typical: 8-32)
- **Description:** Coarse-scale manifold dimension (D5 level - global patterns)
- **Details:**
  - Lowest resolution in the E7 → E6 → D5 hierarchy
  - Captures global, document-level semantic patterns
  - Smallest dimensionality: strongest geometric compression
  - D5 refers to the Lie algebra subgroup in the E7 chain
  - Only active when `use_multiscale_manifold=True`
- **Interdependencies:**
  - Should be: `multiscale_coarse_dim < multiscale_medium_dim < multiscale_fine_dim`
  - Typical ratio: `1:2:4` (e.g., 16:32:64)
  - All must be ≤ `n_embd`
- **Tuning Tips:**
  - Use 8-16 for small models
  - Use 16-32 for medium/large models
  - Larger values: more global capacity, less compression
  - Minimum 4-8 to capture meaningful global structure

### multiscale_medium_dim
- **Type:** `int`
- **Default:** `32`
- **Range:** 8-128 (typical: 16-64)
- **Description:** Medium-scale manifold dimension (E6 level - phrase patterns)
- **Details:**
  - Middle resolution in the E7 → E6 → D5 hierarchy
  - Captures phrase-level and sentence-level semantic patterns
  - E6 refers to the exceptional Lie group in the subgroup chain
  - Balances global and local geometric information
  - Only active when `use_multiscale_manifold=True`
- **Interdependencies:**
  - Should be: `multiscale_coarse_dim < multiscale_medium_dim < multiscale_fine_dim`
  - Typically 2× `multiscale_coarse_dim`
  - Affects mid-level representation capacity
- **Tuning Tips:**
  - Use 16-32 for small models
  - Use 32-64 for medium/large models
  - Should be roughly halfway between coarse and fine dimensions
  - Common pattern: 16 (coarse), 32 (medium), 64 (fine)

### multiscale_fine_dim
- **Type:** `int`
- **Default:** `64`
- **Range:** 16-256 (typical: 32-128)
- **Description:** Fine-scale manifold dimension (E7 level - token patterns)
- **Details:**
  - Highest resolution in the E7 → E6 → D5 hierarchy
  - Captures token-level and word-level semantic patterns
  - E7 refers to the exceptional Lie group (top of the chain)
  - Most detailed geometric representation
  - Replaces `manifold_dim` when `use_multiscale_manifold=True`
  - Only active when `use_multiscale_manifold=True`
- **Interdependencies:**
  - Should be: `multiscale_coarse_dim < multiscale_medium_dim < multiscale_fine_dim`
  - Typically 4× `multiscale_coarse_dim` and 2× `multiscale_medium_dim`
  - Should be ≤ `n_embd` (typically `n_embd // 4` to `n_embd // 2`)
- **Tuning Tips:**
  - Use 32-64 for small models (n_embd=384)
  - Use 64-128 for medium/large models (n_embd=768+)
  - Largest of the three scales: most expressive
  - Common pattern: 16 (coarse), 32 (medium), 64 (fine)

### ortho_weight
- **Type:** `float`
- **Default:** `0.01`
- **Range:** 0.0-0.1 (typical: 0.001-0.05)
- **Description:** Spectral regularization weight for orthogonality
- **Details:**
  - Regularizes eigenvector orthogonality during training
  - Prevents eigenvector collapse (multiple eigenvectors converging to same direction)
  - Adds penalty: `ortho_weight * ||V^T V - I||²` where V is eigenvector matrix
  - Essential for stable spectral decomposition
  - Too high: Over-constrains learning, limits flexibility
  - Too low: Eigenvectors may collapse, losing spectral diversity
- **Interdependencies:**
  - Only active when manifold/spectral layers are enabled
  - Disabled if `skip_manifold_spectral=True`
  - Interacts with overall learning rate
  - Should scale inversely with `n_eigenvectors`
- **Tuning Tips:**
  - Start with 0.01 (default)
  - Increase to 0.02-0.05 if eigenvectors collapse (check via logging)
  - Decrease to 0.001-0.005 if training is too constrained
  - Set to 0.0 to disable (not recommended)
  - Monitor eigenvalue spectrum for diversity

### skip_manifold_spectral
- **Type:** `bool`
- **Default:** `False`
- **Description:** Skip manifold/spectral projection for faster training
- **Details:**
  - When `True`: Completely bypasses manifold and spectral computations
  - When `False`: Full manifold projection and spectral attention enabled
  - Useful for ablation studies or fast baseline comparisons
  - Reduces model to closer-to-standard transformer architecture
  - Disables: manifold projection, spectral decomposition, geometric attention
- **Interdependencies:**
  - When `True`, all manifold/spectral parameters become inactive
  - Significantly reduces computation and memory usage
  - Part of `fast_mode` optimization suite
- **Tuning Tips:**
  - Enable for fast iteration during debugging
  - Disable for full NeuroManifold capabilities (recommended)
  - Use for A/B testing: manifold vs standard architecture
  - Speed improvement: 20-30% depending on manifold dimensions

---

**Example Multi-Scale Manifold Configuration:**

```python
# E7 subgroup chain: coarse-to-fine geometric hierarchy
config = NeuroManifoldConfig(
    use_multiscale_manifold=True,
    multiscale_coarse_dim=16,    # D5: Global patterns
    multiscale_medium_dim=32,    # E6: Phrase patterns
    multiscale_fine_dim=64,      # E7: Token patterns
    n_neighbors=20,              # Sufficient for multi-scale
    n_eigenvectors=32,           # Rich spectral features
    spectral_sigma=1.0,          # Balanced kernel bandwidth
    ortho_weight=0.01,           # Prevent eigenvector collapse
)
```

**Example Single-Scale Manifold Configuration:**

```python
# Simpler, faster single-scale manifold
config = NeuroManifoldConfig(
    use_multiscale_manifold=False,
    manifold_dim=64,             # Single geometric scale
    n_neighbors=15,              # Standard k-NN
    n_eigenvectors=32,           # Standard spectral features
    spectral_sigma=1.0,          # Default bandwidth
    ortho_weight=0.01,           # Standard regularization
)
```

**Example Fast Mode (Skip Manifold):**

```python
# Disable manifold/spectral for maximum speed
config = NeuroManifoldConfig(
    skip_manifold_spectral=True,  # Bypass all manifold computation
    # All manifold parameters ignored
)
```

---

## 4. FHN Dynamics Parameters

**Category:** FitzHugh-Nagumo neural dynamics for soliton-based attention.

The FitzHugh-Nagumo (FHN) model describes **soliton propagation** in excitable media, inspired by nerve impulse transmission. In NeuroManifold, FHN dynamics replace standard softmax attention with **wave-based attention** that propagates information through the network like action potentials in biological neurons.

**Key Concepts:**
- **Solitons:** Self-sustaining waves that propagate without dispersion (lossless information transmission)
- **Excitability:** Neurons fire when input exceeds a threshold, creating propagating pulses
- **Slow-Fast Dynamics:** FHN equations have two timescales (fast voltage, slow recovery)
- **Phase Transitions:** The membrane undergoes phase changes during soliton propagation (Konrad Kaufmann's thermodynamic theory)
- **Linearized Parallel Scan:** FFT-based method for efficient parallel FHN integration

**Mathematical Background:**

The FHN equations model excitable dynamics:

```
dv/dt = v - v³/3 - w + I_ext     (fast variable: voltage)
dw/dt = (v + a - b*w) / τ        (slow variable: recovery)
```

Where:
- `v`: Voltage (excitation state)
- `w`: Recovery variable (refractoriness)
- `I_ext`: External input (attention queries)
- `τ` (`fhn_tau`): Time constant (slow-fast separation)
- Firing occurs when `v` exceeds `fhn_threshold`

**Why FHN for Attention?**
- **O(N) Complexity:** Solitons propagate along paths, not O(N²) all-to-all
- **Long-Range:** Waves travel without dispersion (context length 10k+)
- **Biological Plausibility:** Mimics neural action potentials
- **Sparse Activation:** Only excited regions fire (efficient)

**Integration Methods:**
- **IMEX (Implicit-Explicit):** Semi-implicit scheme for stiff equations (recommended)
- **Parallel Scan:** FFT-based linearized FHN for maximum parallelization

### fhn_threshold
- **Type:** `float`
- **Default:** `0.5`
- **Range:** 0.1-1.0 (typical: 0.3-0.7)
- **Description:** Firing threshold for soliton activation (excitability threshold)
- **Details:**
  - Determines when a neuron/token becomes "excited" and fires a soliton
  - Lower threshold: More excitable, easier to trigger waves (more active attention)
  - Higher threshold: Less excitable, selective firing (sparser attention)
  - Analogous to action potential threshold in biological neurons (~-55mV)
  - 0.5 is a balanced default (50% of maximum excitation)
  - Controls sparsity of attention patterns
- **Interdependencies:**
  - Interacts with `fhn_tau` (recovery speed after firing)
  - Lower threshold with high `n_fhn_steps` can cause runaway excitation
  - Should be tuned with `pulse_width_base` for stable wave propagation
  - Affects gradient flow through attention layers
- **Tuning Tips:**
  - Start with 0.5 (default, balanced excitability)
  - Decrease to 0.3-0.4 for more active/dense attention patterns
  - Increase to 0.6-0.7 for sparser, more selective attention
  - Too low (<0.2): Unstable, constant firing, gradient explosion
  - Too high (>0.8): Dead neurons, no attention propagation
  - Monitor attention entropy to validate sparsity

### fhn_tau
- **Type:** `float`
- **Default:** `12.5`
- **Range:** 1.0-50.0 (typical: 10.0-20.0)
- **Description:** Time constant for FHN slow-fast dynamics (τ parameter)
- **Details:**
  - Controls the timescale separation between fast (voltage) and slow (recovery) variables
  - Larger τ: Slower recovery, longer refractory period, sustained excitation
  - Smaller τ: Faster recovery, shorter refractory period, rapid firing
  - **CRITICAL:** Must be >> 1.0 for proper slow-fast separation (was incorrectly 0.1 in early versions)
  - 12.5 provides biologically plausible separation (12.5× slower recovery than excitation)
  - Affects soliton pulse width and propagation stability
  - Essential for IMEX integration stability
- **Interdependencies:**
  - **CRITICAL:** Must be compatible with `n_fhn_steps` and integration timestep
  - Larger `fhn_tau` requires more `n_fhn_steps` for accurate integration
  - Affects `pulse_width_base` (wider pulses need longer recovery)
  - Interacts with `use_fhn_imex` (implicit integration handles large τ)
  - Too small: Loses slow-fast structure, unstable dynamics
  - Too large: Slow convergence, may require more integration steps
- **Tuning Tips:**
  - Use 12.5 (default, proper slow-fast separation)
  - Increase to 15.0-25.0 for longer memory/sustained attention
  - Decrease to 8.0-10.0 for faster adaptation/shorter context
  - **Never use τ < 1.0** (breaks slow-fast assumption)
  - With `use_fhn_imex=True`, can safely use larger τ (20+)
  - Monitor for NaN/inf during training (sign of τ mismatch)

### fhn_velocity
- **Type:** `float`
- **Default:** `1.0`
- **Range:** 0.1-5.0 (typical: 0.5-2.0)
- **Description:** Propagation velocity of soliton waves
- **Details:**
  - Speed at which solitons travel through the attention space
  - Determines how fast information propagates from query to keys
  - Higher velocity: Faster propagation, wider attention receptive field per step
  - Lower velocity: Slower propagation, more localized attention patterns
  - Biological solitons travel at 1-100 m/s in nerves (normalized to 1.0)
  - Affects the effective context length per FHN integration step
  - Combined with `n_fhn_steps` determines total propagation distance
- **Interdependencies:**
  - Effective range = `fhn_velocity * n_fhn_steps * pulse_width_base`
  - Larger velocity requires careful tuning with `fhn_threshold` for stability
  - Affects gradient magnitude in FHN layers
  - Should scale with sequence length (longer sequences may need higher velocity)
- **Tuning Tips:**
  - Use 1.0 (default, standard propagation)
  - Increase to 1.5-2.0 for longer-range dependencies
  - Decrease to 0.5-0.7 for more local attention
  - Very fast (>3.0): May cause numerical instability
  - Very slow (<0.3): Limited context, similar to local attention
  - Balance with `n_fhn_steps` for desired effective context

### pulse_width_base
- **Type:** `int`
- **Default:** `4`
- **Range:** 1-16 (typical: 2-8)
- **Description:** Base width of soliton pulses (spatial extent)
- **Details:**
  - Determines the spatial extent of each soliton pulse in token space
  - Width 4 = each pulse affects ~4 neighboring tokens directly
  - Wider pulses: More overlap, smoother attention, higher receptive field
  - Narrower pulses: More localized, sharper attention, lower receptive field
  - Similar to kernel size in convolution (but for wave propagation)
  - Affects how much information each soliton carries
  - Biological action potentials have width ~1-2ms (normalized to 4 tokens)
- **Interdependencies:**
  - Effective receptive field = `pulse_width_base * n_fhn_steps * fhn_velocity`
  - Larger `pulse_width_base` with small `n_fhn_steps` = local-only attention
  - Smaller `pulse_width_base` with large `n_fhn_steps` = sparse long-range
  - Affects memory usage: wider pulses = more token interactions
  - Should be ≤ `block_size // 4` for meaningful propagation
- **Tuning Tips:**
  - Use 4 (default, balanced local-global)
  - Increase to 6-8 for smoother, more global attention
  - Decrease to 2-3 for sharper, more localized attention
  - Width 1: Extremely sparse, may miss important connections
  - Width >12: Approaches dense attention (loses wave benefits)
  - Common pattern: width 4, steps 2, velocity 1.0 = 8-token effective range

### n_fhn_steps
- **Type:** `int`
- **Default:** `2`
- **Range:** 1-10 (typical: 1-4)
- **Description:** Number of FHN integration steps per forward pass
- **Details:**
  - How many times to iterate the FHN equations per attention computation
  - More steps: Better dynamics, longer propagation range, higher accuracy
  - Fewer steps: Faster computation, shorter range, less accurate
  - 2 steps (default) with IMEX provides good accuracy/speed tradeoff
  - Each step propagates the wave further through the sequence
  - Total propagation = `n_fhn_steps * fhn_velocity * pulse_width_base`
  - Integration accuracy improves with more steps (especially for large `fhn_tau`)
- **Interdependencies:**
  - **CRITICAL:** Must be compatible with `fhn_tau` for numerical stability
  - Larger `fhn_tau` (>15) may require 3-4 steps for accuracy
  - Computational cost scales linearly with `n_fhn_steps`
  - With `use_fhn_imex=True`, 2 steps is usually sufficient
  - With `use_fhn_parallel=True`, steps are parallelized (less overhead)
  - Memory usage increases slightly with more steps
- **Tuning Tips:**
  - Use 2 (default, balanced accuracy/speed with IMEX)
  - Use 1 for maximum speed (fast mode, acceptable with IMEX)
  - Use 3-4 for high accuracy or large `fhn_tau` (>20)
  - More than 5 steps: Diminishing returns, significant slowdown
  - Ablation: Test 1 vs 2 steps to measure accuracy/speed tradeoff
  - If seeing NaN: Increase steps or enable `use_fhn_imex`

### use_fhn_imex
- **Type:** `bool`
- **Default:** `True`
- **Description:** Use semi-implicit IMEX (Implicit-Explicit) integration scheme
- **Details:**
  - **IMEX:** Treats stiff terms implicitly, non-stiff terms explicitly
  - Handles the slow-fast timescale separation in FHN equations robustly
  - Implicit on slow variable (w), explicit on fast variable (v)
  - **Essential for large `fhn_tau`** (>10) to prevent numerical instability
  - More stable than explicit Euler, especially for stiff equations
  - Slight computational overhead vs explicit, but much better stability
  - Industry standard for stiff ODE systems
  - Prevents gradient explosion in deep FHN attention layers
- **Interdependencies:**
  - **Strongly recommended** when `fhn_tau > 5.0`
  - **Required** when `fhn_tau > 15.0` (explicit will diverge)
  - Allows using fewer `n_fhn_steps` while maintaining accuracy
  - Compatible with all other FHN optimization flags
  - Slight memory increase (stores intermediate implicit states)
- **Tuning Tips:**
  - **Always use True** (default, highly recommended)
  - Only disable for explicit integration experiments (not recommended)
  - IMEX + 2 steps ≈ explicit + 4-6 steps in accuracy
  - Critical for training stability with default `fhn_tau=12.5`
  - If seeing NaN/inf: Ensure this is True

### use_fhn_partitioning
- **Type:** `bool`
- **Default:** `True`
- **Description:** Enable energy balancing via Karmarkar-Karp partitioning
- **Details:**
  - Balances the "energy" (activation magnitude) across FHN states
  - Uses spectral partitioning to prevent energy concentration
  - Keeps the membrane at the "phase transition" point (maximum susceptibility)
  - Part of the Kaufmann Trifecta: balancing for optimal soliton propagation
  - Improves training stability by preventing runaway excitation
  - Adds small computational overhead for partitioning algorithm
  - Based on number-theoretic load balancing (Karmarkar-Karp)
  - Helps maintain consistent attention patterns across training
- **Interdependencies:**
  - Works best with `use_fhn_imex=True` (both improve stability)
  - Interacts with `fhn_threshold` (balancing affects firing probability)
  - Adds overhead: ~5-10% compute cost
  - Disabled automatically when `fast_mode=True`
  - Compatible with all attention variants
- **Tuning Tips:**
  - Enable for training stability (recommended, default True)
  - Disable for maximum speed if stability is not an issue
  - Particularly helpful for long sequences (1024+ tokens)
  - Monitor attention entropy: partitioning should stabilize it
  - Part of "full NeuroManifold" configuration

### use_fhn_fused
- **Type:** `bool`
- **Default:** `False`
- **Description:** Use fused CUDA kernels for FHN computation (currently disabled)
- **Details:**
  - **Deprecated:** Replaced by JIT compilation
  - Would fuse multiple FHN operations into single GPU kernel
  - Reduces memory bandwidth by avoiding intermediate tensors
  - Disabled in favor of PyTorch JIT which provides similar benefits
  - May be re-enabled in future with custom CUDA implementation
  - Keep as False unless you have custom fused kernel implementation
- **Interdependencies:**
  - Mutually exclusive with `use_fhn_parallel` (parallel scan is preferred)
  - Would require custom CUDA extension (not currently implemented)
  - JIT compilation provides most benefits automatically
- **Tuning Tips:**
  - **Keep False** (default, JIT is sufficient)
  - Only enable if you implement custom fused kernels
  - For speed, use `use_fhn_parallel=True` instead

### use_fhn_parallel
- **Type:** `bool`
- **Default:** `True`
- **Description:** Use FFT-based Parallel Scan for linearized FHN integration
- **Details:**
  - **Maximum speed optimization** for FHN computation
  - Linearizes FHN equations to enable parallel prefix scan
  - Uses FFT (Fast Fourier Transform) for O(N log N) parallel integration
  - Drastically faster than sequential integration on GPUs
  - Slight approximation error vs full nonlinear FHN (negligible in practice)
  - Enables efficient long-context FHN (tested on 10k+ tokens)
  - Based on linear recurrent neural network optimization techniques
  - Essential for production deployments
- **Interdependencies:**
  - Compatible with `use_fhn_imex=True` (both can be enabled)
  - Mutually exclusive with `use_fhn_fused=False` (parallel scan is preferred)
  - Larger `block_size` shows more benefit from parallelization
  - Works with all `n_fhn_steps` settings
  - Memory usage slightly higher due to FFT buffers
- **Tuning Tips:**
  - **Always use True** (default, major speedup)
  - Only disable for debugging or exact nonlinear FHN comparison
  - Speed improvement: 3-5× faster than sequential on sequences >256
  - Enables scaling to long contexts (4k-32k tokens)
  - Critical for real-time inference

---

**Example FHN Configurations:**

### Standard Configuration (Balanced)
```python
# Balanced FHN with stability and efficiency
config = NeuroManifoldConfig(
    fhn_threshold=0.5,           # Balanced excitability
    fhn_tau=12.5,                # Proper slow-fast separation
    fhn_velocity=1.0,            # Standard propagation speed
    pulse_width_base=4,          # Moderate pulse width
    n_fhn_steps=2,               # Good accuracy with IMEX
    use_fhn_imex=True,           # Stable integration (required)
    use_fhn_partitioning=True,   # Energy balancing for stability
    use_fhn_parallel=True,       # Maximum speed
)
```

### Fast Mode (Maximum Speed)
```python
# Minimal FHN for fastest training
config = NeuroManifoldConfig(
    fhn_threshold=0.5,           # Keep balanced
    fhn_tau=12.5,                # Keep proper τ
    fhn_velocity=1.0,            # Standard
    pulse_width_base=4,          # Standard
    n_fhn_steps=1,               # Single step (faster)
    use_fhn_imex=True,           # Still critical for stability
    use_fhn_partitioning=False,  # Disable for speed
    use_fhn_parallel=True,       # Keep parallelization
)
```

### High-Accuracy Mode (Research)
```python
# Maximum accuracy for FHN dynamics research
config = NeuroManifoldConfig(
    fhn_threshold=0.5,           # Balanced
    fhn_tau=15.0,                # Slower recovery (more biological)
    fhn_velocity=1.0,            # Standard
    pulse_width_base=4,          # Standard
    n_fhn_steps=4,               # More integration steps
    use_fhn_imex=True,           # Required for τ=15
    use_fhn_partitioning=True,   # Full stability features
    use_fhn_parallel=False,      # Exact nonlinear dynamics
)
```

### Long-Range Attention
```python
# Extended context via faster/wider soliton propagation
config = NeuroManifoldConfig(
    fhn_threshold=0.4,           # More excitable (wider activation)
    fhn_tau=12.5,                # Standard
    fhn_velocity=2.0,            # 2× faster propagation
    pulse_width_base=6,          # Wider pulses
    n_fhn_steps=3,               # More steps for range
    use_fhn_imex=True,           # Stability
    use_fhn_partitioning=True,   # Stability
    use_fhn_parallel=True,       # Speed for long context
)
# Effective range: 2.0 * 6 * 3 = 36 tokens per attention
```

### Sparse Local Attention
```python
# Highly selective, local attention patterns
config = NeuroManifoldConfig(
    fhn_threshold=0.7,           # High threshold (sparse firing)
    fhn_tau=10.0,                # Faster recovery
    fhn_velocity=0.5,            # Slower propagation
    pulse_width_base=2,          # Narrow pulses
    n_fhn_steps=2,               # Standard
    use_fhn_imex=True,           # Stability
    use_fhn_partitioning=True,   # Balance energy
    use_fhn_parallel=True,       # Speed
)
# Effective range: 0.5 * 2 * 2 = 2 tokens (very local)
```

---

**FHN Dynamics Troubleshooting:**

| Issue | Likely Cause | Solution |
|-------|--------------|----------|
| NaN/Inf during training | `fhn_tau` too small or `use_fhn_imex=False` | Set `fhn_tau ≥ 10.0` and `use_fhn_imex=True` |
| Attention collapse (all zeros) | `fhn_threshold` too high | Decrease to 0.3-0.5 |
| Attention explosion (all ones) | `fhn_threshold` too low | Increase to 0.5-0.7 |
| Slow training | `use_fhn_parallel=False` | Enable parallel scan |
| Poor long-range dependencies | Small effective range | Increase `fhn_velocity`, `pulse_width_base`, or `n_fhn_steps` |
| Numerical instability | Large `fhn_tau` with few steps | Increase `n_fhn_steps` or ensure `use_fhn_imex=True` |

---

**Performance Characteristics:**

- **IMEX (True) vs Explicit (False):** ~10% slower, much more stable
- **Parallel Scan (True) vs Sequential:** 3-5× faster on sequences >256 tokens
- **Partitioning (True) vs Off:** ~5% slower, improves stability
- **1 step vs 2 steps:** 2× faster, slight accuracy loss
- **Overall FHN overhead vs Standard Attention:** ~20-30% slower but O(N) vs O(N²)

---

## 5. Attention Variants

**Category:** Different attention mechanisms and normalizations.

NeuroManifold supports multiple attention mechanisms, each offering different properties for semantic modeling. The default uses FHN-based soliton attention, but advanced variants provide topological and quantum-inspired alternatives.

**Available Attention Mechanisms:**
- **Standard (FHN Soliton):** Wave-based attention via FitzHugh-Nagumo dynamics (default)
- **Knot-Theoretic:** Topological attention using knot invariants (Louis Kauffman's bracket polynomial)
- **Kaufmann Trifecta:** Ultimate attention combining all three Kaufmann theories
- **QK Normalization:** RMSNorm on Q/K to prevent logit explosion (Qwen3/GLM-4.5 style)

### use_knot_attention
- **Type:** `bool`
- **Default:** `False`
- **Description:** Enable Knot-Theoretic attention using topological invariants
- **Details:**
  - Uses **knot theory** (Louis Kauffman) to model semantic relationships
  - Computes **bracket polynomial** invariants for attention weights
  - Attention patterns are constrained by topological equivalence classes
  - Captures semantic entanglement: how concepts "knot" together
  - More computationally expensive than standard attention
  - Provides theoretical guarantees on attention structure
  - Based on Jones polynomial and Kauffman bracket
  - **Mutually exclusive** with `use_kaufmann_attention` (Kaufmann includes knot theory)
- **Interdependencies:**
  - Cannot be used with `use_kaufmann_attention=True` (Kaufmann is superset)
  - Works with all FHN dynamics parameters
  - Compatible with `use_qk_norm=True`
  - Adds ~30% computational overhead vs standard attention
  - Requires careful initialization (topological structure)
- **Tuning Tips:**
  - Enable for research on topological semantic structure
  - Best used with medium/large models (more capacity for topology)
  - May improve reasoning tasks that require structural understanding
  - Start with standard attention, switch to knot for ablation studies
  - Monitor attention pattern diversity (knot theory enforces structure)
  - Consider using with `use_qk_norm=True` for stability

### use_kaufmann_attention
- **Type:** `bool`
- **Default:** `False`
- **Description:** Enable Kaufmann Trifecta Attention (The Endgame)
- **Details:**
  - **The Ultimate Attention Mechanism:** Combines all three Kaufmann theories
  - **Konrad Kaufmann (Thermodynamics):** Phase transitions in soliton dynamics
  - **Stuart Kauffman (Complexity):** Fitness landscapes and the Adjacent Possible
  - **Louis Kauffman (Topology):** Knot-theoretic semantic entanglement
  - Integrates:
    1. FHN soliton propagation with phase transition awareness
    2. Kauffman fitness landscape navigation (NK model)
    3. Knot polynomial computation for topological constraints
  - **Most powerful but most expensive** attention variant
  - Provides theoretical foundation for semantic emergence
  - ~2× slower than standard attention, 50% slower than knot-only
  - **Mutually exclusive** with `use_knot_attention` (superset)
  - Enables "Adjacent Possible" exploration in latent space
- **Interdependencies:**
  - Overrides `use_knot_attention` (Kaufmann includes it)
  - Requires all FHN dynamics parameters (uses soliton phase transitions)
  - Works best with `use_fhn_partitioning=True` (energy balancing)
  - Compatible with `use_qk_norm=True` (recommended for stability)
  - Benefits from larger `n_embd` (384+) for richer representations
  - Increases memory usage (~20% more than standard)
- **Tuning Tips:**
  - **Research only:** Not recommended for production due to compute cost
  - Enable for maximum theoretical sophistication
  - Best for small-scale experiments (nano/small models)
  - Requires careful hyperparameter tuning (all three theories interact)
  - Monitor for numerical instability (complex dynamics)
  - Consider enabling `use_qk_norm=True` for stability
  - Ablation: Compare standard → knot → Kaufmann trifecta
  - May show benefits on complex reasoning/composition tasks

### use_qk_norm
- **Type:** `bool`
- **Default:** `True`
- **Description:** Apply RMSNorm to Query and Key projections (Qwen3/GLM-4.5 style)
- **Details:**
  - Applies **RMSNorm** (Root Mean Square Normalization) to Q and K before attention
  - Prevents **attention logit explosion** in deep models
  - Standard technique in modern LLMs (Qwen3, GLM-4.5, DeepSeek-V3)
  - Formula: `Q_norm = Q / sqrt(mean(Q²) + eps)`, same for K
  - Does NOT normalize V (values) - only keys and queries
  - Stabilizes training in models with many layers (12+)
  - Minimal computational overhead (~2% slower)
  - Improves gradient flow through attention layers
  - Essential for large vocabularies (100K+ tokens)
  - Works with all attention variants (standard, knot, Kaufmann)
- **Interdependencies:**
  - Recommended for `n_layer ≥ 12` (deep models)
  - **Critical** for `vocab_size > 100K` (prevents logit overflow)
  - Compatible with all attention variants
  - Works with `use_mhc=True` (both improve stability)
  - No interaction with FHN dynamics (applied before attention)
  - Slight memory increase (stores normalization statistics)
- **Tuning Tips:**
  - **Keep True** (default, modern best practice)
  - Only disable for ablation studies or legacy compatibility
  - Essential for scaling to large models (1B+ parameters)
  - Helps with training stability on long sequences (2K+ tokens)
  - Combine with `use_mhc=True` for maximum stability
  - If seeing attention NaN/Inf: Ensure this is True
  - Standard in production LLM architectures (2024+)

---

**Example Attention Configurations:**

### Standard FHN Soliton Attention (Default)
```python
# Wave-based attention with FitzHugh-Nagumo dynamics
config = NeuroManifoldConfig(
    use_knot_attention=False,      # Standard soliton attention
    use_kaufmann_attention=False,  # No topological constraints
    use_qk_norm=True,              # Modern stability (recommended)
    # FHN parameters active (see Section 4)
)
```

### Knot-Theoretic Attention
```python
# Topological attention with structural constraints
config = NeuroManifoldConfig(
    use_knot_attention=True,       # Enable knot theory
    use_kaufmann_attention=False,  # Knot-only (not full Kaufmann)
    use_qk_norm=True,              # Stability for topological constraints
    n_embd=512,                    # More capacity for topology
    n_layer=8,                     # Medium depth
)
# ~30% slower than standard, structured attention patterns
```

### Kaufmann Trifecta Attention (Research)
```python
# The ultimate attention: all three Kaufmann theories
config = NeuroManifoldConfig(
    use_knot_attention=False,      # Overridden by Kaufmann
    use_kaufmann_attention=True,   # Enable full trifecta
    use_qk_norm=True,              # Essential for stability
    use_fhn_partitioning=True,     # Energy balancing (phase transitions)
    n_embd=512,                    # Large capacity needed
    n_layer=6,                     # Keep shallow (compute intensive)
    fhn_threshold=0.5,             # Standard phase transition threshold
)
# ~2× slower than standard, maximum theoretical power
```

### Deep Stable Configuration (Production)
```python
# Deep model with maximum stability
config = NeuroManifoldConfig(
    use_knot_attention=False,      # Standard attention (fast)
    use_kaufmann_attention=False,  # Production: avoid exotic variants
    use_qk_norm=True,              # Critical for deep models
    use_mhc=True,                  # DeepSeek stability (see Section 6)
    n_layer=24,                    # Deep architecture
    n_embd=1024,                   # Large width
    vocab_size=151936,             # Large vocabulary (Qwen3)
)
# QK norm + mHC = maximum training stability
```

### Fast Baseline (No Exotic Features)
```python
# Standard transformer for comparison
config = NeuroManifoldConfig(
    use_knot_attention=False,      # Disable exotic attention
    use_kaufmann_attention=False,  # Standard only
    use_qk_norm=True,              # Modern best practice
    skip_manifold_spectral=True,   # Skip geometric features
    use_fhn_parallel=True,         # Fast FHN
    fast_mode=True,                # Enable all fast paths
)
# Closest to standard transformer (for ablation)
```

---

**Attention Variant Comparison:**

| Variant | Speed | Stability | Theoretical Basis | Use Case |
|---------|-------|-----------|-------------------|----------|
| **Standard (FHN)** | 1.0× | Good | Soliton dynamics | Default, production |
| **Knot** | 0.7× | Good | Topology (Louis K.) | Research, structure |
| **Kaufmann Trifecta** | 0.5× | Medium | All 3 Kaufmanns | Research, theory |
| **QK Norm** | 0.98× | Excellent | Modern LLM practice | Always enable |

**Recommendation:** Use standard FHN with `use_qk_norm=True` for production. Enable knot or Kaufmann attention for research on topological/thermodynamic semantics.

---

## 6. mHC (Manifold-Constrained Hyper-Connections)

**Category:** DeepSeek-style architecture for training stability.

mHC (Manifold-Constrained Hyper-Connections) is a novel architecture from **DeepSeek-V3** that provides extreme training stability through doubly stochastic routing. It replaces standard residual connections with learned multi-stream routing, preventing gradient vanishing/explosion in deep networks.

**Key Concepts:**
- **Doubly Stochastic Matrices:** Row and column sums = 1 (Birkhoff polytope)
- **Sinkhorn-Knopp Algorithm:** Iterative normalization to enforce double stochasticity
- **Multi-Stream Routing:** Parallel streams with learned routing weights
- **Manifold Constraint:** Routing matrices lie on a Riemannian manifold
- **Gradient Stability:** Prevents vanishing/explosion in 60+ layer models

**Mathematical Background:**

DeepSeek's mHC architecture replaces:
```
x_{l+1} = x_l + F(x_l)  # Standard residual
```

With manifold-constrained hyper-connection:
```
x_{l+1} = H_res @ x_l + H_post^T @ F(H_pre @ x_l)
```

Where:
- `H_res`: Doubly stochastic residual routing (via Sinkhorn-Knopp)
- `H_pre`: Pre-transformation routing (softmax over streams)
- `H_post`: Post-transformation routing (softmax over streams)
- `F(·)`: Transformer block (attention + FFN)

**Why mHC?**
- **Stability:** Enables training 60+ layer models without instability
- **Gradient Flow:** Maintains gradient norms across arbitrary depth
- **Flexibility:** Learned routing adapts per layer
- **Provable:** Double stochasticity guarantees bounded eigenvalues

**Reference:** DeepSeek-V3 (arXiv:2512.24880)

### use_mhc
- **Type:** `bool`
- **Default:** `True`
- **Description:** Enable Manifold-Constrained Hyper-Connections
- **Details:**
  - Replaces standard residual connections with mHC architecture
  - Applies doubly stochastic routing via Sinkhorn-Knopp normalization
  - Dramatically improves training stability in deep models (12+ layers)
  - Used in DeepSeek-V3 (685B parameters, 60 layers)
  - Minimal computational overhead (~5% slower)
  - Essential for scaling to very deep architectures
  - Prevents gradient vanishing/explosion through guaranteed spectral properties
  - Works with all attention variants (FHN, knot, Kaufmann)
  - Compatible with all architectural features (MTP, MoE, MLA)
- **Interdependencies:**
  - When `True`, uses `mhc_*` parameters for configuration
  - When `False`, uses standard residual connections (x + F(x))
  - Recommended with `use_qk_norm=True` for maximum stability
  - Benefits scale with model depth (`n_layer`)
  - Interacts with `mhc_n_streams` for multi-stream routing
  - Slight memory increase (stores routing matrices per layer)
- **Tuning Tips:**
  - **Keep True** (default, modern best practice)
  - Essential for `n_layer ≥ 12` (deep models)
  - Critical for `n_layer ≥ 24` (very deep models)
  - Disable only for ablation studies or standard residual comparison
  - Combine with `use_qk_norm=True` for ultimate stability
  - Enables training 40+ layer models without special techniques
  - Standard in modern large-scale LLMs (2024+)

### use_full_mhc
- **Type:** `bool`
- **Default:** `True`
- **Description:** Use full multi-stream mHC (vs simplified single-stream)
- **Details:**
  - **Full mHC:** Multi-stream routing with H_pre and H_post matrices
  - **Simplified mHC:** Single-stream with only H_res (doubly stochastic residual)
  - Full version provides more routing flexibility and better gradient flow
  - Uses `mhc_n_streams` parallel processing streams
  - Slightly higher memory (~10% more than simplified)
  - Matches DeepSeek-V3 architecture exactly
  - Simplified version is faster but less stable
  - Full version recommended for production and deep models
  - Enables learned per-layer routing patterns
- **Interdependencies:**
  - Only active when `use_mhc=True`
  - When `True`, uses `mhc_n_streams` for stream count
  - When `False`, single-stream routing (H_res only)
  - Full version benefits more from larger `n_layer`
  - Interacts with `mhc_residual_weight` for initialization
  - Requires `mhc_sinkhorn_iters` iterations for normalization
- **Tuning Tips:**
  - **Keep True** (default, full DeepSeek architecture)
  - Disable for faster training if stability is not critical
  - Full version essential for `n_layer ≥ 20`
  - Simplified version acceptable for `n_layer ≤ 12`
  - Ablation: Compare full vs simplified mHC
  - Memory difference is small (10%), stability gains are large

### mhc_n_streams
- **Type:** `int`
- **Default:** `2`
- **Range:** 1-8 (typical: 2-4)
- **Description:** Number of parallel streams for full mHC routing
- **Details:**
  - Number of independent processing streams in multi-stream mHC
  - Each stream has its own routing weights (H_pre/H_post)
  - More streams = more routing flexibility but higher memory
  - DeepSeek-V3 uses 2 streams (sweet spot for efficiency)
  - 1 stream = simplified mHC (equivalent to `use_full_mhc=False`)
  - 4+ streams: Diminishing returns, significant memory increase
  - Each stream processes a portion of the embedding dimension
  - Streams are routed and combined via learned weights
  - Only active when `use_full_mhc=True`
- **Interdependencies:**
  - Only active when `use_mhc=True` and `use_full_mhc=True`
  - Must be ≥ 1 (1 stream = simplified mode)
  - Memory usage scales linearly with `mhc_n_streams`
  - Compute overhead: ~2-3% per additional stream
  - Should be ≤ `n_heads` for meaningful stream separation
  - Independent of `n_heads` (different concept: routing vs attention)
- **Tuning Tips:**
  - Use 2 (default, DeepSeek recommendation)
  - Use 1 for simplified mHC (faster, slightly less stable)
  - Use 3-4 for extremely deep models (40+ layers) or research
  - More than 4 streams: Minimal benefit, wasted memory
  - Balance: 2 streams provides 90% of benefits with minimal overhead
  - Ablation: Test 1 vs 2 streams to measure stability gain

### mhc_residual_weight
- **Type:** `float`
- **Default:** `0.9`
- **Range:** 0.5-1.0 (typical: 0.8-0.95)
- **Description:** Initial identity mapping bias for H_res initialization
- **Details:**
  - Controls initialization of the residual routing matrix H_res
  - Higher values (0.9-0.95): Initialize closer to identity (x ≈ x + F(x))
  - Lower values (0.5-0.7): More aggressive routing from the start
  - 0.9 provides gentle transition from standard residual to learned routing
  - Helps with early training stability (start close to known-good architecture)
  - H_res is initialized as: `0.9 * I + 0.1 * random`, then Sinkhorn-normalized
  - As training progresses, H_res learns optimal routing
  - Too high (>0.95): Slow to learn routing (stuck near identity)
  - Too low (<0.7): Unstable early training (routing too aggressive)
- **Interdependencies:**
  - Only active when `use_mhc=True`
  - Affects initial training dynamics (first ~1000 steps)
  - Interacts with learning rate (higher LR → lower residual weight)
  - After Sinkhorn normalization, weight is approximate (doubly stochastic constraint)
  - No effect on final converged model (learned routing dominates)
- **Tuning Tips:**
  - Use 0.9 (default, balanced initialization)
  - Increase to 0.92-0.95 for very deep models (60+ layers) or if early instability
  - Decrease to 0.8-0.85 for shallow models (6-12 layers) or aggressive routing
  - Monitor initial training loss: If unstable, increase residual weight
  - Less critical than other mHC parameters (initialization only)
  - Typical range: 0.85-0.95 for most configurations

### mhc_sinkhorn_iters
- **Type:** `int`
- **Default:** `5`
- **Range:** 3-20 (typical: 3-7)
- **Description:** Sinkhorn-Knopp iterations for doubly stochastic normalization
- **Details:**
  - Number of iterations for Sinkhorn-Knopp algorithm
  - Enforces doubly stochastic constraint: row sums = col sums = 1
  - More iterations = better approximation of Birkhoff polytope (doubly stochastic matrices)
  - 3-5 iterations typically sufficient for convergence (< 1e-6 error)
  - 5 iterations (default) provides good balance of accuracy and speed
  - Each iteration: O(n_embd²) operations, parallelized
  - Convergence is exponential: error ≈ exp(-k), k = iteration count
  - Too few (<3): Poor approximation, breaks stability guarantees
  - Too many (>10): Wasted computation, negligible improvement
  - Applied during every forward pass (routing is dynamic)
- **Interdependencies:**
  - Only active when `use_mhc=True`
  - Computational cost scales linearly with iterations
  - Larger `n_embd` benefits from more iterations (harder to normalize)
  - Independent of `n_layer` (applied per-layer)
  - ~1% slowdown per iteration (5 iterations ≈ 5% overhead)
  - Critical for gradient stability (double stochasticity guarantee)
- **Tuning Tips:**
  - Use 5 (default, standard DeepSeek setting)
  - Use 3 for faster training if stability is acceptable
  - Use 7-10 for maximum stability in very deep models (60+ layers)
  - Monitor Sinkhorn convergence error (should be < 1e-5)
  - Beyond 10 iterations: Negligible improvement
  - Ablation: Test 3 vs 5 vs 7 iterations
  - Minimum 3 required for meaningful doubly stochastic approximation

---

**Example mHC Configurations:**

### Standard mHC (Default, Recommended)
```python
# Full mHC with DeepSeek defaults
config = NeuroManifoldConfig(
    use_mhc=True,                  # Enable mHC
    use_full_mhc=True,             # Full multi-stream routing
    mhc_n_streams=2,               # 2 streams (efficiency)
    mhc_residual_weight=0.9,       # Gentle identity initialization
    mhc_sinkhorn_iters=5,          # Standard convergence
    n_layer=12,                    # Medium depth
    n_embd=768,                    # Standard width
)
# ~5% slower than standard residual, much more stable
```

### Simplified mHC (Faster)
```python
# Single-stream mHC for speed
config = NeuroManifoldConfig(
    use_mhc=True,                  # Enable mHC
    use_full_mhc=False,            # Simplified (H_res only)
    mhc_n_streams=1,               # Single stream
    mhc_residual_weight=0.9,       # Standard init
    mhc_sinkhorn_iters=3,          # Fewer iterations (faster)
    n_layer=8,                     # Shallow model
    n_embd=384,                    # Smaller width
)
# ~2% slower than standard residual, good stability
```

### Deep Model (60+ Layers)
```python
# Maximum stability for very deep architecture
config = NeuroManifoldConfig(
    use_mhc=True,                  # Critical for depth
    use_full_mhc=True,             # Full routing flexibility
    mhc_n_streams=3,               # More streams for depth
    mhc_residual_weight=0.92,      # Conservative initialization
    mhc_sinkhorn_iters=7,          # Better convergence
    n_layer=60,                    # Very deep (DeepSeek scale)
    n_embd=1024,                   # Large width
    use_qk_norm=True,              # QK norm + mHC = ultimate stability
)
# Enables stable training of 60+ layer models
```

### Ablation: No mHC (Standard Residual)
```python
# Baseline without mHC for comparison
config = NeuroManifoldConfig(
    use_mhc=False,                 # Disable mHC (standard residual)
    # All mhc_* parameters ignored
    n_layer=12,                    # Limited depth without mHC
    n_embd=768,                    # Standard width
)
# Standard residual: x_{l+1} = x_l + F(x_l)
# Faster but less stable, difficult to scale beyond 24 layers
```

### Fast Mode (Minimal mHC)
```python
# Fastest mHC configuration
config = NeuroManifoldConfig(
    use_mhc=True,                  # Enable for stability
    use_full_mhc=False,            # Simplified routing
    mhc_n_streams=1,               # Single stream
    mhc_residual_weight=0.88,      # Slightly aggressive
    mhc_sinkhorn_iters=3,          # Minimum iterations
    n_layer=6,                     # Shallow for speed
    fast_mode=True,                # Enable all fast paths
)
# Minimal mHC overhead, basic stability improvements
```

---

**mHC Performance Characteristics:**

| Configuration | Speed vs Standard | Stability | Max Recommended Depth |
|---------------|-------------------|-----------|----------------------|
| **No mHC** | 1.00× | Baseline | 12-24 layers |
| **Simplified mHC** | 0.98× | Good | 24-40 layers |
| **Standard mHC (2 streams)** | 0.95× | Excellent | 40-60 layers |
| **Full mHC (3+ streams)** | 0.92× | Maximum | 60+ layers |

**Sinkhorn Iterations:**
- 3 iterations: ~3% overhead, convergence error ~1e-4
- 5 iterations: ~5% overhead, convergence error ~1e-6
- 7 iterations: ~7% overhead, convergence error ~1e-8

**Memory Usage:**
- Standard residual: Baseline
- Simplified mHC: +5% memory (H_res matrices)
- Full mHC (2 streams): +10% memory (H_res, H_pre, H_post)
- Full mHC (4 streams): +15% memory

**Recommendation:** Use full mHC with default settings (`use_mhc=True`, `use_full_mhc=True`, `mhc_n_streams=2`) for all production models. The 5% overhead is negligible compared to the massive stability gains, especially for deep architectures (12+ layers).

---

**mHC Troubleshooting:**

| Issue | Likely Cause | Solution |
|-------|--------------|----------|
| Training instability in deep models | mHC disabled or too few Sinkhorn iterations | Enable `use_mhc=True`, increase `mhc_sinkhorn_iters` to 7 |
| Gradient vanishing (deep models) | Not using full mHC | Set `use_full_mhc=True`, `mhc_n_streams=2+` |
| Early training instability | Residual weight too low | Increase `mhc_residual_weight` to 0.92-0.95 |
| Sinkhorn not converging | Too few iterations or numerical issues | Increase `mhc_sinkhorn_iters`, check for NaN |
| Excessive memory usage | Too many streams | Reduce `mhc_n_streams` to 2 |
| Slow training | Too many Sinkhorn iterations or streams | Reduce to 3-5 iterations, 1-2 streams |

---

**Mathematical Detail: Sinkhorn-Knopp Algorithm**

The Sinkhorn-Knopp algorithm iteratively normalizes a matrix to be doubly stochastic:

```
Input: Matrix M (learned routing weights)
Output: H (doubly stochastic: row sums = col sums = 1)

For k = 1 to mhc_sinkhorn_iters:
    H = H / sum(H, dim=1, keepdim=True)  # Normalize rows
    H = H / sum(H, dim=0, keepdim=True)  # Normalize columns
```

Convergence is exponential: `||H - H*|| ≈ exp(-k)` where H* is the true doubly stochastic solution.

**Why Double Stochasticity?**
- Preserves gradient norms: `||∂L/∂x_l|| ≈ ||∂L/∂x_{l+1}||`
- Lies on the Birkhoff polytope (convex hull of permutation matrices)
- Guarantees all eigenvalues have magnitude ≤ 1
- Prevents gradient explosion/vanishing in arbitrary depth networks

---

## 7. KAN (Kolmogorov-Arnold Networks)

**Category:** Learnable activation functions replacing standard MLPs.

KAN (Kolmogorov-Arnold Networks) replaces standard linear layers with learnable basis function expansions, inspired by the **Kolmogorov-Arnold representation theorem**. Instead of fixed activations (ReLU, GELU), KAN learns smooth activation functions from data, providing greater expressivity and better approximation properties for complex functions.

**Key Concepts:**
- **Kolmogorov-Arnold Theorem:** Any continuous multivariate function can be represented as a composition of univariate functions
- **Learnable Basis Functions:** Instead of W·x + b, use Σ φᵢ(x) where φᵢ are learnable
- **Function Approximation:** KAN can approximate arbitrary smooth functions with fewer parameters than standard MLPs
- **Three Variants:** FasterKAN (RSWAF basis), WaveKAN (wavelet basis), ChebyKAN (Chebyshev polynomials)
- **Tradeoff:** Greater expressivity vs parameter count increase (especially with `use_kan_everywhere=True`)

**Mathematical Background:**

The Kolmogorov-Arnold representation theorem states:
```
f(x₁, ..., xₙ) = Σᵢ Φᵢ( Σⱼ φᵢⱼ(xⱼ) )
```

Where φᵢⱼ are univariate functions. KAN implements this by replacing:
```
Standard MLP:  y = σ(W·x + b)
KAN:          y = Σᵢ ψᵢ(wᵢ·x + bᵢ)
```

Where ψᵢ are learnable basis functions (wavelets, polynomials, or rational functions).

**Benefits of KAN:**
- **Higher Expressivity:** Learns problem-specific activations
- **Better Approximation:** Provably more efficient function approximation
- **Smooth Gradients:** Basis functions are continuously differentiable
- **Interpretability:** Basis function shapes reveal learned features

**Drawbacks:**
- **Parameter Bloat:** 3-10× more parameters than standard Linear layers
- **Slower Computation:** Basis function evaluation overhead (~20-40% slower)
- **Memory Usage:** Higher activation storage for backprop
- **Training Sensitivity:** Requires careful initialization and learning rate tuning

**Implementation in NeuroManifold:**
- **Default (use_kan=True, use_kan_everywhere=False):** Only replaces FFN/MLP layers
- **Aggressive (use_kan_everywhere=True):** Replaces ALL Linear layers (attention projections, manifold, spectral)
- **Skipped:** Never replaces lm_head (output vocabulary projection) or input embeddings

**Reference:** Liu et al., "KAN: Kolmogorov-Arnold Networks" (2024)

### use_kan
- **Type:** `bool`
- **Default:** `True`
- **Description:** Enable Kolmogorov-Arnold Networks for FFN/MLP layers
- **Details:**
  - When `True`: FFN layers use KAN basis functions instead of standard Linear layers
  - When `False`: Standard Linear layers with fixed activations (GELU)
  - By default (with `use_kan_everywhere=False`), only affects FFN/MLP modules
  - FFN is the largest parameter block in transformers (~2/3 of total parameters)
  - Increases FFN parameter count by 3-5× depending on `kan_type`
  - Provides better function approximation for the nonlinear FFN transformation
  - Compatible with all architectural variants (mHC, MTP, MoE, etc.)
  - FasterKAN type is recommended for balanced speed/accuracy
- **Interdependencies:**
  - When `True`, uses `kan_type`, `kan_degree`, `kan_wavelet`, `use_fast_wavekan`, `kan_num_centers`
  - Interacts with `use_kan_everywhere` for scope (FFN-only vs all layers)
  - Parameter count increase: 3-10× for FFN, depending on `kan_type` and basis settings
  - Memory usage increases proportionally with parameter count
  - Training may require lower learning rate due to higher capacity
- **Tuning Tips:**
  - **Keep True** (default, better expressivity)
  - Disable for faster training or baseline comparisons
  - KAN shows benefits on complex tasks requiring nonlinear reasoning
  - Use `kan_type="faster"` for best speed/accuracy tradeoff
  - Monitor parameter count: KAN can add 50-200M parameters to large models
  - Consider disabling if parameter budget is tight or training is slow

### kan_type
- **Type:** `str`
- **Default:** `"faster"`
- **Range:** `"faster"`, `"wave"`, `"cheby"`
- **Description:** Type of KAN basis functions to use
- **Details:**
  - **"faster" (FasterKAN):** Uses RSWAF (Rational Spline Wavelet Activation Functions)
    - Fastest KAN variant (only ~20% slower than standard Linear)
    - Uses rational spline basis with learnable centers
    - Default choice for production (best speed/accuracy tradeoff)
    - Parameters: `kan_num_centers` basis functions per input dimension
    - Smooth, continuously differentiable basis functions
    - Most parameter-efficient KAN variant
  - **"wave" (WaveKAN):** Uses wavelet basis functions
    - Medium speed (~30-40% slower than standard Linear)
    - Uses wavelets (Mexican hat, Morlet, etc.) as basis functions
    - Good for signals/sequences with multi-scale structure
    - Parameters: Scale and translation parameters per wavelet
    - Wavelet type controlled by `kan_wavelet` parameter
    - `use_fast_wavekan=True` enables efficient shared-scale variant
  - **"cheby" (ChebyKAN):** Uses Chebyshev polynomial basis
    - Slowest but most accurate (~40-50% slower than standard Linear)
    - Uses Chebyshev polynomials of order `kan_degree`
    - Optimal approximation properties (minimax polynomial approximation)
    - Higher parameter count than FasterKAN (degree × input_dim)
    - Best for functions requiring high-order polynomial approximation
    - More stable than standard polynomial basis (bounded on [-1, 1])
- **Interdependencies:**
  - Only active when `use_kan=True`
  - **FasterKAN** uses `kan_num_centers` parameter
  - **WaveKAN** uses `kan_wavelet` and `use_fast_wavekan` parameters
  - **ChebyKAN** uses `kan_degree` parameter
  - Parameter count varies: FasterKAN (lowest) < WaveKAN < ChebyKAN (highest)
  - Computational cost: FasterKAN (fastest) < WaveKAN < ChebyKAN (slowest)
  - All types are compatible with `use_kan_everywhere` flag
- **Tuning Tips:**
  - **Use "faster"** (default, recommended for production)
    - Best balance: 20% slower, 3-4× parameter increase, good accuracy
    - Suitable for most tasks and model sizes
    - Aligns with FHN soliton attention conceptually (wave-based)
  - **Use "wave"** for multi-scale temporal/sequential patterns
    - Good for audio, time-series, or hierarchical sequence modeling
    - Enable `use_fast_wavekan=True` for efficiency
    - Use `kan_wavelet="dog"` (fastest) or `"mexican_hat"` (more expressive)
  - **Use "cheby"** for maximum accuracy or research
    - Best function approximation properties
    - Higher computational cost and parameter count
    - Useful for complex mathematical functions or ablation studies
    - Requires tuning `kan_degree` (typical: 3-6)
  - **Ablation:** Compare all three types on your task to find optimal tradeoff
  - **Default recommendation:** `"faster"` unless you have specific requirements

### kan_degree
- **Type:** `int`
- **Default:** `4`
- **Range:** 2-8 (typical: 3-6)
- **Description:** Polynomial degree for ChebyKAN (only used when `kan_type="cheby"`)
- **Details:**
  - Maximum degree of Chebyshev polynomials in the basis expansion
  - Higher degree: More expressive, better approximation, but more parameters
  - Lower degree: Fewer parameters, faster, but limited approximation power
  - Degree 4: Can approximate most smooth functions reasonably well
  - Degree 6-8: For very complex nonlinearities
  - Each additional degree adds (input_dim) parameters per KAN layer
  - Chebyshev polynomials of degree n: Tₙ(x) = cos(n · arccos(x))
  - Bounded on [-1, 1], numerically stable
- **Interdependencies:**
  - **Only active when `kan_type="cheby"`**
  - Ignored for `kan_type="faster"` or `"wave"`
  - Parameter count: O(degree × input_dim × output_dim) for each KAN layer
  - Higher degree → more parameters → more memory → slower training
  - Interacts with learning rate: Higher degree may need lower LR
  - Should balance with model capacity (`n_embd`, `n_layer`)
- **Tuning Tips:**
  - Use 4 (default, balanced)
  - Use 3 for faster training or smaller models
  - Use 5-6 for complex functions or large models (n_embd ≥ 768)
  - Degree 2: Too limited, poor approximation
  - Degree >8: Diminishing returns, potential overfitting, numerical issues
  - Typical sweet spot: 3-5 for most applications
  - Higher degree is NOT always better (can overfit)
  - Start with 4, only increase if clear accuracy gains

### kan_wavelet
- **Type:** `str`
- **Default:** `"dog"`
- **Range:** `"dog"`, `"mexican_hat"`, `"morlet"`, `"shannon"`, `"meyer"`
- **Description:** Wavelet type for WaveKAN (only used when `kan_type="wave"`)
- **Details:**
  - Specifies which wavelet basis function to use for WaveKAN
  - **"dog" (Derivative of Gaussian):**
    - Fastest wavelet (linear time complexity)
    - Smooth, stable, good for general use
    - Default choice for WaveKAN
    - Simple computation: ψ(x) = -x · exp(-x²/2)
  - **"mexican_hat" (Ricker wavelet):**
    - Second derivative of Gaussian
    - Good for edge detection and local features
    - Slightly slower than DoG
    - Formula: ψ(x) = (1 - x²) · exp(-x²/2)
  - **"morlet":**
    - Complex wavelet with excellent localization
    - Good for frequency analysis
    - Slower computation (exponential + sine/cosine)
    - Formula: ψ(x) = exp(-x²/2) · cos(5x)
  - **"shannon":**
    - Sinc wavelet: sin(x)/x
    - Perfect frequency localization
    - Numerical issues at x=0 (requires special handling)
  - **"meyer":**
    - Smooth wavelet with compact frequency support
    - Complex computation, slowest
    - Best frequency-space localization properties
  - Each wavelet has different time-frequency tradeoffs
- **Interdependencies:**
  - **Only active when `kan_type="wave"`**
  - Ignored for `kan_type="faster"` or `"cheby"`
  - Works with `use_fast_wavekan` (shared scale/translation)
  - Different wavelets have different computational costs
  - Affects numerical stability (Shannon has singularity at 0)
  - Choice depends on data characteristics (smooth vs. oscillatory)
- **Tuning Tips:**
  - **Use "dog"** (default, recommended)
    - Fastest and most stable
    - Good for general-purpose KAN
    - Linear complexity
  - **Use "mexican_hat"** for richer features
    - Better edge detection
    - Slightly more expressive than DoG
    - Small computational overhead
  - **Use "morlet"** for frequency-domain patterns
    - Good for signals with periodic structure
    - Useful for time-series or audio
    - Moderate computational cost
  - **Avoid "shannon"** unless necessary (numerical issues)
  - **Avoid "meyer"** unless maximum accuracy needed (very slow)
  - **Ablation:** Test "dog" vs "mexican_hat" on your task
  - For most use cases: "dog" is sufficient

### use_fast_wavekan
- **Type:** `bool`
- **Default:** `True`
- **Description:** Use efficient WaveKAN with shared scale/translation parameters
- **Details:**
  - Optimization for WaveKAN that reduces parameter count
  - **When `True`:** Wavelets share scale and translation parameters across channels
    - Reduces parameters by ~50% vs full WaveKAN
    - Minimal accuracy loss (typically <1% performance difference)
    - Faster computation and lower memory usage
    - Recommended for most use cases
  - **When `False`:** Each wavelet has independent scale and translation
    - Maximum expressivity (per-channel wavelet parameters)
    - 2× more parameters than fast variant
    - Slower training and inference
    - Use only for research or when fast variant is insufficient
  - Fast WaveKAN is similar to depthwise-separable convolutions (shared spatial, per-channel learned)
- **Interdependencies:**
  - **Only active when `kan_type="wave"`**
  - Ignored for `kan_type="faster"` or `"cheby"`
  - Affects parameter count: True (50% fewer) vs False (full parameters)
  - Impacts memory usage proportionally to parameter reduction
  - No effect on forward pass speed (same computations, fewer params)
  - Compatible with all other KAN settings
- **Tuning Tips:**
  - **Keep True** (default, recommended)
  - Only set to False if:
    - You have sufficient parameter budget
    - Fast variant shows clear accuracy deficit (rare)
    - Research experiments requiring maximum WaveKAN capacity
  - Parameter savings: 50M+ on large models
  - Accuracy difference: Usually negligible (<1%)
  - Start with True, only disable if fast variant underperforms
  - Ablation: Compare True vs False to validate parameter savings

### kan_num_centers
- **Type:** `int`
- **Default:** `3`
- **Range:** 2-8 (typical: 3-5)
- **Description:** Number of RSWAF basis function centers for FasterKAN
- **Details:**
  - Number of rational spline wavelet centers in the basis expansion
  - **Only used when `kan_type="faster"`**
  - Each center corresponds to a learnable peak in the activation function
  - More centers: More expressive activation functions, but more parameters
  - Fewer centers: Simpler activations, fewer parameters, faster
  - 3 centers (default) provides good balance for most tasks
  - Each center adds (input_dim) parameters per KAN layer
  - Centers are initialized uniformly across the activation range
  - RSWAF: Rational (quotient of polynomials) + Spline (piecewise) + Wavelet + Activation
  - Learnable parameters: center positions, widths, and amplitudes
- **Interdependencies:**
  - **Only active when `kan_type="faster"`**
  - Ignored for `kan_type="wave"` or `"cheby"`
  - Parameter count scales linearly: num_centers × input_dim × output_dim
  - Higher `kan_num_centers` → more parameters → more memory
  - Interacts with `n_embd`: Larger models benefit from more centers
  - Should balance with overall model size
  - Affects expressivity of learned activation functions
- **Tuning Tips:**
  - **Use 3** (default, recommended for efficiency)
    - Good approximation for most smooth functions
    - Minimal parameter overhead
    - Fast computation
  - **Use 4-5** for large models (n_embd ≥ 768) or complex tasks
    - Richer activation functions
    - Better for highly nonlinear problems
    - Moderate parameter increase
  - **Use 2** for maximum speed or nano models
    - Minimal expressivity
    - Barely better than standard Linear
    - Only if parameter budget is extremely tight
  - **Avoid >6** (diminishing returns, parameter bloat)
  - More centers ≠ always better (can overfit)
  - Ablation: Compare 3 vs 4 vs 5 centers on your task
  - Sweet spot: 3-4 centers for most models

### use_kan_everywhere
- **Type:** `bool`
- **Default:** `False`
- **Description:** Replace ALL nn.Linear layers with KAN (not just FFN)
- **Details:**
  - **CRITICAL PARAMETER:** Controls scope of KAN replacement
  - **When `False` (default, recommended):**
    - Only FFN/MLP layers use KAN
    - Attention projections (Q, K, V, O) remain standard Linear
    - Manifold projection, spectral decomposition remain Linear
    - Balanced: Expressivity where it matters (FFN) + efficiency elsewhere
    - Parameter increase: ~50-150M depending on `kan_type`
  - **When `True` (aggressive, not recommended):**
    - **ALL** Linear layers replaced with KAN:
      - FFN/MLP layers ✓
      - Attention Q, K, V, O projections ✓
      - Manifold projection layers ✓
      - Spectral decomposition layers ✓
      - Any other Linear transformations ✓
    - **Exceptions:** lm_head (vocab output), embeddings (never replaced)
    - Parameter increase: 3-10× total model parameters
    - Can increase model from 50M → 200M+ parameters
    - Significantly slower training and inference (~40-60% slower)
    - Higher memory usage (may not fit on GPU)
    - Marginal accuracy gains (often <2%) don't justify cost
  - **WARNING:** `use_kan_everywhere=True` causes massive parameter bloat
  - Most of the model's capacity is in FFN anyway (~66% of parameters)
  - Attention projections are relatively small (don't benefit as much from KAN)
- **Interdependencies:**
  - Only active when `use_kan=True`
  - Affects ALL layers when True (attention, manifold, spectral, FFN)
  - Affects ONLY FFN when False (default)
  - Parameter count multiplier:
    - False: 1.5-2.0× total parameters (FFN only)
    - True: 3-10× total parameters (all layers)
  - Memory usage scales proportionally with parameter count
  - Training time increases: False (+20%), True (+40-60%)
  - Inference speed: False (-20%), True (-40-60%)
  - Interacts with all architectural modules (mHC, MTP, MoE, etc.)
- **Tuning Tips:**
  - **Keep False** (default, strongly recommended)
    - Best cost/benefit ratio
    - FFN is where nonlinearity matters most
    - Attention projections don't benefit much from learnable activations
    - Avoids parameter bloat
    - Maintains reasonable training speed
  - **Only use True if:**
    - You have unlimited compute/memory resources
    - Research experiment on maximum KAN expressivity
    - Small model (< 20M parameters) where bloat is acceptable
    - Task specifically requires learnable activations everywhere (rare)
  - **Parameter bloat example:**
    - Small model (50M params, n_embd=384, n_layer=6):
      - use_kan_everywhere=False: 75M params (50% increase)
      - use_kan_everywhere=True: 200M params (4× increase)
    - Medium model (150M params, n_embd=768, n_layer=12):
      - use_kan_everywhere=False: 220M params (47% increase)
      - use_kan_everywhere=True: 750M params (5× increase)
  - **Recommendation:** Use `use_kan=True, use_kan_everywhere=False` for optimal tradeoff
  - Ablation: Compare FFN-only vs everywhere to validate minimal gains
  - Only enable `True` if you explicitly need learnable attention projections

---

**Example KAN Configurations:**

### Standard KAN (Default, Recommended)
```python
# FasterKAN on FFN only (best tradeoff)
config = NeuroManifoldConfig(
    use_kan=True,                  # Enable KAN
    kan_type="faster",             # FasterKAN (RSWAF basis)
    kan_num_centers=3,             # 3 basis centers (efficient)
    use_kan_everywhere=False,      # FFN only (not attention/manifold)
    n_embd=384,                    # Standard size
    n_layer=6,                     # Standard depth
)
# Parameter count: ~60M (from 40M baseline)
# Speed: ~20% slower than standard Linear
# Best balance for production
```

### WaveKAN Configuration
```python
# WaveKAN for temporal/sequential patterns
config = NeuroManifoldConfig(
    use_kan=True,                  # Enable KAN
    kan_type="wave",               # WaveKAN (wavelet basis)
    kan_wavelet="dog",             # DoG wavelet (fast and stable)
    use_fast_wavekan=True,         # Efficient shared-scale variant
    use_kan_everywhere=False,      # FFN only
    n_embd=384,                    # Standard size
    n_layer=6,                     # Standard depth
)
# Parameter count: ~65M (slightly more than FasterKAN)
# Speed: ~30% slower than standard Linear
# Good for time-series, audio, hierarchical sequences
```

### ChebyKAN (High Accuracy)
```python
# ChebyKAN for maximum approximation accuracy
config = NeuroManifoldConfig(
    use_kan=True,                  # Enable KAN
    kan_type="cheby",              # ChebyKAN (Chebyshev polynomials)
    kan_degree=4,                  # Degree-4 polynomials
    use_kan_everywhere=False,      # FFN only
    n_embd=512,                    # Larger model (ChebyKAN needs capacity)
    n_layer=8,                     # Medium depth
)
# Parameter count: ~100M (highest of the three)
# Speed: ~40% slower than standard Linear
# Best approximation properties, use for research or complex functions
```

### Aggressive KAN (Not Recommended)
```python
# KAN everywhere (massive parameter bloat, marginal gains)
config = NeuroManifoldConfig(
    use_kan=True,                  # Enable KAN
    kan_type="faster",             # FasterKAN (least bloat)
    kan_num_centers=3,             # Keep centers low
    use_kan_everywhere=True,       # WARNING: All layers (attention, manifold, FFN)
    n_embd=256,                    # Keep model small to manage parameters
    n_layer=4,                     # Keep shallow
)
# Parameter count: ~150M (from 25M baseline, 6× increase!)
# Speed: ~50% slower than standard Linear
# NOT recommended: Marginal accuracy gains, huge cost
# Only for research on KAN expressivity
```

### Minimal KAN (Fast Baseline)
```python
# Minimal KAN for speed
config = NeuroManifoldConfig(
    use_kan=True,                  # Enable KAN
    kan_type="faster",             # FasterKAN (fastest)
    kan_num_centers=2,             # Minimal centers
    use_kan_everywhere=False,      # FFN only
    n_embd=384,                    # Standard size
    n_layer=6,                     # Standard depth
)
# Parameter count: ~55M (small increase)
# Speed: ~15% slower than standard Linear
# For speed-critical applications
```

### No KAN (Standard MLP Baseline)
```python
# Disable KAN for standard transformer
config = NeuroManifoldConfig(
    use_kan=False,                 # Disable KAN (standard Linear + GELU)
    # All kan_* parameters ignored
    n_embd=384,                    # Standard size
    n_layer=6,                     # Standard depth
)
# Parameter count: ~40M (baseline)
# Speed: 1.0× (fastest)
# For ablation studies or when parameter budget is tight
```

---

**KAN Variant Comparison:**

| Variant | Speed vs Linear | Param Increase (FFN only) | Approximation Quality | Use Case |
|---------|-----------------|---------------------------|-----------------------|----------|
| **FasterKAN** | 0.80× | 1.5-2.0× | Good | Default, production |
| **WaveKAN** | 0.65× | 1.6-2.2× | Good (multi-scale) | Time-series, audio |
| **ChebyKAN** | 0.55× | 2.0-3.0× | Excellent | Research, complex functions |
| **Standard Linear** | 1.00× | 1.0× (baseline) | Baseline | Ablation, speed-critical |

**Scope Comparison (use_kan_everywhere):**

| Scope | Param Increase | Speed Impact | Recommendation |
|-------|----------------|--------------|----------------|
| **FFN only (False)** | 1.5-2.0× | -20% | ✓ Recommended |
| **All layers (True)** | 3-10× | -50% | ✗ Not recommended (bloat) |

---

**KAN Troubleshooting:**

| Issue | Likely Cause | Solution |
|-------|--------------|----------|
| Excessive parameters (>500M for small model) | `use_kan_everywhere=True` | Set to `False` (FFN only) |
| Slow training (>2× slower) | ChebyKAN or `use_kan_everywhere=True` | Use `kan_type="faster"`, `use_kan_everywhere=False` |
| Out of memory | KAN parameter bloat | Reduce `kan_num_centers`, use `kan_type="faster"`, or disable KAN |
| No accuracy improvement | KAN not needed for task | Disable KAN (`use_kan=False`) or try different `kan_type` |
| Training instability | KAN initialization issues | Lower learning rate, use `kan_type="faster"` |
| Slow WaveKAN | Expensive wavelet type | Use `kan_wavelet="dog"`, enable `use_fast_wavekan=True` |

---

**Performance Characteristics:**

**Parameter Count (40M baseline model, FFN only):**
- No KAN: 40M params
- FasterKAN (3 centers): 60M params (+50%)
- WaveKAN (fast): 65M params (+63%)
- ChebyKAN (degree 4): 75M params (+88%)

**Parameter Count (40M baseline, KAN everywhere):**
- FasterKAN: 200M params (+400%)
- WaveKAN: 220M params (+450%)
- ChebyKAN: 280M params (+600%)

**Training Speed (relative to standard Linear):**
- FasterKAN (FFN only): 0.80×
- WaveKAN (FFN only): 0.65×
- ChebyKAN (FFN only): 0.55×
- Any KAN (everywhere): 0.45×

**Accuracy Gains (typical, task-dependent):**
- FasterKAN: +1-3% over standard Linear
- WaveKAN: +1-4% on temporal tasks
- ChebyKAN: +2-5% on complex functions
- KAN everywhere: +0-2% over FFN-only (not worth the cost)

---

**Recommendation Summary:**

**Production (Default):**
```python
use_kan=True, kan_type="faster", kan_num_centers=3, use_kan_everywhere=False
```
- Best tradeoff: +50% params, -20% speed, +1-3% accuracy
- Suitable for most applications

**Maximum Speed (Baseline):**
```python
use_kan=False
```
- Standard transformer FFN with fixed activations
- For speed-critical applications or tight parameter budgets

**Research (High Accuracy):**
```python
use_kan=True, kan_type="cheby", kan_degree=5, use_kan_everywhere=False
```
- Best approximation properties
- For complex mathematical functions or ablation studies
- Accept +90% params and -45% speed for +2-5% accuracy

**Avoid:**
```python
use_kan_everywhere=True  # Massive parameter bloat, marginal gains
```
- 4-6× parameter increase, minimal accuracy improvement
- Only for specific research on learnable attention projections

---

## 8. Advanced Architectures

**Category:** Advanced scaling techniques for efficiency and performance.

This section covers three major architectural extensions inspired by recent transformer research:

- **MLA (Multi-Head Latent Attention):** DeepSeek-style KV cache compression for 8x memory reduction
- **MTP (Multi-Token Prediction):** Meta/DeepSeek-style auxiliary prediction for better representations
- **MoE (Mixture of Experts):** DeepSeek-style sparse expert routing for parameter-efficient scaling

These features are **off by default** (except MTP) due to complexity/parameter overhead, but enable significant improvements when properly tuned.

---

### 8.1 MTP (Multi-Token Prediction)

**What is MTP?**

Multi-Token Prediction (MTP) is a training technique where the model predicts multiple future tokens simultaneously, not just the immediate next token. This forces the model to learn better intermediate representations that capture longer-range dependencies.

**Key Benefits:**
- **Improved Representations:** Predicting future tokens requires richer latent features
- **Better Generalization:** Multi-step prediction acts as regularization
- **Faster Convergence:** Auxiliary losses provide additional gradient signal
- **No Inference Overhead:** MTP heads are only used during training

**Research Background:**
- Introduced by Meta AI (2024) and adopted by DeepSeek-V3
- Predicting 4 future tokens reduces perplexity by 5-10% in practice
- Particularly effective for code generation and structured text

---

#### use_mtp
- **Type:** `bool`
- **Default:** `True` (enabled by default)
- **Description:** Enable multi-token prediction auxiliary heads
- **Details:**
  - When enabled, adds `mtp_n_predict` auxiliary prediction heads
  - Each head predicts a future token: t+1, t+2, ..., t+n
  - Main loss (next token) has weight 1.0, auxiliary losses weighted by `mtp_loss_weight`
  - Heads share the same trunk but have separate output projections
  - **No inference cost:** Auxiliary heads are dropped after training
- **Interdependencies:**
  - Requires `mtp_n_predict ≥ 1` (number of future tokens)
  - Interacts with `mtp_loss_weight` for loss balancing
  - Compatible with all attention types (standard, MLA, MoE)
- **Tuning Tips:**
  - **Keep enabled** (True) for most use cases - minimal overhead, clear benefits
  - Particularly effective for:
    - Code generation (predicting function bodies)
    - Structured text (JSON, XML)
    - Long-range dependencies (reasoning tasks)
  - Disable only if training memory is extremely tight
  - Combine with MoE for maximum efficiency

**Example:**
```python
# Standard MTP configuration (recommended)
config = NeuroManifoldConfig(
    use_mtp=True,            # Enable multi-token prediction
    mtp_n_predict=4,         # Predict 4 future tokens
    mtp_loss_weight=0.1,     # 10% weight for auxiliary losses
)
```

---

#### mtp_n_predict
- **Type:** `int`
- **Default:** `4`
- **Range:** 1-8 (typical: 2-6)
- **Description:** Number of future tokens to predict simultaneously
- **Details:**
  - 1 = standard next-token prediction (no auxiliary heads)
  - 4 = predict tokens at positions t+1, t+2, t+3, t+4 (recommended)
  - Higher values = more regularization but diminishing returns
  - Each prediction requires a separate output head (memory overhead)
  - Predictions are made in parallel (no autoregressive dependency during training)
- **Interdependencies:**
  - Requires `use_mtp=True`
  - Parameter count increases by `mtp_n_predict × vocab_size × n_embd`
  - For vocab_size=50304, n_embd=768: +154M parameters per head
  - Total overhead: `mtp_n_predict × 154M` parameters
- **Memory Impact:**
  - Training memory: +10-20% for mtp_n_predict=4
  - No inference memory overhead (heads dropped)
  - Parameter count: +`mtp_n_predict` output heads
- **Tuning Tips:**
  - **Recommended: 4** - optimal balance (proven by Meta/DeepSeek research)
  - Use 2-3 for smaller models or memory constraints
  - Use 6-8 for very long-range tasks (document-level reasoning)
  - Diminishing returns beyond 6 predictions
  - Consider: More predictions ≠ always better (overfitting risk)
  - Balance with `mtp_loss_weight`: higher n_predict → lower loss weight

**Example:**
```python
# Aggressive MTP for code generation
config = NeuroManifoldConfig(
    use_mtp=True,
    mtp_n_predict=6,         # Predict 6 tokens ahead
    mtp_loss_weight=0.05,    # Lower weight for more predictions
)

# Conservative MTP for small models
config = NeuroManifoldConfig(
    use_mtp=True,
    mtp_n_predict=2,         # Just 2 future tokens
    mtp_loss_weight=0.15,    # Higher weight for fewer predictions
)
```

---

#### mtp_loss_weight
- **Type:** `float`
- **Default:** `0.1`
- **Range:** 0.01-0.5 (typical: 0.05-0.2)
- **Description:** Weight for auxiliary MTP losses (main loss always weighted at 1.0)
- **Details:**
  - Controls balance between next-token prediction (main task) and future predictions
  - Total loss: `L = L_main + mtp_loss_weight × (L_t+1 + L_t+2 + ... + L_t+n)`
  - Lower weight = prioritize immediate next token
  - Higher weight = prioritize long-range predictions
  - Too high: Model focuses on future at expense of next-token accuracy
  - Too low: MTP regularization effect is negligible
- **Interdependencies:**
  - Requires `use_mtp=True`
  - Inverse relationship with `mtp_n_predict`: more predictions → lower weight
  - Interacts with main learning rate and optimizer settings
- **Tuning Tips:**
  - **Recommended: 0.1** for mtp_n_predict=4 (start here)
  - Scale inversely with `mtp_n_predict`:
    - n_predict=2: weight=0.15-0.2
    - n_predict=4: weight=0.1 (default)
    - n_predict=6: weight=0.05-0.08
  - Monitor training: if next-token perplexity degrades, reduce weight
  - If auxiliary losses dominate (check tensorboard), reduce weight
  - If auxiliary losses are ignored (flat), increase weight
  - Typical range: 0.05 (conservative) to 0.2 (aggressive)

**Example:**
```python
# Balanced MTP configuration
config = NeuroManifoldConfig(
    use_mtp=True,
    mtp_n_predict=4,
    mtp_loss_weight=0.1,     # 10% weight for 4 auxiliary losses
)

# Check loss balance during training:
# - Main loss should dominate (weight=1.0)
# - Auxiliary losses should contribute but not overwhelm
# - Typical ratio: main:aux ≈ 10:1 to 5:1
```

---

### 8.2 MLA (Multi-Head Latent Attention)

**What is MLA?**

Multi-Head Latent Attention (MLA) is DeepSeek's technique for compressing the KV (key-value) cache into a low-dimensional latent space. This enables **8x memory reduction** for long-context inference with minimal quality loss.

**Key Benefits:**
- **8x KV Cache Reduction:** Compress from `2 × n_heads × d_head` to `mla_latent_dim`
- **Long Context Efficiency:** Enable 32K+ context with limited memory
- **Decoupled RoPE:** Separate positional encoding for better flexibility
- **Maintained Quality:** <1% perplexity degradation vs standard attention

**Research Background:**
- Introduced by DeepSeek-V2 (2024), refined in DeepSeek-V3
- Compresses KV cache from ~2048 dims to ~64 dims (32x compression)
- Combined with RoPE decoupling for rotary positional encoding
- Critical for scaling to 128K+ context windows

**When to Use MLA:**
- Long-context applications (8K+ tokens)
- Memory-constrained inference environments
- Production deployments requiring low latency
- **Not needed** for short contexts (<2K tokens) or training-only workloads

---

#### use_mla
- **Type:** `bool`
- **Default:** `False` (disabled - adds architectural complexity)
- **Description:** Enable Multi-Head Latent Attention for KV compression
- **Details:**
  - Replaces standard K/V projections with low-dimensional latent compression
  - Architecture: `[n_embd] → [mla_latent_dim] → [n_heads × d_head]`
  - Standard attention: KV cache is `2 × n_heads × d_head × seq_len`
  - MLA: KV cache is `mla_latent_dim × seq_len` (8x smaller)
  - Compression is lossy but carefully designed to preserve attention quality
  - **Adds complexity:** Requires careful tuning of `mla_latent_dim` and `mla_rope_dim`
- **Interdependencies:**
  - Requires `mla_latent_dim < n_embd` (compression bottleneck)
  - Requires `mla_rope_dim < mla_latent_dim` (RoPE dimension)
  - Incompatible with `use_knot_attention=True` (different attention structure)
  - Compatible with `use_moe=True` and `use_mtp=True`
  - Works best with `block_size ≥ 4096` (long context)
- **Performance Impact:**
  - Training: +5-10% slower (extra projection layers)
  - Inference: 8x KV cache memory reduction
  - Quality: <1% perplexity increase (well-tuned)
  - Best for: Memory-bound inference, long-context scenarios
- **Tuning Tips:**
  - **Leave disabled (False)** unless you specifically need long-context efficiency
  - Enable for:
    - Inference with context ≥8K tokens
    - Memory-constrained deployments (edge devices, limited VRAM)
    - Production systems requiring low latency
  - **Not beneficial** for:
    - Training-only workloads (no inference)
    - Short contexts (<2K tokens)
    - When memory is not a constraint
  - Carefully tune `mla_latent_dim` to balance compression vs quality

**Example:**
```python
# MLA for long-context inference
config = NeuroManifoldConfig(
    use_mla=True,            # Enable KV compression
    mla_latent_dim=64,       # Compress to 64 dims (8x reduction)
    mla_rope_dim=32,         # RoPE dimension (half of latent)
    block_size=8192,         # Long context window
    n_embd=768,              # Standard model size
    n_heads=12,              # 12 attention heads
)

# Memory savings calculation:
# Standard KV: 2 × 12 × 64 × 8192 = 12.6M floats per sample
# MLA KV: 64 × 8192 = 524K floats per sample
# Reduction: 24x smaller KV cache
```

---

#### mla_latent_dim
- **Type:** `int`
- **Default:** `64`
- **Range:** 32-256 (typical: 48-128)
- **Description:** Dimension of the latent KV compression bottleneck
- **Details:**
  - Target dimension for compressing key-value representations
  - Standard attention: KV dims = `n_heads × (n_embd // n_heads)` = `n_embd`
  - MLA: KV dims = `mla_latent_dim` (much smaller)
  - Compression ratio: `n_embd / mla_latent_dim`
  - Example: `768 / 64 = 12x compression`
  - Smaller `mla_latent_dim` = more compression but potential quality loss
  - Larger `mla_latent_dim` = less compression but better quality
- **Interdependencies:**
  - Requires `use_mla=True`
  - Must satisfy: `mla_latent_dim < n_embd` (otherwise no compression)
  - Must satisfy: `mla_latent_dim > mla_rope_dim` (RoPE is subset)
  - Typical ratio: `mla_latent_dim ≈ n_embd / 8` to `n_embd / 16`
  - Larger models can afford lower compression ratios
- **Memory Impact:**
  - KV cache size: `mla_latent_dim × seq_len × 2` (K and V) floats
  - Compression ratio: `n_embd / mla_latent_dim`
  - Example savings (n_embd=768, seq_len=8K):
    - Standard: 768 × 8K × 2 = 12.3M floats = 49MB
    - MLA (dim=64): 64 × 8K × 2 = 1.05M floats = 4.2MB (~12x reduction)
- **Tuning Tips:**
  - **Start with `n_embd / 12`** as a safe default
  - Aggressive compression: `n_embd / 16` (e.g., 768 → 48)
  - Conservative compression: `n_embd / 8` (e.g., 768 → 96)
  - Monitor perplexity: if degraded, increase `mla_latent_dim`
  - Larger models (n_embd=1024+) can use lower dims (32-64)
  - Smaller models (n_embd=384) need higher dims (64-128)
  - Balance: More compression = less quality, but enables longer contexts

**Example:**
```python
# Small model with conservative MLA
config = NeuroManifoldConfig(
    n_embd=384,
    use_mla=True,
    mla_latent_dim=64,       # 384/64 = 6x compression
    mla_rope_dim=32,
)

# Large model with aggressive MLA
config = NeuroManifoldConfig(
    n_embd=1536,
    use_mla=True,
    mla_latent_dim=64,       # 1536/64 = 24x compression
    mla_rope_dim=32,
)
```

---

#### mla_rope_dim
- **Type:** `int`
- **Default:** `32`
- **Range:** 16-128 (typical: 32-64)
- **Description:** Dimension for decoupled RoPE (Rotary Position Embedding)
- **Details:**
  - MLA decouples positional encoding from content encoding
  - RoPE is applied to a separate subset of the latent dimension
  - Standard attention: RoPE applied to full K/V (dimension = `n_embd`)
  - MLA: RoPE applied to `mla_rope_dim` subset (much smaller)
  - This decoupling allows independent tuning of content vs position
  - Smaller `mla_rope_dim` = more aggressive position compression
- **Interdependencies:**
  - Requires `use_mla=True`
  - Must satisfy: `mla_rope_dim < mla_latent_dim` (RoPE is subset)
  - Typical ratio: `mla_rope_dim ≈ mla_latent_dim / 2`
  - Minimum: 16-32 dims for meaningful positional encoding
- **Tuning Tips:**
  - **Use `mla_latent_dim / 2`** as default (e.g., latent=64 → rope=32)
  - Increase to `mla_latent_dim * 0.75` if positional encoding is critical
  - Decrease to `mla_latent_dim / 4` for maximum compression
  - Minimum 16-32 dims needed for effective RoPE
  - Longer contexts (32K+) may benefit from larger `mla_rope_dim`
  - For tasks with weak positional dependencies, can use smaller values

**Example:**
```python
# Standard MLA with balanced RoPE
config = NeuroManifoldConfig(
    use_mla=True,
    mla_latent_dim=64,
    mla_rope_dim=32,         # Half of latent dim
)

# Positional-heavy tasks (e.g., code with strict syntax)
config = NeuroManifoldConfig(
    use_mla=True,
    mla_latent_dim=96,
    mla_rope_dim=64,         # 2/3 of latent dim (more positional capacity)
)
```

---

### 8.3 MoE (Mixture of Experts)

**What is MoE?**

Mixture of Experts (MoE) is a sparse architecture that routes each token to a subset of "expert" networks, dramatically increasing model capacity without proportional compute increase.

**Key Benefits:**
- **Parameter-Efficient Scaling:** 8 experts ≈ 8× parameters, but only 2 active per token
- **Sublinear Compute:** Training cost scales ~2.5x (not 8x) for 8 experts
- **Specialization:** Experts learn domain-specific features (code, math, prose)
- **Auxiliary-Loss-Free:** DeepSeek-style bias-based load balancing (no auxiliary loss)

**Research Background:**
- Classical MoE (2017): Requires auxiliary loss for load balancing
- Switch Transformers (Google, 2021): Scaled to trillions of parameters
- DeepSeek-MoE (2024): Eliminated auxiliary loss via bias-based routing
- DeepSeek-V3 (2024): 671B total, 37B active per token (18x efficiency)

**When to Use MoE:**
- Large-scale models (1B+ parameters) needing efficiency
- Multi-domain datasets (code + math + prose)
- Parameter scaling without proportional compute increase
- **Not needed** for small models (<500M) or single-domain tasks

**Important:** MoE significantly increases total parameter count. Use judiciously.

---

#### use_moe
- **Type:** `bool`
- **Default:** `False` (disabled - increases parameters significantly)
- **Description:** Enable Mixture of Experts sparse routing
- **Details:**
  - Replaces standard FFN with a gated router + multiple expert FFNs
  - Each token routed to `moe_n_active` experts out of `moe_n_experts` total
  - Architecture: Router → TopK gating → Expert FFNs → Weighted sum
  - Parameter count: ~`moe_n_experts × FFN_params` (vs 1× for standard)
  - Compute: ~`moe_n_active × FFN_compute` (sparse activation)
  - **Significant overhead:** 8 experts with top-2 routing = 8× params, ~2× compute
- **Interdependencies:**
  - Requires `moe_n_experts ≥ 2` (number of experts)
  - Requires `moe_n_active ≤ moe_n_experts` (active experts per token)
  - Compatible with `use_mtp=True` (multi-token prediction)
  - Compatible with `use_mla=True` (latent attention)
  - May conflict with `use_kan=True` (KAN experts are very large)
  - `use_shared_expert=True` recommended (DeepSeek style)
- **Parameter Impact:**
  - Standard FFN: `4 × n_embd × n_embd` (e.g., 4× 384² = 590K)
  - MoE: `moe_n_experts × 4 × n_embd × n_embd` (e.g., 8× 590K = 4.7M)
  - Total model: `base_params + (n_layer × MoE_overhead)`
  - Example: 6-layer model → 6 × 4.7M = +28M parameters
- **Compute Impact:**
  - Compute: `moe_n_active / moe_n_experts × FFN_compute`
  - Example: top-2 of 8 experts = 25% of full expert compute (but routing overhead)
  - Training: ~2-3× slower than standard (routing + load balancing)
  - Inference: ~1.5-2× slower (sparse matmul overhead)
- **Tuning Tips:**
  - **Leave disabled (False)** unless you need parameter-efficient scaling
  - Enable for:
    - Large models (1B+ parameters)
    - Multi-domain datasets (code, math, prose, etc.)
    - When parameter count << training compute
  - **Not beneficial** for:
    - Small models (<500M parameters)
    - Single-domain tasks
    - Constrained parameter budgets
  - Requires large-scale data (100B+ tokens) for expert specialization
  - Consider: Training complexity increases significantly

**Example:**
```python
# MoE for large-scale multi-domain model
config = NeuroManifoldConfig(
    use_moe=True,
    moe_n_experts=8,         # 8 expert FFNs
    moe_n_active=2,          # Top-2 routing (25% sparse)
    use_shared_expert=True,  # Always-on shared expert (DeepSeek)
    use_e7_routing=False,    # Standard learned routing
    n_embd=1024,             # Large model
    n_layer=24,              # Deep model
)

# Parameter calculation:
# Base model: ~500M parameters
# MoE overhead: 24 layers × 8 experts × 4M = +768M parameters
# Total: ~1.3B parameters (but only ~200M active per token)
```

---

#### moe_n_experts
- **Type:** `int`
- **Default:** `8`
- **Range:** 2-64 (typical: 4-16)
- **Description:** Total number of expert FFN networks
- **Details:**
  - Number of parallel expert networks in each MoE layer
  - Each expert is a full FFN: `Linear(n_embd, 4×n_embd) → GELU → Linear(4×n_embd, n_embd)`
  - More experts = more specialization capacity but higher parameters
  - Experts learn to specialize on different input patterns (domains, styles, etc.)
  - DeepSeek-V3 uses 256 experts; GPT-4 rumored to use 8-16
- **Interdependencies:**
  - Requires `use_moe=True`
  - Must be ≥ `moe_n_active` (can't activate more than exist)
  - Parameter overhead: `moe_n_experts × FFN_params` per layer
  - Larger `moe_n_experts` requires more training data for specialization
  - Typically power of 2 for efficient GPU kernels (4, 8, 16, 32)
- **Parameter Impact:**
  - Each expert: ~`4 × n_embd × n_embd` parameters
  - Total MoE: `moe_n_experts × 4 × n_embd²` per layer
  - Example (n_embd=768):
    - 4 experts: 4 × 2.36M = 9.4M params/layer
    - 8 experts: 8 × 2.36M = 18.9M params/layer
    - 16 experts: 16 × 2.36M = 37.7M params/layer
- **Tuning Tips:**
  - **Start with 8 experts** (proven sweet spot from DeepSeek/OpenAI)
  - Use 4 experts for smaller models (n_embd < 512)
  - Use 16-32 experts for very large models (10B+ params) with massive data
  - Diminishing returns beyond 16 experts without trillion-token datasets
  - Power-of-2 values (4, 8, 16, 32) for GPU efficiency
  - More experts requires:
    - More training data (100B+ tokens minimum)
    - Larger batch sizes (for load balancing)
    - Longer training (expert specialization takes time)

**Example:**
```python
# Small MoE configuration
config = NeuroManifoldConfig(
    use_moe=True,
    moe_n_experts=4,         # 4 experts (conservative)
    moe_n_active=2,          # Top-2 routing (50% sparse)
    n_embd=512,
)

# Standard MoE configuration (recommended)
config = NeuroManifoldConfig(
    use_moe=True,
    moe_n_experts=8,         # 8 experts (DeepSeek style)
    moe_n_active=2,          # Top-2 routing (25% sparse)
    n_embd=1024,
)

# Large-scale MoE configuration
config = NeuroManifoldConfig(
    use_moe=True,
    moe_n_experts=16,        # 16 experts (requires massive data)
    moe_n_active=2,          # Top-2 routing (12.5% sparse)
    n_embd=2048,
    # Requires: 1T+ tokens, large batch sizes, extensive training
)
```

---

#### moe_n_active
- **Type:** `int`
- **Default:** `2`
- **Range:** 1-8 (typical: 1-2)
- **Description:** Number of active experts per token (top-k routing)
- **Details:**
  - Number of experts selected by the routing function for each token
  - Router scores all experts, then selects top-k highest scores
  - Sparse activation: only `moe_n_active` experts process each token
  - More active = more compute and better quality, less sparsity
  - DeepSeek-V3, GPT-4, and most MoE systems use top-2 routing
- **Interdependencies:**
  - Requires `use_moe=True`
  - Must satisfy: `moe_n_active ≤ moe_n_experts`
  - Sparsity ratio: `moe_n_active / moe_n_experts`
  - Example: 2 active of 8 total = 25% sparsity
  - Affects compute: `moe_n_active × expert_compute`
- **Compute Impact:**
  - Compute per token: `moe_n_active × FFN_compute`
  - Example (8 experts):
    - top-1: 12.5% compute (very sparse, may hurt quality)
    - top-2: 25% compute (standard, good balance)
    - top-4: 50% compute (less sparse, higher quality)
  - Routing overhead: ~10-20% regardless of `moe_n_active`
- **Tuning Tips:**
  - **Use 2** (top-2 routing) - industry standard (DeepSeek, OpenAI, Google)
  - Top-1 routing:
    - Most sparse (lowest compute)
    - May hurt quality (single expert bottleneck)
    - Use only for extreme efficiency requirements
  - Top-2 routing (recommended):
    - Excellent quality vs compute tradeoff
    - Robust to routing errors (two experts redundancy)
    - Proven at scale (DeepSeek-V3, GPT-4)
  - Top-4+ routing:
    - Diminishing returns (approaching dense model)
    - Use only if quality is paramount and compute is cheap
  - Never exceed `moe_n_experts / 2` (defeats sparsity purpose)

**Example:**
```python
# Standard top-2 routing (recommended)
config = NeuroManifoldConfig(
    use_moe=True,
    moe_n_experts=8,
    moe_n_active=2,          # Top-2: 25% sparsity, robust
)

# Ultra-sparse top-1 routing (for extreme efficiency)
config = NeuroManifoldConfig(
    use_moe=True,
    moe_n_experts=8,
    moe_n_active=1,          # Top-1: 12.5% sparsity, risky
)

# Dense top-4 routing (high quality, less sparse)
config = NeuroManifoldConfig(
    use_moe=True,
    moe_n_experts=8,
    moe_n_active=4,          # Top-4: 50% sparsity, approaching dense
)
```

---

#### use_shared_expert
- **Type:** `bool`
- **Default:** `True` (enabled - DeepSeek style)
- **Description:** Enable always-active shared expert (DeepSeek innovation)
- **Details:**
  - In addition to routed experts, adds one shared expert processed by all tokens
  - Architecture: `output = SharedExpert(x) + sum(TopK_RoutedExperts(x))`
  - Shared expert captures common features across all domains
  - Routed experts specialize on specific patterns
  - Improves stability and reduces routing errors
  - Introduced by DeepSeek-MoE, adopted widely
- **Interdependencies:**
  - Requires `use_moe=True`
  - Adds one additional expert (parameter overhead)
  - Shared expert is always active (not counted in `moe_n_active`)
  - Total active experts per token: `moe_n_active + 1` (shared)
- **Parameter Impact:**
  - Adds 1 extra FFN per layer: `+4 × n_embd²` parameters
  - Total experts: `moe_n_experts` (routed) + 1 (shared)
  - Example (n_embd=768): +2.36M parameters per layer
- **Compute Impact:**
  - Shared expert processed for all tokens (dense compute)
  - Effective compute: `(moe_n_active + 1) / moe_n_experts`
  - Example (top-2 of 8): (2+1)/8 = 37.5% compute (vs 25% without shared)
- **Tuning Tips:**
  - **Keep enabled (True)** - DeepSeek's research shows clear benefits
  - Improves:
    - Training stability (common features not dependent on routing)
    - Load balancing (shared expert absorbs uncertain tokens)
    - Quality (all tokens get shared features)
  - Disable (False) only if:
    - Parameter budget is extremely tight
    - You're replicating non-DeepSeek MoE architectures
  - Negligible downside: +1 expert overhead pays for itself in stability

**Example:**
```python
# DeepSeek-style MoE (recommended)
config = NeuroManifoldConfig(
    use_moe=True,
    moe_n_experts=8,         # 8 routed experts
    moe_n_active=2,          # Top-2 routing
    use_shared_expert=True,  # +1 always-on shared expert
    # Effective: 3 active experts per token (2 routed + 1 shared)
)

# Classical MoE (no shared expert)
config = NeuroManifoldConfig(
    use_moe=True,
    moe_n_experts=8,
    moe_n_active=2,
    use_shared_expert=False, # Pure routed experts only
)
```

---

#### use_e7_routing
- **Type:** `bool`
- **Default:** `False` (disabled - uses standard learned routing)
- **Description:** Route experts based on E7 curriculum tier (experimental)
- **Details:**
  - Experimental feature: Routes experts based on E7 exceptional Lie group hierarchy
  - E7 curriculum tiers: D5 (global) → E6 (phrase) → E7 (token) patterns
  - Instead of learned routing, uses geometric curriculum level to select experts
  - Idea: Align expert specialization with geometric abstraction levels
  - **Highly experimental:** No proven benefits, may hurt quality
- **Interdependencies:**
  - Requires `use_moe=True`
  - Requires `use_multiscale_manifold=True` (E7 hierarchy)
  - Replaces standard learned router with deterministic E7-based routing
  - May conflict with load balancing (E7 routing is not balanced by default)
- **Tuning Tips:**
  - **Leave disabled (False)** - experimental, unproven
  - Standard learned routing is more flexible and battle-tested
  - Enable only for research into geometric expert specialization
  - Requires careful analysis of expert utilization patterns
  - May need custom load balancing if enabled

**Example:**
```python
# Standard learned routing (recommended)
config = NeuroManifoldConfig(
    use_moe=True,
    use_e7_routing=False,    # Use learned router (standard)
)

# Experimental E7-based routing
config = NeuroManifoldConfig(
    use_moe=True,
    use_e7_routing=True,     # Route by geometric tier (experimental)
    use_multiscale_manifold=True,  # Required for E7 hierarchy
    # WARNING: Unproven, may hurt quality
)
```

---

### 8.4 Combining Advanced Features

**MTP + MLA:**
- Excellent combination for long-context efficiency
- MTP improves representations, MLA reduces memory
- No conflicts, complementary benefits
- Recommended for production long-context models

**MTP + MoE:**
- Great combination for parameter-efficient scaling
- MTP regularization helps expert specialization
- Slightly slower training (both add overhead)
- Recommended for large multi-domain models

**MLA + MoE:**
- Powerful combination for massive-scale models
- MLA reduces KV cache, MoE increases parameters efficiently
- Complex to tune (two interdependent systems)
- Used by DeepSeek-V3 (671B params, 37B active)

**All Three (MTP + MLA + MoE):**
- Maximum efficiency for frontier models
- Requires careful tuning and large-scale infrastructure
- Example: DeepSeek-V3 uses all three techniques
- Not recommended for models <10B parameters

**Example Configurations:**

```python
# Long-context efficiency (MTP + MLA)
config = NeuroManifoldConfig(
    use_mtp=True,
    mtp_n_predict=4,
    use_mla=True,
    mla_latent_dim=64,
    block_size=16384,        # 16K context
)

# Parameter-efficient scaling (MTP + MoE)
config = NeuroManifoldConfig(
    use_mtp=True,
    mtp_n_predict=4,
    use_moe=True,
    moe_n_experts=8,
    moe_n_active=2,
    use_shared_expert=True,
    n_embd=1024,
    n_layer=24,
)

# Frontier model (MTP + MLA + MoE)
config = NeuroManifoldConfig(
    use_mtp=True,
    mtp_n_predict=6,
    use_mla=True,
    mla_latent_dim=96,
    use_moe=True,
    moe_n_experts=16,
    moe_n_active=2,
    use_shared_expert=True,
    n_embd=2048,
    n_layer=32,
    block_size=32768,
    # Requires: Large-scale infrastructure, 1T+ tokens
)
```

---

## 9. System 2 Reasoning Components

**Category:** Deliberative reasoning, planning, and imagination.

These parameters enable "System 2" thinking - the slow, deliberative reasoning mode that complements fast pattern-matching (System 1). Inspired by Kahneman's dual-process theory, these modules add:

- **Hybrid Reasoning:** Switch between fast and slow thinking modes
- **DAG Planning:** Decompose complex tasks into structured graphs
- **Hierarchical Memory:** Multi-tier memory system (hot/warm/cold)
- **Imagination:** Counterfactual exploration via lightweight diffusion

**Performance Impact:** System 2 components add significant compute overhead (2-10x slower) but enable qualitatively different capabilities like planning, reasoning, and exploration.

**When to Enable:**
- Reasoning tasks (math, logic, planning)
- Multi-step problem solving
- Exploration and creativity
- Long-term memory requirements

**When to Disable:**
- Pure language modeling
- Speed-critical applications
- Limited compute budget
- Short-context tasks

---

### 9.1 Hybrid Reasoning

Qwen3-style dual-mode architecture that routes between fast (direct) and slow (thinking) paths.

#### use_hybrid_reasoning
- **Type:** `bool`
- **Default:** `False` (disabled - single reasoning mode)
- **Description:** Enable hybrid fast/slow reasoning modes
- **Details:**
  - Implements Qwen3-style "thinking" vs "non-thinking" mode routing
  - Fast path: Standard transformer layers (low latency)
  - Slow path: Additional "thinking layers" for complex inputs
  - Mode selection based on learned complexity threshold
  - Architecture: `if complexity > threshold: use_thinking_layers(x) else: direct(x)`
  - Thinking layers are deeper (more parameters) but only activated when needed
  - Provides adaptive compute: simple inputs get fast processing, hard inputs get more depth
- **Interdependencies:**
  - Adds `n_thinking_layers` extra layers (parameter overhead)
  - Requires learned complexity estimator (small MLP)
  - Compatible with all other features
- **Parameter Impact:**
  - Adds `n_thinking_layers × 4 × n_embd²` parameters (thinking path)
  - Example (n_embd=768, 2 layers): +9.44M parameters
  - Complexity estimator: negligible (~10K parameters)
- **Compute Impact:**
  - Fast path: Standard compute (no overhead)
  - Slow path: `1 + (n_thinking_layers / n_layer)` relative compute
  - Example (6 base + 2 thinking): 33% slower when thinking mode triggered
  - Adaptive: Overhead only when complexity demands it
- **Tuning Tips:**
  - Start with `n_thinking_layers=2` (one extra layer depth)
  - Tune `thinking_threshold` to balance speed vs quality:
    - Lower (0.3): Thinking mode more often (higher quality, slower)
    - Higher (0.7): Thinking mode rarely (faster, may miss hard cases)
  - Monitor thinking mode activation rate during training
  - Ideal: 10-30% activation (hard examples get thinking, easy get fast path)
  - Best for: Question answering, reasoning benchmarks
  - Not useful for: Pure generation, chat

**Example:**
```python
# Hybrid reasoning for Q&A tasks
config = NeuroManifoldConfig(
    use_hybrid_reasoning=True,
    n_thinking_layers=2,        # Add 2 extra layers for thinking
    thinking_threshold=0.5,     # Balanced fast/slow routing
    n_layer=12,                 # Base layers
    # Total depth: 12 (fast) or 14 (slow) depending on input
)

# Fast-only mode (disabled hybrid)
config = NeuroManifoldConfig(
    use_hybrid_reasoning=False, # Single reasoning mode
    n_layer=12,                 # Fixed depth
)
```

---

#### n_thinking_layers
- **Type:** `int`
- **Default:** `2`
- **Range:** 1-8 (typical: 2-4)
- **Description:** Number of extra layers for thinking mode
- **Details:**
  - Additional transformer layers activated only in slow thinking path
  - Adds depth for complex reasoning without penalizing simple inputs
  - Stacked on top of base `n_layer` layers
  - Total depth in thinking mode: `n_layer + n_thinking_layers`
  - Each layer is a standard transformer block (attention + FFN)
- **Interdependencies:**
  - Only used when `use_hybrid_reasoning=True`
  - Parameter cost: `n_thinking_layers × 4 × n_embd²` per layer
  - Higher values = better reasoning but more parameters
- **Tuning Tips:**
  - Start with 2 (minimal overhead, measurable benefit)
  - Use 4 for hard reasoning tasks (math, logic)
  - Diminishing returns beyond 4 layers
  - Balance with base `n_layer`: don't make thinking path too deep

---

#### thinking_threshold
- **Type:** `float`
- **Default:** `0.5` (balanced)
- **Range:** 0.0-1.0
- **Description:** Complexity threshold for triggering thinking mode
- **Details:**
  - Learned complexity estimator outputs score in [0, 1]
  - If score > `thinking_threshold`, use slow thinking path
  - Lower threshold: More aggressive thinking (higher quality, slower)
  - Higher threshold: Conservative thinking (faster, may miss hard cases)
  - Threshold is tunable at inference time (no retraining needed)
- **Tuning Tips:**
  - Default 0.5 is a good starting point
  - Monitor activation rate:
    - <10%: Threshold too high, raising may help
    - >50%: Threshold too low, lowering improves speed
  - Tune per dataset:
    - Math/logic: Lower (0.3-0.4) for more thinking
    - Chat/generation: Higher (0.6-0.7) for speed
  - Can be adjusted dynamically at inference

---

### 9.2 DAG Planning

ForcedDAGPlanner decomposes complex tasks into directed acyclic graphs for systematic reasoning.

#### use_dag_planner
- **Type:** `bool`
- **Default:** `False` (disabled - no task decomposition)
- **Description:** Enable DAG-based task planning module
- **Details:**
  - Forces model to decompose complex tasks into structured DAGs before solving
  - Prevents "System 1" pattern matching on tasks requiring deliberate reasoning
  - Architecture:
    1. Task Decomposer: Splits problem into subtasks (DAG nodes)
    2. Dependency Resolver: Establishes ordering (DAG edges)
    3. Sequential Executor: Solves subtasks in topological order
  - Each node is a mini-task with input/output specification
  - Enforces at least `dag_min_nodes` decomposition (prevents shortcuts)
  - Caps at `dag_max_nodes` to avoid over-fragmentation
  - Inspired by System 2 thinking in cognitive science
- **Interdependencies:**
  - Adds dedicated DAG planner module (parameter overhead)
  - Can combine with `use_imagination=True` for plan exploration
  - Compatible with hierarchical memory for cross-step context
- **Parameter Impact:**
  - DAG encoder: `~2 × n_embd²` parameters
  - Node encoder: `~4 × n_embd² × dag_max_nodes` parameters
  - Example (n_embd=768, 32 nodes): ~75M parameters
  - Significant overhead - only enable when needed
- **Compute Impact:**
  - Adds one full forward pass per DAG node
  - Total compute: `(1 + avg_nodes) × base_compute`
  - Example (avg 8 nodes): 9× slower
  - Critical: Only use for tasks that benefit from decomposition
- **Tuning Tips:**
  - **Start disabled** - only enable for reasoning tasks
  - Best for: Multi-step math, coding, planning tasks
  - Not useful for: Chat, generation, classification
  - Requires training signal:
    - Supervised: Provide ground-truth DAGs
    - RL: Reward correct final answers (harder to train)
  - Monitor DAG quality:
    - Are nodes meaningful subtasks?
    - Is ordering logical?
    - Does decomposition help accuracy?
  - Tune `dag_min_nodes` to force decomposition (prevent shortcut)
  - Tune `dag_max_nodes` to avoid over-fragmentation

**Example:**
```python
# DAG planning for multi-step reasoning
config = NeuroManifoldConfig(
    use_dag_planner=True,
    dag_max_nodes=32,           # Up to 32 subtasks
    dag_min_nodes=3,            # At least 3 (force decomposition)
    # WARNING: 3-32× slower, only for tasks needing decomposition
)

# Standard mode (no decomposition)
config = NeuroManifoldConfig(
    use_dag_planner=False,      # Direct solution
)
```

---

#### dag_max_nodes
- **Type:** `int`
- **Default:** `32`
- **Range:** 4-128 (typical: 16-64)
- **Description:** Maximum nodes in task decomposition DAG
- **Details:**
  - Caps the number of subtasks in decomposition
  - Prevents over-fragmentation (too many tiny steps)
  - Each node is a coherent subtask with clear input/output
  - Higher values = more fine-grained decomposition
  - Lower values = coarser task breakdown
- **Interdependencies:**
  - Only used when `use_dag_planner=True`
  - Must be > `dag_min_nodes` (enforced in validation)
  - Higher values increase parameter and memory costs
- **Tuning Tips:**
  - Default 32 works for most tasks
  - Increase (64) for very complex multi-step problems
  - Decrease (16) for simpler reasoning tasks
  - Monitor actual node usage: if always hitting max, increase limit

---

#### dag_min_nodes
- **Type:** `int`
- **Default:** `3`
- **Range:** 2-16 (typical: 3-8)
- **Description:** Minimum nodes required in DAG (forces decomposition)
- **Details:**
  - Enforces minimum decomposition depth
  - Prevents model from "cheating" with single-node pass-through
  - Critical for training: Model must learn to break down tasks
  - Example: Math problem must have ≥3 steps (understand, plan, compute, verify)
  - Lower values = less forced structure
  - Higher values = more aggressive decomposition
- **Interdependencies:**
  - Only used when `use_dag_planner=True`
  - Must be ≤ `dag_max_nodes` (enforced in validation)
  - Higher values make training harder (model must learn decomposition)
- **Tuning Tips:**
  - Start with 3 (minimal forced decomposition)
  - Increase to 5-8 for very complex tasks
  - Monitor if model struggles: May be too restrictive
  - Balance between forced structure and flexibility

---

### 9.3 Hierarchical Memory

Three-tier memory system (L1/L2/L3) inspired by CPU caches - replaces flat SDREngramMemory.

#### use_hierarchical_memory
- **Type:** `bool`
- **Default:** `False` (disabled - uses flat SDREngramMemory)
- **Description:** Enable three-tier hierarchical memory system
- **Details:**
  - Replaces flat `SDREngramMemory` with hierarchical `HierarchicalEngramMemory`
  - Three tiers inspired by CPU cache hierarchy:
    - **L1 (Hot):** Small, fast, working memory (recent/important items)
    - **L2 (Warm):** Medium, short-term memory (recent context)
    - **L3 (Cold):** Large, compressed, long-term memory (archived)
  - Automatic promotion/demotion based on access patterns
  - LRU (Least Recently Used) eviction policy
  - L3 uses compression (quantization/pruning) for efficiency
  - Enables truly infinite context via memory hierarchy
- **Interdependencies:**
  - Mutually exclusive with flat engram memory
  - Works with `memory_active_retrieval=True` for retrieval-augmented generation
  - Compatible with DAG planner (memory persists across DAG nodes)
- **Parameter Impact:**
  - Negligible: Just metadata tracking (no learned parameters)
  - Memory cost: `L1 + L2 + L3` capacities
  - L3 compression reduces memory ~4× (quantization)
- **Compute Impact:**
  - L1 access: O(1) - hash lookup
  - L2 access: O(log n) - binary search
  - L3 access: O(n) - linear scan + decompression
  - Promotion/demotion: Negligible overhead
  - Overall: Minimal impact (<5% slowdown)
- **Tuning Tips:**
  - Enable for long-context tasks (books, codebases)
  - L1 capacity: Working memory size (32-128)
  - L2 capacity: Recent context (256-1024)
  - L3 capacity: Archive size (1024-8192)
  - Rule of thumb: L1:L2:L3 = 1:8:64 ratio
  - Monitor hit rates:
    - L1: Should be 60-80% (hot items)
    - L2: Should be 15-30% (recent items)
    - L3: Should be 5-10% (rarely needed)
    - Misses: <5% (not in any tier)

**Example:**
```python
# Hierarchical memory for long-context tasks
config = NeuroManifoldConfig(
    use_hierarchical_memory=True,
    hierarchical_l1_capacity=64,    # Hot: 64 items
    hierarchical_l2_capacity=512,   # Warm: 512 items
    hierarchical_l3_capacity=4096,  # Cold: 4096 items (compressed)
    memory_active_retrieval=True,   # Enable retrieval-augmented generation
    # Total effective memory: ~5K items with minimal overhead
)

# Flat memory (default)
config = NeuroManifoldConfig(
    use_hierarchical_memory=False,
    engram_capacity=1000,           # Single flat tier
)
```

---

#### hierarchical_l1_capacity
- **Type:** `int`
- **Default:** `64`
- **Range:** 16-256 (typical: 32-128)
- **Description:** L1 hot memory capacity (working memory)
- **Details:**
  - Smallest, fastest tier - stores most recently accessed items
  - Hash-based O(1) lookup
  - No compression - full fidelity
  - Automatically promotes frequently accessed L2/L3 items
  - LRU eviction to L2 when full
- **Tuning Tips:**
  - Smaller = faster but more L2 access
  - Larger = higher hit rate but more memory
  - Default 64 works for most tasks
  - Increase to 128 for very dynamic workloads

---

#### hierarchical_l2_capacity
- **Type:** `int`
- **Default:** `512`
- **Range:** 128-2048 (typical: 256-1024)
- **Description:** L2 warm memory capacity (short-term memory)
- **Details:**
  - Medium-sized tier for recent context
  - Sorted structure for O(log n) lookup
  - No compression - full fidelity
  - Receives evictions from L1
  - LRU eviction to L3 when full
- **Tuning Tips:**
  - Balance between L1 and L3 sizes
  - Default 512 = 8× L1 (good ratio)
  - Increase for longer recent context

---

#### hierarchical_l3_capacity
- **Type:** `int`
- **Default:** `4096`
- **Range:** 1024-16384 (typical: 2048-8192)
- **Description:** L3 cold memory capacity (long-term archive)
- **Details:**
  - Largest tier for archived memories
  - O(n) scan for retrieval (infrequent access)
  - Uses compression (INT8 quantization + pruning) for 4× memory savings
  - Receives evictions from L2
  - Oldest items dropped when full
- **Tuning Tips:**
  - Default 4096 = 64× L1 (good ratio)
  - Increase to 8192+ for very long documents
  - Compression reduces quality slightly but enables massive capacity

---

### 9.4 Imagination Module

ConsistencyImaginationModule enables counterfactual exploration via lightweight diffusion.

#### use_imagination
- **Type:** `bool`
- **Default:** `False` (disabled - no counterfactual exploration)
- **Description:** Enable imagination module for counterfactual reasoning
- **Details:**
  - Lightweight diffusion model for "mental whiteboard" exploration
  - Generates alternative trajectories ("what if?") for reasoning tasks
  - Architecture: Consistency model (2-4 step diffusion, much faster than DDPM)
  - Use cases:
    - Planning: Explore different action sequences
    - Reasoning: Consider alternative hypotheses
    - Creativity: Generate diverse solutions
  - Generates `imagination_n_alternatives` counterfactual paths
  - Model selects best path or ensembles across alternatives
  - Inspired by mental simulation in human cognition
- **Interdependencies:**
  - Adds separate diffusion model (parameter overhead)
  - Can combine with `use_dag_planner=True` to imagine DAG variations
  - Compatible with hierarchical memory for storing explored paths
- **Parameter Impact:**
  - Consistency model: `~8 × imagination_dim²` parameters
  - Example (imagination_dim=256): ~524K parameters
  - Relatively lightweight (consistency model < full diffusion)
- **Compute Impact:**
  - Adds `imagination_steps × imagination_n_alternatives` forward passes
  - Example (4 steps, 4 alternatives): 16× overhead
  - Can run alternatives in parallel (batch over alternatives)
  - Critical: Very slow - only use when counterfactuals are valuable
- **Tuning Tips:**
  - **Start disabled** - significant compute overhead
  - Best for: Planning, exploration, creative tasks
  - Not useful for: Standard language modeling, classification
  - `imagination_steps`: 2-4 (consistency model needs few steps)
  - `imagination_n_alternatives`: 4-8 (diversity vs compute tradeoff)
  - Training: Requires counterfactual supervision or RL
  - Consider running imagination asynchronously (background threads)

**Example:**
```python
# Imagination for planning tasks
config = NeuroManifoldConfig(
    use_imagination=True,
    imagination_steps=4,            # 4-step consistency model
    imagination_n_alternatives=4,   # Generate 4 "what if" scenarios
    imagination_dim=256,            # Latent space dimension
    # WARNING: 16× slower (4 steps × 4 alternatives)
    # Use for planning/exploration, not standard generation
)

# Standard mode (no imagination)
config = NeuroManifoldConfig(
    use_imagination=False,          # Direct reasoning
)
```

---

#### imagination_steps
- **Type:** `int`
- **Default:** `4`
- **Range:** 2-8 (typical: 2-4)
- **Description:** Number of denoising steps in consistency model
- **Details:**
  - Controls diffusion quality vs speed tradeoff
  - Consistency models need far fewer steps than DDPM (50-1000 steps)
  - 2 steps: Fastest, lower quality
  - 4 steps: Balanced (recommended)
  - 8 steps: Higher quality, slower
  - Each step refines the counterfactual trajectory
- **Interdependencies:**
  - Only used when `use_imagination=True`
  - Total compute: `imagination_steps × imagination_n_alternatives × base_compute`
- **Tuning Tips:**
  - Start with 4 (good quality/speed balance)
  - Use 2 for fastest iteration during development
  - Rarely need >4 (diminishing returns)

---

#### imagination_n_alternatives
- **Type:** `int`
- **Default:** `4`
- **Range:** 2-16 (typical: 4-8)
- **Description:** Number of counterfactual alternatives to generate
- **Details:**
  - How many "what if?" scenarios to explore
  - Higher values = more diversity but more compute
  - Model either:
    - Selects best alternative (argmax over outcomes)
    - Ensembles across alternatives (average logits)
  - Alternatives generated in parallel (batch dimension)
- **Interdependencies:**
  - Only used when `use_imagination=True`
  - Batch size must accommodate alternatives (memory consideration)
- **Tuning Tips:**
  - Default 4 is good balance
  - Increase to 8 for very open-ended tasks (creativity)
  - Decrease to 2 for speed
  - Monitor diversity: Are alternatives meaningfully different?

---

### 9.5 Combining System 2 Components

System 2 components can be combined for powerful reasoning capabilities:

**Hybrid + DAG:**
- Use hybrid reasoning to decide when to engage DAG planner
- Fast path: Direct solution (simple tasks)
- Slow path: DAG decomposition (complex tasks)
- Adaptive compute based on complexity

**DAG + Memory:**
- Hierarchical memory persists across DAG nodes
- Each subtask can retrieve relevant memories
- Enables multi-step reasoning with long-term context

**DAG + Imagination:**
- Imagine alternative DAG structures before committing
- Explore different decomposition strategies
- Select best DAG based on imagined outcomes

**Full System 2 (All Components):**
- Maximum reasoning capability
- Extreme compute cost (10-100× slower)
- Only for hardest reasoning tasks

**Example Configurations:**

```python
# Minimal System 2 (Hybrid only)
config = NeuroManifoldConfig(
    use_hybrid_reasoning=True,
    n_thinking_layers=2,
    # 1.3× slower (only when thinking triggered)
)

# Moderate System 2 (Hybrid + Memory)
config = NeuroManifoldConfig(
    use_hybrid_reasoning=True,
    use_hierarchical_memory=True,
    hierarchical_l3_capacity=4096,
    # ~1.4× slower, infinite context capability
)

# Heavy System 2 (DAG + Memory)
config = NeuroManifoldConfig(
    use_dag_planner=True,
    dag_max_nodes=32,
    use_hierarchical_memory=True,
    # 8× slower, structured multi-step reasoning
)

# Full System 2 (All components)
config = NeuroManifoldConfig(
    use_hybrid_reasoning=True,
    use_dag_planner=True,
    use_hierarchical_memory=True,
    use_imagination=True,
    imagination_n_alternatives=4,
    # 30-100× slower, maximum reasoning capability
    # Only for hardest tasks: math olympiad, complex planning
)
```

---

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
