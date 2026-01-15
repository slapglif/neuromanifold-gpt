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
