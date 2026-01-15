# NeuroManifoldGPT

**A Neuromorphic Manifold-Constrained Language Model**

NeuroManifoldGPT is an experimental transformer architecture that synthesizes insights from neuroscience, topology, and theoretical physics to create a more efficient and biologically-plausible language model. Unlike standard GPT models, this implementation replaces traditional attention with **FitzHugh-Nagumo (FHN) soliton dynamics**, uses **Sparse Distributed Representations (SDR)** for semantic encoding, constrains information flow through **manifold projections**, and employs **topological knot theory** for sparse attention gating.

This is **not** standard nanoGPT—it's a complete architectural reimagining built on the nanoGPT foundation.

---

## Architecture Overview

NeuroManifoldGPT implements the **Kaufmann Trifecta Attention Model**, a unified theory of efficient attention that combines three fundamental mechanisms:

### Core Innovations

1. **FHN Attention (Soliton Dynamics)**
   - Replaces standard softmax attention with FitzHugh-Nagumo neural dynamics
   - Models attention as propagating soliton waves (acoustic density waves) through a phase-transitioning membrane
   - Achieves O(N) complexity along propagation paths instead of O(N²) broadcast
   - Implements IMEX (Implicit-Explicit) solver with balanced partitioning for stability

2. **SDR Encoding (Sparse Distributed Representations)**
   - Semantic Folding Encoder projects tokens into high-dimensional sparse binary space
   - SDR Engram Memory system provides content-addressable retrieval with "breadcrumb" trails
   - Semantic Retina extracts context-aware features from SDR patterns
   - Enables robust, noise-tolerant semantic representations (inspired by Jeff Hawkins' HTM)

3. **WaveKAN (Kolmogorov-Arnold Networks)**
   - Replaces standard MLP with wavelet-based KAN for "rugged fitness landscape" navigation
   - Uses Difference of Gaussians (DoG) or Mexican Hat wavelets
   - Provides tunable non-linearity to shape embedding space topology
   - Implements the "Adjacent Possible" constraint (Stuart Kauffman complexity theory)

4. **Knot Attention (Topological Gating)**
   - Uses discrete linking numbers to determine semantic entanglement between tokens
   - Hard gates attention to O(N·k) where k is the number of "topologically linked" neighbors
   - Based on Louis Kauffman's knot theory—semantically unlinked concepts have zero interaction
   - Projects token trajectories to 3D manifold subspace for linking number computation

5. **Manifold Projections**
   - Projects embeddings onto learned geometric manifolds
   - Constrains token trajectories to follow manifold geodesics
   - Inspired by E7 Lie group manifolds (133-dimensional exceptional Lie group)
   - Future: Geometric Algebra (Clifford) rotors for efficient Lie group approximation

6. **Spectral Decomposition**
   - Graph Laplacian spectral analysis of token connectivity
   - Identifies community structure in semantic space
   - Balances FHN input currents using Karmarkar-Karp number partitioning
   - Enables larger ODE solver timesteps for faster computation

7. **Manifold Hyper-Connections (mHC)**
   - Multi-stream residual connections constrained to manifold geometry
   - Replaces simple residual connections with manifold-aware information flow
   - Implements parallel processing streams with geometric routing

8. **E7 Curriculum & MLA**
   - E7-inspired curriculum learning (progressive complexity on rugged fitness landscape)
   - Multi-Layer Aggregation (MLA) for hierarchical feature fusion
   - Ramanujan positional embeddings using discrete prime frequencies

---

## Key Components

### Attention Mechanisms
- **FHNAttention**: Core FitzHugh-Nagumo dynamics with voltage (v) and recovery (w) variables
- **FHNDynamics**: Standalone FHN oscillator implementing reaction-diffusion equations
- **KnotAttention**: Topological linking number computation for sparse gating
- **KaufmannAttention**: Combined FHN + Knot + WaveKAN unified attention

### Manifold & Geometry
- **ManifoldProjection**: Projects embeddings onto learned manifold surfaces
- **SpectralDecomposition**: Graph Laplacian eigendecomposition for community detection
- **SpectralPartitioner**: Karmarkar-Karp balanced partitioning for ODE stability

### SDR & Memory
- **SDROperations**: Core SDR operations (union, intersection, threshold, distance)
- **SDREngramMemory**: Hebbian-style associative memory with breadcrumb trails
- **SemanticFoldingEncoder**: Encodes tokens into high-dimensional SDR space
- **SemanticRetina**: Extracts contextual features from SDR patterns
- **ContextEncoder**: Hierarchical context aggregation

### Hyper-Connections
- **HyperConnections**: Manifold-constrained multi-stream residuals (mHC)
- **Residual**: Fallback simple residual connections

---

## Theoretical Foundations

### The Kaufmann Trifecta

This architecture synthesizes the work of three visionaries named Kaufmann/Kauffman:

#### 1. Konrad Kaufmann (Thermodynamics & Solitons)
- **Theory**: Nerve impulses are solitons (acoustic density waves) in a 2D phase-transitioning membrane
- **Role**: Attention as propagating wave of information density
- **Mechanism**: FHN Dynamics simulating phase transition
- **Efficiency**: Solitons are lossless; information moves O(N) not O(N²)

#### 2. Stuart Kauffman (Complexity & Fitness)
- **Theory**: Evolution on rugged fitness landscapes at the "Edge of Chaos"
- **Role**: Embedding space is a landscape; model navigates to find "fittest" next token
- **Mechanism**: WaveKAN provides rugged, tunable non-linearity
- **Efficiency**: "Adjacent Possible" constrains search space

#### 3. Louis Kauffman (Topology & Knots)
- **Theory**: Quantum entanglement as knots; topology defines interaction constraints
- **Role**: Semantic relationships as topological links; "meaning" is an invariant
- **Mechanism**: KnotAttention (Linking Number Gating)
- **Efficiency**: Topology is sparse; unlinked concepts have zero interaction

### Unified Attention Formula

```
Attention(Q, K, V) = SolitonPropagate(TopologyGate(Q, K) · Landscape(V))
```

1. **Landscape**: Input V projected onto rugged manifold using WaveKAN
2. **Topology**: Q and K determine Linking Number; if unlinked, path is closed
3. **Dynamics**: Soliton pulse triggered at Q, propagates through linked paths of V
4. **Resonance**: Ramanujan frequencies ensure lossless transmission of morphological patterns

### Why This Is Efficient

- **No O(N²)**: Topology gates interaction to O(N·k) (sparse)
- **No Dispersion**: Solitons preserve signal over long distances (T=10k+ tokens)
- **Fast Optimization**: Partitioning balances load, allowing larger timesteps
- **Biologically Plausible**: FHN models real neural dynamics; SDR mirrors cortical columns

### E7 Lie Group Manifolds

The architecture is designed to support E7 Lie group manifolds:
- **E7**: Exceptional Lie group of dimension 133 (U-duality of N=8 supergravity)
- **Encoding**: Embeddings in E7/SU(8) coset space capture hyper-entanglement
- **Practical**: Uses Geometric Algebra (Clifford) rotors for efficient approximation
- **Future**: E7 root lattices for token quantization

---

## Installation

### Dependencies

```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm pytorch-lightning loguru
```

**Core Requirements:**
- [PyTorch](https://pytorch.org) ≥2.0 (for compile support)
- [numpy](https://numpy.org)
- `transformers` (HuggingFace)
- `datasets` (HuggingFace)
- `tiktoken` (OpenAI BPE)
- `wandb` (optional logging)
- `tqdm` (progress bars)
- `pytorch-lightning` (training framework)
- `loguru` (logging)

**Optional (for GPU optimization):**
- CUDA ≥11.8 for custom FHN kernels
- `triton` for fused FHN operations

### Installation from Source

```bash
git clone https://github.com/yourusername/neuromanifold-gpt.git
cd neuromanifold-gpt
pip install -e .
```

---

## Quick Start

### 1. Training on Shakespeare (Character-Level)

The fastest way to see NeuroManifoldGPT in action is to train on the Shakespeare character dataset:

```bash
# Prepare data
python data/shakespeare_char/prepare.py

# Train with Shakespeare preset
python train.py --config neuromanifold_gpt/config/presets/shakespeare_char.py
```

This preset (`neuromanifold_gpt/config/presets/shakespeare_char.py`) trains a compact NeuroManifoldGPT with:
- **Model**: 6 layers, 6 heads, 384 embedding dim
- **NeuroManifold features**: SDR encoding (1024-dim), FHN Attention, WaveKAN, Manifold projections (64-dim)
- **Training**: 5,000 iterations, batch size 64
- **Time**: ~10-20 minutes on a single GPU

**Expected output:**
```
Epoch 0: 100%|████████| 5000/5000 [15:30<00:00, 5.38it/s, loss=1.2, v_num=0]
```

### 2. Sampling from a Trained Model

Generate text using a trained model:

```bash
python sample.py \
  --out_dir=out-neuromanifold-shakespeare \
  --start="\n" \
  --num_samples=5 \
  --max_new_tokens=500 \
  --temperature=0.8 \
  --top_k=200
```

**Example output:**
```
DUKE VINCENTIO:
What say you to this part of this night?

First Citizen:
The king hath cause to think of us,
And yet the manifold ways of thought converge
Upon the great decision of our time...
---------------
```

**Sampling Parameters:**
- `--out_dir`: Directory containing the trained model checkpoint
- `--start`: Starting text prompt (use `"\n"` for character-level models, or `"FILE:prompt.txt"` to load from file)
- `--num_samples`: Number of samples to generate
- `--max_new_tokens`: Length of generated text (in tokens)
- `--temperature`: Sampling temperature (0.8 = more focused, 1.2 = more creative)
- `--top_k`: Top-k sampling filter (200 = balanced, 40 = more focused)

### 3. Training with Different Model Sizes

NeuroManifoldGPT includes four preset configurations (located in `neuromanifold_gpt/config/presets/`):

#### Nano (Fast Experimentation)
```bash
python train.py --config neuromanifold_gpt/config/presets/nano.py
```
- **Size**: 4 layers, 4 heads, 128 dims (~1M parameters)
- **Use case**: Quick tests, debugging, CPU training
- **Training time**: ~5-10 minutes on single GPU
- **Memory**: ~2GB VRAM

#### Shakespeare (Character-Level)
```bash
python train.py --config neuromanifold_gpt/config/presets/shakespeare_char.py
```
- **Size**: 6 layers, 6 heads, 384 dims (~10M parameters)
- **Use case**: Character-level language modeling, quick validation
- **Training time**: ~15-20 minutes on single GPU
- **Memory**: ~4GB VRAM

#### Small (Standard Training)
```bash
python train.py --config neuromanifold_gpt/config/presets/small.py
```
- **Size**: 12 layers, 12 heads, 768 dims (~85M parameters, similar to GPT-2 small)
- **Use case**: Full-scale experiments on consumer GPUs
- **Training time**: ~1-2 days for 600k iterations
- **Memory**: ~12GB VRAM with gradient accumulation

#### Medium (Research Scale)
```bash
python train.py --config neuromanifold_gpt/config/presets/medium.py
```
- **Size**: 24 layers, 16 heads, 1024 dims (~350M parameters, similar to GPT-2 medium)
- **Use case**: Large-scale research experiments
- **Training time**: ~3-5 days for 600k iterations
- **Memory**: ~24GB VRAM with gradient accumulation

### 4. Multi-GPU Training

PyTorch Lightning automatically handles distributed training:

```bash
# Train on 4 GPUs with DDP
python train.py --config neuromanifold_gpt/config/presets/small.py --devices 4

# Train on 8 GPUs with mixed precision
python train.py --config neuromanifold_gpt/config/presets/medium.py --devices 8 --precision bf16-mixed
```

### 5. Customizing Training

Override any config parameter via command line:

```bash
python train.py \
  --config neuromanifold_gpt/config/presets/small.py \
  --max_iters 10000 \
  --learning_rate 6e-4 \
  --batch_size 16 \
  --wandb_log true \
  --wandb_run_name my-experiment
```

**Common command-line overrides:**
- `--max_iters`: Total training iterations
- `--learning_rate`: Peak learning rate
- `--batch_size`: Batch size per GPU
- `--gradient_accumulation_steps`: Accumulation steps (effective batch = batch_size × accum × devices)
- `--devices`: Number of GPUs
- `--precision`: Mixed precision (`bf16-mixed`, `fp16-mixed`, or `fp32`)
- `--wandb_log`: Enable Weights & Biases logging
- `--out_dir`: Output directory for checkpoints

### 6. Configuration File Anatomy

All preset configs follow this structure (see `neuromanifold_gpt/config/presets/*.py`):

```python
# Model architecture
n_layer = 12              # Number of transformer layers
n_head = 12               # Number of attention heads
n_embd = 768              # Embedding dimension
block_size = 1024         # Context window size

# NeuroManifold specific
sdr_size = 2048           # SDR dimensionality (high-dim sparse space)
manifold_dim = 128        # Manifold projection dimension
n_eigenvectors = 64       # Spectral decomposition eigenvectors

# Training hyperparameters
batch_size = 12
gradient_accumulation_steps = 40  # Effective batch = 12 × 40 = 480
max_iters = 600000
learning_rate = 6e-4
min_lr = 6e-5

# Optimization
warmup_iters = 2000       # LR warmup steps
lr_decay_iters = 600000   # Cosine decay schedule

# Evaluation
eval_interval = 2000      # Validation frequency
eval_iters = 200          # Validation iterations

# Output
out_dir = "out-neuromanifold-small"
wandb_run_name = "neuromanifold-small"
```

**Key NeuroManifold Parameters:**
- `use_sdr`: Enable SDR encoding (default: False, memory intensive)
- `use_kan`: Enable WaveKAN FFN (default: True)
- `kan_wavelet`: Wavelet type (`"dog"` = Difference of Gaussians, `"mex"` = Mexican Hat)
- `use_fhn_imex`: Use IMEX solver for FHN (default: True, recommended for stability)
- `use_fhn_partitioning`: Use Karmarkar-Karp partitioning (default: True, enables larger timesteps)
- `use_kaufmann_attention`: Enable full Kaufmann Trifecta (FHN + Knot + WaveKAN, default: False, experimental)
- `use_mhc`: Enable manifold hyper-connections (default: True)
- `skip_manifold_spectral`: Skip manifold/spectral processing for faster training (default: False)

### 7. Complete Training Example

Here's a full workflow from data preparation to text generation:

```bash
# 1. Prepare dataset
python data/shakespeare_char/prepare.py

# 2. Train model (with WandB logging)
python train.py \
  --config neuromanifold_gpt/config/presets/shakespeare_char.py \
  --wandb_log true \
  --wandb_project neuromanifold-experiments \
  --wandb_run_name shakespeare-test-1

# 3. Generate samples during training (automatic with sample_interval in config)
# Samples are logged every 500 steps by default

# 4. After training, generate longer samples
python sample.py \
  --out_dir=out-neuromanifold-shakespeare \
  --start="ROMEO:\n" \
  --num_samples=3 \
  --max_new_tokens=1000 \
  --temperature=0.9

# 5. Resume training from checkpoint
python train.py \
  --config neuromanifold_gpt/config/presets/shakespeare_char.py \
  --out_dir out-neuromanifold-shakespeare
  # Lightning automatically resumes from last.ckpt if present
```

### 8. Configuration Presets Reference

All configuration presets are located in `neuromanifold_gpt/config/presets/`:

| Preset | File | Parameters | Context | Use Case |
|--------|------|------------|---------|----------|
| **Nano** | `nano.py` | ~1M | 256 | Fast experimentation, debugging |
| **Shakespeare** | `shakespeare_char.py` | ~10M | 256 | Character-level training, validation |
| **Small** | `small.py` | ~85M | 1024 | GPT-2 small comparison, single-GPU |
| **Medium** | `medium.py` | ~350M | 1024 | GPT-2 medium comparison, multi-GPU |

**Creating custom presets:** Copy any preset file and modify the parameters. All `.py` files in the presets directory can be loaded with `--config`

---

## Reproducing Results

### Baseline GPT-2 Comparison

Train standard GPT (no NeuroManifold features) for comparison:

```bash
python train.py --config neuromanifold_gpt/config/training/train_baseline_nanogpt.py
```

### Ablation Studies

We provide ablation configs to study individual components:

```bash
# No SDR
python train.py --config neuromanifold_gpt/config/training/ablation_no_sdr.py

# Standard MLP (no WaveKAN)
python train.py --config neuromanifold_gpt/config/training/ablation_swiglu.py
```

### Benchmarks

Run comprehensive benchmarks:

```bash
# FHN vs Standard Attention
python neuromanifold_gpt/bench_fhn_fusion.py

# WaveKAN vs SwiGLU
python neuromanifold_gpt/bench_kan_vs_swiglu.py
```

---

## Research & Theory

Detailed theoretical documentation is available in `neuromanifold_gpt/research/`:

- **`kaufmann_attention.md`**: Complete Kaufmann Trifecta theory and implementation strategy
- **`lie_algebra_e7.md`**: E7 Lie group manifolds and geometric algebra approximations
- **`efficiency_theory_integration.md`**: Knot-theoretic gating and number partitioning for ODE stability

---

## Project Status

**Current Status: Experimental Research Code**

This is an active research project. The architecture is functional and trains successfully, but it is:
- ✅ Theoretically grounded in neuroscience, topology, and physics
- ✅ Fully implemented with all core components
- ✅ Capable of training and generating coherent text
- ⚠️ Not yet optimized for production use
- ⚠️ Undergoing active experimentation and iteration
- ⚠️ Some components (E7 full implementation, geometric algebra) are planned but not yet complete

### Known Limitations
- FHN attention is slower than standard attention (research tradeoff for biological plausibility)
- SDR encoding adds memory overhead
- Kaufmann attention (full unified) is computationally intensive
- E7 manifolds currently approximated; full implementation in progress

### Future Work
- [ ] Full E7 Lie group implementation with geometric algebra rotors
- [ ] Triton-optimized fused FHN kernels
- [ ] Adaptive knot-theoretic gating with learned linking thresholds
- [ ] Ramanujan positional embeddings with prime frequency bases
- [ ] Parallel scan FHN solver for linear-time inference
- [ ] Multi-scale hierarchical SDR encoding

---

## Citation

If you use NeuroManifoldGPT in your research, please cite:

```bibtex
@software{neuromanifoldgpt2025,
  title={NeuroManifoldGPT: A Neuromorphic Manifold-Constrained Language Model},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/neuromanifold-gpt}
}
```

---

## Acknowledgments

This project builds on:
- **nanoGPT** by Andrej Karpathy (foundation codebase)
- **Kaufmann/Kauffman Trifecta**: Konrad Kaufmann (soliton theory), Stuart Kauffman (complexity), Louis Kauffman (knot theory)
- **Hierarchical Temporal Memory (HTM)** by Jeff Hawkins (SDR encoding)
- **FitzHugh-Nagumo model** (neuroscience)
- **Kolmogorov-Arnold Networks** (WaveKAN)
- **E7 Lie group theory** (theoretical physics)

---

## License

MIT License (same as nanoGPT)

---

## Architecture Diagram

```
Token Input
    ↓
[SDR Semantic Folding] → High-dimensional sparse encoding
    ↓
[Embedding Layer] → Dense embedding space
    ↓
[Ramanujan Positional Encoding] → Prime frequency basis
    ↓
┌─────────────────────────────────────────┐
│  NeuroManifoldBlock (x N layers)        │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │ Manifold Projection              │   │
│  │ (E7-inspired geometry)           │   │
│  └──────────────┬──────────────────┘   │
│                 ↓                       │
│  ┌─────────────────────────────────┐   │
│  │ Spectral Decomposition           │   │
│  │ (Graph Laplacian, Partitioning)  │   │
│  └──────────────┬──────────────────┘   │
│                 ↓                       │
│  ┌─────────────────────────────────┐   │
│  │ Kaufmann Attention               │   │
│  │  ├─ Knot Attention (Topology)    │   │
│  │  ├─ FHN Dynamics (Solitons)      │   │
│  │  └─ WaveKAN Landscape (Fitness)  │   │
│  └──────────────┬──────────────────┘   │
│                 ↓                       │
│  ┌─────────────────────────────────┐   │
│  │ Manifold Hyper-Connections (mHC) │   │
│  │ (Multi-stream residuals)         │   │
│  └──────────────┬──────────────────┘   │
│                 ↓                       │
│  ┌─────────────────────────────────┐   │
│  │ WaveKAN FFN                      │   │
│  │ (Wavelet-based MLP)              │   │
│  └──────────────┬──────────────────┘   │
│                 ↓                       │
│  [Residual + LayerNorm]                │
└──────────────────┬──────────────────────┘
                   ↓
[SDR Engram Memory] → Associative retrieval
                   ↓
[Output Head] → Token predictions
```

---

**"Smarter, Smaller, Faster"** — The Kaufmann Trifecta
