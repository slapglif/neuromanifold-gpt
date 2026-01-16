# Weight Initialization Strategies

**A Comprehensive Guide to Modern Neural Network Initialization**

This document explains the weight initialization strategies implemented in NeuroManifoldGPT and baseline GPT models, providing guidance on when to use each strategy and how to leverage muP for hyperparameter transfer across model scales.

## Overview

Proper weight initialization is critical for:
- **Training Stability**: Prevents gradient explosion/vanishing in early training
- **Convergence Speed**: Well-initialized weights reach good solutions faster
- **Hyperparameter Transfer**: muP enables transferring optimal hyperparameters across model sizes
- **Final Performance**: Better initialization leads to better final loss values

## Available Strategies

### 1. DeepSeek (Default for NeuroManifoldGPT)

**Theory:** Based on DeepSeek-V3 initialization approach using very small standard deviation for faster early convergence.

**Initialization:**
- Linear layers: `std = 0.006`
- Embeddings: `std = 0.006`
- Biases: Zero initialization

**When to Use:**
- Default for NeuroManifoldGPT models
- When you want faster early training convergence
- For models with novel architectures that benefit from conservative initialization
- Works well with specialized components (SDR, manifolds, soliton dynamics)

**Configuration:**
```python
from neuromanifold_gpt.config.base import NeuroManifoldConfig
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT

config = NeuroManifoldConfig(
    vocab_size=50304,
    n_layer=12,
    n_heads=8,
    n_embd=768,
    init_strategy='deepseek'  # Default, can be omitted
)
model = NeuroManifoldGPT(config)
```

**Characteristics:**
- Very conservative initialization (small std)
- Reduces risk of early training instability
- May require slightly more iterations to fully converge
- Excellent for architectures with complex attention mechanisms

---

### 2. GPT-2 Standard

**Theory:** Original GPT-2 initialization from Radford et al. (2019). Uses uniform standard deviation across all layers.

**Initialization:**
- Linear layers: `std = 0.02`
- Embeddings: `std = 0.02`
- Biases: Zero initialization

**When to Use:**
- Baseline GPT model (default)
- When replicating GPT-2 results
- For standard transformer architectures
- When you have well-tested hyperparameters from GPT-2 literature

**Configuration:**
```python
from model import GPT, GPTConfig

config = GPTConfig(
    vocab_size=50304,
    n_layer=12,
    n_head=12,
    n_embd=768,
    init_strategy='gpt2'
)
model = GPT(config)
```

**Characteristics:**
- Proven effective for standard transformers
- Good baseline for comparison
- Well-documented in literature
- May need adjustment for very deep networks (>24 layers)

---

### 3. GPT-2 Scaled (Residual Scaling)

**Theory:** GPT-2 with depth-dependent residual projection scaling. Scales down residual branch contributions by `1/sqrt(2*n_layer)` to prevent activation growth with depth.

**Initialization:**
- Linear layers: `std = 0.02`
- Embeddings: `std = 0.02`
- Residual projections (`c_proj`, `out_proj`): `std = 0.02 / sqrt(2 * n_layer)`
- Biases: Zero initialization

**When to Use:**
- Deep models (>12 layers)
- When training stability issues occur with standard GPT-2 init
- Models with many residual connections
- When you observe gradient explosion or NaN losses

**Configuration:**
```python
from neuromanifold_gpt.config.base import NeuroManifoldConfig
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT

config = NeuroManifoldConfig(
    vocab_size=50304,
    n_layer=24,  # Deep model benefits from scaled init
    n_heads=16,
    n_embd=1024,
    init_strategy='gpt2_scaled'
)
model = NeuroManifoldGPT(config)
```

**Characteristics:**
- Better stability for deep networks
- Prevents residual branch dominance
- Maintains signal variance through depth
- Recommended for n_layer > 12

---

### 4. muP (Maximal Update Parametrization)

**Theory:** Based on Yang et al. (2022) "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer". Enables **hyperparameter transfer** across model widths by maintaining optimal learning rate and initialization variance relationships.

**Initialization:**
- Embeddings: `std = 1 / sqrt(d_model)`
- Hidden layers: `std = 1 / d_model` (width-independent scaling)
- Output head (lm_head): `std = 1 / sqrt(d_model)` (no width scaling)
- Residual branches: `std = (1 / d_model) * sqrt(base_width / d_model)`

**Key Innovation:**
muP changes the scaling relationship so that:
1. **Learning rate** becomes width-independent
2. **Optimal hyperparameters** transfer from small → large models
3. **Feature learning** remains consistent across widths

**When to Use:**
- **Hyperparameter search** on small models, transfer to large models
- **Scaling experiments** (e.g., testing 125M → 1.3B → 13B parameter models)
- When computational budget limits large-scale hyperparameter tuning
- Research projects exploring model scaling properties

**Configuration:**
```python
from neuromanifold_gpt.config.base import NeuroManifoldConfig
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT

# Small base model for hyperparameter tuning
base_config = NeuroManifoldConfig(
    vocab_size=50304,
    n_layer=6,
    n_heads=4,
    n_embd=128,  # This is the base_width
    init_strategy='mup',
    mup_base_width=128,  # Must match n_embd of base model
    mup_scale_attn_by_d=True  # Scale attention by d (not sqrt(d))
)
base_model = NeuroManifoldGPT(base_config)

# Large model with transferred hyperparameters
large_config = NeuroManifoldConfig(
    vocab_size=50304,
    n_layer=12,
    n_heads=12,
    n_embd=768,  # Wider model
    init_strategy='mup',
    mup_base_width=128,  # Same base_width as tuning model
    mup_scale_attn_by_d=True,
    # Learning rate stays the same as base model!
    learning_rate=3e-4  # Tuned on base model, transfers to large model
)
large_model = NeuroManifoldGPT(large_config)
```

**Characteristics:**
- Enables zero-shot hyperparameter transfer
- Requires tracking `mup_base_width` (the width of your tuning model)
- Learning rate must be scaled differently (width-independent)
- Attention should scale by d instead of sqrt(d) when `mup_scale_attn_by_d=True`
- More complex but highly valuable for scaling research

---

## muP Hyperparameter Transfer Workflow

### Step 1: Choose a Base Width

Select a small model width for hyperparameter tuning:
```python
base_width = 128  # Small enough to train quickly
```

### Step 2: Tune on Small Model

Train multiple configurations with different learning rates, warmup schedules, etc.:
```python
for lr in [1e-4, 3e-4, 1e-3]:
    config = NeuroManifoldConfig(
        n_embd=base_width,
        init_strategy='mup',
        mup_base_width=base_width,
        learning_rate=lr
    )
    # Train and evaluate...
```

### Step 3: Transfer to Large Model

Use the **exact same hyperparameters** on your large model:
```python
# Best config from small model: lr=3e-4, warmup_steps=1000, etc.
large_config = NeuroManifoldConfig(
    n_embd=768,  # 6x wider
    init_strategy='mup',
    mup_base_width=128,  # Same as tuning model
    learning_rate=3e-4,  # Same as tuning model!
    # All other hyperparameters stay the same
)
```

### Step 4: Verify Transfer

Compare loss curves between small and large models. With proper muP:
- Training should be equally stable
- Optimal learning rate should be the same
- Convergence patterns should be similar

**Important Notes:**
- The base_width must be consistent across all models in a family
- Do NOT scale learning rate when changing width (that's the point of muP!)
- Attention scaling (`mup_scale_attn_by_d`) must be consistent
- Track which base_width each experiment uses

---

## Strategy Comparison Table

| Strategy | Std Dev (Linear) | Depth Scaling | Width Scaling | Best For |
|----------|------------------|---------------|---------------|----------|
| **DeepSeek** | 0.006 | No | No | Novel architectures, NeuroManifold |
| **GPT-2** | 0.02 | No | No | Standard transformers, baselines |
| **GPT-2 Scaled** | 0.02 → 0.02/√(2n) | Yes (residual) | No | Deep models (>12 layers) |
| **muP** | 1/d → varies | Yes (implicit) | Yes | Scaling experiments, transfer |

---

## Analyzing Initialization

Use the `analyze_init.py` script to verify weight distributions before training:

### Basic Analysis
```bash
python neuromanifold_gpt/research/analyze_init.py \
    --strategy deepseek \
    --n-layer 12 \
    --n-embd 768
```

### Compare All Strategies
```bash
python neuromanifold_gpt/research/analyze_init.py \
    --strategy mup \
    --n-layer 12 \
    --n-embd 768 \
    --compare-all
```

### Visualize Distributions
```bash
python neuromanifold_gpt/research/analyze_init.py \
    --strategy gpt2_scaled \
    --n-layer 24 \
    --n-embd 1024 \
    --visualize \
    --output-dir ./init_analysis
```

**What to Look For:**
- **Mean close to 0**: All strategies should have near-zero mean
- **Reasonable std**: Values between 0.001 and 0.1 (strategy-dependent)
- **No NaN/Inf**: Check for numerical issues
- **Depth scaling**: For scaled init, verify residual projections have smaller std
- **Width scaling**: For muP, verify std scales with model width

---

## Troubleshooting Guide

### Problem: Loss is NaN or Inf

**Symptoms:**
- Loss becomes NaN within first few iterations
- Gradients explode to infinity

**Solutions:**
1. Switch to more conservative initialization:
   ```python
   init_strategy='deepseek'  # Smallest std=0.006
   ```
2. Enable gradient clipping:
   ```python
   grad_clip=1.0  # Default, increase if still unstable
   ```
3. Reduce learning rate:
   ```python
   learning_rate=1e-4  # Down from default 3e-4
   ```

### Problem: Loss Not Decreasing

**Symptoms:**
- Loss stays constant or decreases very slowly
- Model appears "stuck" in early training

**Solutions:**
1. Increase initialization std (less conservative):
   ```python
   init_strategy='gpt2'  # std=0.02 instead of 0.006
   ```
2. Check learning rate isn't too small
3. Verify data preprocessing (tokenization, batching)
4. Run `analyze_init.py` to check for dead neurons (zero gradients)

### Problem: Training Unstable in Deep Models

**Symptoms:**
- Training stable for first epoch, then spikes
- Gradients fluctuate wildly after initial steps

**Solutions:**
1. Use scaled initialization:
   ```python
   init_strategy='gpt2_scaled'  # Depth-aware scaling
   ```
2. Add more aggressive gradient clipping:
   ```python
   grad_clip=0.5  # Down from default 1.0
   ```
3. Use learning rate warmup:
   ```python
   warmup_steps=2000  # Gradual LR increase
   ```

### Problem: muP Transfer Not Working

**Symptoms:**
- Large model doesn't train well with small model's hyperparameters
- Learning rate that worked on small model fails on large model

**Solutions:**
1. Verify `mup_base_width` is consistent:
   ```python
   # Both models must have same base_width
   small: mup_base_width=128, n_embd=128
   large: mup_base_width=128, n_embd=768  # ✓ Correct
   ```
2. Check attention scaling:
   ```python
   mup_scale_attn_by_d=True  # Must be same for all models
   ```
3. Ensure all hyperparameters transfer (not just learning rate):
   - Warmup steps
   - Weight decay
   - Beta1/Beta2
   - Dropout (if any)

### Problem: Weights Too Small/Large

**Symptoms:**
- `analyze_init.py` shows std << 0.001 or std >> 0.1
- Gradients vanish or explode

**Solutions:**
1. Check model configuration matches strategy requirements
2. For muP, verify base_width calculation
3. Try different strategy if current one doesn't match architecture
4. Inspect specific layer types (embeddings vs linear vs residual)

---

## Best Practices

### 1. Start with Defaults
- **NeuroManifoldGPT**: Use `deepseek` (default)
- **Baseline GPT**: Use `gpt2` (default)
- Only change if you have specific issues

### 2. Scale with Depth
- **≤12 layers**: Standard initialization usually fine
- **>12 layers**: Use `gpt2_scaled` or `mup`
- **>24 layers**: Strongly consider scaled initialization

### 3. Use muP for Scaling Studies
- Planning to scale from 125M → 1B+ parameters? Use muP from the start
- Define your base_width early (e.g., 128 or 256)
- Keep detailed records of which base_width each experiment uses

### 4. Always Analyze Before Training
```bash
# Quick check before long training run
python neuromanifold_gpt/research/analyze_init.py \
    --strategy $STRATEGY \
    --n-layer $N_LAYER \
    --n-embd $N_EMBD
```

### 5. Monitor Early Training
- First 100-500 steps are most sensitive to initialization
- Watch for gradient norms (should stay roughly constant)
- Early loss spike? Your initialization may be too aggressive

---

## References

1. **GPT-2 Paper**: Radford et al. (2019) "Language Models are Unsupervised Multitask Learners"
   - Standard transformer initialization approach

2. **muP Paper**: Yang et al. (2022) "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer"
   - Theoretical foundation for width-independent hyperparameter transfer
   - arXiv: 2203.03466

3. **DeepSeek-V3**: DeepSeek AI (2024) "DeepSeek-V3 Technical Report"
   - Small initialization std for stable training of complex architectures

4. **Residual Scaling**: He et al. (2016) "Identity Mappings in Deep Residual Networks"
   - Depth-dependent scaling for residual branches

---

## Advanced: Creating Custom Initialization

If you need a custom initialization strategy:

### 1. Add Strategy to Config
```python
# neuromanifold_gpt/config/base.py
init_strategy: str = 'my_custom_strategy'
```

### 2. Implement in _init_weights
```python
# neuromanifold_gpt/model/gpt.py
def _init_weights(self, module: nn.Module) -> None:
    init_strategy = getattr(self.config, 'init_strategy', 'deepseek')

    if init_strategy == 'my_custom_strategy':
        if isinstance(module, nn.Linear):
            # Your custom initialization logic
            std = compute_custom_std(module)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Your custom embedding initialization
            pass
```

### 3. Test with analyze_init.py
```bash
python neuromanifold_gpt/research/analyze_init.py \
    --strategy my_custom_strategy \
    --n-layer 12 \
    --n-embd 768 \
    --visualize
```

### 4. Document Your Strategy
Add to this file:
- Theory/motivation
- Initialization formulas
- When to use
- Configuration examples

---

## Summary

**Quick Reference:**

| Use Case | Recommended Strategy |
|----------|---------------------|
| NeuroManifoldGPT default | `deepseek` |
| Baseline GPT default | `gpt2` |
| Deep models (>12 layers) | `gpt2_scaled` |
| Hyperparameter transfer | `mup` |
| Unstable training | `deepseek` |
| Replicating GPT-2 | `gpt2` |
| Scaling research | `mup` |

**Remember:**
- Initialization is critical but often overlooked
- Use `analyze_init.py` to verify before long training runs
- muP enables powerful hyperparameter transfer but requires careful setup
- When in doubt, start with defaults and only change if needed
