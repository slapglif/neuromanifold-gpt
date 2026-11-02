# Hyperparameter Optimization Guide

**Automate hyperparameter search with Optuna to find optimal configurations faster**

This guide shows you how to use automated hyperparameter optimization (HPO) to systematically search for the best learning rates, architecture sizes, and NeuroManifold component parameters. Whether you're exploring novel architectures or fine-tuning for production, HPO accelerates the search process.

## Table of Contents

1. [What is Hyperparameter Optimization?](#what-is-hyperparameter-optimization)
2. [Quick Start: Run HPO in 3 Steps](#quick-start-run-hpo-in-3-steps)
3. [Configuration Format](#configuration-format)
4. [Search Space Definition](#search-space-definition)
5. [Running HPO](#running-hpo)
6. [Interpreting Results](#interpreting-results)
7. [Best Practices](#best-practices)
8. [Advanced Features](#advanced-features)
9. [Troubleshooting](#troubleshooting)
10. [FAQ](#faq)

---

## What is Hyperparameter Optimization?

Hyperparameter optimization automates the search for optimal model configurations. Instead of manually testing different learning rates, layer counts, or feature flags, HPO systematically explores the search space and finds configurations that minimize validation loss.

**How it works:**
1. **Define search space** - Specify ranges for parameters (e.g., learning_rate: 1e-5 to 1e-2)
2. **Run trials** - HPO trains models with different parameter combinations
3. **Smart sampling** - Algorithm learns from previous trials to suggest better parameters
4. **Early pruning** - Stops unpromising trials early to save compute
5. **Export best config** - Automatically generates config file with optimal parameters

**Benefits:**
- **Faster exploration** - Test dozens of configurations automatically
- **Better results** - Systematic search often finds better configs than manual tuning
- **Save time** - No manual trial-and-error
- **Reproducible** - All trials tracked with full parameter history

**Common HPO use cases:**
- Finding optimal learning rates for new datasets
- Tuning NeuroManifold features (SDR, KAN, FHN parameters)
- Architecture search (layers, heads, embedding dimensions)
- Balancing model size vs. performance

---

## Quick Start: Run HPO in 3 Steps

Want to jump right in? Here's the fastest path to automated hyperparameter search:

### Step 1: Review Example Configuration

The repository includes a ready-to-use HPO configuration:

```sh
# View the example config
cat hpo_config_example.yaml
```

This config searches over 18 parameters including learning rate, architecture size, and NeuroManifold features.

### Step 2: Run HPO

```sh
# Run with default settings (50 trials)
python run_hpo.py --config hpo_config_example.yaml

# Or run a quick test (5 trials)
python run_hpo.py --config hpo_config_example.yaml --n-trials 5
```

**What happens:**
- HPO creates an Optuna study
- Runs multiple training trials with different hyperparameters
- Uses TPE (Tree-structured Parzen Estimator) sampler for intelligent search
- Prunes unpromising trials early (if enabled)
- Tracks all trials and results

**What to expect:**
- Each trial trains for `max_iters` iterations (default: 1000)
- Progress updates for each trial
- Takes ~5-30 minutes for 5 trials on GPU (depends on max_iters)
- Output includes best trial number and validation loss

### Step 3: Use Best Configuration

```sh
# Best config automatically saved to config/hpo_best.py
python train.py config/hpo_best.py
```

The exported config matches the format of `config/ralph_iter*.py` files and is ready to use for full training runs.

**Visualizations:**
- Optimization history plot: `hpo_results/optimization_history.png`
- Parameter importance plot: `hpo_results/param_importances.png`

That's it! You now have an optimized configuration found through automated search.

---

## Configuration Format

HPO uses YAML configuration files with four main sections:

### Basic Structure

```yaml
# Search space: parameters to optimize
search_space:
  learning_rate:
    type: float
    low: 1.0e-5
    high: 1.0e-2
    log: true

  n_layer:
    type: int
    low: 2
    high: 8

  use_sdr:
    type: categorical
    choices: [true, false]

# Fixed parameters: not optimized
fixed_params:
  dataset: shakespeare_char
  max_iters: 1000
  batch_size: 32

# Study configuration
study:
  name: my-hpo-study
  direction: minimize
  n_trials: 50
  sampler: tpe

# Pruning configuration (optional)
pruning:
  enabled: true
  algorithm: median
  n_startup_trials: 5
  n_warmup_steps: 50
```

### Parameter Types

**Float parameters:**
```yaml
learning_rate:
  type: float
  low: 1.0e-5      # Minimum value
  high: 1.0e-2     # Maximum value
  log: true        # Use log scale (recommended for learning rates)
```

**Integer parameters:**
```yaml
n_layer:
  type: int
  low: 2           # Minimum value (inclusive)
  high: 8          # Maximum value (inclusive)
```

**Categorical parameters:**
```yaml
# Boolean choices
use_sdr:
  type: categorical
  choices: [true, false]

# String choices
kan_type:
  type: categorical
  choices: ["faster", "wave", "cheby"]

# Numeric choices (discrete values)
batch_size:
  type: categorical
  choices: [16, 32, 64, 128]
```

### Study Configuration

```yaml
study:
  # Study name (used for storage/resumption)
  name: neuromanifold-hpo

  # Optimization direction
  direction: minimize  # or "maximize"

  # Number of trials to run
  n_trials: 50

  # Sampler algorithm
  sampler: tpe  # Options: tpe, random, grid, cmaes

  # Storage for resumption (optional)
  storage: "sqlite:///hpo_study.db"
```

**Sampler options:**
- **tpe** (default) - Tree-structured Parzen Estimator, good for most cases
- **random** - Random search, useful as baseline
- **grid** - Grid search over discrete values
- **cmaes** - CMA-ES, good for continuous optimization

### Pruning Configuration

```yaml
pruning:
  enabled: true
  algorithm: median       # Options: median, hyperband, percentile
  n_startup_trials: 5     # Trials before pruning starts
  n_warmup_steps: 50      # Validation steps before pruning
  interval_steps: 10      # Check pruning every N steps
```

**Pruning algorithms:**
- **median** - Prune if trial is worse than median of previous trials
- **hyperband** - Aggressive early stopping based on successive halving
- **percentile** - Prune if trial is in bottom percentile

---

## Search Space Definition

### Choosing What to Optimize

**Start with these parameters** (highest impact):
1. **learning_rate** - Most important hyperparameter
2. **n_layer, n_head, n_embd** - Architecture size
3. **dropout** - Regularization
4. **batch_size** - Training dynamics

**NeuroManifold-specific parameters:**
- **use_sdr, use_kan, use_fhn** - Feature flags
- **kan_type, kan_num_centers** - KAN configuration
- **fhn_threshold, fhn_tau, n_fhn_steps** - FHN dynamics
- **use_mhc, use_full_mhc** - Manifold hyper-connections

### Search Space Sizing

**Small search space** (3-5 parameters):
- Faster search, easier to interpret
- Good for focused optimization
- Example: learning_rate, n_layer, dropout

**Medium search space** (6-12 parameters):
- Balanced exploration
- Good for architecture + training hyperparameters
- Example: learning_rate, architecture params, dropout, batch_size

**Large search space** (13+ parameters):
- Comprehensive search
- Requires more trials (50-100+)
- Example: All parameters in `hpo_config_example.yaml`

### Parameter Range Guidelines

**Learning rate:**
```yaml
learning_rate:
  type: float
  low: 1.0e-5    # Conservative lower bound
  high: 1.0e-2   # Aggressive upper bound
  log: true      # Always use log scale
```

**Architecture size:**
```yaml
n_layer:
  type: int
  low: 2         # Minimum for meaningful model
  high: 8        # Balance quality vs. compute

n_head:
  type: int
  low: 4         # Standard minimum
  high: 8        # Typical maximum

n_embd:
  type: int
  low: 128       # Small models
  high: 512      # Medium models
```

**Dropout:**
```yaml
dropout:
  type: float
  low: 0.0       # No dropout
  high: 0.3      # Moderate regularization
  log: false     # Linear scale
```

**Batch size:**
```yaml
batch_size:
  type: categorical
  choices: [16, 32, 64, 128]  # Powers of 2
```

---

## Running HPO

### Basic Usage

```sh
# Run with config file
python run_hpo.py --config hpo_config_example.yaml

# Override number of trials
python run_hpo.py --config hpo_config_example.yaml --n-trials 100

# Specify output location for best config
python run_hpo.py --config hpo_config_example.yaml --output-config config/my_best.py

# Enable verbose logging
python run_hpo.py --config hpo_config_example.yaml --verbose
```

### Resuming Studies

If you configured storage URL, you can resume interrupted studies:

```yaml
# In config file
study:
  storage: "sqlite:///hpo_study.db"
  name: my-study
```

```sh
# Resume from database
python run_hpo.py --config hpo_config_example.yaml --resume
```

### Monitoring Progress

**Console output shows:**
```
Trial 5/50 - Parameters: {'learning_rate': 0.00023, 'n_layer': 4, ...}
Training trial 5...
iter 0: loss 3.28, time 125ms
iter 100: loss 2.15, time 120ms
...
Trial 5 complete - val_loss: 1.85
```

**Key metrics:**
- **Trial number** - Current trial (e.g., 5/50)
- **Parameters** - Suggested hyperparameters for this trial
- **Training loss** - Training progress
- **val_loss** - Final validation loss (objective to minimize)

**Progress indicators:**
- ✓ Completed trials
- ✂ Pruned trials (stopped early)
- ✗ Failed trials (errors)

### Expected Runtime

**Time per trial** depends on `max_iters` in fixed_params:

| max_iters | GPU Time | CPU Time |
|-----------|----------|----------|
| 100 | ~30s | ~2-5 min |
| 500 | ~2 min | ~10-20 min |
| 1000 | ~5 min | ~20-40 min |
| 5000 | ~20 min | ~2-4 hours |

**Total HPO time = trials × time_per_trial**

**Recommendations:**
- **Quick exploration** - 10 trials × 100 iters = ~5 minutes
- **Standard search** - 50 trials × 1000 iters = ~4 hours
- **Thorough search** - 100 trials × 5000 iters = ~33 hours

**Tip:** Use short `max_iters` (100-1000) for HPO, then train best config with full iterations.

---

## Interpreting Results

### Study Summary

After HPO completes, you'll see:

```
Optimization Summary
======================================================================
Total trials: 50
Completed: 45
Pruned: 4
Failed: 1

Best trial: #23
Best validation loss: 1.234

Best parameters:
  learning_rate: 0.000234
  n_layer: 6
  n_head: 8
  n_embd: 384
  dropout: 0.12
  use_sdr: True
  use_kan: False
  ...
======================================================================
```

**What to look for:**
- **Completed trials** - More is better, aim for 80%+ completion rate
- **Pruned trials** - Normal if pruning enabled, indicates early stopping working
- **Failed trials** - Should be <5%, investigate if higher
- **Best val_loss** - Lower is better, compare to baseline

### Visualizations

**Optimization History** (`hpo_results/optimization_history.png`):
- Shows objective value (val_loss) over trials
- Best value line shows improvement over time
- Markers indicate completed/pruned/failed trials

**What to look for:**
- **Downward trend** - HPO is finding better configurations
- **Plateau** - May need more trials or larger search space
- **Early convergence** - Good! Best config found quickly
- **No improvement** - Search space may be wrong or too narrow

**Parameter Importances** (`hpo_results/param_importances.png`):
- Horizontal bar chart showing which parameters matter most
- Uses fANOVA importance analysis
- Most important parameter highlighted in red

**What to look for:**
- **learning_rate usually #1** - Expected, most impactful parameter
- **Architecture params high** - n_layer, n_embd, n_head matter
- **Low importance params** - Consider fixing these in future searches
- **Unexpected high importance** - Interesting insights about your model

### Using Best Configuration

**Exported config** (`config/hpo_best.py`):

```python
# Best configuration from HPO
# Trial: 23
# Validation loss: 1.234

# Data
dataset = 'shakespeare_char'
...

# Model
n_layer = 6
n_head = 8
n_embd = 384
dropout = 0.12
...

# NeuroManifold features
use_sdr = True
use_kan = False
...

# Training
learning_rate = 0.000234
...
```

**Next steps:**
1. **Train with best config:**
   ```sh
   python train.py config/hpo_best.py
   ```

2. **Increase max_iters** for full training:
   ```python
   # Edit config/hpo_best.py
   max_iters = 20000  # Increase from HPO's 1000
   ```

3. **Enable checkpointing and logging:**
   ```python
   save_checkpoints = True
   wandb_log = True
   ```

### Comparing to Baseline

**Calculate improvement:**
```
Improvement = (baseline_loss - best_loss) / baseline_loss × 100%
```

**Example:**
- Baseline val_loss: 2.50 (default config)
- HPO best val_loss: 1.85
- Improvement: (2.50 - 1.85) / 2.50 × 100% = **26% better**

---

## Best Practices

### 1. Start Small, Scale Up

**Phase 1: Quick exploration** (30 min)
```yaml
study:
  n_trials: 10

fixed_params:
  max_iters: 100
```

**Phase 2: Standard search** (4 hours)
```yaml
study:
  n_trials: 50

fixed_params:
  max_iters: 1000
```

**Phase 3: Refinement** (optional)
```yaml
# Narrow search space around best config from Phase 2
search_space:
  learning_rate:
    low: 2.0e-4  # ±50% of best from Phase 2
    high: 3.0e-4
```

### 2. Choose Search Space Wisely

**Do:**
- ✅ Include learning_rate (always)
- ✅ Include 3-5 most impactful parameters
- ✅ Use log scale for learning_rate, weight_decay
- ✅ Use categorical for boolean flags

**Don't:**
- ❌ Optimize too many parameters (>15) without many trials
- ❌ Use linear scale for learning_rate
- ❌ Include parameters you know don't matter
- ❌ Set ranges too narrow (limits exploration)

### 3. Set Appropriate Fixed Parameters

**Keep these fixed during HPO:**
```yaml
fixed_params:
  dataset: shakespeare_char  # Dataset stays same
  max_iters: 1000           # Short for HPO
  eval_interval: 200        # Frequent validation
  eval_iters: 50            # Quick validation
  save_checkpoints: false   # Don't save every trial
  wandb_log: false          # Avoid clutter
  compile_model: false      # Faster trial startup
```

**Why:**
- Faster trials = more exploration
- Consistent comparison across trials
- Reduce I/O and logging overhead

### 4. Use Pruning Effectively

**Enable pruning for:**
- Large search spaces
- Limited compute budget
- Quick exploration

**Recommended settings:**
```yaml
pruning:
  enabled: true
  algorithm: median
  n_startup_trials: 5     # Learn from 5 trials first
  n_warmup_steps: 50      # Wait 50 steps before pruning
  interval_steps: 10      # Check every 10 steps
```

**Disable pruning when:**
- Small number of trials (<20)
- Training is very fast (<1 min per trial)
- You want to see full training curves

### 5. Monitor and Adjust

**After first 10 trials, check:**

1. **Completion rate**
   - <50% → Increase `n_warmup_steps` or disable pruning
   - >95% → Pruning may be too conservative

2. **Val_loss range**
   - All trials similar → Search space too narrow
   - Huge variation → Good exploration

3. **Parameter importances**
   - Update search space to focus on important params

### 6. GPU vs CPU Considerations

**GPU (recommended):**
```yaml
fixed_params:
  devices: 1
  precision: "bf16-mixed"  # Fast training
  compile_model: false     # Faster startup
```

**CPU (if no GPU):**
```yaml
fixed_params:
  devices: 1
  precision: "32"
  compile_model: false

study:
  n_trials: 10  # Fewer trials due to slow training
```

---

## Advanced Features

### Multi-Objective Optimization

While the current implementation focuses on single-objective (validation loss), you can manually balance objectives:

**Example: Speed vs. Accuracy**
1. Run HPO to minimize val_loss
2. Filter trials by model size: `n_layer * n_embd < 3000`
3. Choose best trial meeting size constraint

### Custom Search Spaces

**Example: Coupled parameters**

Some parameters should be related (e.g., n_head must divide n_embd):

```yaml
# Define compatible pairs
search_space:
  n_embd:
    type: categorical
    choices: [256, 384, 512]  # Divisible by common n_head values

  n_head:
    type: categorical
    choices: [4, 8]  # Works with all n_embd values above
```

### Conditional Parameters

**Example: KAN parameters only matter when use_kan=True**

Strategy: Include all parameters but interpret results knowing some are irrelevant for certain trials.

Alternatively, run separate HPO studies:
1. Study A: `use_kan: false` (optimize other params)
2. Study B: `use_kan: true` (optimize KAN params too)

### Warm-Starting from Previous Studies

```yaml
# First study
study:
  name: initial-search
  n_trials: 50
  storage: "sqlite:///hpo.db"
```

```sh
# Run initial study
python run_hpo.py --config hpo_v1.yaml

# Modify search space (e.g., narrow around best)
# Edit hpo_v2.yaml with refined ranges

# Continue with same study name to combine trials
python run_hpo.py --config hpo_v2.yaml --resume
```

### Distributed HPO

For very large searches, run multiple HPO processes in parallel:

```sh
# Terminal 1
python run_hpo.py --config hpo_config.yaml --n-trials 25

# Terminal 2 (simultaneously)
python run_hpo.py --config hpo_config.yaml --n-trials 25 --resume
```

Both will share the same study via storage database and coordinate trials.

**Requirements:**
```yaml
study:
  storage: "sqlite:///hpo.db"  # Shared database
  name: parallel-hpo
```

### Analyzing Study Database

```python
import optuna

# Load study
study = optuna.load_study(
    study_name="neuromanifold-hpo",
    storage="sqlite:///hpo.db"
)

# Get all trials
trials = study.trials

# Filter trials
good_trials = [t for t in trials if t.value < 2.0]

# Analyze parameter distributions
learning_rates = [t.params['learning_rate'] for t in good_trials]
import numpy as np
print(f"Good LR range: {np.min(learning_rates):.6f} - {np.max(learning_rates):.6f}")
```

---

## Troubleshooting

### Issue 1: All Trials Failing

**Symptoms:**
```
Trial 1 failed: CUDA out of memory
Trial 2 failed: CUDA out of memory
...
```

**Solution:**
Reduce batch size in fixed_params:
```yaml
fixed_params:
  batch_size: 16  # Reduce from 32/64
  gradient_accumulation_steps: 2  # Compensate
```

### Issue 2: No Improvement Over Trials

**Symptoms:**
- Best val_loss not improving
- Random-looking trial results

**Possible causes:**

**Cause 1: Search space too narrow**
```yaml
# Too narrow - all values similar
learning_rate:
  low: 1.0e-4
  high: 2.0e-4

# Better - wider exploration
learning_rate:
  low: 1.0e-5
  high: 1.0e-2
```

**Cause 2: max_iters too low**
```yaml
fixed_params:
  max_iters: 1000  # Increase from 100
```

**Cause 3: Wrong parameters in search space**
- Verify you're optimizing impactful parameters
- Check parameter importances plot

### Issue 3: Too Many Pruned Trials

**Symptoms:**
```
Completed: 10
Pruned: 40
```

**Solution:**
Adjust pruning to be less aggressive:
```yaml
pruning:
  enabled: true
  n_startup_trials: 10  # Increase from 5
  n_warmup_steps: 100   # Increase from 50
  algorithm: percentile # Try different algorithm
```

Or disable pruning:
```yaml
pruning:
  enabled: false
```

### Issue 4: HPO Taking Too Long

**Symptoms:**
- Hours per trial
- Can't complete study in reasonable time

**Solutions:**

**Solution 1: Reduce max_iters**
```yaml
fixed_params:
  max_iters: 500  # Reduce from 1000+
```

**Solution 2: Enable pruning**
```yaml
pruning:
  enabled: true
  algorithm: hyperband  # More aggressive
```

**Solution 3: Reduce number of trials**
```yaml
study:
  n_trials: 20  # Start smaller
```

**Solution 4: Disable compilation**
```yaml
fixed_params:
  compile_model: false  # Faster startup
```

### Issue 5: "No improvement" Warning in Visualizations

**Symptoms:**
```
WARNING: No parameter importances calculated
```

**Cause:**
- Fewer than 2 completed trials
- All trials have same objective value

**Solution:**
- Complete more trials
- Widen search space to get variation

### Issue 6: Study Won't Resume

**Symptoms:**
```
Study not found: neuromanifold-hpo
```

**Solutions:**

1. **Check storage path:**
   ```yaml
   study:
     storage: "sqlite:///hpo_study.db"  # File path must exist
   ```

2. **Verify study name:**
   ```sh
   # List studies in database
   python -c "import optuna; print(optuna.get_all_study_names('sqlite:///hpo_study.db'))"
   ```

3. **Create new study if needed:**
   ```sh
   # Remove --resume flag
   python run_hpo.py --config hpo_config.yaml
   ```

---

## FAQ

### Q: How many trials should I run?

**A:** Depends on search space size:
- **Small space (3-5 params):** 20-30 trials
- **Medium space (6-12 params):** 50-100 trials
- **Large space (13+ params):** 100-200 trials

Rule of thumb: ~10 trials per parameter in search space.

### Q: What's the best sampler to use?

**A:** TPE (default) works well for most cases:
- **TPE** - Best for most use cases, learns from previous trials
- **Random** - Good baseline, use to verify TPE is helping
- **CmaEs** - Good for continuous optimization with many trials

Start with TPE unless you have specific needs.

### Q: Should I use pruning?

**A:** Yes, if:
- Running >30 trials
- Compute budget is limited
- Quick exploration desired

No, if:
- Running <20 trials
- You want complete training curves
- Trials are already very fast (<2 min)

### Q: Can I use HPO for architecture search (NAS)?

**A:** Yes, include architecture parameters:
```yaml
search_space:
  n_layer: {type: int, low: 2, high: 12}
  n_head: {type: int, low: 4, high: 16}
  n_embd: {type: int, low: 128, high: 768}
```

But note: This searches discrete architectures, not true NAS with weight sharing.

### Q: How do I optimize for multiple metrics?

**A:** Current implementation is single-objective (val_loss). For multi-objective:

1. **Manual filtering:** Run HPO for val_loss, then filter trials by secondary metric
2. **Weighted objective:** Modify code to return `val_loss + α * latency`
3. **Pareto optimization:** Use Optuna's multi-objective study (requires code changes)

### Q: Can I use HPO with my own dataset?

**A:** Yes! Just specify in fixed_params:
```yaml
fixed_params:
  dataset: my_dataset  # Your dataset name
  # Ensure data/my_dataset/train.bin and val.bin exist
```

### Q: What if I want to optimize different parameters?

**A:** Edit the YAML config:
1. Copy `hpo_config_example.yaml` to `my_hpo_config.yaml`
2. Modify `search_space` section with your parameters
3. Ensure parameter names match those in `TrainConfig`

### Q: How do I reproduce results from a specific trial?

**A:** Each trial's parameters are saved. To reproduce:
1. Find trial number from study summary or plots
2. Load study and extract params:
   ```python
   import optuna
   study = optuna.load_study(study_name="...", storage="...")
   trial_params = study.trials[23].params  # Trial 23
   ```
3. Create config file with those exact parameters

### Q: Can I pause and resume HPO?

**A:** Yes, if using storage:
1. **Ctrl+C** to stop
2. **Resume:** `python run_hpo.py --config hpo_config.yaml --resume`

If not using storage, you'll lose progress.

### Q: Why is my best trial not trial #0?

**A:** HPO explores the space, early trials are often random. The sampler learns from these and suggests better params in later trials. Best trial can be anywhere, often in the middle range.

### Q: What's a good validation loss?

**A:** Depends on dataset:
- **Shakespeare:** 0.8-1.2 is excellent
- **Your dataset:** Compare to baseline (training without HPO)

Focus on **relative improvement** not absolute values.

---

## Quick Reference

**Start HPO:**
```sh
python run_hpo.py --config hpo_config_example.yaml
```

**Quick test (5 trials):**
```sh
python run_hpo.py --config hpo_config_example.yaml --n-trials 5
```

**Resume study:**
```sh
python run_hpo.py --config hpo_config.yaml --resume
```

**Use best config:**
```sh
python train.py config/hpo_best.py
```

**Key files:**
- `hpo_config_example.yaml` - Example configuration
- `config/hpo_best.py` - Best config (generated after HPO)
- `hpo_results/` - Visualization plots
- `hpo_study.db` - Study database (if storage enabled)

**Configuration sections:**
- `search_space:` - Parameters to optimize
- `fixed_params:` - Parameters to keep constant
- `study:` - HPO settings (trials, sampler)
- `pruning:` - Early stopping settings

**Common parameter types:**
```yaml
learning_rate: {type: float, low: 1e-5, high: 1e-2, log: true}
n_layer: {type: int, low: 2, high: 8}
use_sdr: {type: categorical, choices: [true, false]}
```

---

## Next Steps

Now that you understand HPO, try:

1. **Run example config** - Get familiar with the workflow
2. **Customize search space** - Focus on parameters relevant to your task
3. **Analyze results** - Study parameter importances
4. **Iterate** - Refine search space based on findings
5. **Scale up** - Run larger studies for production configs

**Additional resources:**
- **Example config:** `hpo_config_example.yaml` - Full configuration reference
- **Optuna documentation:** [optuna.org](https://optuna.org) - Advanced features
- **Training guide:** `docs/finetuning-guide.md` - Using HPO best configs

**Happy optimizing! 🎯**
