# Callback Migration Guide

## Overview

The monolithic `TrainingHealthCallback` (630 lines, 8+ concerns) has been refactored into **6 focused single-purpose callbacks**. This enables selective callback composition, clearer ownership of each monitoring feature, and easier maintenance.

### Benefits of the New Architecture

- ✅ **Single Responsibility Principle**: Each callback handles one concern
- ✅ **Selective Composition**: Use only the monitoring you need
- ✅ **Easier Testing**: Each callback is independently testable
- ✅ **Better Maintainability**: Smaller, focused files are easier to understand and modify
- ✅ **Backward Compatible**: Old `TrainingHealthCallback` still works (with deprecation warning)

### The New Callbacks

| Callback | Responsibility | Key Metrics |
|----------|---------------|-------------|
| `GradientMonitorCallback` | Track gradient norms and clipping | `train/grad_norm`, `train/clip_rate`, gradient explosion detection |
| `MemoryMonitorCallback` | Track GPU memory usage | `train/memory_current_mb`, `train/memory_peak_mb` |
| `ThroughputMonitorCallback` | Track training speed and ETA | `train/step_time`, `train/tokens_per_sec`, `train/eta` |
| `LossMonitorCallback` | Track loss history and spikes | `train/loss_avg`, loss spike detection |
| `AnomalyDetectorCallback` | Detect NaN/Inf in gradients and loss | Critical warnings for training instability |
| `TrainingDashboardCallback` | Rich console display | Visual dashboard (reads metrics from other callbacks) |

---

## Quick Start

### Option 1: Full Monitoring (Recommended)

Get comprehensive monitoring similar to the old `TrainingHealthCallback`:

```python
from neuromanifold_gpt.callbacks import (
    GradientMonitorCallback,
    MemoryMonitorCallback,
    ThroughputMonitorCallback,
    LossMonitorCallback,
    AnomalyDetectorCallback,
    TrainingDashboardCallback,
)

callbacks = [
    GradientMonitorCallback(),
    MemoryMonitorCallback(),
    ThroughputMonitorCallback(),
    LossMonitorCallback(),
    AnomalyDetectorCallback(),
    TrainingDashboardCallback(),
]

trainer.fit(model, callbacks=callbacks)
```

### Option 2: Minimal Monitoring

Only track essential metrics:

```python
from neuromanifold_gpt.callbacks import (
    LossMonitorCallback,
    AnomalyDetectorCallback,
)

callbacks = [
    LossMonitorCallback(),
    AnomalyDetectorCallback(),
]

trainer.fit(model, callbacks=callbacks)
```

### Option 3: Custom Selection

Choose exactly what you need:

```python
from neuromanifold_gpt.callbacks import (
    GradientMonitorCallback,
    ThroughputMonitorCallback,
    TrainingDashboardCallback,
)

callbacks = [
    GradientMonitorCallback(log_interval=50, grad_explosion_threshold=2.5),
    ThroughputMonitorCallback(warmup_steps=5),
    TrainingDashboardCallback(refresh_rate=50),
]

trainer.fit(model, callbacks=callbacks)
```

---

## Old vs New: Side-by-Side Comparison

### Before (Monolithic Callback)

```python
from neuromanifold_gpt.callbacks import TrainingHealthCallback

# All monitoring features bundled together
# Cannot disable individual features
# Single configuration for everything
callback = TrainingHealthCallback(
    log_interval=100,
    gradient_norm_history_size=100,
    enable_dashboard=True,
)

trainer.fit(model, callbacks=[callback])
```

**Issues:**
- 😕 Cannot disable gradient monitoring without losing memory monitoring
- 😕 Cannot customize individual features independently
- 😕 Hard to test individual monitoring features
- 😕 630 lines in a single file - difficult to maintain

### After (Composed Callbacks)

```python
from neuromanifold_gpt.callbacks import (
    GradientMonitorCallback,
    MemoryMonitorCallback,
    ThroughputMonitorCallback,
    LossMonitorCallback,
    AnomalyDetectorCallback,
    TrainingDashboardCallback,
)

# Each callback is independently configurable
# Use only what you need
# Easy to customize each feature
callbacks = [
    # Fine-tune gradient monitoring
    GradientMonitorCallback(
        log_interval=100,
        gradient_norm_history_size=100,
        grad_explosion_threshold=3.0,
    ),

    # Simple memory tracking
    MemoryMonitorCallback(log_interval=100),

    # Custom throughput settings
    ThroughputMonitorCallback(
        step_time_history_size=20,
        warmup_steps=10,
    ),

    # Loss spike detection
    LossMonitorCallback(
        loss_history_size=100,
        loss_spike_threshold=3.0,
    ),

    # Critical anomaly detection (always recommended)
    AnomalyDetectorCallback(),

    # Optional: Rich console dashboard
    TrainingDashboardCallback(refresh_rate=100),
]

trainer.fit(model, callbacks=callbacks)
```

**Benefits:**
- ✅ Each callback configured independently
- ✅ Use only what you need (e.g., skip dashboard for headless environments)
- ✅ Easy to test and maintain each component
- ✅ Clear separation of concerns

---

## Individual Callback Usage

### 1. GradientMonitorCallback

**Purpose:** Track gradient norms, clipping events, and detect gradient explosions.

```python
from neuromanifold_gpt.callbacks import GradientMonitorCallback

callback = GradientMonitorCallback(
    log_interval=100,                      # Log metrics every 100 steps
    gradient_norm_history_size=100,        # Keep last 100 gradient norms for statistics
    grad_explosion_threshold=3.0,          # Alert if grad_norm > mean + 3*std
    min_grad_history_for_detection=20,     # Wait for 20 samples before detecting explosions
)
```

**Metrics Logged:**
- `train/grad_norm` - Current gradient norm
- `train/grad_norm_avg` - Average gradient norm over history
- `train/clip_rate` - Percentage of steps where clipping occurred
- `train/total_clip_events` - Total number of clipping events
- `train/avg_clip_ratio` - Average ratio of clipped gradients

**Warnings:**
- Prints warning when gradient explosion detected (grad_norm > mean + threshold*std)

### 2. MemoryMonitorCallback

**Purpose:** Track GPU memory usage during training.

```python
from neuromanifold_gpt.callbacks import MemoryMonitorCallback

callback = MemoryMonitorCallback(
    log_interval=100,  # Log metrics every 100 steps
)
```

**Metrics Logged:**
- `train/memory_current_mb` - Current GPU memory allocated (MB)
- `train/memory_peak_mb` - Peak GPU memory allocated (MB)

**Note:** Only tracks GPU memory. CPU memory monitoring can be added if needed.

### 3. ThroughputMonitorCallback

**Purpose:** Track training speed, step times, and estimated time to completion.

```python
from neuromanifold_gpt.callbacks import ThroughputMonitorCallback

callback = ThroughputMonitorCallback(
    log_interval=100,              # Log metrics every 100 steps
    step_time_history_size=20,     # Keep last 20 step times for rolling average
    warmup_steps=10,               # Wait 10 steps before showing ETA
)
```

**Metrics Logged:**
- `train/step_time` - Time per step (seconds)
- `train/tokens_per_sec` - Throughput in tokens/second (if batch size available)
- `train/eta` - Estimated time to completion (formatted as "2h 34m")
- `train/progress_pct` - Training progress percentage

**Note:** ETA only shown after warmup period to ensure accurate estimates.

### 4. LossMonitorCallback

**Purpose:** Track loss history and detect anomalous loss spikes.

```python
from neuromanifold_gpt.callbacks import LossMonitorCallback

callback = LossMonitorCallback(
    log_interval=100,                     # Log metrics every 100 steps
    loss_history_size=100,                # Keep last 100 loss values for statistics
    loss_spike_threshold=3.0,             # Alert if loss > mean + 3*std
    min_loss_history_for_detection=20,    # Wait for 20 samples before detecting spikes
)
```

**Metrics Logged:**
- `train/loss_avg` - Average loss over history window

**Warnings:**
- Prints warning when loss spike detected (loss > mean + threshold*std)

**Note:** Spike detection helps identify training instability early.

### 5. AnomalyDetectorCallback

**Purpose:** Detect critical training issues (NaN/Inf in gradients and loss).

```python
from neuromanifold_gpt.callbacks import AnomalyDetectorCallback

callback = AnomalyDetectorCallback()  # No configuration needed
```

**Detects:**
- NaN values in loss
- Inf values in loss
- NaN values in gradients
- Inf values in gradients

**Warnings:**
- Prints critical warnings when anomalies detected
- Provides diagnostic suggestions (reduce learning rate, check data, etc.)

**Recommendation:** Always include this callback - it catches critical errors.

### 6. TrainingDashboardCallback

**Purpose:** Display training metrics in a Rich console dashboard.

```python
from neuromanifold_gpt.callbacks import TrainingDashboardCallback

callback = TrainingDashboardCallback(
    refresh_rate=100,          # Update dashboard every 100 steps
    enable_dashboard=True,     # Set to False to disable (e.g., for headless environments)
)
```

**Features:**
- Live-updating Rich console display
- Reads metrics from `trainer.logged_metrics` (logged by other callbacks)
- Shows step/epoch, loss, learning rate, gradients, memory, throughput, ETA
- Automatically disabled in non-TTY environments (e.g., CI/CD pipelines)

**Note:** This callback doesn't track metrics itself - it only displays metrics logged by other callbacks. Use it with other monitoring callbacks for best results.

---

## Composing Callbacks: Common Patterns

### Pattern 1: Production Training (Full Monitoring)

For production training runs where you want comprehensive observability:

```python
from neuromanifold_gpt.callbacks import (
    GradientMonitorCallback,
    MemoryMonitorCallback,
    ThroughputMonitorCallback,
    LossMonitorCallback,
    AnomalyDetectorCallback,
    TrainingDashboardCallback,
)

callbacks = [
    GradientMonitorCallback(),
    MemoryMonitorCallback(),
    ThroughputMonitorCallback(),
    LossMonitorCallback(),
    AnomalyDetectorCallback(),
    TrainingDashboardCallback(),
]

trainer = Trainer(max_steps=10000, callbacks=callbacks)
trainer.fit(model)
```

### Pattern 2: Debugging / Experimentation

For debugging or experimenting with hyperparameters:

```python
from neuromanifold_gpt.callbacks import (
    GradientMonitorCallback,
    LossMonitorCallback,
    AnomalyDetectorCallback,
)

# Focus on gradient and loss behavior
callbacks = [
    GradientMonitorCallback(
        grad_explosion_threshold=2.0,  # More sensitive detection
        min_grad_history_for_detection=10,
    ),
    LossMonitorCallback(
        loss_spike_threshold=2.0,  # More sensitive detection
        min_loss_history_for_detection=10,
    ),
    AnomalyDetectorCallback(),
]

trainer = Trainer(max_steps=1000, callbacks=callbacks)
trainer.fit(model)
```

### Pattern 3: Headless / CI/CD Environments

For automated runs without interactive terminals:

```python
from neuromanifold_gpt.callbacks import (
    LossMonitorCallback,
    AnomalyDetectorCallback,
    ThroughputMonitorCallback,
)

# Skip dashboard (no TTY), focus on metrics
callbacks = [
    LossMonitorCallback(),
    AnomalyDetectorCallback(),
    ThroughputMonitorCallback(),
    # Note: TrainingDashboardCallback automatically disables in non-TTY environments
]

trainer = Trainer(max_steps=10000, callbacks=callbacks)
trainer.fit(model)
```

### Pattern 4: Memory-Constrained Environments

When GPU memory is tight and you need to monitor it closely:

```python
from neuromanifold_gpt.callbacks import (
    MemoryMonitorCallback,
    AnomalyDetectorCallback,
)

callbacks = [
    MemoryMonitorCallback(log_interval=10),  # Frequent memory checks
    AnomalyDetectorCallback(),
]

trainer = Trainer(max_steps=10000, callbacks=callbacks)
trainer.fit(model)
```

### Pattern 5: Fast Iteration (Minimal Overhead)

For rapid experimentation where callback overhead matters:

```python
from neuromanifold_gpt.callbacks import AnomalyDetectorCallback

# Only critical anomaly detection
callbacks = [
    AnomalyDetectorCallback(),
]

trainer = Trainer(max_steps=1000, callbacks=callbacks)
trainer.fit(model)
```

---

## Step-by-Step Migration Guide

### Step 1: Identify Current Usage

Find where you're using `TrainingHealthCallback`:

```python
# Old code
from neuromanifold_gpt.callbacks import TrainingHealthCallback

callback = TrainingHealthCallback(
    log_interval=100,
    gradient_norm_history_size=100,
    enable_dashboard=True,
)

trainer.fit(model, callbacks=[callback])
```

### Step 2: Choose Your Replacement Pattern

Decide which monitoring features you need:

| Feature | Old Callback | New Callback |
|---------|--------------|--------------|
| Gradient tracking | ✅ Included | `GradientMonitorCallback` |
| Memory tracking | ✅ Included | `MemoryMonitorCallback` |
| Throughput/ETA | ✅ Included | `ThroughputMonitorCallback` |
| Loss tracking | ✅ Included | `LossMonitorCallback` |
| NaN/Inf detection | ✅ Included | `AnomalyDetectorCallback` |
| Dashboard display | ✅ Included | `TrainingDashboardCallback` |

### Step 3: Update Imports

Replace the single import with multiple imports:

```python
# New code
from neuromanifold_gpt.callbacks import (
    GradientMonitorCallback,
    MemoryMonitorCallback,
    ThroughputMonitorCallback,
    LossMonitorCallback,
    AnomalyDetectorCallback,
    TrainingDashboardCallback,
)
```

### Step 4: Create Callback List

Replace the single callback with a list of focused callbacks:

```python
# New code
callbacks = [
    GradientMonitorCallback(
        log_interval=100,
        gradient_norm_history_size=100,
    ),
    MemoryMonitorCallback(log_interval=100),
    ThroughputMonitorCallback(),
    LossMonitorCallback(),
    AnomalyDetectorCallback(),
    TrainingDashboardCallback(enable_dashboard=True),
]

trainer.fit(model, callbacks=callbacks)
```

### Step 5: Verify Behavior

Run a short training session and verify:

- ✅ Metrics are logged correctly
- ✅ Dashboard displays (if enabled)
- ✅ Anomaly detection works (test with intentionally bad hyperparameters)
- ✅ No regressions in training performance

### Step 6: Fine-Tune Configuration

Adjust individual callback parameters as needed:

```python
callbacks = [
    GradientMonitorCallback(
        grad_explosion_threshold=2.5,  # More sensitive
    ),
    MemoryMonitorCallback(log_interval=50),  # More frequent
    ThroughputMonitorCallback(warmup_steps=5),  # Faster ETA
    LossMonitorCallback(loss_spike_threshold=2.0),  # More sensitive
    AnomalyDetectorCallback(),
    TrainingDashboardCallback(refresh_rate=50),  # More frequent updates
]
```

---

## Configuration Examples

### Example 1: High-Sensitivity Monitoring

For critical training runs where you want to catch issues early:

```python
callbacks = [
    GradientMonitorCallback(
        log_interval=50,                       # Log frequently
        grad_explosion_threshold=2.0,          # Lower threshold (more sensitive)
        min_grad_history_for_detection=10,     # Detect early
    ),
    MemoryMonitorCallback(log_interval=50),
    ThroughputMonitorCallback(),
    LossMonitorCallback(
        log_interval=50,
        loss_spike_threshold=2.0,              # Lower threshold (more sensitive)
        min_loss_history_for_detection=10,     # Detect early
    ),
    AnomalyDetectorCallback(),
    TrainingDashboardCallback(refresh_rate=50),
]
```

### Example 2: Low-Overhead Monitoring

For long training runs where you want minimal callback overhead:

```python
callbacks = [
    GradientMonitorCallback(log_interval=500),  # Less frequent logging
    MemoryMonitorCallback(log_interval=500),
    ThroughputMonitorCallback(log_interval=500),
    LossMonitorCallback(log_interval=500),
    AnomalyDetectorCallback(),  # Always keep this - negligible overhead
    # Skip dashboard to reduce overhead
]
```

### Example 3: Gradient-Focused Monitoring

When debugging gradient-related issues:

```python
callbacks = [
    GradientMonitorCallback(
        log_interval=10,                       # Very frequent
        gradient_norm_history_size=200,        # Larger history
        grad_explosion_threshold=2.0,
        min_grad_history_for_detection=5,      # Detect very early
    ),
    AnomalyDetectorCallback(),  # Catch NaN/Inf in gradients
    TrainingDashboardCallback(refresh_rate=10),
]
```

### Example 4: Memory-Focused Monitoring

When optimizing memory usage:

```python
callbacks = [
    MemoryMonitorCallback(log_interval=10),  # Very frequent memory checks
    AnomalyDetectorCallback(),
    TrainingDashboardCallback(refresh_rate=10),
]
```

---

## Troubleshooting

### Issue: Dashboard not displaying

**Symptom:** No Rich dashboard appears during training.

**Possible Causes:**
1. Running in non-TTY environment (e.g., Jupyter notebook, CI/CD)
2. `enable_dashboard=False` in `TrainingDashboardCallback`
3. Dashboard callback not included in callbacks list

**Solution:**
```python
# Check if TTY is available
import sys
print(f"Is TTY: {sys.stdout.isatty()}")

# Ensure dashboard is enabled
callbacks = [
    TrainingDashboardCallback(enable_dashboard=True),
]
```

### Issue: Metrics not showing in dashboard

**Symptom:** Dashboard displays but some metrics are missing.

**Possible Cause:** Dashboard only displays metrics logged by other callbacks.

**Solution:** Ensure you include the corresponding monitoring callbacks:
```python
callbacks = [
    GradientMonitorCallback(),  # Required for gradient metrics
    MemoryMonitorCallback(),    # Required for memory metrics
    ThroughputMonitorCallback(), # Required for throughput/ETA metrics
    LossMonitorCallback(),      # Required for loss_avg metric
    TrainingDashboardCallback(),
]
```

### Issue: Too many warnings about gradient explosions

**Symptom:** Constant gradient explosion warnings during training.

**Possible Causes:**
1. Threshold too low (too sensitive)
2. Normal gradient variance in your model
3. Learning rate too high

**Solution:**
```python
# Option 1: Increase threshold (less sensitive)
GradientMonitorCallback(
    grad_explosion_threshold=5.0,  # Default is 3.0
)

# Option 2: Reduce learning rate
# Option 3: Enable gradient clipping in model config
```

### Issue: No warnings despite obvious training issues

**Symptom:** Training diverges but no anomaly warnings.

**Possible Causes:**
1. Not enough history for detection
2. Threshold too high (not sensitive enough)
3. AnomalyDetectorCallback not included

**Solution:**
```python
# Ensure AnomalyDetectorCallback is included
callbacks = [
    AnomalyDetectorCallback(),  # Critical - always include
]

# Lower thresholds for more sensitive detection
callbacks = [
    GradientMonitorCallback(grad_explosion_threshold=2.0),
    LossMonitorCallback(loss_spike_threshold=2.0),
]

# Reduce minimum history for earlier detection
callbacks = [
    GradientMonitorCallback(min_grad_history_for_detection=10),
    LossMonitorCallback(min_loss_history_for_detection=10),
]
```

### Issue: Performance degradation with callbacks

**Symptom:** Training is slower with new callbacks.

**Possible Causes:**
1. Too frequent logging (small `log_interval`)
2. Too many callbacks enabled
3. Dashboard overhead in non-TTY environment

**Solution:**
```python
# Reduce logging frequency
callbacks = [
    GradientMonitorCallback(log_interval=500),  # Default: 100
    MemoryMonitorCallback(log_interval=500),
    # ... other callbacks with increased log_interval
]

# Disable dashboard in non-TTY environments
callbacks = [
    # ... monitoring callbacks
    TrainingDashboardCallback(enable_dashboard=False),
]

# Or use only essential callbacks
callbacks = [
    AnomalyDetectorCallback(),  # Minimal overhead, critical functionality
]
```

### Issue: Deprecation warning with old callback

**Symptom:** `DeprecationWarning: TrainingHealthCallback is deprecated...`

**Solution:** This is expected - migrate to the new callbacks as shown in this guide.

---

## FAQ

### Q: Can I still use the old `TrainingHealthCallback`?

**A:** Yes, it still works but is deprecated. You'll see a deprecation warning. We recommend migrating to the new callbacks for better maintainability and flexibility.

### Q: Do I need to use all 6 new callbacks?

**A:** No! That's the beauty of the new architecture. Use only what you need. At minimum, we recommend `AnomalyDetectorCallback` for critical error detection.

### Q: Will the new callbacks affect training performance?

**A:** Minimal impact. Each callback has configurable `log_interval` to control overhead. For production, use `log_interval=100` or higher.

### Q: Can I create custom callbacks that work with these?

**A:** Absolutely! All callbacks follow standard PyTorch Lightning callback patterns. You can create custom callbacks that read from `trainer.logged_metrics` or log their own metrics.

### Q: How do I test my callback configuration?

**A:** Run a short training session (e.g., 100 steps) and verify metrics are logged correctly:
```python
trainer = Trainer(max_steps=100, callbacks=callbacks)
trainer.fit(model)
```

### Q: What if I don't need the dashboard?

**A:** Simply omit `TrainingDashboardCallback` from your callbacks list. The dashboard callback is optional.

### Q: Can I use these callbacks with non-PyTorch-Lightning code?

**A:** These callbacks are designed for PyTorch Lightning. For pure PyTorch, you'd need to adapt them or use different monitoring solutions.

### Q: Where are metrics logged?

**A:** Metrics are logged via `pl_module.log()` which writes to PyTorch Lightning's logger (TensorBoard, WandB, etc.). The dashboard reads from `trainer.logged_metrics`.

---

## Summary

The refactored callback architecture provides:

1. **Flexibility**: Use only the monitoring you need
2. **Clarity**: Each callback has a single, well-defined purpose
3. **Testability**: Each callback is independently testable
4. **Maintainability**: Smaller, focused files are easier to maintain
5. **Backward Compatibility**: Old callback still works during migration

**Recommended Migration Path:**
1. Start with full monitoring (all 6 callbacks)
2. Run a test training session
3. Remove callbacks you don't need
4. Fine-tune configuration based on your requirements

**Minimum Recommended Setup:**
```python
callbacks = [
    AnomalyDetectorCallback(),  # Critical error detection
    LossMonitorCallback(),      # Track training progress
]
```

**Questions or Issues?**
- Check the troubleshooting section above
- Review the test file: `neuromanifold_gpt/tests/test_callbacks.py`
- Examine callback source code for detailed implementation
