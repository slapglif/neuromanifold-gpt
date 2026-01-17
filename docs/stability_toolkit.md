# Training Stability Toolkit

**Comprehensive debugging utilities for training stability: SDR collapse detection, automatic checkpoint rollback, attention visualization, and log diagnostics**

This guide explains how to use NeuroManifoldGPT's Training Stability Toolkit to detect and recover from common training issues. The toolkit provides automated monitoring, recovery mechanisms, and diagnostic tools to accelerate experimentation with novel architectures.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [SDR Collapse Detection](#sdr-collapse-detection)
4. [Automatic Checkpoint Rollback](#automatic-checkpoint-rollback)
5. [Attention Pattern Visualization](#attention-pattern-visualization)
6. [CLI Diagnostic Tool](#cli-diagnostic-tool)
7. [Configuration Reference](#configuration-reference)
8. [Warning Interpretation](#warning-interpretation)
9. [Troubleshooting Guide](#troubleshooting-guide)
10. [Best Practices](#best-practices)

---

## Overview

The Training Stability Toolkit provides four main components:

| Component | Purpose | Use Case |
|-----------|---------|----------|
| **SDR Collapse Monitor** | Detects when SDR representations degenerate | Early warning for representation collapse |
| **Divergence Rollback** | Automatically rolls back to stable checkpoint | Automatic recovery from training instabilities |
| **Attention Visualization** | Exports attention patterns as PNG images | Visual debugging of attention mechanisms |
| **CLI Diagnostic Tool** | Analyzes training logs for common issues | Post-mortem analysis and CI/CD integration |

### Key Features

- **Proactive Monitoring**: Detect issues early before they cause complete training failure
- **Automatic Recovery**: Roll back to stable checkpoints without manual intervention
- **Visual Debugging**: Export attention patterns for visual analysis
- **Rich Console Output**: Color-coded warnings with actionable remediation steps
- **CI/CD Integration**: Non-zero exit codes for critical issues detected in logs

---

## Quick Start

### Enable All Stability Features

The fastest way to enable all stability monitoring during training:

```bash
python train.py \
  --enable_sdr_collapse_monitor \
  --enable_divergence_rollback \
  --enable_attention_viz
```

### Basic Configuration Example

```python
from neuromanifold_gpt.training.config import TrainConfig

config = TrainConfig(
    # Enable stability toolkit
    enable_sdr_collapse_monitor=True,
    enable_divergence_rollback=True,
    enable_attention_viz=True,

    # Basic training config
    batch_size=64,
    max_iters=10000,
    out_dir="out-stable",
)
```

### Analyze Training Logs

After training, analyze logs for stability issues:

```bash
python -m neuromanifold_gpt.cli.diagnose logs/training.log
```

---

## SDR Collapse Detection

The `SDRCollapseMonitor` callback tracks SDR representation health and warns when representations degenerate.

### What is SDR Collapse?

SDR (Sparse Distributed Representation) collapse occurs when:
- **Low diversity**: Most SDRs become similar or identical
- **Bit concentration**: Only a few bits are actively used
- **Pattern repetition**: Limited unique patterns in the representation space
- **Temperature issues**: Temperature parameter moves out of optimal range

### Configuration

Enable via command line:

```bash
python train.py \
  --enable_sdr_collapse_monitor \
  --sdr_check_interval=100 \
  --sdr_collapse_threshold=0.3
```

Or via Python config:

```python
from neuromanifold_gpt.training.config import TrainConfig

config = TrainConfig(
    enable_sdr_collapse_monitor=True,
    sdr_check_interval=100,          # Check every 100 steps
    sdr_collapse_threshold=0.3,      # Warn if <30% unique patterns
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_sdr_collapse_monitor` | bool | False | Enable the SDR collapse monitor |
| `sdr_check_interval` | int | 100 | Number of steps between SDR health checks |
| `sdr_collapse_threshold` | float | 0.3 | Unique pattern ratio threshold (0.0-1.0) |

### Monitored Metrics

The callback tracks these metrics automatically:

- **Active Bits**: Number of active bits per SDR (should match target sparsity)
- **Unique Patterns**: Ratio of unique SDR patterns in batch
- **Bit Usage**: Fraction of total bits that are actively used
- **Temperature**: Current temperature parameter value
- **Duty Cycle**: Distribution of bit activation frequency

### Example Output

When SDR collapse is detected:

```
⚠ SDR COLLAPSE DETECTED at step 1500: Only 18.3% unique patterns.
Consider: (1) Increasing temperature, (2) Checking token discrimination loss,
(3) Reviewing SDR dimensionality
```

When diversity is low but not critical:

```
⚠ Low SDR diversity (severity: medium) at step 2200:
Unique pattern ratio dropped to 28.5%. Active bits: 156.2/512 (30.5%)
```

### Training Summary

At the end of training, a summary is printed:

```
╭─────────── SDR Collapse Monitor Summary ───────────╮
│                                                     │
│  Average Active Bits: 164.3                         │
│  Average Unique Pattern Ratio: 82.4%                │
│  Average Bit Usage Ratio: 76.2%                     │
│  Average Temperature: 0.023450                      │
│                                                      │
│  ✓ No SDR collapse detected                         │
│  Total warnings: 0                                  │
╰─────────────────────────────────────────────────────╯
```

---

## Automatic Checkpoint Rollback

The `DivergenceRollbackCallback` automatically detects training divergence and rolls back to a stable checkpoint.

### What Triggers Rollback?

Rollback is triggered when:
- **Loss spikes**: Loss > 2x recent average for N consecutive steps (configurable)
- **NaN/Inf detection**: Loss or gradients become NaN or infinite
- **Gradient explosion**: Gradients exceed normal range significantly

### Configuration

Enable via command line:

```bash
python train.py \
  --enable_divergence_rollback \
  --divergence_threshold=2.0 \
  --rollback_checkpoint_interval=500
```

Or via Python config:

```python
from neuromanifold_gpt.training.config import TrainConfig

config = TrainConfig(
    enable_divergence_rollback=True,
    divergence_threshold=2.0,            # Trigger at 2x average loss
    rollback_checkpoint_interval=500,    # Save checkpoint every 500 steps
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_divergence_rollback` | bool | False | Enable automatic rollback |
| `divergence_threshold` | float | 2.0 | Loss multiplier to trigger divergence detection |
| `rollback_checkpoint_interval` | int | 500 | Steps between rollback checkpoint saves |

### How It Works

1. **Periodic Checkpointing**: Saves checkpoints every N steps to `out_dir/rollback_checkpoints/`
2. **Loss Monitoring**: Tracks rolling loss statistics (mean and standard deviation)
3. **Divergence Detection**: Detects when loss exceeds threshold for consecutive steps
4. **Automatic Rollback**: Loads best recent checkpoint (lowest loss)
5. **State Restoration**: Restores both model weights and optimizer state
6. **Training Continuation**: Resumes training from stable checkpoint

### Example Output

When divergence is detected:

```
⚠ Divergence warning (1/3) at step 3450:
Loss=8.2340 > 2.0x avg (4.1123)
```

```
⚠ Divergence warning (2/3) at step 3451:
Loss=9.1234 > 2.0x avg (4.2156)
```

```
🔄 ROLLBACK TRIGGERED at step 3452
Finding best checkpoint...
Rolling back to checkpoint at step 3000 (loss=3.8921)
✓ Model and optimizer state restored successfully
```

### Checkpoint Management

Rollback checkpoints are saved to `out_dir/rollback_checkpoints/`:

```
out/rollback_checkpoints/
├── rollback_ckpt-000500-3.8234.ckpt
├── rollback_ckpt-001000-3.6123.ckpt
├── rollback_ckpt-001500-3.5012.ckpt
└── rollback_ckpt-002000-3.4567.ckpt
```

The callback maintains a fixed number of recent checkpoints (default: 10) and automatically cleans up old ones.

### Training Summary

```
╭────── Divergence Rollback Summary ──────╮
│                                          │
│  Total Rollbacks: 2                      │
│  Best Checkpoint: step 4500 (loss=2.89)  │
│                                          │
╰──────────────────────────────────────────╯
```

---

## Attention Pattern Visualization

The `AttentionVisualizationCallback` exports attention patterns as PNG heatmaps during training.

### Why Visualize Attention?

Attention visualization helps debug:
- **Attention collapse**: All queries attend to the same position
- **Diagonal patterns**: Model only attends to current position
- **Scattered patterns**: Model fails to learn meaningful attention
- **Head specialization**: Different heads learn different patterns

### Configuration

Enable via command line:

```bash
python train.py \
  --enable_attention_viz \
  --attention_viz_interval=500 \
  --attention_viz_max_seq_len=64
```

Or via Python config:

```python
from neuromanifold_gpt.training.config import TrainConfig

config = TrainConfig(
    enable_attention_viz=True,
    attention_viz_interval=500,        # Save every 500 steps
    attention_viz_max_seq_len=64,     # Visualize up to 64 tokens
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_attention_viz` | bool | False | Enable attention visualization |
| `attention_viz_interval` | int | 500 | Steps between visualization saves |
| `attention_viz_max_seq_len` | int | 64 | Maximum sequence length to visualize |

### Output Files

Visualizations are saved to `out_dir/attention_viz/`:

```
out/attention_viz/
├── step_000500_layer0.png
├── step_000500_layer1.png
├── step_000500_layer2.png
├── step_001000_layer0.png
└── ...
```

### Visualization Features

The attention visualizations include:

- **Heatmaps**: Color-coded attention weights (darker = higher attention)
- **Multi-head support**: Separate plots for each attention head or averaged
- **Layer-wise tracking**: Visualize attention evolution across layers
- **High resolution**: 300 DPI PNG images suitable for publications

### Programmatic Usage

You can also use the visualization utilities directly:

```python
from neuromanifold_gpt.utils.attention_viz import (
    visualize_attention_pattern,
    compare_attention_patterns,
    plot_attention_entropy,
)

# Visualize single attention pattern
visualize_attention_pattern(
    attention_weights,  # Shape: (num_heads, seq_len, seq_len)
    save_path="attention_step1000.png",
    title="Attention Pattern at Step 1000",
)

# Compare two attention patterns
compare_attention_patterns(
    attention_a,  # Standard attention
    attention_b,  # NeuroManifold attention
    save_path="attention_comparison.png",
    title_a="Standard Attention",
    title_b="NeuroManifold Attention",
)

# Plot attention entropy to detect collapse
plot_attention_entropy(
    attention_weights,
    save_path="attention_entropy.png",
    title="Attention Entropy Over Sequence",
)
```

---

## CLI Diagnostic Tool

The `diagnose` CLI command analyzes training logs to identify stability issues.

### Basic Usage

Analyze a single log file:

```bash
python -m neuromanifold_gpt.cli.diagnose logs/training.log
```

Analyze all logs in a directory:

```bash
python -m neuromanifold_gpt.cli.diagnose logs/
```

### Command Line Options

```bash
python -m neuromanifold_gpt.cli.diagnose --help
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `path` | str | required | Log file or directory to analyze |
| `--verbose` | flag | False | Show detailed analysis with metrics |
| `--issue-type` | str | all | Filter by issue type (loss_spike, nan, gradient_explosion, sdr_collapse) |
| `--loss-spike-threshold` | float | 3.0 | Standard deviations above mean to trigger loss spike detection |
| `--gradient-explosion-threshold` | float | 3.0 | Standard deviations above mean for gradient explosion |
| `--output-format` | str | rich | Output format: rich, json, or csv |

### Detected Issues

The diagnostic tool detects:

| Issue Type | Severity | Description |
|------------|----------|-------------|
| `nan_detected` | CRITICAL | NaN values in loss or gradients |
| `inf_detected` | CRITICAL | Inf values in loss or gradients |
| `sdr_collapse` | CRITICAL | SDR collapse detected by monitor |
| `loss_spike` | WARNING | Sudden large increase in loss |
| `gradient_explosion` | WARNING | Gradient norm exceeds normal range |
| `divergence` | WARNING | Loss divergence pattern detected |
| `rollback` | INFO | Automatic rollback occurred |

### Example Output

#### Summary Report

```
╭─────────────────────────────────────────╮
│  Analyzed 1 log file(s)                 │
│  Found 8 issues                         │
╰─────────────────────────────────────────╯

Issue Summary:
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━┓
┃ Issue Type          ┃ Count ┃ Severity ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━┩
│ loss_spike          │ 3     │ WARNING  │
│ gradient_explosion  │ 2     │ WARNING  │
│ sdr_collapse        │ 1     │ CRITICAL │
│ nan_detected        │ 1     │ CRITICAL │
│ rollback            │ 1     │ INFO     │
└─────────────────────┴───────┴──────────┘

Timeline (first 50 issues):
┏━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Step  ┃ Issue Type          ┃ Severity  ┃ Message                    ┃
┡━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1234  │ loss_spike          │ WARNING   │ Loss spike: 12.34 vs ...   │
│ 1235  │ gradient_explosion  │ WARNING   │ Gradient explosion: 145... │
│ 2456  │ sdr_collapse        │ CRITICAL  │ SDR collapse: 15.3% unique │
│ 2457  │ nan_detected        │ CRITICAL  │ NaN detected in loss       │
│ 2458  │ rollback            │ INFO      │ Rollback to step 2000      │
└───────┴─────────────────────┴───────────┴────────────────────────────┘

⚠ Critical Issues Detected
Suggested Actions:
• Reduce learning rate to improve training stability
• Enable gradient clipping (consider max_norm=1.0)
• Check for data quality issues causing extreme loss values
• Enable automatic checkpoint rollback for recovery
```

#### Verbose Output

With `--verbose`, get detailed analysis:

```bash
python -m neuromanifold_gpt.cli.diagnose logs/training.log --verbose
```

```
Detailed Analysis:

loss_spike (3 occurrences):
  Step 1234: loss=12.3456 (mean=3.2345, std=1.2345)
  Step 2345: loss=15.6789 (mean=3.4567, std=1.3456)
  Step 3456: loss=11.2345 (mean=3.5678, std=1.4567)

  Remediation:
  • Reduce learning rate by 2-5x
  • Enable gradient clipping (max_norm=1.0)
  • Check batch composition for outliers
  • Enable divergence rollback callback

gradient_explosion (2 occurrences):
  Step 1235: grad_norm=145.23 (mean=2.34, std=1.56)
  Step 2346: grad_norm=203.45 (mean=2.45, std=1.67)

  Remediation:
  • Enable gradient clipping immediately
  • Reduce learning rate
  • Check for numerical instability in loss calculation
  • Review model architecture for unstable components
```

### Exit Codes

The diagnostic tool returns meaningful exit codes for CI/CD integration:

- `0`: No issues detected or only INFO-level issues
- `1`: WARNING or CRITICAL issues detected

Use in CI/CD:

```bash
#!/bin/bash
python -m neuromanifold_gpt.cli.diagnose logs/ || {
    echo "Training stability issues detected!"
    exit 1
}
```

### JSON Output

For programmatic processing:

```bash
python -m neuromanifold_gpt.cli.diagnose logs/ --output-format=json > report.json
```

```json
{
  "summary": {
    "total_files": 1,
    "total_issues": 8,
    "critical_count": 2,
    "warning_count": 5,
    "info_count": 1
  },
  "issues": [
    {
      "step": 1234,
      "issue_type": "loss_spike",
      "severity": "WARNING",
      "message": "Loss spike detected",
      "metrics": {
        "loss": 12.3456,
        "mean_loss": 3.2345,
        "std_loss": 1.2345
      }
    }
  ]
}
```

---

## Configuration Reference

### Complete Example

All stability toolkit options with recommended values:

```python
from neuromanifold_gpt.training.config import TrainConfig

config = TrainConfig(
    # Basic training
    out_dir="out-stable",
    batch_size=64,
    max_iters=100000,
    learning_rate=3e-4,

    # SDR Collapse Monitoring
    enable_sdr_collapse_monitor=True,
    sdr_check_interval=100,              # Check every 100 steps
    sdr_collapse_threshold=0.3,          # Warn if <30% unique patterns

    # Automatic Rollback
    enable_divergence_rollback=True,
    divergence_threshold=2.0,            # Trigger at 2x average loss
    rollback_checkpoint_interval=500,    # Save checkpoint every 500 steps

    # Attention Visualization
    enable_attention_viz=True,
    attention_viz_interval=1000,         # Visualize every 1000 steps
    attention_viz_max_seq_len=64,       # Visualize up to 64 tokens
)
```

### Recommended Settings by Use Case

#### Research / Experimentation

```python
config = TrainConfig(
    enable_sdr_collapse_monitor=True,
    sdr_check_interval=50,              # Frequent checks
    enable_divergence_rollback=True,
    divergence_threshold=1.5,           # Aggressive detection
    enable_attention_viz=True,
    attention_viz_interval=500,         # Frequent visualization
)
```

#### Production / Long Training Runs

```python
config = TrainConfig(
    enable_sdr_collapse_monitor=True,
    sdr_check_interval=500,             # Less frequent checks
    enable_divergence_rollback=True,
    divergence_threshold=2.5,           # More tolerant
    rollback_checkpoint_interval=1000,  # Less frequent checkpoints
    enable_attention_viz=False,         # Disable for performance
)
```

#### Debugging / Issue Investigation

```python
config = TrainConfig(
    enable_sdr_collapse_monitor=True,
    sdr_check_interval=10,              # Very frequent checks
    sdr_collapse_threshold=0.4,         # Early warnings
    enable_divergence_rollback=True,
    divergence_threshold=1.2,           # Very aggressive
    rollback_checkpoint_interval=100,   # Very frequent checkpoints
    enable_attention_viz=True,
    attention_viz_interval=100,         # Frequent visualization
)
```

---

## Warning Interpretation

### SDR Collapse Warnings

#### Low Diversity (Severity: High)

```
⚠ SDR COLLAPSE DETECTED at step 1500: Only 18.3% unique patterns.
```

**Meaning**: Most SDRs in the batch are identical or very similar. The model is losing representational capacity.

**Actions**:
1. **Increase temperature**: Try 2-3x current value
2. **Check token discrimination loss**: Verify it's decreasing
3. **Review SDR dimensionality**: May need larger SDR dimension
4. **Inspect training data**: Check for data quality issues

#### Low Diversity (Severity: Medium)

```
⚠ Low SDR diversity (severity: medium) at step 2200:
Unique pattern ratio dropped to 28.5%. Active bits: 156.2/512 (30.5%)
```

**Meaning**: Diversity is below threshold but not critically low. Monitor closely.

**Actions**:
1. **Monitor trend**: Check if it's improving or degrading
2. **Consider temperature adjustment**: Small increase (1.2-1.5x)
3. **Review recent changes**: Did you modify hyperparameters recently?

#### Bit Usage Warning

```
⚠ Poor bit usage (severity: medium) at step 3100:
Only 45.2% of bits actively used. Average duty cycle std: 0.156
```

**Meaning**: The model is only using a subset of available bits, limiting capacity.

**Actions**:
1. **Check duty cycle regularization**: Ensure it's enabled and properly weighted
2. **Increase SDR sparsity**: Try higher k value
3. **Review initialization**: Poor initialization can cause bit concentration

#### Temperature Warning

```
⚠ Temperature out of range (severity: high) at step 4200:
Temperature=0.001234 is very low. SDR selection may be too deterministic.
```

**Meaning**: Temperature has drifted to extreme values, affecting SDR sampling.

**Actions**:
1. **Check temperature schedule**: Verify it's configured correctly
2. **Review temperature bounds**: May need to constrain temperature updates
3. **Inspect gradient flow**: Very low temperature can cause gradient issues

### Divergence Warnings

#### Gradual Divergence

```
⚠ Divergence warning (1/3) at step 3450:
Loss=8.2340 > 2.0x avg (4.1123)
```

**Meaning**: Loss is elevated but rollback hasn't triggered yet. You have time to investigate.

**Actions**:
1. **Monitor next steps**: See if it recovers or continues diverging
2. **Check learning rate**: May be too high
3. **Review batch composition**: Could be a difficult batch
4. **Inspect gradients**: Look for gradient explosion signs

#### NaN/Inf Detection

```
⚠ NaN/Inf detected at step 5600. Immediate rollback required.
```

**Meaning**: Training has become numerically unstable. Immediate action required.

**Actions**:
1. **Enable automatic rollback**: Will handle this automatically
2. **Enable gradient clipping**: Prevents gradient explosion
3. **Check loss calculation**: Look for division by zero or log(0)
4. **Review learning rate**: Often caused by excessive learning rate

#### Rollback Event

```
🔄 ROLLBACK TRIGGERED at step 3452
Rolling back to checkpoint at step 3000 (loss=3.8921)
✓ Model and optimizer state restored successfully
```

**Meaning**: Automatic recovery in progress. Training will continue from stable checkpoint.

**Actions**:
1. **Monitor post-rollback**: Ensure divergence doesn't recur
2. **Adjust hyperparameters**: Fix root cause (learning rate, clipping, etc.)
3. **Review trigger cause**: Check logs to understand what caused divergence

---

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. SDR Collapse Not Detected

**Symptom**: You suspect collapse but no warnings appear.

**Solutions**:
- Check `sdr_check_interval`: May be too infrequent
- Lower `sdr_collapse_threshold`: Current threshold may be too lenient
- Verify `enable_sdr_collapse_monitor=True`
- Check model architecture: Ensure it has SDR components (`model.encoder`)

#### 2. Too Many False Positive Warnings

**Symptom**: Constant warnings for minor fluctuations.

**Solutions**:
- Increase `sdr_collapse_threshold`: Try 0.4 or 0.5 for more tolerance
- Increase `sdr_check_interval`: Reduce check frequency
- Adjust `divergence_threshold`: Increase to 2.5 or 3.0 for less sensitivity
- Review data quality: Ensure training data is clean and consistent

#### 3. Rollback Not Triggering

**Symptom**: Training diverges but rollback doesn't activate.

**Solutions**:
- Verify `enable_divergence_rollback=True`
- Check `divergence_threshold`: May be too high (try 1.5 instead of 2.0)
- Reduce `consecutive_divergence_steps`: Try 2 instead of 3
- Ensure sufficient loss history: Need at least 20 samples for detection

#### 4. Rollback Happens Too Frequently

**Symptom**: Training constantly rolls back, making no progress.

**Solutions**:
- Increase `divergence_threshold`: Try 2.5 or 3.0
- Increase `consecutive_divergence_steps`: Require more consecutive divergent steps
- Reduce learning rate: Root cause is likely excessive learning rate
- Enable gradient clipping: Prevents gradient explosion

#### 5. Attention Visualizations Not Saving

**Symptom**: No PNG files appear in output directory.

**Solutions**:
- Verify `enable_attention_viz=True`
- Check `out_dir` exists and is writable
- Ensure matplotlib is installed: `pip install matplotlib`
- Check model architecture: Must have standard transformer attention layers
- Review logs for error messages

#### 6. CLI Diagnose Finds No Issues

**Symptom**: Log analysis returns no issues despite training problems.

**Solutions**:
- Check log format: Ensure logs contain metrics (loss, grad_norm, etc.)
- Adjust thresholds: Try `--loss-spike-threshold=2.0` for more sensitivity
- Use `--verbose`: See detailed statistics and detection logic
- Verify log file path: Ensure pointing to correct log file
- Check that metrics are being logged: Training must log loss and grad_norm

#### 7. High Memory Usage

**Symptom**: Stability toolkit causes OOM (Out of Memory) errors.

**Solutions**:
- Reduce checkpoint frequency: Increase `rollback_checkpoint_interval`
- Disable attention visualization: Set `enable_attention_viz=False`
- Reduce history size: Callbacks keep limited history (100 samples by default)
- Limit attention visualization: Reduce `attention_viz_max_seq_len` to 32

#### 8. Performance Impact

**Symptom**: Training is noticeably slower with stability toolkit.

**Solutions**:
- Increase check intervals: `sdr_check_interval=500`, `rollback_checkpoint_interval=1000`
- Disable attention visualization: Most expensive component
- Reduce `attention_viz_max_seq_len`: Smaller visualizations are faster
- Use conditional enabling: Only enable during known problematic periods

---

## Best Practices

### Development Workflow

1. **Initial Experimentation**
   ```python
   config = TrainConfig(
       enable_sdr_collapse_monitor=True,
       enable_divergence_rollback=True,
       enable_attention_viz=True,
       # Aggressive settings for rapid iteration
       sdr_check_interval=50,
       divergence_threshold=1.5,
       attention_viz_interval=500,
   )
   ```

2. **Stability Verification**
   ```bash
   # After training completes
   python -m neuromanifold_gpt.cli.diagnose logs/ --verbose
   ```

3. **Production Deployment**
   ```python
   config = TrainConfig(
       enable_sdr_collapse_monitor=True,
       enable_divergence_rollback=True,
       enable_attention_viz=False,  # Disable for performance
       # Conservative settings for long runs
       sdr_check_interval=500,
       divergence_threshold=2.5,
   )
   ```

### Hyperparameter Tuning

When experimenting with novel architectures:

1. **Start Conservative**: Use default thresholds and aggressive monitoring
2. **Baseline First**: Train baseline GPT to establish normal behavior
3. **Compare Metrics**: Use CLI tool to compare logs between runs
4. **Iterative Adjustment**: Gradually adjust thresholds based on observed behavior
5. **Document Issues**: Keep notes on what warnings correspond to actual problems

### CI/CD Integration

Include stability checks in your CI pipeline:

```bash
#!/bin/bash
# train_and_validate.sh

# Run training
python train.py \
  --enable_sdr_collapse_monitor \
  --enable_divergence_rollback \
  --max_iters=1000

# Check for stability issues
python -m neuromanifold_gpt.cli.diagnose logs/ --output-format=json > report.json

# Fail CI if critical issues detected
exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "Training stability issues detected!"
    cat report.json
    exit 1
fi

echo "Training completed successfully with no stability issues"
```

### Logging Best Practices

Ensure your training setup logs the metrics needed for diagnosis:

```python
# In your training loop
trainer.log("loss", loss)
trainer.log("grad_norm", grad_norm)
trainer.log("learning_rate", current_lr)

# For SDR monitoring
if hasattr(model, 'encoder'):
    trainer.log("sdr/active_bits", active_bits)
    trainer.log("sdr/unique_patterns", unique_pattern_ratio)
    trainer.log("sdr/temperature", temperature)
```

### Visualization Analysis

When analyzing attention patterns:

1. **Compare Across Steps**: Look at evolution over training
2. **Check All Layers**: Different layers learn different patterns
3. **Look for Trends**: Gradual degradation vs. sudden collapse
4. **Compare to Baseline**: Train standard GPT for comparison
5. **Use Entropy Plots**: Quantify attention diversity numerically

### Resource Management

Balance monitoring depth with resource constraints:

| Resource Constraint | Recommended Settings |
|---------------------|---------------------|
| **Memory Limited** | `rollback_checkpoint_interval=2000`, `enable_attention_viz=False` |
| **Disk Limited** | `max_checkpoints=5`, `attention_viz_interval=2000` |
| **Compute Limited** | `sdr_check_interval=500`, `enable_attention_viz=False` |
| **All Constrained** | Only enable during suspected issues, disable otherwise |

### Debugging Checklist

When investigating training instability:

- [ ] Enable all stability monitors
- [ ] Set aggressive thresholds (low divergence_threshold, high sdr_collapse_threshold)
- [ ] Enable attention visualization with frequent saves
- [ ] Log all available metrics
- [ ] Run CLI diagnostic on logs after each experiment
- [ ] Compare against baseline GPT training
- [ ] Document all warnings and when they occur
- [ ] Correlate warnings with hyperparameter changes
- [ ] Check for data quality issues
- [ ] Verify gradient clipping is enabled

---

## Conclusion

The Training Stability Toolkit provides comprehensive monitoring and recovery mechanisms for stable training. By enabling proactive detection, automatic recovery, and rich diagnostics, it accelerates experimentation with novel architectures where stability is not guaranteed.

For additional support:
- Review training logs with the CLI diagnostic tool
- Consult the [Configuration Reference](configuration-reference.md) for full parameter documentation
- Check [Checkpoint Management](checkpoint-management.md) for checkpoint-related details
- See example configs in `neuromanifold_gpt/config/presets/`

Happy stable training! 🚀
