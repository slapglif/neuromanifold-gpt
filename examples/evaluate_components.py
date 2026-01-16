"""
Evaluate component-specific metrics from a trained NeuroManifoldGPT model.

This script demonstrates how to:
- Load a trained model from checkpoint
- Run a forward pass on sample data
- Compute component-specific metrics (SDR, FHN, MTP, Memory)
- Display the metrics in a readable format

Usage:
    python examples/evaluate_components.py
    python examples/evaluate_components.py --out_dir=out-custom
    python examples/evaluate_components.py --num_batches=5 --batch_size=8
"""
import os
import sys

# Handle --help before any imports that require dependencies
if '--help' in sys.argv or '-h' in sys.argv:
    print(__doc__)
    print("\nConfiguration parameters:")
    print("  --out_dir=<path>        Model checkpoint directory (default: 'out')")
    print("  --dataset=<name>        Dataset name for evaluation (default: 'shakespeare_char')")
    print("  --num_batches=<int>     Number of batches to evaluate (default: 10)")
    print("  --batch_size=<int>      Batch size (default: 12)")
    print("  --block_size=<int>      Context length (default: 1024)")
    print("  --device=<str>          Device to use: 'cpu', 'cuda', etc. (default: 'cuda')")
    print("  --dtype=<str>           Data type: 'float32', 'bfloat16', 'float16' (default: auto)")
    print("  --seed=<int>            Random seed (default: 1337)")
    print("  --compile=<bool>        Use PyTorch 2.0 compilation (default: False)")
    sys.exit(0)

from contextlib import nullcontext
from dataclasses import dataclass
import numpy as np
import torch
import pickle

# Add parent directory to path to import neuromanifold_gpt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neuromanifold_gpt.model.gpt import NeuroManifoldGPT
from neuromanifold_gpt.config.base import NeuroManifoldConfig
from neuromanifold_gpt.config.loader import load_config
from neuromanifold_gpt.evaluation import ComponentMetricsAggregator

# -----------------------------------------------------------------------------
@dataclass
class EvaluationConfig:
    """Configuration for component metrics evaluation."""
    out_dir: str = 'out'  # directory containing ckpt.pt
    dataset: str = 'shakespeare_char'  # dataset name for loading eval data
    num_batches: int = 10  # number of batches to evaluate
    batch_size: int = 12  # batch size for evaluation
    block_size: int = 1024  # context length
    device: str = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    dtype: str = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    seed: int = 1337
    compile: bool = False  # use PyTorch 2.0 to compile the model to be faster

# Load config with CLI overrides
config = load_config(EvaluationConfig, show_help=True)
# -----------------------------------------------------------------------------

torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in config.device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config.dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Load model from checkpoint
print(f"Loading model from {config.out_dir}...")
ckpt_path = os.path.join(config.out_dir, 'ckpt.pt')
if not os.path.exists(ckpt_path):
    print(f"Error: Checkpoint not found at {ckpt_path}")
    print("Please train a model first or specify a valid --out_dir")
    sys.exit(1)

# Weights only load issue in PyTorch 2.6+ with custom configs (trust local source)
checkpoint = torch.load(ckpt_path, map_location=config.device, weights_only=False)

# Check if it's a NeuroManifold checkpoint (has 'config' object)
if 'config' not in checkpoint or not isinstance(checkpoint['config'], NeuroManifoldConfig):
    print("Error: This script requires a NeuroManifoldGPT checkpoint with component metrics support")
    print("Standard GPT checkpoints are not supported.")
    sys.exit(1)

print("Loading NeuroManifoldGPT model...")
nm_config = checkpoint['config']
model = NeuroManifoldGPT(nm_config)

state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict)
model.eval()
model.to(config.device)

if config.compile:
    print("Compiling model...")
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

print(f"Model loaded successfully!")
print(f"  - Layers: {nm_config.n_layer}")
print(f"  - Embedding dim: {nm_config.n_embd}")
print(f"  - Heads: {nm_config.n_head}")
print(f"  - SDR enabled: {nm_config.use_sdr}")
print(f"  - Block size: {nm_config.block_size}")

# Load data for evaluation
data_dir = os.path.join('data', config.dataset)
if os.path.exists(os.path.join(data_dir, 'val.bin')):
    print(f"\nLoading validation data from {data_dir}...")
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    def get_batch(split):
        data = val_data
        ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+config.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+config.block_size]).astype(np.int64)) for i in ix])
        x, y = x.pin_memory().to(config.device, non_blocking=True), y.pin_memory().to(config.device, non_blocking=True)
        return x, y
else:
    print(f"\nWarning: Validation data not found at {data_dir}")
    print("Using random data for demonstration...")

    # Get vocab size from config
    vocab_size = nm_config.vocab_size

    def get_batch(split):
        x = torch.randint(vocab_size, (config.batch_size, config.block_size), device=config.device)
        y = torch.randint(vocab_size, (config.batch_size, config.block_size), device=config.device)
        return x, y

# Initialize metrics aggregator
aggregator = ComponentMetricsAggregator()

# Evaluate model and collect component metrics
print(f"\nEvaluating {config.num_batches} batches...")
all_metrics = {
    'loss': [],
    'sdr': {},
    'fhn': {},
    'mtp': {},
    'memory': {}
}

with torch.no_grad():
    with ctx:
        for batch_idx in range(config.num_batches):
            X, Y = get_batch('val')

            # Forward pass with component metrics
            logits, loss, info = model(X, Y)

            # Compute component-specific metrics
            batch_metrics = aggregator.compute(info, nm_config, logits=logits, targets=Y)

            # Accumulate loss
            all_metrics['loss'].append(loss.item())

            # Accumulate component metrics
            for component, metrics_dict in batch_metrics.items():
                for metric_name, metric_value in metrics_dict.items():
                    if metric_name not in all_metrics[component]:
                        all_metrics[component][metric_name] = []
                    all_metrics[component][metric_name].append(metric_value)

            # Print progress
            if (batch_idx + 1) % max(1, config.num_batches // 10) == 0:
                print(f"  Batch {batch_idx + 1}/{config.num_batches} - Loss: {loss.item():.4f}")

# Compute averages
print("\n" + "="*60)
print("COMPONENT-SPECIFIC EVALUATION METRICS")
print("="*60)

# Overall loss
avg_loss = np.mean(all_metrics['loss'])
print(f"\nOverall Loss: {avg_loss:.4f}")

# SDR Metrics
if all_metrics['sdr']:
    print("\nSDR Metrics (Semantic Folding):")
    print("-" * 60)
    for metric_name, values in all_metrics['sdr'].items():
        avg_value = np.mean(values)
        std_value = np.std(values)
        print(f"  {metric_name:30s}: {avg_value:8.4f} ± {std_value:.4f}")

# FHN Metrics
if all_metrics['fhn']:
    print("\nFHN Metrics (Wave Stability):")
    print("-" * 60)
    for metric_name, values in all_metrics['fhn'].items():
        avg_value = np.mean(values)
        std_value = np.std(values)
        print(f"  {metric_name:30s}: {avg_value:8.4f} ± {std_value:.4f}")

# MTP Metrics
if all_metrics['mtp']:
    print("\nMTP Metrics (Token Prediction):")
    print("-" * 60)
    for metric_name, values in all_metrics['mtp'].items():
        avg_value = np.mean(values)
        std_value = np.std(values)
        print(f"  {metric_name:30s}: {avg_value:8.4f} ± {std_value:.4f}")

# Memory Metrics
if all_metrics['memory']:
    print("\nMemory Metrics (Engram Utilization):")
    print("-" * 60)
    for metric_name, values in all_metrics['memory'].items():
        avg_value = np.mean(values)
        std_value = np.std(values)
        print(f"  {metric_name:30s}: {avg_value:8.4f} ± {std_value:.4f}")

print("\n" + "="*60)
print("Evaluation complete!")
print("="*60)
