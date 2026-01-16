"""
Zero-shot benchmark evaluation script for NeuroManifoldGPT and GPT models.

Evaluate trained models on standard NLP benchmarks:
- LAMBADA: Perplexity on final word prediction
- HellaSwag: Commonsense reasoning (multiple choice)
- PIQA: Physical commonsense reasoning (multiple choice)
- WinoGrande: Winograd schema challenge (multiple choice)

Usage:
    python eval.py --help
    python eval.py --out_dir=out --benchmark=lambada
    python eval.py --out_dir=out --benchmark=all
    python eval.py config/eval_lambada.py
    python eval.py --out_dir=out --benchmark=hellaswag --device=cpu
"""

import sys
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from rich.console import Console
from rich.table import Table

from neuromanifold_gpt.config.loader import load_config
from neuromanifold_gpt.config.training import EvalConfig
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT
from neuromanifold_gpt.config.base import NeuroManifoldConfig
from neuromanifold_gpt.utils.checkpoints import select_checkpoint
from neuromanifold_gpt.utils.progress import checkpoint_progress
from neuromanifold_gpt.utils.logging import get_logger
from neuromanifold_gpt.benchmarks.zero_shot import evaluate_lambada, evaluate_multiple_choice
from model import GPT, GPTConfig

logger = get_logger(__name__)
console = Console()

# -----------------------------------------------------------------------------
# Load Configuration
# -----------------------------------------------------------------------------
config = load_config(EvalConfig)

# Set random seeds
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in config.device else 'cpu'  # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config.dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Validate benchmark
valid_benchmarks = ['lambada', 'hellaswag', 'piqa', 'winogrande', 'all']
if config.benchmark not in valid_benchmarks:
    raise ValueError(f"Invalid benchmark: {config.benchmark}. Must be one of {valid_benchmarks}")

# Determine which benchmarks to run
if config.benchmark == 'all':
    benchmarks_to_run = ['lambada', 'hellaswag', 'piqa', 'winogrande']
else:
    benchmarks_to_run = [config.benchmark]

# Load model from checkpoint
logger.section(f"Loading model from {config.out_dir}")
ckpt_path = select_checkpoint(config.out_dir)
if ckpt_path is None:
    raise FileNotFoundError(f"No checkpoints found in {config.out_dir}")

# Weights only load issue in PyTorch 2.6+ with custom configs (trust local source)
with checkpoint_progress("Loading checkpoint from disk"):
    checkpoint = torch.load(ckpt_path, map_location=config.device, weights_only=False)

# Check if it's a NeuroManifold checkpoint (has 'config' object) or legacy nanoGPT
if 'config' in checkpoint and isinstance(checkpoint['config'], (NeuroManifoldConfig, type(None))):
    logger.info("Loading NeuroManifoldGPT model...")
    nm_config = checkpoint['config']
    model = NeuroManifoldGPT(nm_config)
else:
    # Legacy/Standard GPT
    logger.info("Loading standard GPT model...")
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)

state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

with checkpoint_progress("Loading model weights"):
    model.load_state_dict(state_dict)

model.eval()
model.to(config.device)

if config.compile:
    logger.info("Compiling model with torch.compile...")
    model = torch.compile(model)  # requires PyTorch 2.0 (optional)

logger.success("Model loaded successfully!")

# Set up tokenizer
# Try to load meta.pkl from dataset if available
load_meta = False
dataset_name = None

# Handle NeuroManifold config (object) vs Dict config
if 'config' in checkpoint:
    if hasattr(checkpoint['config'], 'dataset'):
        dataset_name = checkpoint['config'].dataset
    elif isinstance(checkpoint['config'], dict) and 'dataset' in checkpoint['config']:
        dataset_name = checkpoint['config']['dataset']

# Check if meta.pkl exists for this dataset
if dataset_name:
    meta_path = os.path.join('data', dataset_name, 'meta.pkl')
    load_meta = os.path.exists(meta_path)

# Create tokenizer
class Tokenizer:
    """Simple wrapper for encoding/decoding text."""
    def __init__(self, encode_fn, decode_fn):
        self._encode = encode_fn
        self._decode = decode_fn

    def encode(self, text):
        return self._encode(text)

    def decode(self, tokens):
        return self._decode(tokens)

if load_meta:
    logger.info(f"Loading tokenizer from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode_fn = lambda s: [stoi[c] for c in s]
    decode_fn = lambda l: ''.join([itos[i] for i in l])
    tokenizer = Tokenizer(encode_fn, decode_fn)
else:
    # Use GPT-2 encodings by default (standard for benchmarks)
    logger.info("No meta.pkl found, using GPT-2 tokenizer...")
    enc = tiktoken.get_encoding("gpt2")
    encode_fn = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode_fn = lambda l: enc.decode(l)
    tokenizer = Tokenizer(encode_fn, decode_fn)

# Initialize wandb if requested
if config.wandb_log:
    import wandb
    run_name = config.wandb_run_name or f"eval-{config.benchmark}"
    wandb.init(project=config.wandb_project, name=run_name, config={
        'out_dir': config.out_dir,
        'benchmark': config.benchmark,
        'eval_iters': config.eval_iters,
        'device': config.device,
        'dtype': config.dtype,
        'seed': config.seed,
        'checkpoint': os.path.basename(ckpt_path),
    })

# Run evaluations
all_results = {}

for bench in benchmarks_to_run:
    logger.section(f"Evaluating {bench.upper()}")

    if bench == 'lambada':
        results = evaluate_lambada(
            model=model,
            tokenizer=tokenizer,
            device=config.device,
            dtype=config.dtype,
            max_examples=config.eval_iters,
            verbose=True,
        )
    else:
        results = evaluate_multiple_choice(
            model=model,
            tokenizer=tokenizer,
            benchmark=bench,
            device=config.device,
            dtype=config.dtype,
            max_examples=config.eval_iters,
            verbose=True,
        )

    all_results[bench] = results

    # Log to wandb
    if config.wandb_log:
        wandb_results = {f"{bench}/{k}": v for k, v in results.items()}
        wandb.log(wandb_results)

# Print summary using rich table
logger.section("Evaluation Summary")

# Create table for results
table = Table(title="Benchmark Evaluation Results", show_header=True, header_style="bold magenta")
table.add_column("Benchmark", style="cyan", width=15)
table.add_column("Metric", style="white", width=25)
table.add_column("Value", justify="right", style="yellow", width=15)

for bench, results in all_results.items():
    # Add benchmark rows
    first_row = True
    for metric, value in results.items():
        if isinstance(value, float):
            value_str = f"{value:.4f}"
        else:
            value_str = str(value)

        if first_row:
            table.add_row(bench.upper(), metric, value_str)
            first_row = False
        else:
            table.add_row("", metric, value_str)

    # Add separator between benchmarks (if not last)
    if bench != list(all_results.keys())[-1]:
        table.add_row("", "", "", end_section=True)

console.print(table)

# Log summary to wandb
if config.wandb_log:
    # Create summary metrics for wandb
    summary_metrics = {}
    for bench, results in all_results.items():
        for metric, value in results.items():
            summary_metrics[f"summary/{bench}_{metric}"] = value

    wandb.log(summary_metrics)

    # Also log as wandb summary (persisted metrics)
    for bench, results in all_results.items():
        for metric, value in results.items():
            wandb.run.summary[f"{bench}/{metric}"] = value

    wandb.finish()

logger.success("Evaluation complete!")
