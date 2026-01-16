"""
Zero-shot benchmark evaluation script for NeuroManifoldGPT and GPT models.

Evaluate trained models on standard NLP benchmarks:
- LAMBADA: Perplexity on final word prediction
- HellaSwag: Commonsense reasoning (multiple choice)
- PIQA: Physical commonsense reasoning (multiple choice)
- WinoGrande: Winograd schema challenge (multiple choice)
- MMLU: Massive Multitask Language Understanding (multiple choice)
- ARC: AI2 Reasoning Challenge (multiple choice)

Usage:
    python eval.py --help
    python eval.py --out_dir=out --benchmark=lambada
    python eval.py --out_dir=out --benchmark=all
    python eval.py config/eval_lambada.py
    python eval.py --out_dir=out --benchmark=hellaswag --device=cpu
"""

import sys
import os

# Handle --help before any imports that require dependencies
if '--help' in sys.argv or '-h' in sys.argv:
    print(__doc__)
    print("\nConfiguration parameters:")
    print("  --out_dir=<path>           Checkpoint directory (default: 'out')")
    print("  --benchmark=<name>         Benchmark to evaluate: lambada, hellaswag, piqa, winogrande, mmlu, arc, arc-easy, arc-challenge, all (default: 'lambada')")
    print("  --eval_iters=<int>         Max examples to evaluate, None=all (default: None)")
    print("  --device=<str>             Device: 'cpu', 'cuda', 'cuda:0', etc. (default: 'cuda')")
    print("  --dtype=<str>              Data type: 'float32', 'bfloat16', 'float16' (default: auto)")
    print("  --seed=<int>               Random seed (default: 1337)")
    print("  --compile=<bool>           Use PyTorch 2.0 compilation (default: False)")
    print("  --wandb_log=<bool>         Log results to wandb (default: False)")
    print("  --wandb_project=<str>      WandB project name (default: 'neuromanifold-eval')")
    print("  --wandb_run_name=<str>     WandB run name (default: auto)")
    print("\nExamples:")
    print("  python eval.py --out_dir=out --benchmark=lambada")
    print("  python eval.py --out_dir=out --benchmark=all --eval_iters=100")
    print("  python eval.py config/eval_lambada.py --wandb_log=True")
    sys.exit(0)

import pickle
from contextlib import nullcontext
import torch
import tiktoken
from rich.console import Console
from rich.table import Table

from neuromanifold_gpt.model.gpt import NeuroManifoldGPT
from neuromanifold_gpt.config.base import NeuroManifoldConfig
from neuromanifold_gpt.utils.checkpoints import select_checkpoint
from neuromanifold_gpt.utils.progress import checkpoint_progress
from neuromanifold_gpt.utils.logging import get_logger
from neuromanifold_gpt.benchmarks.zero_shot import evaluate_lambada, evaluate_multiple_choice, evaluate_mmlu, evaluate_arc
from model import GPT, GPTConfig

logger = get_logger(__name__)
console = Console()

# -----------------------------------------------------------------------------
# Default Configuration
# -----------------------------------------------------------------------------
out_dir = 'out'  # checkpoint directory
benchmark = 'lambada'  # benchmark to evaluate: lambada, hellaswag, piqa, winogrande, mmlu, arc, arc-easy, arc-challenge, all
eval_iters = None  # max examples to evaluate (None = all examples)
device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # 'float32' or 'bfloat16' or 'float16'
seed = 1337
compile = False  # use PyTorch 2.0 to compile the model to be faster
wandb_log = False  # log results to wandb
wandb_project = 'neuromanifold-eval'
wandb_run_name = None  # auto-generated if None

exec(open('configurator.py').read())  # overrides from command line or config file
# -----------------------------------------------------------------------------

# Set random seeds
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Validate benchmark
valid_benchmarks = ['lambada', 'hellaswag', 'piqa', 'winogrande', 'mmlu', 'arc', 'arc-easy', 'arc-challenge', 'all']
if benchmark not in valid_benchmarks:
    raise ValueError(f"Invalid benchmark: {benchmark}. Must be one of {valid_benchmarks}")

# Determine which benchmarks to run
if benchmark == 'all':
    benchmarks_to_run = ['lambada', 'hellaswag', 'piqa', 'winogrande', 'mmlu', 'arc-easy', 'arc-challenge']
else:
    benchmarks_to_run = [benchmark]

# Expand 'arc' into both easy and challenge variants
if 'arc' in benchmarks_to_run:
    idx = benchmarks_to_run.index('arc')
    benchmarks_to_run[idx:idx+1] = ['arc-easy', 'arc-challenge']

# Load model from checkpoint
logger.section(f"Loading model from {out_dir}")
ckpt_path = select_checkpoint(out_dir)
if ckpt_path is None:
    raise FileNotFoundError(f"No checkpoints found in {out_dir}")

# Weights only load issue in PyTorch 2.6+ with custom configs (trust local source)
with checkpoint_progress("Loading checkpoint from disk"):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

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
model.to(device)

if compile:
    logger.info("Compiling model with torch.compile...")
    model = torch.compile(model)  # requires PyTorch 2.0 (optional)

logger.info("Model loaded successfully!")

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
if wandb_log:
    import wandb
    run_name = wandb_run_name or f"eval-{benchmark}"
    wandb.init(project=wandb_project, name=run_name, config={
        'out_dir': out_dir,
        'benchmark': benchmark,
        'eval_iters': eval_iters,
        'device': device,
        'dtype': dtype,
        'seed': seed,
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
            device=device,
            dtype=dtype,
            max_examples=eval_iters,
            verbose=True,
        )
    elif bench == 'mmlu':
        results = evaluate_mmlu(
            model=model,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            max_examples=eval_iters,
            verbose=True,
        )
    elif bench in ['arc-easy', 'arc-challenge']:
        # Extract variant from benchmark name (arc-easy -> easy, arc-challenge -> challenge)
        variant = bench.split('-')[1]
        results = evaluate_arc(
            model=model,
            tokenizer=tokenizer,
            variant=variant,
            device=device,
            dtype=dtype,
            max_examples=eval_iters,
            verbose=True,
        )
    else:
        # Standard multiple choice benchmarks: hellaswag, piqa, winogrande
        results = evaluate_multiple_choice(
            model=model,
            tokenizer=tokenizer,
            benchmark=bench,
            device=device,
            dtype=dtype,
            max_examples=eval_iters,
            verbose=True,
        )

    all_results[bench] = results

    # Log to wandb
    if wandb_log:
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
if wandb_log:
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

logger.info("Evaluation complete!")
