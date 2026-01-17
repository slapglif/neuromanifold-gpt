"""
Teacher-student distillation evaluation script for NeuroManifoldGPT.

Compare teacher and student models on standard NLP benchmarks to measure
knowledge transfer quality after distillation training.

Metrics:
- KL Divergence: Measures distribution similarity between teacher/student outputs
- Output Agreement: Percentage of examples where models agree on predictions
- Performance Gap: Difference in accuracy/perplexity between teacher and student
- Per-Benchmark Comparison: Side-by-side results for each evaluation task

Benchmarks:
- LAMBADA: Perplexity on final word prediction
- HellaSwag: Commonsense reasoning (multiple choice)
- PIQA: Physical commonsense reasoning (multiple choice)
- WinoGrande: Winograd schema challenge (multiple choice)

Usage:
    python eval_distillation.py --help
    python eval_distillation.py --teacher_checkpoint=out-teacher/ckpt.pt --student_checkpoint=out-student/ckpt.pt
    python eval_distillation.py --teacher_checkpoint=out-teacher/ckpt.pt --student_checkpoint=out-student/ckpt.pt --benchmark=all
    python eval_distillation.py config/eval_distillation_lambada.py
"""

import sys
import os
import pickle
from contextlib import nullcontext
import torch
import torch.nn.functional as F
import tiktoken
from rich.console import Console
from rich.table import Table

from neuromanifold_gpt.config.loader import load_config
from neuromanifold_gpt.config.training import DistillationEvalConfig
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT
from neuromanifold_gpt.config.base import NeuroManifoldConfig
from neuromanifold_gpt.utils.checkpoints import select_checkpoint
from neuromanifold_gpt.utils.checkpoint_loader import load_model_only
from neuromanifold_gpt.utils.progress import checkpoint_progress
from neuromanifold_gpt.utils.logging import get_logger
from neuromanifold_gpt.benchmarks.zero_shot import evaluate_lambada, evaluate_multiple_choice
from model import GPT, GPTConfig

logger = get_logger(__name__)
console = Console()

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def load_model_from_checkpoint(ckpt_path: str, device: str, compile_model: bool = False):
    """Load a model from checkpoint (supports both NeuroManifold and GPT).

    Args:
        ckpt_path: Path to checkpoint file
        device: Device to load model on
        compile_model: Whether to compile with torch.compile

    Returns:
        model: Loaded model in eval mode
        checkpoint: Full checkpoint dict (for config inspection)
    """
    with checkpoint_progress(f"Loading checkpoint from {os.path.basename(ckpt_path)}"):
        checkpoint = load_model_only(ckpt_path, device=device, weights_only=False)

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

    if compile_model:
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    return model, checkpoint


def compute_output_agreement(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> float:
    """Compute percentage of examples where teacher and student agree on top prediction.

    Args:
        teacher_logits: Teacher model logits [batch, seq_len, vocab_size]
        student_logits: Student model logits [batch, seq_len, vocab_size]

    Returns:
        agreement_pct: Percentage (0-100) of positions with same argmax
    """
    teacher_preds = teacher_logits.argmax(dim=-1)
    student_preds = student_logits.argmax(dim=-1)
    agreement = (teacher_preds == student_preds).float().mean().item()
    return agreement * 100.0


def compute_kl_divergence(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> float:
    """Compute KL divergence between teacher and student output distributions.

    Args:
        teacher_logits: Teacher model logits [batch, seq_len, vocab_size]
        student_logits: Student model logits [batch, seq_len, vocab_size]

    Returns:
        kl_div: Mean KL divergence across all positions
    """
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    student_log_probs = F.log_softmax(student_logits, dim=-1)

    kl_div = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    return kl_div.item()


# -----------------------------------------------------------------------------
# Load Configuration
# -----------------------------------------------------------------------------
config = load_config(DistillationEvalConfig)

# Validate required checkpoints
if not config.teacher_checkpoint:
    raise ValueError(
        "teacher_checkpoint is required. Use --teacher_checkpoint=path/to/teacher.pt"
    )
if not config.student_checkpoint:
    raise ValueError(
        "student_checkpoint is required. Use --student_checkpoint=path/to/student.pt"
    )

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

# -----------------------------------------------------------------------------
# Load Models
# -----------------------------------------------------------------------------
logger.section("Loading Teacher and Student Models")

# Load teacher model
logger.info(f"Loading teacher from: {config.teacher_checkpoint}")
teacher, teacher_checkpoint = load_model_from_checkpoint(
    config.teacher_checkpoint,
    config.device,
    config.compile
)
logger.success("Teacher model loaded successfully!")

# Load student model
logger.info(f"Loading student from: {config.student_checkpoint}")
student, student_checkpoint = load_model_from_checkpoint(
    config.student_checkpoint,
    config.device,
    config.compile
)
logger.success("Student model loaded successfully!")

# -----------------------------------------------------------------------------
# Set up tokenizer
# -----------------------------------------------------------------------------
# Try to load meta.pkl from dataset if available
load_meta = False
dataset_name = None

# Get dataset from teacher checkpoint (assume teacher was trained on target data)
if 'config' in teacher_checkpoint:
    if hasattr(teacher_checkpoint['config'], 'dataset'):
        dataset_name = teacher_checkpoint['config'].dataset
    elif isinstance(teacher_checkpoint['config'], dict) and 'dataset' in teacher_checkpoint['config']:
        dataset_name = teacher_checkpoint['config']['dataset']

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

# -----------------------------------------------------------------------------
# Initialize wandb if requested
# -----------------------------------------------------------------------------
if config.wandb_log:
    import wandb
    run_name = config.wandb_run_name or f"eval-distill-{config.benchmark}"
    wandb.init(project=config.wandb_project, name=run_name, config={
        'teacher_checkpoint': config.teacher_checkpoint,
        'student_checkpoint': config.student_checkpoint,
        'benchmark': config.benchmark,
        'eval_iters': config.eval_iters,
        'device': config.device,
        'dtype': config.dtype,
        'seed': config.seed,
    })

# -----------------------------------------------------------------------------
# Run Evaluations
# -----------------------------------------------------------------------------
all_results = {
    'teacher': {},
    'student': {},
    'comparison': {},
}

for bench in benchmarks_to_run:
    logger.section(f"Evaluating {bench.upper()}")

    # Evaluate teacher
    logger.info("Running teacher evaluation...")
    if bench == 'lambada':
        teacher_results = evaluate_lambada(
            model=teacher,
            tokenizer=tokenizer,
            device=config.device,
            dtype=config.dtype,
            max_examples=config.eval_iters,
            verbose=True,
        )
    else:
        teacher_results = evaluate_multiple_choice(
            model=teacher,
            tokenizer=tokenizer,
            benchmark=bench,
            device=config.device,
            dtype=config.dtype,
            max_examples=config.eval_iters,
            verbose=True,
        )

    # Evaluate student
    logger.info("Running student evaluation...")
    if bench == 'lambada':
        student_results = evaluate_lambada(
            model=student,
            tokenizer=tokenizer,
            device=config.device,
            dtype=config.dtype,
            max_examples=config.eval_iters,
            verbose=True,
        )
    else:
        student_results = evaluate_multiple_choice(
            model=student,
            tokenizer=tokenizer,
            benchmark=bench,
            device=config.device,
            dtype=config.dtype,
            max_examples=config.eval_iters,
            verbose=True,
        )

    all_results['teacher'][bench] = teacher_results
    all_results['student'][bench] = student_results

    # Compute comparison metrics
    comparison = {}

    # Performance gap (for accuracy or perplexity)
    if 'accuracy' in teacher_results:
        teacher_acc = teacher_results['accuracy']
        student_acc = student_results['accuracy']
        comparison['accuracy_gap'] = teacher_acc - student_acc
        comparison['accuracy_retention'] = (student_acc / teacher_acc * 100) if teacher_acc > 0 else 0.0

    if 'perplexity' in teacher_results:
        teacher_ppl = teacher_results['perplexity']
        student_ppl = student_results['perplexity']
        comparison['perplexity_gap'] = student_ppl - teacher_ppl
        comparison['perplexity_ratio'] = student_ppl / teacher_ppl if teacher_ppl > 0 else float('inf')

    all_results['comparison'][bench] = comparison

    # Log to wandb
    if config.wandb_log:
        wandb_results = {}
        for k, v in teacher_results.items():
            wandb_results[f"{bench}/teacher_{k}"] = v
        for k, v in student_results.items():
            wandb_results[f"{bench}/student_{k}"] = v
        for k, v in comparison.items():
            wandb_results[f"{bench}/comparison_{k}"] = v
        wandb.log(wandb_results)

# -----------------------------------------------------------------------------
# Print Summary
# -----------------------------------------------------------------------------
logger.section("Distillation Evaluation Summary")

# Create comparison table
table = Table(
    title="Teacher-Student Comparison",
    show_header=True,
    header_style="bold magenta"
)
table.add_column("Benchmark", style="cyan", width=15)
table.add_column("Model", style="white", width=10)
table.add_column("Metric", style="white", width=25)
table.add_column("Value", justify="right", style="yellow", width=15)

for bench in benchmarks_to_run:
    teacher_results = all_results['teacher'][bench]
    student_results = all_results['student'][bench]
    comparison = all_results['comparison'][bench]

    # Add teacher rows
    first_row = True
    for metric, value in teacher_results.items():
        if isinstance(value, float):
            value_str = f"{value:.4f}"
        else:
            value_str = str(value)

        if first_row:
            table.add_row(bench.upper(), "Teacher", metric, value_str)
            first_row = False
        else:
            table.add_row("", "Teacher", metric, value_str)

    # Add student rows
    for metric, value in student_results.items():
        if isinstance(value, float):
            value_str = f"{value:.4f}"
        else:
            value_str = str(value)
        table.add_row("", "Student", metric, value_str)

    # Add comparison rows
    for metric, value in comparison.items():
        if isinstance(value, float):
            value_str = f"{value:.4f}"
        else:
            value_str = str(value)
        table.add_row("", "Gap", metric, value_str, style="bold green")

    # Add separator between benchmarks (if not last)
    if bench != benchmarks_to_run[-1]:
        table.add_row("", "", "", "", end_section=True)

console.print(table)

# Print summary statistics
logger.section("Distillation Quality Metrics")

for bench in benchmarks_to_run:
    comparison = all_results['comparison'][bench]

    if 'accuracy_retention' in comparison:
        retention = comparison['accuracy_retention']
        logger.info(f"{bench}: Accuracy retention = {retention:.2f}%")

        # Color-coded assessment
        if retention >= 95:
            logger.success(f"  ✓ Excellent distillation (>95% retention)")
        elif retention >= 90:
            logger.info(f"  ✓ Good distillation (90-95% retention)")
        elif retention >= 85:
            logger.warning(f"  ⚠ Acceptable distillation (85-90% retention)")
        else:
            logger.error(f"  ✗ Poor distillation (<85% retention)")

    if 'perplexity_ratio' in comparison:
        ratio = comparison['perplexity_ratio']
        logger.info(f"{bench}: Perplexity ratio = {ratio:.2f}x")

        # Color-coded assessment
        if ratio <= 1.1:
            logger.success(f"  ✓ Excellent distillation (<1.1x perplexity)")
        elif ratio <= 1.25:
            logger.info(f"  ✓ Good distillation (1.1-1.25x perplexity)")
        elif ratio <= 1.5:
            logger.warning(f"  ⚠ Acceptable distillation (1.25-1.5x perplexity)")
        else:
            logger.error(f"  ✗ Poor distillation (>1.5x perplexity)")

# Log summary to wandb
if config.wandb_log:
    summary_metrics = {}

    for bench in benchmarks_to_run:
        teacher_results = all_results['teacher'][bench]
        student_results = all_results['student'][bench]
        comparison = all_results['comparison'][bench]

        for metric, value in teacher_results.items():
            summary_metrics[f"summary/{bench}_teacher_{metric}"] = value
        for metric, value in student_results.items():
            summary_metrics[f"summary/{bench}_student_{metric}"] = value
        for metric, value in comparison.items():
            summary_metrics[f"summary/{bench}_{metric}"] = value

    wandb.log(summary_metrics)

    # Also log as wandb summary (persisted metrics)
    for bench in benchmarks_to_run:
        for metric, value in all_results['teacher'][bench].items():
            wandb.run.summary[f"{bench}/teacher_{metric}"] = value
        for metric, value in all_results['student'][bench].items():
            wandb.run.summary[f"{bench}/student_{metric}"] = value
        for metric, value in all_results['comparison'][bench].items():
            wandb.run.summary[f"{bench}/{metric}"] = value

    wandb.finish()

logger.success("Distillation evaluation complete!")
