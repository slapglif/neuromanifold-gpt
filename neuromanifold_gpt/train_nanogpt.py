"""
Training script for NeuroManifoldGPT following Karpathy's exact nanoGPT methodology.

Supports:
- Single GPU training
- Distributed Data Parallel (DDP) across multiple GPUs/nodes
- Mixed precision (float16/bfloat16)
- Gradient accumulation
- Cosine LR with warmup
- Checkpointing and resume

Single GPU:
    $ python neuromanifold_gpt/train_nanogpt.py --batch_size=32 --compile_model=False

DDP on 4 GPUs:
    $ torchrun --standalone --nproc_per_node=4 neuromanifold_gpt/train_nanogpt.py

DDP across 2 nodes (4 GPUs each):
    # Master node:
    $ torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=<IP> --master_port=1234 neuromanifold_gpt/train_nanogpt.py
    # Worker node:
    $ torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=<IP> --master_port=1234 neuromanifold_gpt/train_nanogpt.py
"""
import math
import os
import time

from neuromanifold_gpt.cli.help_formatter import (
    create_parser_from_defaults,
    parse_args_with_config_override,
)

# Lazy imports for heavy dependencies (allows --help to work without them)
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Progress utilities (imported at top since they're lightweight)
from neuromanifold_gpt.utils.progress import progress_bar

# -----------------------------------------------------------------------------
# Default config values designed to train NeuroManifoldGPT
defaults = {
    # I/O
    "out_dir": "out-neuromanifold",
    "eval_interval": 2000,
    "log_interval": 1,
    "eval_iters": 200,
    "eval_only": False,
    "always_save_checkpoint": True,
    "init_from": "scratch",  # 'scratch' or 'resume'
    # Logging
    "wandb_log": False,
    "wandb_project": "neuromanifold-gpt",
    "wandb_run_name": f"run{int(time.time())}",
    # Data
    "dataset": "openwebtext",
    "gradient_accumulation_steps": 40,  # 5 * 8 - simulate larger batches
    "batch_size": 12,  # micro-batch size
    "block_size": 1024,
    # Model - NeuroManifold specific
    "use_nano_config": False,  # Use NeuroManifoldConfigNano for fast experimentation
    "n_layer": 6,
    "n_head": 8,
    "n_embd": 384,
    "sdr_size": 2048,
    "sdr_sparsity": 0.02,
    "manifold_dim": 64,
    "n_eigenvectors": 32,
    "dropout": 0.0,
    "bias": False,
    # Optimizer
    "learning_rate": 6e-4,
    "max_iters": 600000,
    "weight_decay": 1e-1,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,
    "early_stopping_patience": 0,  # 0 to disable, >0 for patience steps
    # Learning Rate Schedule
    "decay_lr": True,
    "warmup_iters": 2000,
    "lr_decay_iters": 600000,
    "min_lr": 6e-5,  # ~= learning_rate/10 per Chinchilla
    # DDP
    "backend": "nccl",
    # System
    "device": "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu",
    "dtype": "bfloat16"
    if TORCH_AVAILABLE and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16",
    "compile_model": False,  # Disabled: Python 3.13 doesn't support torch.compile
}

# Define argument groups for organized help output
argument_groups = {
    "I/O": [
        "out_dir",
        "eval_interval",
        "log_interval",
        "eval_iters",
        "eval_only",
        "always_save_checkpoint",
        "init_from",
    ],
    "Logging": ["wandb_log", "wandb_project", "wandb_run_name"],
    "Data": ["dataset", "gradient_accumulation_steps", "batch_size", "block_size"],
    "Model": [
        "use_nano_config",
        "n_layer",
        "n_head",
        "n_embd",
        "sdr_size",
        "sdr_sparsity",
        "manifold_dim",
        "n_eigenvectors",
        "dropout",
        "bias",
    ],
    "Optimizer": [
        "learning_rate",
        "max_iters",
        "weight_decay",
        "beta1",
        "beta2",
        "grad_clip",
        "early_stopping_patience",
    ],
    "Learning Rate Schedule": ["decay_lr", "warmup_iters", "lr_decay_iters", "min_lr"],
    "DDP": ["backend"],
    "System": ["device", "dtype", "compile_model"],
}

# Usage examples
examples = [
    "python neuromanifold_gpt/train_nanogpt.py --batch_size=32",
    "python neuromanifold_gpt/train_nanogpt.py config/my_config.py --learning_rate=1e-4",
    "torchrun --standalone --nproc_per_node=4 neuromanifold_gpt/train_nanogpt.py",
]

# -----------------------------------------------------------------------------
# Parse config from command line or config file
if __name__ == "__main__":
    # Create parser with rich formatting
    parser = create_parser_from_defaults(
        defaults=defaults,
        description="Train NeuroManifoldGPT following nanoGPT methodology",
        groups=argument_groups,
        examples=examples,
    )

    # Parse arguments with config file override support
    args = parse_args_with_config_override(parser)

    # Convert to config dict for backward compatibility
    config = vars(args)
    # Remove 'config' key (the config file path) from the dict
    config.pop("config", None)

    # Update module-level variables for use in the script
    for key, value in config.items():
        globals()[key] = value

    # Import heavy dependencies after argparse (so --help works without them)
    import pickle
    from contextlib import nullcontext

    import numpy as np
    import torch
    from torch.distributed import destroy_process_group, init_process_group
    from torch.nn.parallel import DistributedDataParallel as DDP

    from neuromanifold_gpt.config import NeuroManifoldConfig, NeuroManifoldConfigNano
    from neuromanifold_gpt.model.gpt import NeuroManifoldGPT
else:
    # When imported as a module, use defaults
    for key, value in defaults.items():
        globals()[key] = value
    config = defaults.copy()

    # Import heavy dependencies
    import pickle
    from contextlib import nullcontext

    import numpy as np
    import torch
    from torch.distributed import destroy_process_group, init_process_group
    from torch.nn.parallel import DistributedDataParallel as DDP

    from neuromanifold_gpt.config import NeuroManifoldConfig, NeuroManifoldConfigNano
    from neuromanifold_gpt.model.gpt import NeuroManifoldGPT

# -----------------------------------------------------------------------------
# DDP setup
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
if master_process:
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

# -----------------------------------------------------------------------------
# Setup
if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)


# -----------------------------------------------------------------------------
# Poor man's data loader
data_dir = os.path.join("data", dataset)


def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a batch of data.

    Uses memmap to avoid memory leak.
    """
    if split == "train":
        data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    else:
        data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )
    if device_type == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# -----------------------------------------------------------------------------
# Model init
iter_num = 0
best_val_loss = 1e9
early_stopping_counter = 0

# Attempt to derive vocab_size from dataset
meta_path = os.path.join(data_dir, "meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    if master_process:
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

vocab_size = meta_vocab_size if meta_vocab_size is not None else 50304

if init_from == "scratch":
    if master_process:
        print("Initializing a new NeuroManifoldGPT model from scratch")

    if use_nano_config:
        # Use nano config for fast experimentation
        model_config = NeuroManifoldConfigNano()
        model_config.vocab_size = vocab_size
        model_config.block_size = min(block_size, 256)  # nano has 256 max
    else:
        # Full config
        model_config = NeuroManifoldConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layer=n_layer,
            n_heads=n_head,
            n_embd=n_embd,
            sdr_size=sdr_size,
            sdr_sparsity=sdr_sparsity,
            manifold_dim=manifold_dim,
            n_eigenvectors=n_eigenvectors,
            dropout=dropout,
            bias=bias,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            beta1=beta1,
            beta2=beta2,
            grad_clip=grad_clip,
        )

    model = NeuroManifoldGPT(model_config)

elif init_from == "resume":
    if master_process:
        print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    # Recreate config from checkpoint
    checkpoint_config = checkpoint["model_config"]
    model_config = NeuroManifoldConfig(**checkpoint_config)
    model = NeuroManifoldGPT(model_config)
    state_dict = checkpoint["model"]
    # Fix state dict keys if needed
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
else:
    raise ValueError(f"Unknown init_from: {init_from}")

# Adjust block size if needed
if block_size < model_config.block_size:
    model_config.block_size = block_size
    if master_process:
        print(f"Cropped block_size to {block_size}")

model.to(device)

if master_process:
    print(f"Model has {model.num_parameters() / 1e6:.2f}M parameters")

# -----------------------------------------------------------------------------
# GradScaler for fp16
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

# Optimizer
optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device_type
)
if init_from == "resume" and "optimizer" in checkpoint:
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # Free memory

# Compile
if compile_model and device_type == "cuda":
    if master_process:
        print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)

# Wrap in DDP
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


# -----------------------------------------------------------------------------
# Evaluation
@torch.no_grad()
def estimate_loss() -> dict[str, float]:
    """Estimate loss on train and val splits."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        # Only show progress on master process to avoid duplicate output in DDP
        iterator = range(eval_iters)
        if master_process:
            iterator = progress_bar(
                iterator, description=f"Evaluating {split}", total=eval_iters
            )

        for k in iterator:
            X, Y = get_batch(split)
            with ctx:
                logits, loss, info = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# -----------------------------------------------------------------------------
# Learning rate scheduler (cosine with warmup)
def get_lr(it: int) -> float:
    """Cosine learning rate with linear warmup."""
    # Linear warmup
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # After decay, return min_lr
    if it > lr_decay_iters:
        return min_lr
    # Cosine decay
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# -----------------------------------------------------------------------------
# MFU estimation for NeuroManifold
def estimate_mfu(batch_size: int, dt: float) -> float:
    """Estimate model flops utilization (MFU).

    Adapted for NeuroManifold architecture:
    - SDR ops are mostly sparse (~2% active bits)
    - Spectral attention is O(n*k^2) not O(n^2)
    - Soliton attention has wave propagation overhead
    """
    raw_model = model.module if ddp else model
    raw_model.num_parameters()
    cfg = raw_model.config
    L, H, Q, T = cfg.n_layer, cfg.n_heads, cfg.n_embd // cfg.n_heads, cfg.block_size

    # Approximate FLOPs per token:
    # - SDR encoding: sparse ops, roughly sdr_size * n_active * 2
    # - Per layer: spectral attention O(T * n_eigenvectors^2) + MLP O(4 * n_embd^2)
    # - Total per token: L * (spectral + soliton + mlp)
    sdr_flops = cfg.sdr_size * cfg.sdr_n_active * 2
    spectral_flops = T * cfg.n_eigenvectors * cfg.n_eigenvectors
    soliton_flops = H * Q * T  # Wave propagation
    mlp_flops = 4 * cfg.n_embd * cfg.n_embd
    layer_flops = spectral_flops + soliton_flops + mlp_flops
    flops_per_token = sdr_flops + L * layer_flops

    # Forward + backward is ~3x forward
    flops_per_iter = 3 * flops_per_token * batch_size * T
    flops_achieved = flops_per_iter / dt  # per second

    # A100 GPU: ~312 TFLOPS bf16
    # RTX 4090: ~83 TFLOPS bf16
    flops_promised = 312e12  # Assume A100
    mfu = flops_achieved / flops_promised
    return mfu


# -----------------------------------------------------------------------------
# Wandb
if wandb_log and master_process:
    import wandb

    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# -----------------------------------------------------------------------------
# Training loop
X, Y = get_batch("train")
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

if master_process:
    print(f"Starting training from iteration {iter_num}")
    print(f"Configuration: {config}")

while True:
    # Set learning rate
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # Evaluate and checkpoint
    # IMPORTANT: skip eval at iter 0 (startup must be <60s)
    if iter_num > 0 and iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
        val_perplexity = math.exp(losses["val"])
        print(f"step {iter_num}: val perplexity {val_perplexity:.4f}")
        if wandb_log:
            wandb.log(
                {
                    "iter": iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "val/perplexity": val_perplexity,
                    "lr": lr,
                    "mfu": running_mfu * 100,
                }
            )
        # Early stopping logic
        improved = losses["val"] < best_val_loss
        if improved:
            best_val_loss = losses["val"]
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if (
            early_stopping_patience > 0
            and early_stopping_counter >= early_stopping_patience
        ):
            if master_process:
                print(
                    f"Early stopping at step {iter_num} (patience {early_stopping_patience} exceeded)"
                )
            break

        if improved or always_save_checkpoint:
            if iter_num > 0:
                # Save all config fields needed for model reconstruction
                # (excluding computed fields like sdr_n_active which __post_init__ derives)
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_config": {
                        # Core architecture
                        "vocab_size": model_config.vocab_size,
                        "block_size": model_config.block_size,
                        "n_layer": model_config.n_layer,
                        "n_heads": model_config.n_heads,
                        "n_embd": model_config.n_embd,
                        "dropout": model_config.dropout,
                        "bias": model_config.bias,
                        # SDR configuration
                        "sdr_size": model_config.sdr_size,
                        "sdr_sparsity": model_config.sdr_sparsity,
                        "sdr_embed_dim": model_config.sdr_embed_dim,
                        "sdr_context_size": model_config.sdr_context_size,
                        # Manifold and spectral
                        "manifold_dim": model_config.manifold_dim,
                        "n_neighbors": model_config.n_neighbors,
                        "n_eigenvectors": model_config.n_eigenvectors,
                        "spectral_sigma": model_config.spectral_sigma,
                        # Soliton dynamics
                        "soliton_threshold": model_config.soliton_threshold,
                        "soliton_tau": model_config.soliton_tau,
                        "soliton_velocity": model_config.soliton_velocity,
                        "pulse_width_base": model_config.pulse_width_base,
                        # Engram memory
                        "engram_capacity": model_config.engram_capacity,
                        "engram_threshold": model_config.engram_threshold,
                        "l1_capacity": model_config.l1_capacity,
                        "l2_capacity": model_config.l2_capacity,
                        "l3_capacity": model_config.l3_capacity,
                        # DAG planning and imagination
                        "max_dag_depth": model_config.max_dag_depth,
                        "imagination_steps": model_config.imagination_steps,
                        "imagination_dim": model_config.imagination_dim,
                        # Training config (for reference)
                        "learning_rate": model_config.learning_rate,
                        "weight_decay": model_config.weight_decay,
                        "beta1": model_config.beta1,
                        "beta2": model_config.beta2,
                        "grad_clip": model_config.grad_clip,
                    },
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))

    if iter_num == 0 and eval_only:
        break

    # Forward backward with gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (
                micro_step == gradient_accumulation_steps - 1
            )
        with ctx:
            logits, loss, info = model(X, Y)
            loss = loss / gradient_accumulation_steps
        # Prefetch next batch
        X, Y = get_batch("train")
        # Backward
        scaler.scale(loss).backward()

    # Gradient clipping
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Clear engram memory periodically to prevent unbounded growth
    if iter_num % 1000 == 0 and iter_num > 0:
        raw_model.memory.clear()

    # Timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(
            f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%"
        )
    iter_num += 1
    local_iter_num += 1

    # Termination
    if iter_num > max_iters:
        break

# Cleanup
if ddp:
    destroy_process_group()

if master_process:
    print("Training complete!")
