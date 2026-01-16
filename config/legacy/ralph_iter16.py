# Ralph Loop Iteration 16 - Gradient accumulation for better convergence
# GOALS: val_loss < 1.5, training_time < 100s
# Strategy: gradient_accumulation_steps=2 for effective batch=128
# Fewer steps needed with larger effective batch

import torch
torch.set_float32_matmul_precision('medium')

# Data
dataset = "shakespeare_char"
batch_size = 64
block_size = 256
num_workers = 4

# Model - BASELINE
model_type = "baseline"
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1
bias = False

# NeuroManifold features - all disabled
use_sdr = False
use_kan = False
use_mhc = False
use_full_mhc = False
n_fhn_steps = 0
use_fhn_imex = False
use_fhn_partitioning = False
use_fhn_fused = False

# Training - gradient accumulation
# With accum=2, each "step" is 2 forward passes
# So 450 steps * 2 = 900 forward passes, should be ~100s
max_iters = 450
gradient_accumulation_steps = 2  # Effective batch = 128
learning_rate = 3e-3  # Higher LR for larger batch
min_lr = 3e-4
weight_decay = 0.1
warmup_iters = 15  # Half warmup for half steps
lr_decay_iters = 450
grad_clip = 1.0

# Early stopping
early_stopping_patience = 0

# Eval/logging - minimal overhead
eval_interval = 100
log_interval = 50
eval_iters = 5
sample_interval = 0

# Output
out_dir = "out-ralph-iter16"
save_checkpoints = False

# Hardware
devices = 1
precision = "bf16-mixed"
compile_model = False

# Logging
wandb_log = False
