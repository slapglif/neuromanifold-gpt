# Ralph Loop Iteration 50 - Optimized learning schedule
# GOALS: val_loss < 1.5, training_time < 100s
# Strategy: Reduce warmup for more effective training time

import torch
torch.set_float32_matmul_precision('medium')

# Data
dataset = "shakespeare_char"
batch_size = 64
block_size = 128
num_workers = 4

# Model - 7.43M (optimal size)
model_type = "neuromanifold"
n_layer = 6
n_head = 5
n_embd = 320
dropout = 0.1
bias = False

# NeuroManifold features - disabled for speed
use_sdr = False
use_kan = False
use_mhc = False
use_full_mhc = False

# FHN - DISABLED (enables Flash Attention fast path)
n_fhn_steps = 0
fhn_threshold = 0.5
fhn_tau = 12.5
use_fhn_imex = True
use_fhn_partitioning = False
use_fhn_fused = False

# Skip expensive computations
skip_manifold_spectral = True

# Training - 2100 iterations, optimized schedule
max_iters = 2100
gradient_accumulation_steps = 1
learning_rate = 2.5e-3  # Slightly higher LR
min_lr = 1e-4           # Lower min LR for longer decay
weight_decay = 0.1
warmup_iters = 50       # Reduced from 65
lr_decay_iters = 2100
grad_clip = 1.0

# Early stopping
early_stopping_patience = 0

# Eval/logging
eval_interval = 700
log_interval = 350
eval_iters = 5
sample_interval = 0

# Output
out_dir = "out-ralph-iter50"
save_checkpoints = False

# Hardware
devices = 1
precision = "bf16-mixed"
compile_model = True

# Logging
wandb_log = False
