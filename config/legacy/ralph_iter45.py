# Ralph Loop Iteration 45 - Larger batch for throughput
# GOALS: val_loss < 1.5, training_time < 100s
# Strategy: batch_size=96 for better GPU utilization, 2100 iterations

import torch
torch.set_float32_matmul_precision('medium')

# Data - larger batch for throughput
dataset = "shakespeare_char"
batch_size = 96  # Increased for better GPU utilization
block_size = 128
num_workers = 4

# Model
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

# Training - 2100 iterations with larger batch
max_iters = 2100
gradient_accumulation_steps = 1
learning_rate = 2.5e-3  # Slightly higher LR for larger batch
min_lr = 2.5e-4
weight_decay = 0.1
warmup_iters = 70
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
out_dir = "out-ralph-iter45"
save_checkpoints = False

# Hardware
devices = 1
precision = "bf16-mixed"
compile_model = True

# Logging
wandb_log = False
