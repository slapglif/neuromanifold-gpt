# Ralph Loop Iteration 47 - Slightly larger model for better capacity
# GOALS: val_loss < 1.5, training_time < 100s
# Strategy: n_embd=384 for more capacity, 1800 iterations

import torch
torch.set_float32_matmul_precision('medium')

# Data
dataset = "shakespeare_char"
batch_size = 64
block_size = 128
num_workers = 4

# Model - slightly larger
model_type = "neuromanifold"
n_layer = 6
n_head = 6  # 6 heads for 384
n_embd = 384  # Larger for better capacity
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

# Training - 1800 iterations
max_iters = 1800
gradient_accumulation_steps = 1
learning_rate = 2e-3
min_lr = 2e-4
weight_decay = 0.1
warmup_iters = 55
lr_decay_iters = 1800
grad_clip = 1.0

# Early stopping
early_stopping_patience = 0

# Eval/logging
eval_interval = 600
log_interval = 300
eval_iters = 5
sample_interval = 0

# Output
out_dir = "out-ralph-iter47"
save_checkpoints = False

# Hardware
devices = 1
precision = "bf16-mixed"
compile_model = True

# Logging
wandb_log = False
