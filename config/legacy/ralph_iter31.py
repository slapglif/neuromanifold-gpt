# Ralph Loop Iteration 31 - Medium NeuroManifold balancing speed/quality
# GOALS: val_loss < 1.5, training_time < 100s
# Strategy: Medium model with 2000 iterations in ~90s budget

import torch
torch.set_float32_matmul_precision('medium')

# Data
dataset = "shakespeare_char"
batch_size = 64
block_size = 256
num_workers = 4

# Model - medium size for quality
model_type = "neuromanifold"
n_layer = 6
n_head = 6
n_embd = 384  # Standard size
dropout = 0.1
bias = False

# NeuroManifold features - all disabled for speed
use_sdr = False
use_kan = False
use_mhc = False
use_full_mhc = False

# FHN - DISABLED
n_fhn_steps = 0
fhn_threshold = 0.5
fhn_tau = 12.5
use_fhn_imex = True
use_fhn_partitioning = False
use_fhn_fused = False

# Skip expensive computations
skip_manifold_spectral = True

# Training - higher learning rate for faster convergence
max_iters = 1200
gradient_accumulation_steps = 1
learning_rate = 6e-4
min_lr = 6e-5
weight_decay = 0.1
warmup_iters = 40
lr_decay_iters = 1200
grad_clip = 1.0

# Early stopping
early_stopping_patience = 0

# Eval/logging
eval_interval = 400
log_interval = 200
eval_iters = 5
sample_interval = 0

# Output
out_dir = "out-ralph-iter31"
save_checkpoints = False

# Hardware
devices = 1
precision = "bf16-mixed"
compile_model = False

# Logging
wandb_log = False
