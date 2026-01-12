# Ralph Loop Iteration 29 - NeuroManifold with skip_manifold_spectral
# GOALS: val_loss < 1.5, training_time < 100s
# Strategy: Skip manifold/spectral computations for speed while keeping NeuroManifold

import torch
torch.set_float32_matmul_precision('medium')

# Data
dataset = "shakespeare_char"
batch_size = 64
block_size = 256
num_workers = 4

# Model - NeuroManifold architecture (but with fast path)
model_type = "neuromanifold"
n_layer = 6
n_head = 5
n_embd = 320
dropout = 0.1
bias = False

# NeuroManifold features - minimal
use_sdr = False
use_kan = False  # Use SwiGLU (faster than KAN)
use_mhc = False
use_full_mhc = False

# FHN - DISABLED (skip modulation)
n_fhn_steps = 0
fhn_threshold = 0.5
fhn_tau = 12.5
use_fhn_imex = True
use_fhn_partitioning = False
use_fhn_fused = False

# CRITICAL: Skip expensive manifold/spectral computations
skip_manifold_spectral = True

# Training
max_iters = 975
gradient_accumulation_steps = 1
learning_rate = 2e-3
min_lr = 2e-4
weight_decay = 0.1
warmup_iters = 35
lr_decay_iters = 975
grad_clip = 1.0

# Early stopping
early_stopping_patience = 0

# Eval/logging - minimal
eval_interval = 300
log_interval = 150
eval_iters = 5
sample_interval = 0

# Output
out_dir = "out-ralph-iter29"

save_checkpoints = False

# Hardware
devices = 1
precision = "bf16-mixed"
compile_model = False

# Logging
wandb_log = False
