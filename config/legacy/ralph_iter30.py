# Ralph Loop Iteration 30 - Smaller NeuroManifold for speed
# GOALS: val_loss < 1.5, training_time < 100s
# Strategy: Smaller model to increase it/s, more iterations in time budget

import torch
torch.set_float32_matmul_precision('medium')

# Data - smaller batch/block for speed
dataset = "shakespeare_char"
batch_size = 48
block_size = 128
num_workers = 4

# Model - smaller for speed
model_type = "neuromanifold"
n_layer = 4  # Fewer layers
n_head = 4
n_embd = 256  # Smaller
dropout = 0.1
bias = False

# NeuroManifold features - all disabled for pure speed
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

# Training - more iterations since we're faster
max_iters = 1500
gradient_accumulation_steps = 1
learning_rate = 3e-3
min_lr = 3e-4
weight_decay = 0.1
warmup_iters = 50
lr_decay_iters = 1500
grad_clip = 1.0

# Early stopping
early_stopping_patience = 0

# Eval/logging - minimal
eval_interval = 500
log_interval = 250
eval_iters = 5
sample_interval = 0

# Output
out_dir = "out-ralph-iter30"
save_checkpoints = False

# Hardware
devices = 1
precision = "bf16-mixed"
compile_model = False

# Logging
wandb_log = False
