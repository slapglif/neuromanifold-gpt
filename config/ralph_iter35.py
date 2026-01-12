# Ralph Loop Iteration 35 - NeuroManifold optimized for goals
# GOALS: val_loss < 1.5, training_time < 100s
# Strategy: Slightly fewer iterations to stay under 100s, higher LR for faster convergence

import torch
torch.set_float32_matmul_precision('medium')

# Data
dataset = "shakespeare_char"
batch_size = 64
block_size = 256
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

# FHN - DISABLED
n_fhn_steps = 0
fhn_threshold = 0.5
fhn_tau = 12.5
use_fhn_imex = True
use_fhn_partitioning = False
use_fhn_fused = False

# Skip expensive computations
skip_manifold_spectral = True

# Training - 950 iterations to fit under 100s
max_iters = 950
gradient_accumulation_steps = 1
learning_rate = 2.5e-3  # Slightly higher LR
min_lr = 2.5e-4
weight_decay = 0.1
warmup_iters = 30
lr_decay_iters = 950
grad_clip = 1.0

# Early stopping
early_stopping_patience = 0

# Eval/logging
eval_interval = 300
log_interval = 150
eval_iters = 5
sample_interval = 0

# Output
out_dir = "out-ralph-iter35"
save_checkpoints = False

# Hardware - ENABLE COMPILE
devices = 1
precision = "bf16-mixed"
compile_model = True

# Logging
wandb_log = False
