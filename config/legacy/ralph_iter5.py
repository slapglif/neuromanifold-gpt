# Ralph Loop Iteration 5 - Fast model, max iterations in 100s
# GOALS: val_loss < 1.5, training_time < 100s

import torch
torch.set_float32_matmul_precision('medium')

# Data - optimized for throughput
dataset = "shakespeare_char"
batch_size = 64  # Larger batch for better GPU utilization
block_size = 64  # Shorter sequences = faster
num_workers = 4

# Model - 3M params, fast
model_type = "neuromanifold"
n_layer = 4  # More layers helps quality
n_head = 4
n_embd = 192
dropout = 0.0  # No dropout for faster training
bias = False

# NeuroManifold features - disabled for speed
use_sdr = False
use_kan = False
use_mhc = False  # Disable all novel features
use_full_mhc = False
mhc_n_streams = 2

# FHN - minimal
fhn_threshold = 0.5
fhn_tau = 12.5
n_fhn_steps = 1
use_fhn_imex = True
use_fhn_partitioning = False
use_fhn_fused = False

# Training - max quality in 100s
# 3M model at ~17 iter/s = ~1700 iters in 100s
max_iters = 1200  # ~100s at 12 it/s
gradient_accumulation_steps = 1
learning_rate = 6e-3  # Higher LR for faster convergence
min_lr = 6e-4
weight_decay = 0.1
warmup_iters = 30
lr_decay_iters = 1200
grad_clip = 1.0

# Early stopping
early_stopping_patience = 0

# Eval/logging - minimal
eval_interval = 300
log_interval = 100
eval_iters = 10
sample_interval = 0

# Output
out_dir = "out-ralph-iter5"
save_checkpoints = False

# Hardware
devices = 1
precision = "bf16-mixed"
compile_model = False

# Logging
wandb_log = False
