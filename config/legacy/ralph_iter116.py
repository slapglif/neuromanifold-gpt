# Ralph Loop Iteration 116 - Push for val_loss < 1.5 AND time < 100s
# Strategy: Slightly more iterations (2900), larger batch for faster throughput

import torch
torch.set_float32_matmul_precision('medium')

# Data
dataset = "shakespeare_char"
batch_size = 96  # Larger batch for faster throughput
block_size = 80
num_workers = 4

# Model - 7.43M
model_type = "neuromanifold"
n_layer = 6
n_head = 5
n_embd = 320
dropout = 0.06  # Lower dropout for better val loss
bias = False

# NeuroManifold features - disabled
use_sdr = False
use_kan = False
use_mhc = False
use_full_mhc = False

# FHN - DISABLED
n_fhn_steps = 0
skip_manifold_spectral = True

# Training - 2700 iters (fewer but larger batches = same data, faster)
max_iters = 2700
gradient_accumulation_steps = 1
learning_rate = 2.5e-3  # Slightly higher LR
min_lr = 2.5e-4
weight_decay = 0.1
warmup_iters = 80
lr_decay_iters = 2700
grad_clip = 1.0

# Early stopping
early_stopping_patience = 0

# Eval/logging - minimal
eval_interval = 2700
log_interval = 900
eval_iters = 3
sample_interval = 0

# Output
out_dir = "out-ralph-iter116"
save_checkpoints = True

# Hardware
devices = 1
precision = "bf16-mixed"
compile_model = True

# Logging
wandb_log = False
