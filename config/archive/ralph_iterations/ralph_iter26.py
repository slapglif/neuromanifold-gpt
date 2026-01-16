# Ralph Loop Iteration 26 - Larger NeuroManifold for more capacity
# GOALS: val_loss < 1.5, training_time < 100s
# Strategy: Larger model, reduced batch/block for memory

import torch
torch.set_float32_matmul_precision('medium')

# Data - reduced for memory
dataset = "shakespeare_char"
batch_size = 16  # Small batch
block_size = 64  # Short context
num_workers = 4

# Model - Larger NeuroManifold
model_type = "neuromanifold"
n_layer = 6
n_head = 4
n_embd = 256
dropout = 0.1
bias = False

# NeuroManifold features - all disabled
use_sdr = False
use_kan = False
use_mhc = False
use_full_mhc = False

# FHN - minimal steps
n_fhn_steps = 1
fhn_threshold = 0.5
fhn_tau = 12.5
use_fhn_imex = True
use_fhn_partitioning = False
use_fhn_fused = False

# Training - very aggressive
max_iters = 3000
gradient_accumulation_steps = 2  # Effective batch = 32
learning_rate = 3e-3
min_lr = 3e-4
weight_decay = 0.1
warmup_iters = 150
lr_decay_iters = 3000
grad_clip = 1.0

# Early stopping
early_stopping_patience = 0

# Eval/logging
eval_interval = 500
log_interval = 250
eval_iters = 5
sample_interval = 0

# Output
out_dir = "out-ralph-iter26"
save_checkpoints = False

# Hardware
devices = 1
precision = "bf16-mixed"
compile_model = False

# Logging
wandb_log = False
