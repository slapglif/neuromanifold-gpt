# Ralph Loop Iteration 24 - Minimal NeuroManifold to fit in memory
# GOALS: val_loss < 1.5, training_time < 100s

import torch
torch.set_float32_matmul_precision('medium')

# Data - reduced for memory
dataset = "shakespeare_char"
batch_size = 32
block_size = 64  # Much shorter context
num_workers = 4

# Model - Tiny NeuroManifold
model_type = "neuromanifold"
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.1
bias = False

# NeuroManifold features - all disabled
use_sdr = False
use_kan = False
use_mhc = False
use_full_mhc = False

# FHN - minimal
n_fhn_steps = 1
fhn_threshold = 0.5
fhn_tau = 12.5
use_fhn_imex = True
use_fhn_partitioning = False
use_fhn_fused = False

# Training - aggressive
max_iters = 2000
gradient_accumulation_steps = 1
learning_rate = 5e-3  # Very high LR for small model
min_lr = 5e-4
weight_decay = 0.05
warmup_iters = 100
lr_decay_iters = 2000
grad_clip = 1.0

# Early stopping
early_stopping_patience = 0

# Eval/logging
eval_interval = 400
log_interval = 200
eval_iters = 5
sample_interval = 0

# Output
out_dir = "out-ralph-iter24"
save_checkpoints = False

# Hardware
devices = 1
precision = "bf16-mixed"
compile_model = False

# Logging
wandb_log = False
