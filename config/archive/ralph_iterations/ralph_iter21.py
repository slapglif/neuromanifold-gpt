# Ralph Loop Iteration 21 - NeuroManifold with minimal FHN
# GOALS: val_loss < 1.5, training_time < 100s
# Must use NeuroManifold, not baseline!
# Strategy: Minimal FHN (1 step), no SDR/KAN/MHC

import torch
torch.set_float32_matmul_precision('medium')

# Data
dataset = "shakespeare_char"
batch_size = 64
block_size = 256
num_workers = 4

# Model - NeuroManifold with minimal overhead
model_type = "neuromanifold"
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1
bias = False

# NeuroManifold features - minimal
use_sdr = False
use_kan = False
use_mhc = False
use_full_mhc = False

# FHN - minimal (1 step only)
n_fhn_steps = 1
fhn_threshold = 0.5
fhn_tau = 12.5
use_fhn_imex = True  # Faster integration
use_fhn_partitioning = False
use_fhn_fused = False

# Training - aggressive
max_iters = 900
gradient_accumulation_steps = 1
learning_rate = 3e-3  # Higher LR
min_lr = 3e-4
weight_decay = 0.1
warmup_iters = 30
lr_decay_iters = 900
grad_clip = 1.0

# Early stopping
early_stopping_patience = 0

# Eval/logging - minimal
eval_interval = 200
log_interval = 100
eval_iters = 5
sample_interval = 0

# Output
out_dir = "out-ralph-iter21"
save_checkpoints = False

# Hardware
devices = 1
precision = "bf16-mixed"
compile_model = False

# Logging
wandb_log = False
