# Ralph Loop Iteration 22 - Smaller NeuroManifold
# GOALS: val_loss < 1.5, training_time < 100s
# Strategy: Smaller model to avoid OOM

import torch
torch.set_float32_matmul_precision('medium')

# Data
dataset = "shakespeare_char"
batch_size = 32  # Reduced for memory
block_size = 128  # Reduced for memory
num_workers = 4

# Model - Smaller NeuroManifold
model_type = "neuromanifold"
n_layer = 4
n_head = 4
n_embd = 192
dropout = 0.1
bias = False

# NeuroManifold features - minimal
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
eval_interval = 300
log_interval = 150
eval_iters = 5
sample_interval = 0

# Output
out_dir = "out-ralph-iter22"
save_checkpoints = False

# Hardware
devices = 1
precision = "bf16-mixed"
compile_model = False

# Logging
wandb_log = False
