# Ralph Loop Iteration 3 - Balanced model for quality
# GOALS: val_loss < 1.5, training_time < 100s

import torch
torch.set_float32_matmul_precision('medium')

# Data
dataset = "shakespeare_char"
batch_size = 32  # Reduced for memory
block_size = 128  # Shorter context for memory
num_workers = 4

# Model - ~6M params for balance
model_type = "neuromanifold"
n_layer = 4
n_head = 6
n_embd = 288
dropout = 0.1
bias = False

# NeuroManifold features - minimal
use_sdr = False
use_kan = False
use_mhc = False
use_full_mhc = False
mhc_n_streams = 2

# FHN - minimal
fhn_threshold = 0.5
fhn_tau = 12.5
n_fhn_steps = 1
use_fhn_imex = True
use_fhn_partitioning = False
use_fhn_fused = False

# Training
max_iters = 1000  # Balance speed/quality
gradient_accumulation_steps = 1  # No accumulation for speed
learning_rate = 3e-3  # Higher LR for faster convergence
min_lr = 3e-4
weight_decay = 0.1
warmup_iters = 50
lr_decay_iters = 1000
grad_clip = 1.0

# Early stopping
early_stopping_patience = 0

# Eval/logging
eval_interval = 250
log_interval = 100
eval_iters = 10
sample_interval = 0

# Output
out_dir = "out-ralph-iter3"
save_checkpoints = False

# Hardware
devices = 1
precision = "bf16-mixed"
compile_model = False

# Logging
wandb_log = False
