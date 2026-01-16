# Ralph Loop Iteration 11 - Higher LR to converge faster
# GOALS: val_loss < 1.5, training_time < 100s
# Strategy: 900 iters fit in 98s, use higher LR to converge faster

import torch
torch.set_float32_matmul_precision('medium')

# Data
dataset = "shakespeare_char"
batch_size = 64
block_size = 256
num_workers = 4

# Model - BASELINE (no NeuroManifold features)
model_type = "baseline"
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.15
bias = False

# NeuroManifold features - all disabled (baseline)
use_sdr = False
use_kan = False
use_mhc = False
use_full_mhc = False
n_fhn_steps = 0
use_fhn_imex = False
use_fhn_partitioning = False
use_fhn_fused = False

# Training - higher LR for faster convergence
max_iters = 900
gradient_accumulation_steps = 1
learning_rate = 2.5e-3  # Increased from 2e-3
min_lr = 2.5e-4
weight_decay = 0.1
warmup_iters = 30
lr_decay_iters = 900
grad_clip = 1.0

# Early stopping
early_stopping_patience = 0

# Eval/logging
eval_interval = 100
log_interval = 50
eval_iters = 10
sample_interval = 0

# Output
out_dir = "out-ralph-iter11"
save_checkpoints = False

# Hardware
devices = 1
precision = "bf16-mixed"
compile_model = False

# Logging
wandb_log = False
