# Ralph Loop Iteration 7 - Baseline GPT (no NeuroManifold)
# GOALS: val_loss < 1.5, training_time < 100s

import torch
torch.set_float32_matmul_precision('medium')

# Data
dataset = "shakespeare_char"
batch_size = 64
block_size = 256  # Longer context helps quality
num_workers = 4

# Model - BASELINE (no NeuroManifold features)
model_type = "baseline"  # Use standard GPT
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
bias = False

# NeuroManifold features - all disabled (baseline)
use_sdr = False
use_kan = False
use_mhc = False
use_full_mhc = False

# FHN - disabled for baseline
n_fhn_steps = 0
use_fhn_imex = False
use_fhn_partitioning = False
use_fhn_fused = False

# Training
max_iters = 875  # Stay under 100s
gradient_accumulation_steps = 1
learning_rate = 1.5e-3  # Higher LR for faster convergence
min_lr = 1e-4
weight_decay = 0.1
warmup_iters = 50
lr_decay_iters = 875
grad_clip = 1.0

# Early stopping
early_stopping_patience = 0

# Eval/logging
eval_interval = 100
log_interval = 50
eval_iters = 10
sample_interval = 0

# Output
out_dir = "out-ralph-iter7"
save_checkpoints = False

# Hardware
devices = 1
precision = "bf16-mixed"
compile_model = False

# Logging
wandb_log = False
