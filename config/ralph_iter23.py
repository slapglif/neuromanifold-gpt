# Ralph Loop Iteration 23 - NeuroManifold without FHN
# GOALS: val_loss < 1.5, training_time < 100s
# Test: n_fhn_steps=0 to diagnose FHN issue

import torch
torch.set_float32_matmul_precision('medium')

# Data
dataset = "shakespeare_char"
batch_size = 64
block_size = 256
num_workers = 4

# Model - NeuroManifold without FHN
model_type = "neuromanifold"
n_layer = 6
n_head = 5
n_embd = 320
dropout = 0.1
bias = False

# NeuroManifold features - all disabled
use_sdr = False
use_kan = False
use_mhc = False
use_full_mhc = False

# FHN - DISABLED
n_fhn_steps = 0
use_fhn_imex = False
use_fhn_partitioning = False
use_fhn_fused = False

# Training
max_iters = 1000
gradient_accumulation_steps = 1
learning_rate = 2e-3
min_lr = 2e-4
weight_decay = 0.1
warmup_iters = 35
lr_decay_iters = 1000
grad_clip = 1.0

# Early stopping
early_stopping_patience = 0

# Eval/logging - minimal
eval_interval = 250
log_interval = 125
eval_iters = 5
sample_interval = 0

# Output
out_dir = "out-ralph-iter23"
save_checkpoints = False

# Hardware
devices = 1
precision = "bf16-mixed"
compile_model = False

# Logging
wandb_log = False
