# Ralph Loop Iteration 2 - Optimized for speed
# GOALS: val_loss < 1.5, training_time < 100s

import torch
torch.set_float32_matmul_precision('medium')

# Data - optimized for throughput
dataset = "shakespeare_char"
batch_size = 64  # Increased for better GPU utilization
block_size = 64  # Shorter sequences = faster
num_workers = 4  # More data loading parallelism

# Model - TINY for speed
model_type = "neuromanifold"
n_layer = 4  # Slightly more capacity
n_head = 4
n_embd = 192  # Smaller embedding
dropout = 0.0
bias = False

# NeuroManifold features - minimal
use_sdr = False
use_kan = False
kan_type = "faster"
use_mhc = True
use_full_mhc = False
mhc_n_streams = 2

# FHN - minimal
fhn_threshold = 0.5
fhn_tau = 12.5
n_fhn_steps = 1
use_fhn_imex = True
use_fhn_partitioning = False
use_fhn_fused = False

# Training - aggressive
max_iters = 1200  # Try more iterations within time budget
gradient_accumulation_steps = 1  # No accumulation for speed
learning_rate = 3e-3  # Higher LR for faster convergence
min_lr = 3e-4
weight_decay = 0.1
warmup_iters = 50
lr_decay_iters = 1200
grad_clip = 1.0

# Early stopping
early_stopping_patience = 0

# Eval/logging - minimal for speed
eval_interval = 250  # Less frequent eval
log_interval = 100
eval_iters = 10  # Fewer eval batches
sample_interval = 0  # No sample generation during training

# Output
out_dir = "out-ralph-iter2"
save_checkpoints = False

# Hardware
devices = 1
precision = "bf16-mixed"
compile_model = False  # Skip for now - can cause issues

# Logging
wandb_log = False
