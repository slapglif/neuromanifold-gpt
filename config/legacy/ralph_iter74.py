# Ralph Loop Iteration 74 - 2820 iterations for sub-100s
# GOALS: val_loss < 1.5, training_time < 100s

import torch
torch.set_float32_matmul_precision('medium')

# Data
dataset = "shakespeare_char"
batch_size = 64
block_size = 80
num_workers = 4

# Model - 7.43M
model_type = "neuromanifold"
n_layer = 6
n_head = 5
n_embd = 320
dropout = 0.08
bias = False

# NeuroManifold features - disabled
use_sdr = False
use_kan = False
use_mhc = False
use_full_mhc = False

# FHN - DISABLED
n_fhn_steps = 0
fhn_threshold = 0.5
fhn_tau = 12.5
use_fhn_imex = True
use_fhn_partitioning = False
use_fhn_fused = False

# Skip expensive computations
skip_manifold_spectral = True

# Training - 2820 iterations
max_iters = 2820
gradient_accumulation_steps = 1
learning_rate = 2e-3
min_lr = 2e-4
weight_decay = 0.1
warmup_iters = 85
lr_decay_iters = 2820
grad_clip = 1.0

# Early stopping
early_stopping_patience = 0

# Eval/logging - minimal
eval_interval = 2820
log_interval = 940
eval_iters = 3
sample_interval = 0

# Output
out_dir = "out-ralph-iter74"
save_checkpoints = False

# Hardware
devices = 1
precision = "bf16-mixed"
compile_model = True

# Logging
wandb_log = False
