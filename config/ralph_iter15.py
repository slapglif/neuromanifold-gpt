# Ralph Loop Iteration 15 - Fine-tune iteration count
# GOALS: val_loss < 1.5, training_time < 100s
# Strategy: 895 iters should be ~99s at 8 it/s

import torch
torch.set_float32_matmul_precision('medium')

# Data
dataset = "shakespeare_char"
batch_size = 64
block_size = 256
num_workers = 4

# Model - BASELINE
model_type = "baseline"
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1
bias = False

# NeuroManifold features - all disabled
use_sdr = False
use_kan = False
use_mhc = False
use_full_mhc = False
n_fhn_steps = 0
use_fhn_imex = False
use_fhn_partitioning = False
use_fhn_fused = False

# Training - 895 iters for ~99s
max_iters = 895
gradient_accumulation_steps = 1
learning_rate = 2e-3
min_lr = 2e-4
weight_decay = 0.1
warmup_iters = 30
lr_decay_iters = 895
grad_clip = 1.0

# Early stopping
early_stopping_patience = 0

# Eval/logging - minimal overhead
eval_interval = 200
log_interval = 100
eval_iters = 5
sample_interval = 0

# Output
out_dir = "out-ralph-iter15"
save_checkpoints = False

# Hardware
devices = 1
precision = "bf16-mixed"
compile_model = False

# Logging
wandb_log = False
