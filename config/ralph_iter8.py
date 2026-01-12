# Ralph Loop Iteration 8 - Aggressive baseline GPT
# GOALS: val_loss < 1.5, training_time < 100s
# Last run: 107s @ 920 iters, val_loss=1.566

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
dropout = 0.15  # Slightly less regularization
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

# Training - aggressive settings
# 920 iters = 107s, so ~8.6 it/s
# 860 iters should be ~100s
max_iters = 850  # Target ~99s
gradient_accumulation_steps = 1
learning_rate = 2e-3  # More aggressive LR
min_lr = 2e-4
weight_decay = 0.1
warmup_iters = 30  # Shorter warmup for faster ramp
lr_decay_iters = 850
grad_clip = 1.0

# Early stopping
early_stopping_patience = 0

# Eval/logging
eval_interval = 100
log_interval = 50
eval_iters = 10
sample_interval = 0

# Output
out_dir = "out-ralph-iter8"
save_checkpoints = False

# Hardware
devices = 1
precision = "bf16-mixed"
compile_model = False

# Logging
wandb_log = False
