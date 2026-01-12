# Ralph Loop Iteration 6 - Compiled model for max speed
# GOALS: val_loss < 1.5, training_time < 100s

import torch
torch.set_float32_matmul_precision('medium')

# Data
dataset = "shakespeare_char"
batch_size = 64
block_size = 64
num_workers = 4

# Model - 3M params
model_type = "neuromanifold"
n_layer = 4
n_head = 4
n_embd = 192
dropout = 0.0
bias = False

# NeuroManifold features - disabled
use_sdr = False
use_kan = False
use_mhc = False
use_full_mhc = False
mhc_n_streams = 2

# FHN
fhn_threshold = 0.5
fhn_tau = 12.5
n_fhn_steps = 1
use_fhn_imex = True
use_fhn_partitioning = False
use_fhn_fused = False

# Training - more iterations with compiled model speedup
max_iters = 2000  # Try more iterations
gradient_accumulation_steps = 1
learning_rate = 3e-3
min_lr = 3e-4
weight_decay = 0.1
warmup_iters = 50
lr_decay_iters = 2000
grad_clip = 1.0

# Early stopping
early_stopping_patience = 0

# Eval/logging - minimal
eval_interval = 400
log_interval = 100
eval_iters = 10
sample_interval = 0

# Output
out_dir = "out-ralph-iter6"
save_checkpoints = False

# Hardware
devices = 1
precision = "bf16-mixed"
compile_model = True  # ENABLE torch.compile()

# Logging
wandb_log = False
