# Ralph Loop Iteration 1 - Tiny config for 6GB GPU, sub-100s training
# GOALS: val_loss < 1.5, training_time < 100s

# Data
dataset = "shakespeare_char"
batch_size = 32  # Reduced for memory
block_size = 128  # Reduced for memory
num_workers = 2

# Model - TINY for speed and memory
model_type = "neuromanifold"
n_layer = 2  # Minimal layers
n_head = 4  # Fewer heads
n_embd = 256  # Smaller embedding
dropout = 0.0
bias = False

# NeuroManifold features - minimal for speed
use_sdr = False  # Skip SDR
use_kan = False  # Use standard MLP for now (KAN OOM)
kan_type = "faster"
kan_num_centers = 2
use_mhc = True  # Keep mHC for stability
use_full_mhc = False  # Simplified mHC
mhc_n_streams = 2

# FHN settings - minimal steps
fhn_threshold = 0.5
fhn_tau = 12.5
n_fhn_steps = 1
use_fhn_imex = True
use_fhn_partitioning = False  # Disable for speed
use_fhn_fused = False

# Training - aggressive for speed
max_iters = 500  # Short run to verify completion
gradient_accumulation_steps = 2  # Simulate larger batch
learning_rate = 1e-3
min_lr = 1e-4
weight_decay = 0.1
warmup_iters = 100
lr_decay_iters = 500
grad_clip = 1.0

# Early stopping
early_stopping_patience = 0

# Eval/logging
eval_interval = 200
log_interval = 50
eval_iters = 20  # Faster eval
sample_interval = 500

# Output
out_dir = "out-ralph-iter1"
save_checkpoints = False

# Hardware
devices = 1
precision = "bf16-mixed"
compile_model = False

# Logging
wandb_log = False
