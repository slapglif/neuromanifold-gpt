# NeuroManifoldGPT - General Text Training with FineWeb-Edu Streaming
# Uses GPT-2 BPE tokenizer (50257 vocab) for general English competence
# Streams data from HuggingFace - minimal disk usage

import torch
torch.set_float32_matmul_precision('medium')

# Data - streaming from FineWeb-Edu
dataset = "fineweb-edu"  # Educational web content
streaming = True  # Enable streaming mode
batch_size = 32  # Smaller batch for streaming
block_size = 256  # Longer context for general text
num_workers = 2

# Model - scaled up for general text (15M params)
model_type = "neuromanifold"
n_layer = 8
n_head = 8
n_embd = 512
vocab_size = 50257  # GPT-2 BPE tokenizer
dropout = 0.1
bias = False

# NeuroManifold features - start simple
use_sdr = False
use_kan = False
use_mhc = False
use_full_mhc = False
n_fhn_steps = 0
skip_manifold_spectral = True

# Training - longer for general competence
max_iters = 10000
gradient_accumulation_steps = 4  # Effective batch = 128
learning_rate = 3e-4
min_lr = 3e-5
weight_decay = 0.1
warmup_iters = 500
lr_decay_iters = 10000
grad_clip = 1.0

# Early stopping
early_stopping_patience = 0

# Eval/logging
eval_interval = 500
log_interval = 100
eval_iters = 20
sample_interval = 1000

# Output
out_dir = "out-general-streaming"
save_checkpoints = True

# Hardware
devices = 1
precision = "bf16-mixed"
compile_model = True

# Logging
wandb_log = False
