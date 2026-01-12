# Nano preset for fast experimentation and testing
# ~1M parameters, trains in minutes on a single GPU

use_nano_config = True
batch_size = 8
block_size = 256
gradient_accumulation_steps = 1

# Reduced training
max_iters = 5000
eval_interval = 250
eval_iters = 50
warmup_iters = 100
lr_decay_iters = 5000

# Higher LR for small model
learning_rate = 1e-3
min_lr = 1e-4

# Output
out_dir = "out-neuromanifold-nano"
wandb_run_name = "neuromanifold-nano"
