# Medium preset - similar to GPT-2 medium (350M)
# Requires ~24GB VRAM with gradient accumulation

use_nano_config = False
n_layer = 24
n_head = 16
n_embd = 1024
sdr_size = 4096
manifold_dim = 256
n_eigenvectors = 128

batch_size = 8
block_size = 1024
gradient_accumulation_steps = 80  # ~640 effective batch size

max_iters = 600000
eval_interval = 2000
warmup_iters = 2000
lr_decay_iters = 600000

learning_rate = 3e-4
min_lr = 3e-5

out_dir = "out-neuromanifold-medium"
wandb_run_name = "neuromanifold-medium"
