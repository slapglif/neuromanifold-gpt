# Shakespeare character-level training preset
# Good for quick sanity checks

use_nano_config = False
dataset = "shakespeare_char"
out_dir = "out-neuromanifold-shakespeare"
wandb_run_name = "neuromanifold-shakespeare"

# Smaller model for char-level
n_layer = 6
n_head = 6
n_embd = 384
block_size = 256
sdr_size = 1024
manifold_dim = 64
n_eigenvectors = 32

batch_size = 64
gradient_accumulation_steps = 1

# Faster training
max_iters = 5000
eval_interval = 250
eval_iters = 200
warmup_iters = 100
lr_decay_iters = 5000

learning_rate = 1e-3
min_lr = 1e-4
