# Small preset - similar to GPT-2 small (124M)
# Good for single-GPU training on consumer hardware

use_nano_config = False
n_layer = 12
n_head = 12
n_embd = 768
sdr_size = 2048
manifold_dim = 128
n_eigenvectors = 64

batch_size = 12
block_size = 1024
gradient_accumulation_steps = 40  # ~480 effective batch size

max_iters = 600000
eval_interval = 2000
warmup_iters = 2000
lr_decay_iters = 600000

learning_rate = 6e-4
min_lr = 6e-5

out_dir = "out-neuromanifold-small"
wandb_run_name = "neuromanifold-small"
