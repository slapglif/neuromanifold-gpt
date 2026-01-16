# config for training GPT-2 (124M) on multi-GPU setup
# optimized for training on 8x A100 40GB GPUs
# launch as: torchrun --standalone --nproc_per_node=8 train.py config/defaults/gpt2_124m_multi_gpu.py

wandb_log = True
wandb_project = 'gpt2'
wandb_run_name = 'gpt2-124M-multi-gpu'

# these make the total batch size be ~0.5M tokens
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1
