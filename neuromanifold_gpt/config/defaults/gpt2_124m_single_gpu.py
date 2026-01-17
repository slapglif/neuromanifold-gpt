# config for training GPT-2 (124M) on a single GPU
# optimized for training on 1x GPU with ~16-24GB VRAM
# launch as: python train.py config/defaults/gpt2_124m_single_gpu.py

wandb_log = True
wandb_project = "gpt2"
wandb_run_name = "gpt2-124M-single-gpu"

# these make the total batch size be ~49K tokens
# 4 batch size * 1024 block size * 12 gradaccum * 1 GPU = 49,152
# (about 1/10th of the 8-GPU setup, reasonable for single GPU)
batch_size = 4
block_size = 1024
gradient_accumulation_steps = 12

# this makes total number of tokens be 300B (same as multi-GPU)
# but will take proportionally longer
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1
