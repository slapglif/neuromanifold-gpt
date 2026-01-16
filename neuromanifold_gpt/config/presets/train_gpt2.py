# Config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# Launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py neuromanifold_gpt.config.presets.train_gpt2

wandb_log = True
wandb_project = 'owt'
wandb_run_name = 'gpt2-124M'

# These make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# This makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# Eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# Weight decay
weight_decay = 1e-1

# Export all config as a dict for compatibility with verification
config = {
    'wandb_log': wandb_log,
    'wandb_project': wandb_project,
    'wandb_run_name': wandb_run_name,
    'batch_size': batch_size,
    'block_size': block_size,
    'gradient_accumulation_steps': gradient_accumulation_steps,
    'max_iters': max_iters,
    'lr_decay_iters': lr_decay_iters,
    'eval_interval': eval_interval,
    'eval_iters': eval_iters,
    'log_interval': log_interval,
    'weight_decay': weight_decay,
}
