import time

# GPT-2 Large finetuning recipe
# n_layer=36, n_head=20, n_embd=1280
# 774M parameters

out_dir = 'out-finetune-gpt2-large'
eval_interval = 250
eval_iters = 200
wandb_log = False # feel free to turn on
wandb_project = 'finetune-gpt2'
wandb_run_name = 'gpt2-large-' + str(time.time())

# start from GPT-2 Large (774M) checkpoint
init_from = 'gpt2-large'

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# batch size and gradient accumulation
# 2 batch_size * 8 grad_accum * 1024 tokens = 16,384 tokens/iter
batch_size = 2
gradient_accumulation_steps = 8
max_iters = 5000

# learning rate schedule for finetuning
learning_rate = 3e-5  # lower than pretraining for finetuning
decay_lr = True
warmup_iters = 100
lr_decay_iters = 5000
min_lr = 3e-6  # learning_rate / 10
