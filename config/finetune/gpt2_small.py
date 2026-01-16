import time

# GPT-2 Small finetuning recipe
# n_layer=12, n_head=12, n_embd=768
# 124M parameters

out_dir = 'out-finetune-gpt2-small'
eval_interval = 250
eval_iters = 200
wandb_log = False # feel free to turn on
wandb_project = 'finetune-gpt2'
wandb_run_name = 'gpt2-small-' + str(time.time())

# start from GPT-2 Small (124M) checkpoint
init_from = 'gpt2'

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# batch size and gradient accumulation
# 8 batch_size * 8 grad_accum * 1024 tokens = 65,536 tokens/iter
batch_size = 8
gradient_accumulation_steps = 8
max_iters = 5000

# learning rate schedule for finetuning
learning_rate = 3e-5  # lower than pretraining for finetuning
decay_lr = True
warmup_iters = 100
lr_decay_iters = 5000
min_lr = 3e-6  # learning_rate / 10
