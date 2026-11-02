import time

out_dir = "out-finetune-gpt2-medium"
eval_interval = 5
eval_iters = 40
wandb_log = False  # feel free to turn on
wandb_project = "gpt2-finetune"
wandb_run_name = "ft-gpt2-medium-" + str(time.time())

init_from = "gpt2-medium"  # 345M parameter model

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# batch size and gradient accumulation
# 2 batch_size * 16 grad_accum * 1024 tokens = 32,768 tokens/iter
# slightly larger batch size than gpt2-xl since medium is smaller
batch_size = 2
gradient_accumulation_steps = 16
max_iters = 20

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False
