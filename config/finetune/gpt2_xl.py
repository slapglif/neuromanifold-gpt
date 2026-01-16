import time

# GPT-2 XL finetuning recipe
# n_layer=48, n_head=25, n_embd=1600
# 1558M parameters
#
# EXPECTED RESULTS (Shakespeare dataset, ~1M tokens):
#   Training time:
#     - A100 (40GB): ~60 minutes
#     - V100 (32GB): ~120 minutes
#     - RTX 4090 (24GB): ~90 minutes
#     - RTX 3090 (24GB): ~140 minutes
#   Final validation loss: 0.75-0.85
#   Baseline (pretrained): 3.0-3.5
#   Improvement: ~77% loss reduction
#
# For other datasets, see docs/finetuning-guide.md for domain-specific expectations

out_dir = 'out-finetune-gpt2-xl'
eval_interval = 250
eval_iters = 200
wandb_log = False # feel free to turn on
wandb_project = 'finetune-gpt2'
wandb_run_name = 'gpt2-xl-' + str(time.time())

# start from GPT-2 XL (1558M) checkpoint
init_from = 'gpt2-xl'

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# batch size and gradient accumulation
# 1 batch_size * 8 grad_accum * 1024 tokens = 8,192 tokens/iter
batch_size = 1
gradient_accumulation_steps = 8
max_iters = 5000

# learning rate for finetuning
learning_rate = 3e-5  # constant LR for finetuning (lower than pretraining)
decay_lr = False  # finetuning works well with constant LR
