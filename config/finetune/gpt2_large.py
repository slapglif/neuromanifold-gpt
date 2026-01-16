import time

# GPT-2 Large finetuning recipe
# n_layer=36, n_head=20, n_embd=1280
# 774M parameters
#
# EXPECTED RESULTS (Shakespeare dataset, ~1M tokens):
#   Training time:
#     - A100 (40GB): ~30 minutes
#     - V100 (32GB): ~60 minutes
#     - RTX 4090 (24GB): ~45 minutes
#     - RTX 3090 (24GB): ~70 minutes
#   Final validation loss: 0.80-0.90
#   Baseline (pretrained): 3.0-3.5
#   Improvement: ~75% loss reduction
#
# For other datasets, see docs/finetuning-guide.md for domain-specific expectations

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

# learning rate for finetuning
learning_rate = 3e-5  # constant LR for finetuning (lower than pretraining)
decay_lr = False  # finetuning works well with constant LR
