import time

# GPT-2 Small finetuning recipe
# n_layer=12, n_head=12, n_embd=768
# 124M parameters
#
# EXPECTED RESULTS (Shakespeare dataset, ~1M tokens):
#   Training time:
#     - A100 (40GB): ~5 minutes
#     - V100 (32GB): ~10 minutes
#     - RTX 4090 (24GB): ~7 minutes
#     - RTX 3090 (24GB): ~12 minutes
#   Final validation loss: 1.00-1.10
#   Baseline (pretrained): 3.0-3.5
#   Improvement: ~66% loss reduction
#
# For other datasets, see docs/finetuning-guide.md for domain-specific expectations

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

# learning rate for finetuning
learning_rate = 3e-5  # constant LR for finetuning (lower than pretraining)
decay_lr = False  # finetuning works well with constant LR
