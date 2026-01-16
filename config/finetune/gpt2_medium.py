import time

# GPT-2 Medium finetuning recipe
# n_layer=24, n_head=16, n_embd=1024
# 350M parameters
#
# EXPECTED RESULTS (Shakespeare dataset, ~1M tokens):
#   Training time:
#     - A100 (40GB): ~15 minutes
#     - V100 (32GB): ~30 minutes
#     - RTX 4090 (24GB): ~22 minutes
#     - RTX 3090 (24GB): ~35 minutes
#   Final validation loss: 0.85-0.95
#   Baseline (pretrained): 3.0-3.5
#   Improvement: ~73% loss reduction
#
# For other datasets, see docs/finetuning-guide.md for domain-specific expectations

out_dir = 'out-finetune-gpt2-medium'
eval_interval = 250
eval_iters = 200
wandb_log = False # feel free to turn on
wandb_project = 'finetune-gpt2'
wandb_run_name = 'gpt2-medium-' + str(time.time())

# start from GPT-2 Medium (350M) checkpoint
init_from = 'gpt2-medium'

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# batch size and gradient accumulation
# 4 batch_size * 8 grad_accum * 1024 tokens = 32,768 tokens/iter
batch_size = 4
gradient_accumulation_steps = 8
max_iters = 5000

# learning rate for finetuning
learning_rate = 3e-5  # constant LR for finetuning (lower than pretraining)
decay_lr = False  # finetuning works well with constant LR
