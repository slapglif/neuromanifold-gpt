# train a miniature character-level shakespeare model
# Baseline: Standard NanoGPT architecture (SwiGLU, Softmax Attention)

out_dir = "out-baseline-shakespeare"
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = "shakespeare-char"
wandb_run_name = "baseline-nano"

dataset = "shakespeare_char"
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

# Standard GPT Config
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0
bias = False

# Optimizer
learning_rate = 1e-3
max_iters = 1000
lr_decay_iters = 1000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

# device = 'cuda'
# compile = True # We'll override this in CLI

# Select model type for train.py
model_type = "gpt"
