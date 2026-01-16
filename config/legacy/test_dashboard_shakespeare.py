# Test config for dashboard testing with standard GPT model
# Uses shakespeare_char dataset and standard GPT (not NeuroManifold)

out_dir = 'out-test-dashboard'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = True
save_checkpoints = True

wandb_log = False
wandb_project = 'shakespeare-char'
wandb_run_name = 'dashboard-test'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

# Use standard GPT model (not NeuroManifold)
model_type = "gpt"

# Small GPT model for fast testing
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0
bias = False

learning_rate = 3e-3
max_iters = 500
lr_decay_iters = 500
min_lr = 3e-4
beta2 = 0.99

warmup_iters = 10

# Enable dashboard (default)
quiet = False
