# train a miniature character-level shakespeare model with model-only checkpoints
# demonstrates how to save just model weights without optimizer state

out_dir = 'out-shakespeare-model-only'
eval_interval = 250
eval_iters = 200
log_interval = 10

# Enable model-only checkpoint saving
# This saves ONLY model weights (no optimizer state)
# Benefits:
# - 50%+ smaller checkpoint files
# - Perfect for inference, evaluation, or sharing models
# - Reduced storage requirements
# Note: Cannot resume training from model-only checkpoints
save_model_only = True

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'shakespeare-char'
wandb_run_name = 'model-only-checkpoints'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
