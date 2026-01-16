# train a miniature character-level shakespeare model with separated checkpoints
# demonstrates how to save model weights and optimizer state to separate files

out_dir = 'out-shakespeare-separated'
eval_interval = 250
eval_iters = 200
log_interval = 10

# Enable separated checkpoint saving
# This saves model.pt and optimizer.pt as separate files
# Benefits:
# - Smaller model-only files for inference/sharing
# - Flexible optimizer state management
# - Full training resumption capability
save_separate_optimizer = True

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'shakespeare-char'
wandb_run_name = 'separated-checkpoints'

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
