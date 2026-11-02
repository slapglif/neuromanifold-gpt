# Train a miniature character-level Shakespeare model
# Good for debugging and playing on macbooks and such

out_dir = "out-shakespeare-char"
eval_interval = 250  # Keep frequent because we'll overfit
eval_iters = 200
log_interval = 10  # Don't print too often

# We expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False  # Override via command line if you like
wandb_project = "shakespeare-char"
wandb_run_name = "mini-gpt"

dataset = "shakespeare_char"
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256  # Context of up to 256 previous characters

# Baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3  # With baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000  # Make equal to max_iters usually
min_lr = 1e-4  # learning_rate / 10 usually
beta2 = 0.99  # Make a bit bigger because number of tokens per iter is small

warmup_iters = 100  # Not super necessary potentially

# On macbook also add:
# device = 'cpu'  # Run on cpu only
# compile = False  # Do not torch compile the model

# Export all config as a dict for compatibility with verification
config = {
    "out_dir": out_dir,
    "eval_interval": eval_interval,
    "eval_iters": eval_iters,
    "log_interval": log_interval,
    "always_save_checkpoint": always_save_checkpoint,
    "wandb_log": wandb_log,
    "wandb_project": wandb_project,
    "wandb_run_name": wandb_run_name,
    "dataset": dataset,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "batch_size": batch_size,
    "block_size": block_size,
    "n_layer": n_layer,
    "n_head": n_head,
    "n_embd": n_embd,
    "dropout": dropout,
    "learning_rate": learning_rate,
    "max_iters": max_iters,
    "lr_decay_iters": lr_decay_iters,
    "min_lr": min_lr,
    "beta2": beta2,
    "warmup_iters": warmup_iters,
}
