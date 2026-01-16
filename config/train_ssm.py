# Train a Pure SSM (Mamba) model on Shakespeare
# Disables Attention, KAN, mHC, and Manifold/Spectral components for pure SSM evaluation.

out_dir = 'out-ssm-pure'
eval_interval = 250
eval_iters = 200
log_interval = 10

save_checkpoints = True

wandb_log = False # turned off by default
wandb_project = 'shakespeare-char'
wandb_run_name = 'ssm-pure'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 16
batch_size = 4
block_size = 256 # context of up to 256 previous characters

# Model configuration
model_type = 'neuromanifold'
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
bias = False # do we use bias inside LayerNorm and Linear layers?

# SSM (Mamba) Configuration
use_ssm = True
ssm_state_dim = 16
ssm_conv_kernel = 4
ssm_expand = 2

# Disable other components for Pure SSM
use_kan = False
use_mhc = False
use_fhn_partitioning = False # Not used if attention replaced, but good to be explicit
use_full_mhc = False
skip_manifold_spectral = True # Disable manifold projection for speed

# Learning rate settings
learning_rate = 1e-4 # Reduced from 1e-3 to prevent NaN
grad_clip = 0.5 # Tighter clipping for stability
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-5 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary for small datasets, but can't hurt

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
