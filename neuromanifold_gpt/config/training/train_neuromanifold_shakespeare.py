# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = "out-neuromanifold-shakespeare"
eval_interval = 250  # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10  # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True
save_checkpoints = True

wandb_log = False  # override via command line if you like
wandb_project = "shakespeare-char"
wandb_run_name = "neuromanifold-nano"

dataset = "shakespeare_char"
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256  # context of up to 256 previous characters

# NeuroManifold Nano Config
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0

# NeuroManifold Specifics (Explicitly set to defaults, but good for documentation)
sdr_size = 1024
manifold_dim = 32
n_eigenvectors = 16
use_sdr = True
kan_type = "wave"
kan_wavelet = "dog"
use_fast_wavekan = True
# NeuroManifold FHN tuning
fhn_threshold = 0.1  # Lower threshold for better signal flow
fhn_tau = 12.5
n_fhn_steps = 2
use_fhn_imex = True
use_fhn_partitioning = True
use_fhn_fused = False

learning_rate = 3e-3  # Aggressive learning rate
max_iters = 10000  # Extended run for full convergence
lr_decay_iters = 10000  # Match max_iters
eval_interval = 500  # Checkpoint every ~500 iters (approx 75s)

# device = 'cuda'
# compile = True
