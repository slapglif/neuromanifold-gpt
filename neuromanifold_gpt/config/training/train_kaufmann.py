# train a miniature character-level shakespeare model
# Kaufmann Trifecta Attention Run

out_dir = "out-kaufmann-shakespeare"
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = "shakespeare-char"
wandb_run_name = "kaufmann-nano"

dataset = "shakespeare_char"
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

# NeuroManifold Nano Config
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0

# NeuroManifold Specifics
sdr_size = 1024
manifold_dim = 32
n_eigenvectors = 16
use_sdr = True
kan_type = "wave"
kan_wavelet = "dog"
use_fast_wavekan = True

# Kaufmann Features
use_kaufmann_attention = True  # Enable Trifecta (FHN + Knot + Ramanujan)
use_knot_attention = True  # Included in Kaufmann
use_ramanujan = True  # Included

# FHN Tuning
fhn_threshold = 0.1
fhn_tau = 12.5
n_fhn_steps = 2
use_fhn_imex = True
use_fhn_partitioning = True
use_fhn_fused = False
use_fhn_parallel = (
    False  # Disable FFT Scan for stability (Fallback to sequential 2-step)
)

learning_rate = 1e-3
max_iters = 1000
lr_decay_iters = 1000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

model_type = "neuromanifold"
