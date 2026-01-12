# train a miniature character-level shakespeare model
# Kaufmann Trifecta Attention Run (V2: Reaction-Diffusion)

out_dir = 'out-kaufmann-v2'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'shakespeare-char'
wandb_run_name = 'kaufmann-v2'

dataset = 'shakespeare_char'
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
# WaveKAN (kept for FFN)
kan_type = "wave"
kan_wavelet = "dog"
use_fast_wavekan = True

# Kaufmann Features
use_kaufmann_attention = True # Enable V2 Reaction-Diffusion
use_knot_attention = False # Internal to Kaufmann
use_ramanujan = True # Keep Ramanujan (it works)

# FHN Tuning (Reaction-Diffusion)
fhn_threshold = 0.5
fhn_tau = 12.5
n_fhn_steps = 2 # Reaction steps
use_fhn_imex = True
use_fhn_partitioning = False # Not needed for spatial V2? Or keep for balance? Keep off for simplicity.
use_fhn_fused = False
use_fhn_parallel = False

learning_rate = 1e-3
max_iters = 1000
lr_decay_iters = 1000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

model_type = 'neuromanifold'
