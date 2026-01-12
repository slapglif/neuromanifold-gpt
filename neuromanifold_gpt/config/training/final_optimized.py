# Final Optimized NeuroManifold Config
# "Smarter and Smaller" - Full Architecture, Optimized for Speed/Size

out_dir = "out-neuromanifold-optimized"
eval_interval = 250
eval_iters = 200
log_interval = 50

always_save_checkpoint = True
save_checkpoints = True

wandb_log = False
wandb_project = "shakespeare-char"
wandb_run_name = "neuromanifold-optimized"

dataset = "shakespeare_char"
gradient_accumulation_steps = 2
batch_size = 32
block_size = 256

# NeuroManifold Nano Config
n_layer = 3
n_head = 6
n_embd = 384
dropout = 0.0

# === The "Smarter" Parts ===

# 1. Semantic Folding (SDR) & HTM
use_sdr = False  # Dense for speed
sdr_size = 2048
sdr_sparsity = 0.10

# 2. Manifold & Spectral
manifold_dim = 64
n_eigenvectors = 16

# 3. FHN Attention (Soliton Dynamics)
fhn_threshold = 0.0
fhn_tau = 10.0
n_fhn_steps = 1  # Reduced steps for speed (Semi-Implicit handles this)
use_fhn_imex = True
use_fhn_partitioning = True
use_fhn_fused = False

# 4. KAN FFN (Replacing MLP)
use_kan = True
kan_type = "faster"
kan_num_centers = 2

# 5. Hyper-Connections (mHC)
use_mhc = True
use_full_mhc = True
mhc_n_streams = 2

# Training
learning_rate = 3e-3  # Aggressive LR for shallow net
max_iters = 1000
lr_decay_iters = 1000
min_lr = 3e-4

# Sampling (New)
sample_interval = 0
sample_iters = 1
sample_max_tokens = 200

# System
device = "cuda"
compile = True  # Inductor compilation for speed
