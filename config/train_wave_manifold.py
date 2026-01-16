# Train WaveManifoldGPT on Shakespeare
out_dir = "out-wave-shakespeare"
eval_interval = 50
eval_iters = 20
log_interval = 10

# Data
dataset = "shakespeare_char"
batch_size = 16
block_size = 256
vocab_size = 0  # Auto-detect from meta.pkl (65 for Shakespeare)

# Model
model_type = "wave_manifold"
n_layer = 2
n_head = 4
n_embd = 128
dropout = 0.1

# Wave Manifold Specifics
use_fno_encoder = True
fno_modes = 16
use_mamba_backbone = True
mamba_state_dim = 16
use_soliton_mixing = True
soliton_type = "sine_gordon"
use_topological_loss = True
use_continuous_head = False # Keep simple first

# Optimization
learning_rate = 1e-3
max_iters = 600  # Approx 100-120 seconds at current speed (5it/s)
warmup_iters = 50 # Increased warmup for stability
weight_decay = 0.01

# Hardware
devices = 1
precision = "bf16-mixed"
compile_model = False
wandb_log = False
