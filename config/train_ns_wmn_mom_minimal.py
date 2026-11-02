"""NS-WMN with FNO + Minimal MoM (fits in 6GB VRAM)."""

out_dir = "out-ns-wmn-mom-minimal"
eval_interval = 100
eval_iters = 20
log_interval = 1
save_checkpoints = True

dataset = "shakespeare_char"
batch_size = 16  # Reduced from 32
block_size = 128  # Reduced from 256

model_type = "wave_manifold"
n_layer = 4  # Reduced from 6
n_head = 4  # Reduced from 6
n_embd = 256  # Reduced from 384
dropout = 0.1
bias = False

use_fno_encoder = True
fno_modes = 8  # Reduced from 16

# MoM enabled with minimal config
use_mixture_of_mamba = True
mom_num_experts = 2  # Minimal experts (was 4)
mom_top_k = 1
mom_state_dim = 8  # Reduced state dim

use_soliton_mixing = False  # Disable for pure MoM test
use_hybrid_reasoning = False
use_topological_loss = False
use_continuous_head = False
use_sac_output = False

learning_rate = 1e-3
max_iters = 1500
gradient_accumulation_steps = 4  # Compensate for smaller batch
lr_decay_iters = 1500
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 25

devices = 1
compile_model = False
