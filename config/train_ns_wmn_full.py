"""
Configuration for training full NeuroManifoldGPT on Shakespeare.

Enables all advanced features:
- FNO Encoder
- Mixture-of-Mamba Backbone
- Soliton Physics (Sine-Gordon, KdV)
- Hybrid Reasoning (System 1/2)
- Topological Head (Braid Theory)
- Continuous Output (SAC/Diffusion)
"""

# I/O
out_dir = "out-ns-wmn-shakespeare"
eval_interval = 100
eval_iters = 20
log_interval = 1
save_checkpoints = True

# Data
dataset = "shakespeare_char"
batch_size = 16
block_size = 256  # Context window

# Model Architecture
model_type = "wave_manifold"  # Use WaveManifoldGPT
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
bias = False

# NeuroManifold Features
# -----------------------------------------------------------------------------

# 1. FNO Encoder (Continuous Input)
use_fno_encoder = True
fno_modes = 16
fno_width = 32

# 2. Mixture-of-Mamba Backbone (Modality-Aware SSM)
use_mixture_of_mamba = True
mom_num_experts = 4
mom_top_k = 2
mom_state_dim = 16

# 3. Soliton Physics (Non-linear Dynamics)
use_soliton_mixing = True
soliton_types = ["sine_gordon", "kdv", "heimburg_jackson"]

# 4. Hybrid Reasoning (System 1 vs System 2)
use_hybrid_reasoning = True
n_thinking_layers = 2
thinking_threshold = 0.5
use_e7_prior = True

# 5. Topological Head (Braid Theory)
use_topological_loss = True
braid_dim = 64

# 6. Continuous Output (SAC/Diffusion) - DISABLED for stable discrete training
use_continuous_head = False
use_sac_output = False

# Training
learning_rate = 1e-3
max_iters = 5000
gradient_accumulation_steps = 4
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

# System
devices = 1
compile_model = False  # Disabled: complex ops (FNO/soliton) cause slow compilation
