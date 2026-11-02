"""Baseline NS-WMN - standard embeddings, no FNO."""

out_dir = "out-ns-wmn-baseline"
eval_interval = 100
eval_iters = 20
log_interval = 1
save_checkpoints = True

dataset = "shakespeare_char"
batch_size = 64
block_size = 256

model_type = "wave_manifold"
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
bias = False

# Standard embeddings (no FNO)
use_fno_encoder = False

# Disable all advanced features
use_mixture_of_mamba = False
use_soliton_mixing = False
use_hybrid_reasoning = False
use_topological_loss = False
use_continuous_head = False
use_sac_output = False

learning_rate = 1e-3
max_iters = 1000
gradient_accumulation_steps = 1
lr_decay_iters = 1000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 50

devices = 1
compile_model = False
