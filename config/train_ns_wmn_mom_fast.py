"""NS-WMN with optimized MoM - reduced experts for speed on 6GB GPU."""

out_dir = "out-ns-wmn-mom"
eval_interval = 100
eval_iters = 20
log_interval = 1
save_checkpoints = True

dataset = "shakespeare_char"
batch_size = 16
block_size = 256

model_type = "wave_manifold"
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
bias = False

use_fno_encoder = True
fno_modes = 16
backbone_type = "mom"

use_mixture_of_mamba = True
mom_num_experts = 2
mom_top_k = 1
mom_state_dim = 8
mamba_state_dim = 8

use_soliton_mixing = False
use_hybrid_reasoning = False
use_topological_loss = False
use_continuous_head = False
use_sac_output = False

learning_rate = 1e-3
max_iters = 1000
gradient_accumulation_steps = 2
lr_decay_iters = 1000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 50

devices = 1
compile_model = False
