out_dir = "out-ns-wmn-combined"
eval_interval = 100
eval_iters = 20
log_interval = 1
save_checkpoints = True

dataset = "shakespeare_char"
batch_size = 16
block_size = 128

model_type = "wave_manifold"
n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.1
bias = False

use_fno_encoder = True
fno_modes = 8
fno_blend_alpha = 0.0

use_mixture_of_mamba = True
mom_num_experts = 2
mom_top_k = 1
mom_state_dim = 8

use_soliton_mixing = True
soliton_type = "sine_gordon"

use_hybrid_reasoning = False
use_topological_loss = False
use_continuous_head = False
use_sac_output = False

learning_rate = 1e-3
max_iters = 500
gradient_accumulation_steps = 4
lr_decay_iters = 500
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 25

devices = 1
compile_model = False
