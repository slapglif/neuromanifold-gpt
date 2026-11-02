"""NS-WMN with FNO + Soliton Mixing enabled."""

out_dir = "out-ns-wmn-soliton"
eval_interval = 100
eval_iters = 20
log_interval = 1
save_checkpoints = True

dataset = "shakespeare_char"
batch_size = 32
block_size = 256

model_type = "wave_manifold"
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
bias = False

use_fno_encoder = True
fno_modes = 16

use_mixture_of_mamba = False
use_soliton_mixing = True
soliton_type = "sine_gordon"

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
