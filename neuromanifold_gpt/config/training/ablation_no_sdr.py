# Ablation: No SDR (Dense Embeddings)
# Based on train_neuromanifold_shakespeare.py

out_dir = "out-neuromanifold-ablation-no-sdr"
eval_interval = 250  # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10  # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True
save_checkpoints = True

wandb_log = False  # override via command line if you like
wandb_project = "shakespeare-char"
wandb_run_name = "neuromanifold-ablation-no-sdr"

dataset = "shakespeare_char"
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256  # context of up to 256 previous characters

# NeuroManifold Nano Config
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0

# NeuroManifold Specifics
sdr_size = 1024
manifold_dim = 32
n_eigenvectors = 16
use_sdr = False  # <--- ABLATION: DISABLED
kan_type = "faster"  # FasterKAN with RSWAF basis (fastest)
use_kan_everywhere = False  # Keep Linear for projections, FasterKAN only for FFN
kan_num_centers = 3  # Reduced from 8 for efficiency

# NeuroManifold FHN tuning
fhn_threshold = 0.1  # Lower threshold for better signal flow
fhn_tau = 12.5
n_fhn_steps = 2
use_fhn_imex = True
use_fhn_partitioning = True
use_fhn_fused = False

learning_rate = 3e-3  # Aggressive learning rate
max_iters = 1000  # <--- ABLATION: SHORT RUN
lr_decay_iters = 1000  # Match max_iters
eval_interval = 250  # Checkpoint frequently
