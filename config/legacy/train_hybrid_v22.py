# Hybrid Training Config V22
# MERGED: V21 (SOTA Pipeline) + V18 (SDR Tuning) + Hybrid Architecture
#
# OBJECTIVE: Verify full integration of DeepSeek SOTA techniques with our custom
# Hybrid NeuroManifold architecture (SDR + Ramanujan + KAN + mHC).
#
# V22 Configuration Stack:
# 1. Pipeline: QK-Norm, WSD, MTP, FP32 Head, DeepSeek Init, AdamW-MiniMax
# 2. SDR Tuning: Sparsity=0.08, Context=8 (from V18 findings)
# 3. Architecture: FasterKAN + Ramanujan PE + DeepSeek mHC
# 4. Memory: Active SDR Retrieval (Read-after-Write)

out_dir = 'out-hybrid-v22'

# Dataset - FineWeb-Edu streaming
dataset = 'fineweb-edu'
data_recipe = 'knowledge-heavy'
streaming = True

# ==== MODEL ARCHITECTURE ====
n_layer = 6
n_head = 6
n_embd = 384
block_size = 256
dropout = 0.05
bias = False  # Qwen3 style - no QKV bias

# Qwen3 tokenizer (151K vocab)
tokenizer = 'qwen3'
vocab_size = 151680

# ==== SOTA ARCHITECTURE ENHANCEMENTS (V21) ====
use_qk_norm = True        # Qwen3/GLM-4.5
lm_head_fp32 = True       # MiniMax 151K stability
init_std = 0.006          # DeepSeek-V3 convergence

# ==== HYBRID COMPONENTS ====

# 1. SDR Memory (V18 Tuning)
use_sdr = True
sdr_size = 512
sdr_sparsity = 0.08       # V18: Increased active neurons
sdr_context_size = 8      # V18: Expanded context
sdr_embed_dim = 384

# Memory Active Retrieval (SDR Read-after-Write)
memory_active_retrieval = True  # Enable SDR memory retrieval
memory_retrieval_top_k = 3      # Retrieve top 3 memories
memory_retrieval_weight = 0.1   # Mix weight for retrieved content

# 2. FasterKAN (Wave-based FFN)
use_kan = True
kan_type = 'faster'
kan_num_centers = 3

# 3. mHC (DeepSeek V3.2 Topology)
use_mhc = True            # Verified correct topology in block.py
use_full_mhc = True
mhc_n_streams = 2
mhc_sinkhorn_iters = 5

# 4. FHN Dynamics
n_fhn_steps = 2
use_fhn_imex = True
use_fhn_partitioning = True

# 5. Ramanujan PE (Implicitly enabled via Architecture definition)
# No specific config flag needed if hardcoded in GPT class, but documented here.

# Skip heavy spectral computations for speed
skip_manifold_spectral = True

# ==== AUXILIARY LOSSES ====
# MTP (DeepSeek-V3) - Densify training signal
use_mtp = True
mtp_n_predict = 2
mtp_loss_weight = 0.1

# ==== REASONING COMPONENTS ====
use_dag_planner = True
dag_max_nodes = 6
dag_min_nodes = 3

use_hierarchical_memory = True
hierarchical_l1_capacity = 32
hierarchical_l2_capacity = 128

use_imagination = True
imagination_steps = 2

# ==== OPTIMIZER (MiniMax Recipe) ====
learning_rate = 6e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95              # MiniMax: faster adaptation
optimizer_eps = 1e-15     # MiniMax: tiny gradient handling
grad_clip = 1.0

# ==== WSD SCHEDULE (Warmup-Stable-Decay) ====
lr_schedule = 'wsd'
warmup_ratio = 0.05
stable_ratio = 0.65
decay_ratio = 0.30

# Training Setup
batch_size = 4
gradient_accumulation_steps = 16  # Effective batch = 64
num_workers = 0
max_iters = 5000          # Short run for verification
min_lr = 6e-5

# Logging & Checkpointing
eval_interval = 100
eval_iters = 20
log_interval = 10
compile_model = False
save_checkpoints = True
