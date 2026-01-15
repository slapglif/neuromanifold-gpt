# Reasoning Training Config V21
# SOTA TECHNIQUES: DeepSeek-V3 + Qwen3 + GLM-4.5 + MiniMax-M1
#
# V21 SOTA Stack:
# 1. QK-Norm (Qwen3/GLM-4.5) - Prevents attention logit explosion
# 2. FP32 LM Head (MiniMax) - 151K vocab stability
# 3. Init std=0.006 (DeepSeek-V3) - Faster early convergence
# 4. AdamW eps=1e-15, beta2=0.95 (MiniMax) - Handles tiny gradients
# 5. WSD LR schedule (MiniMax/DeepSeek) - Better final loss
# 6. MTP loss=0.1 (DeepSeek-V3) - Densified training signals
# 7. mHC ON (DeepSeek validated) - Proven convergence help
# 8. All reasoning components ON

out_dir = 'out-reasoning-v21'

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

# Qwen3 tokenizer (151K vocab - MANDATORY per user)
tokenizer = 'qwen3'
vocab_size = 151680

# ==== SOTA ARCHITECTURE ENHANCEMENTS ====
# QK-Norm (Qwen3/GLM-4.5 style) - prevents attention logit explosion
use_qk_norm = True

# FP32 LM Head (MiniMax critical) - numerical stability for 151K vocab
lm_head_fp32 = True

# Small initialization (DeepSeek-V3 style) - faster early convergence
init_std = 0.006

# ==== SDR - Sparse Distributed Representations ====
use_sdr = True
sdr_size = 512

# ==== FasterKAN - Wave-based FFN ====
use_kan = True
kan_type = 'faster'
kan_num_centers = 3

# ==== mHC - Manifold Hyper-Connections (DeepSeek validated) ====
# DO NOT DISABLE - DeepSeek explicitly validates this improves convergence
use_mhc = True
use_full_mhc = True
mhc_n_streams = 2
mhc_sinkhorn_iters = 5

# ==== FHN Dynamics - Neural wave propagation ====
n_fhn_steps = 2  # Minimum 2 for proper dynamics
use_fhn_imex = True
use_fhn_partitioning = True

# Skip spectral for memory efficiency
skip_manifold_spectral = True

# ==== MTP - Multi-Token Prediction (DeepSeek-V3) ====
# Densifies training signals - each token contributes to multiple losses
use_mtp = True
mtp_n_predict = 2
mtp_loss_weight = 0.1  # DeepSeek: NOT zero, positive weight densifies training

# ==== REASONING COMPONENTS - ALL ON ====
use_dag_planner = True
dag_max_nodes = 6
dag_min_nodes = 3

use_hierarchical_memory = True
hierarchical_l1_capacity = 32
hierarchical_l2_capacity = 128
hierarchical_l3_capacity = 512

use_imagination = True
imagination_steps = 2
imagination_n_alternatives = 2

use_multiscale_manifold = True
multiscale_coarse_dim = 16
multiscale_medium_dim = 32
multiscale_fine_dim = 64

# ==== OPTIMIZER - MiniMax Critical Discovery ====
# Most gradients are smaller than 1e-14
# Default eps=1e-8 masks tiny gradients, beta2=0.999 too slow
learning_rate = 6e-4  # Conservative for stability
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95  # MiniMax: lower than 0.999 for faster adaptation
optimizer_eps = 1e-15  # MiniMax critical: handles tiny gradients
grad_clip = 1.0

# ==== WSD LR SCHEDULE (MiniMax/DeepSeek) ====
# Warmup-Stable-Decay instead of cosine for better final loss
lr_schedule = 'wsd'
warmup_ratio = 0.05  # 5% warmup
stable_ratio = 0.65  # 65% stable at peak (the key innovation)
decay_ratio = 0.30   # 30% linear decay

# Batch size
batch_size = 2
gradient_accumulation_steps = 32  # Effective batch = 64
num_workers = 0

# Training iterations
max_iters = 10000
min_lr = 6e-5  # 10% of max

# Evaluation
eval_interval = 250
eval_iters = 20
log_interval = 50

# Checkpointing
compile_model = False
save_checkpoints = True
