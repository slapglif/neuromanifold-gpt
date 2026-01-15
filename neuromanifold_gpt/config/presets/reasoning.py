# Reasoning preset - System 2 reasoning components enabled
# Designed for deliberate, systematic reasoning with DAG planning,
# hierarchical memory, imagination, and multi-token prediction

# ========================================
# System 2 Reasoning Components
# ========================================

# ForcedDAGPlanner - Decomposes tasks into DAGs for systematic reasoning
use_dag_planner = True
dag_max_nodes = 32
dag_min_nodes = 3

# HierarchicalEngramMemory - L1/L2/L3 tiered memory system
use_hierarchical_memory = True
hierarchical_l1_capacity = 64
hierarchical_l2_capacity = 512
hierarchical_l3_capacity = 4096

# ConsistencyImaginationModule - Counterfactual exploration
use_imagination = True
imagination_steps = 4
imagination_n_alternatives = 4

# Multi-Token Prediction - Predicting multiple future tokens
use_mtp = True
mtp_n_predict = 4
mtp_loss_weight = 0.1

# ========================================
# Model Configuration
# ========================================

# Use reasonable model size for reasoning tasks
n_layer = 12
n_head = 12
n_embd = 768
manifold_dim = 128
n_eigenvectors = 64

# ========================================
# Training Configuration
# ========================================

batch_size = 8
block_size = 1024
gradient_accumulation_steps = 16  # ~128 effective batch size

max_iters = 100000
eval_interval = 1000
warmup_iters = 1000
lr_decay_iters = 100000

learning_rate = 3e-4
min_lr = 3e-5

# Output
out_dir = "out-neuromanifold-reasoning"
wandb_run_name = "neuromanifold-reasoning"
