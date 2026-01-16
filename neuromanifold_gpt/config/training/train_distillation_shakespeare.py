# train_distillation_shakespeare.py
# Knowledge distillation config for compressing a teacher model into a smaller student
# Distills from a trained NeuroManifold model into a more compact variant

from neuromanifold_gpt.training.config import TrainConfig

# Output and checkpointing
out_dir = 'out-distillation-shakespeare'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = True
save_checkpoints = True

# Logging
wandb_log = False
wandb_project = 'shakespeare-char'
wandb_run_name = 'distillation-nano'

# Dataset (same as teacher)
dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

# Student Model - Smaller than teacher (2 layers vs 4)
n_layer = 2  # Half the layers of the teacher
n_head = 4
n_embd = 128
dropout = 0.0

# NeuroManifold Specifics (match teacher architecture)
sdr_size = 1024
manifold_dim = 32
n_eigenvectors = 16
use_sdr = True
kan_type = "wave"
kan_wavelet = "dog"
use_fast_wavekan = True

# NeuroManifold FHN tuning
fhn_threshold = 0.1
fhn_tau = 12.5
n_fhn_steps = 2
use_fhn_imex = True
use_fhn_partitioning = True
use_fhn_fused = False

# Distillation-specific settings
enable_distillation = True
teacher_checkpoint = 'out-neuromanifold-shakespeare/ckpt.pt'  # Path to trained teacher model
distillation_alpha = 0.5  # Balance between task loss and distillation loss
distillation_temperature = 2.0  # Temperature for softening teacher predictions

# Training schedule (similar to teacher)
learning_rate = 3e-3
max_iters = 10000
lr_decay_iters = 10000
eval_interval = 500


def get_config() -> TrainConfig:
    """Get distillation training configuration.

    Returns:
        TrainConfig: Configuration with distillation settings and enable_distillation flag
    """
    config = TrainConfig(
        # Output
        out_dir=out_dir,
        eval_interval=eval_interval,
        eval_iters=eval_iters,
        log_interval=log_interval,
        always_save_checkpoint=always_save_checkpoint,
        save_checkpoints=save_checkpoints,

        # Logging
        wandb_log=wandb_log,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,

        # Dataset
        dataset=dataset,
        gradient_accumulation_steps=gradient_accumulation_steps,
        batch_size=batch_size,
        block_size=block_size,

        # Student Model
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,

        # NeuroManifold specifics
        sdr_size=sdr_size,
        manifold_dim=manifold_dim,
        n_eigenvectors=n_eigenvectors,
        use_sdr=use_sdr,
        kan_type=kan_type,
        kan_wavelet=kan_wavelet,
        use_fast_wavekan=use_fast_wavekan,

        # FHN tuning
        fhn_threshold=fhn_threshold,
        fhn_tau=fhn_tau,
        n_fhn_steps=n_fhn_steps,
        use_fhn_imex=use_fhn_imex,
        use_fhn_partitioning=use_fhn_partitioning,
        use_fhn_fused=use_fhn_fused,

        # Distillation
        teacher_checkpoint=teacher_checkpoint,
        distillation_alpha=distillation_alpha,
        distillation_temperature=distillation_temperature,

        # Training
        learning_rate=learning_rate,
        max_iters=max_iters,
        lr_decay_iters=lr_decay_iters,
    )

    # Add enable_distillation flag as an attribute
    config.enable_distillation = enable_distillation

    return config
