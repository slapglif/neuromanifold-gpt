"""
PyTorch Lightning distillation script for NeuroManifoldGPT.

Knowledge distillation training that compresses larger trained models into
smaller, faster variants while preserving performance.

Supports:
- Teacher-student distillation with KL divergence
- Different student architectures than teacher
- Combined loss (task loss + distillation loss)
- Single GPU and multi-GPU (DDP) training
- Mixed precision (bf16/fp16/fp32)
- All standard training features (checkpointing, early stopping, logging)

Usage:
    # Basic distillation
    python distill.py --teacher_checkpoint=out-teacher/ckpt.pt --n_layer=2

    # With config preset
    python distill.py neuromanifold_gpt.config.training.train_distillation_shakespeare

    # Multi-GPU distillation
    python distill.py --teacher_checkpoint=out-teacher/ckpt.pt --devices=4
"""

# Validate dependencies before heavy imports to fail fast with clear messages
import sys
import os as _os
_validation_path = _os.path.join(_os.path.dirname(__file__), 'neuromanifold_gpt', 'config')
sys.path.insert(0, _validation_path)
try:
    import validation as _validation_module
    _validation_module.validate_dependencies(verbose=True)
finally:
    sys.path.pop(0)  # Clean up sys.path

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger
from loguru import logger

from neuromanifold_gpt.model.gpt import NeuroManifoldGPT
from neuromanifold_gpt.model.wave_manifold_gpt import WaveManifoldGPT
from neuromanifold_gpt.config.base import NeuroManifoldConfig
from neuromanifold_gpt.config.wave_manifold_config import WaveManifoldConfig
from neuromanifold_gpt.config.loader import load_config
from neuromanifold_gpt.training.config import TrainConfig
from neuromanifold_gpt.training.data_modules import StreamingDataModule, TextDataModule
from neuromanifold_gpt.training.distillation_module import DistillationLitModule
from neuromanifold_gpt.training.callbacks import SampleGenerationCallback, MFUCallback
from model import GPTConfig, GPT


def train_distillation(config: TrainConfig) -> None:
    """Main distillation training entry point.

    Orchestrates knowledge distillation from a teacher model to a student model:
    1. Sets up data module (streaming or local memmap)
    2. Configures student model (can be different architecture than teacher)
    3. Creates DistillationLitModule with teacher checkpoint
    4. Configures callbacks (checkpointing, early stopping, sampling, MFU)
    5. Initializes PyTorch Lightning Trainer
    6. Runs distillation training loop with validation
    7. Generates final sample

    Args:
        config: TrainConfig with all training and distillation hyperparameters
    """
    # Validate required distillation parameters
    if not config.teacher_checkpoint:
        raise ValueError(
            "teacher_checkpoint is required for distillation training. "
            "Use --teacher_checkpoint=path/to/teacher.pt"
        )

    # Set matmul precision for stability
    torch.set_float32_matmul_precision('medium')

    pl.seed_everything(1337)

    # Setup data
    if config.streaming:
        # Use streaming data module for HuggingFace datasets
        data_module = StreamingDataModule(
            dataset_name=config.dataset,
            block_size=config.block_size,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )
        data_module.setup()
        logger.info(f"Using streaming data from: {config.dataset}")
    else:
        # Use local memmap data
        data_dir = os.path.join("data", config.dataset)
        data_module = TextDataModule(
            data_dir=data_dir,
            block_size=config.block_size,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )
        data_module.setup()

    # Override vocab_size if specified
    if config.vocab_size > 0:
        data_module.vocab_size = config.vocab_size

    # Build student model config
    if config.model_type == "wave_manifold":
        model_config = WaveManifoldConfig(
            vocab_size=data_module.vocab_size,
            block_size=config.block_size,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            dropout=config.dropout,
            bias=config.bias,
            # Wave specific
            use_fno_encoder=config.use_fno_encoder,
            fno_modes=config.fno_modes,
            use_mamba_backbone=config.use_mamba_backbone,
            mamba_state_dim=config.mamba_state_dim,
            mamba_expand=config.mamba_expand,
            use_soliton_mixing=config.use_soliton_mixing,
            soliton_type=config.soliton_type,
            use_topological_loss=config.use_topological_loss,
            use_continuous_head=config.use_continuous_head,
        )
    elif config.model_type == "neuromanifold":
        model_config = NeuroManifoldConfig(
            vocab_size=data_module.vocab_size,
            block_size=config.block_size,
            n_layer=config.n_layer,
            n_heads=config.n_head,
            n_embd=config.n_embd,
            dropout=config.dropout,
            bias=config.bias,
            # SDR
            use_sdr=config.use_sdr,
            sdr_size=config.sdr_size,
            # Manifold
            manifold_dim=config.manifold_dim,
            n_eigenvectors=config.n_eigenvectors,
            # KAN
            use_kan=config.use_kan,
            kan_type=config.kan_type,
            kan_wavelet=config.kan_wavelet,
            use_fast_wavekan=config.use_fast_wavekan,
            kan_num_centers=config.kan_num_centers,
            # FHN
            fhn_threshold=config.fhn_threshold,
            fhn_tau=config.fhn_tau,
            n_fhn_steps=config.n_fhn_steps,
            use_fhn_imex=config.use_fhn_imex,
            use_fhn_partitioning=config.use_fhn_partitioning,
            use_fhn_fused=config.use_fhn_fused,
            # mHC
            use_mhc=config.use_mhc,
            use_full_mhc=config.use_full_mhc,
            mhc_n_streams=config.mhc_n_streams,
            # Attention
            use_kaufmann_attention=config.use_kaufmann_attention,
            # SSM (Mamba)
            use_ssm=config.use_ssm,
            ssm_state_dim=config.ssm_state_dim,
            ssm_conv_kernel=config.ssm_conv_kernel,
            ssm_expand=config.ssm_expand,
            # Speed optimization
            skip_manifold_spectral=config.skip_manifold_spectral,
            # Training
            gradient_checkpointing=config.gradient_checkpointing,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            beta1=config.beta1,
            beta2=config.beta2,
            grad_clip=config.grad_clip,
        )
    else:
        model_config = GPTConfig(
            vocab_size=data_module.vocab_size,
            block_size=config.block_size,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            dropout=config.dropout,
            bias=config.bias,
            gradient_checkpointing=config.gradient_checkpointing,
        )

    logger.info(f"Student model: {config.model_type} with {config.n_layer} layers")
    logger.info(f"Teacher checkpoint: {config.teacher_checkpoint}")
    logger.info(
        f"Distillation: alpha={config.distillation_alpha}, "
        f"temperature={config.distillation_temperature}"
    )

    # Build Distillation Lightning module
    lit_module = DistillationLitModule(
        config=model_config,
        teacher_checkpoint=config.teacher_checkpoint,
        distillation_alpha=config.distillation_alpha,
        distillation_temperature=config.distillation_temperature,
    )

    # Compile model if requested
    if config.compile_model:
        logger.info("Compiling student model with torch.compile()...")
        lit_module.model = torch.compile(lit_module.model)

    # Callbacks
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
    ]

    if config.save_checkpoints:
        callbacks.append(
            ModelCheckpoint(
                dirpath=config.out_dir,
                filename="ckpt-{step:06d}-{val/loss:.4f}",
                monitor="val/loss",
                mode="min",
                save_top_k=3,
                save_last=True,
            )
        )

    if config.early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val/loss",
                patience=config.early_stopping_patience,
                mode="min",
                verbose=True,
            )
        )

    if config.sample_interval > 0 and data_module.itos:
        callbacks.append(
            SampleGenerationCallback(
                sample_interval=config.sample_interval,
                max_tokens=config.sample_max_tokens,
                temperature=config.sample_temperature,
                top_k=config.sample_top_k,
                itos=data_module.itos,
                stoi=data_module.stoi,
            )
        )

    callbacks.append(MFUCallback(log_interval=config.log_interval))

    # Logger
    pl_logger = None
    if config.wandb_log:
        pl_logger = WandbLogger(
            project=config.wandb_project,
            name=config.wandb_run_name,
            save_dir=config.out_dir,
        )

    # Trainer
    trainer = pl.Trainer(
        max_steps=config.max_iters,
        accelerator="auto",
        devices=config.devices,
        precision=config.precision,
        gradient_clip_val=config.grad_clip,
        accumulate_grad_batches=config.gradient_accumulation_steps,
        callbacks=callbacks,
        logger=pl_logger,
        default_root_dir=config.out_dir,
        enable_progress_bar=True,
        log_every_n_steps=config.log_interval,
        val_check_interval=config.eval_interval,
        limit_val_batches=config.eval_iters,
        enable_checkpointing=config.save_checkpoints,
    )

    # Train
    logger.info(f"Starting distillation: {config.model_type} on {config.dataset}")
    logger.info(f"Output dir: {config.out_dir}")
    trainer.fit(lit_module, data_module)

    # Final sample generation
    if data_module.itos:
        logger.info("\n=== Final Sample (Student Model) ===")
        device = lit_module.device
        start_idx = data_module.stoi.get("\n", 0) if data_module.stoi else 0
        start = torch.tensor([[start_idx]], device=device, dtype=torch.long)

        try:
            out_idx = lit_module.generate(
                start,
                max_new_tokens=400,
                temperature=config.sample_temperature,
                top_k=config.sample_top_k,
            )
            text = "".join([data_module.itos[i] for i in out_idx[0].tolist()])
            logger.info(f"\n{text}")
        except Exception as e:
            logger.warning(f"Final sample generation failed: {e}")


# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Load configuration with type-safe CLI overrides
    config = load_config(TrainConfig, sys.argv[1:])

    # Run distillation training
    train_distillation(config)
