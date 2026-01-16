"""
Main training orchestration function for NeuroManifoldGPT.

This module provides the train() entry point that coordinates:
- Data module setup (streaming or local memmap)
- Model configuration (NeuroManifold or baseline GPT)
- Lightning module creation
- Callbacks (checkpointing, early stopping, sampling, MFU)
- PyTorch Lightning Trainer initialization and execution
"""

import os
from typing import Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger
from loguru import logger

from neuromanifold_gpt.config.base import NeuroManifoldConfig
from neuromanifold_gpt.training.config import TrainConfig
from neuromanifold_gpt.training.data_modules import StreamingDataModule, TextDataModule
from neuromanifold_gpt.training.lightning_module import NeuroManifoldLitModule
from neuromanifold_gpt.training.callbacks import SampleGenerationCallback, MFUCallback
from model import GPTConfig, GPT


def train(config: TrainConfig) -> None:
    """Main training entry point for NeuroManifoldGPT.

    Orchestrates the complete training pipeline:
    1. Sets up data module (streaming or local memmap)
    2. Configures model (NeuroManifold or baseline GPT)
    3. Creates Lightning module with optimizer and scheduler
    4. Configures callbacks (checkpointing, early stopping, sampling, MFU)
    5. Initializes PyTorch Lightning Trainer
    6. Runs training loop with validation
    7. Generates final sample

    Args:
        config: TrainConfig with all training hyperparameters and settings
    """
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

    # Build model config
    if config.model_type == "neuromanifold":
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
            # Speed optimization
            skip_manifold_spectral=config.skip_manifold_spectral,
            # Training
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
        )

    # Build Lightning module
    lit_module = NeuroManifoldLitModule(
        model_config=model_config,
        train_config=config,
        itos=data_module.itos,
    )

    # Compile model if requested
    if config.compile_model:
        logger.info("Compiling model with torch.compile()...")
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
    logger.info(f"Starting training: {config.model_type} on {config.dataset}")
    logger.info(f"Output dir: {config.out_dir}")
    trainer.fit(lit_module, data_module)

    # Final sample generation
    if data_module.itos:
        logger.info("\n=== Final Sample ===")
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
