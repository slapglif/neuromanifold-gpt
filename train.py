"""
PyTorch Lightning training script for NeuroManifoldGPT and baseline GPT.

Supports:
- Single GPU and multi-GPU (DDP) training
- Mixed precision (bf16/fp16/fp32)
- Gradient accumulation
- WandB logging
- Checkpoint resume
- Early stopping
- Sample generation during training

Usage:
    # Single GPU
    python train.py --config config/train_neuromanifold_shakespeare.py

    # Multi-GPU (Lightning handles DDP)
    python train.py --config config/train_neuromanifold_shakespeare.py --devices 4
"""

import os
import math
import pickle
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    Callback,
)
from pytorch_lightning.loggers import WandbLogger
from loguru import logger

from neuromanifold_gpt.model.gpt import NeuroManifoldGPT
from neuromanifold_gpt.config.base import NeuroManifoldConfig
from model import GPTConfig, GPT


# -----------------------------------------------------------------------------
# Default Configuration
# -----------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """Training configuration with all hyperparameters."""
    # I/O
    out_dir: str = "out-lightning"
    eval_interval: int = 2000
    log_interval: int = 10
    eval_iters: int = 200
    save_checkpoints: bool = True

    # Data
    dataset: str = "shakespeare_char"
    batch_size: int = 64
    block_size: int = 256
    num_workers: int = 4
    streaming: bool = False  # Use HuggingFace streaming for general text
    vocab_size: int = 0  # 0 = auto-detect, 50257 = GPT-2 BPE

    # Model
    model_type: str = "neuromanifold"  # "neuromanifold" or "gpt"
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = False

    # NeuroManifold specific
    sdr_size: int = 2048
    manifold_dim: int = 64
    n_eigenvectors: int = 32
    use_sdr: bool = False
    use_kan: bool = True
    kan_type: str = "faster"
    kan_wavelet: str = "dog"
    use_fast_wavekan: bool = True
    kan_num_centers: int = 3
    fhn_threshold: float = 0.5
    fhn_tau: float = 12.5
    n_fhn_steps: int = 2
    use_fhn_imex: bool = True
    use_fhn_partitioning: bool = True
    use_fhn_fused: bool = False
    use_mhc: bool = True
    use_full_mhc: bool = True
    mhc_n_streams: int = 2
    use_kaufmann_attention: bool = False
    attention: str = "standard"  # Attention mechanism: standard, soliton, sdr, fast-spectral

    # Speed optimization
    skip_manifold_spectral: bool = False  # Skip manifold/spectral for faster training

    # Training
    max_iters: int = 5000
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_iters: int = 100
    lr_decay_iters: int = 5000

    # Early stopping
    early_stopping_patience: int = 50

    # Sampling during training
    sample_interval: int = 500
    sample_max_tokens: int = 200
    sample_temperature: float = 1.0
    sample_top_k: int = 40

    # Hardware
    devices: int = 1
    precision: str = "bf16-mixed"
    compile_model: bool = False

    # Logging
    wandb_log: bool = False
    wandb_project: str = "neuromanifold-gpt"
    wandb_run_name: str = "neuromanifold"


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class MemmapDataset(Dataset):
    """Memory-mapped dataset for efficient large file handling."""

    def __init__(self, data_path: str, block_size: int):
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __len__(self) -> int:
        return max(1, len(self.data) - self.block_size - 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Random sampling within data (ignore idx for randomness like original)
        pos = torch.randint(len(self.data) - self.block_size, (1,)).item()
        chunk = torch.from_numpy(
            self.data[pos : pos + self.block_size + 1].astype(np.int64)
        )
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


class TextDataModule(pl.LightningDataModule):
    """Lightning DataModule for text datasets."""

    def __init__(
        self,
        data_dir: str,
        block_size: int,
        batch_size: int,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.block_size = block_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_size = None
        self.itos = None
        self.stoi = None

    def setup(self, stage: Optional[str] = None) -> None:
        # Load vocab info if available
        meta_path = os.path.join(self.data_dir, "meta.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            self.vocab_size = meta["vocab_size"]
            self.itos = meta.get("itos")
            self.stoi = meta.get("stoi")
            logger.info(f"Loaded vocab_size={self.vocab_size} from {meta_path}")
        else:
            self.vocab_size = 50304  # GPT-2 default
            logger.info(f"No meta.pkl found, using default vocab_size={self.vocab_size}")

        self.train_ds = MemmapDataset(
            os.path.join(self.data_dir, "train.bin"), self.block_size
        )
        self.val_ds = MemmapDataset(
            os.path.join(self.data_dir, "val.bin"), self.block_size
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )


class StreamingDataModule(pl.LightningDataModule):
    """Lightning DataModule for streaming HuggingFace datasets."""

    def __init__(
        self,
        dataset_name: str,
        block_size: int,
        batch_size: int,
        num_workers: int = 2,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.block_size = block_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_size = 50257  # GPT-2 BPE tokenizer

    def setup(self, stage: Optional[str] = None) -> None:
        from neuromanifold_gpt.data.streaming import create_streaming_dataset
        self.train_ds = create_streaming_dataset(
            self.dataset_name,
            block_size=self.block_size
        )
        self.val_ds = create_streaming_dataset(
            self.dataset_name,
            block_size=self.block_size
        )
        logger.info(f"Streaming dataset: {self.dataset_name}, vocab_size={self.vocab_size}")

    def train_dataloader(self) -> DataLoader:
        from neuromanifold_gpt.data.streaming import create_streaming_dataset
        dataset = create_streaming_dataset(self.dataset_name, self.block_size)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        from neuromanifold_gpt.data.streaming import create_streaming_dataset
        dataset = create_streaming_dataset(self.dataset_name, self.block_size)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=1,
        )


# -----------------------------------------------------------------------------
# Lightning Module
# -----------------------------------------------------------------------------
class NeuroManifoldLitModule(pl.LightningModule):
    """PyTorch Lightning module for NeuroManifold/GPT training."""

    def __init__(
        self,
        model_config: NeuroManifoldConfig | GPTConfig,
        train_config: TrainConfig,
        itos: Optional[Dict[int, str]] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["itos"])

        self.train_config = train_config
        self.model_config = model_config
        self.itos = itos

        # Build model
        if isinstance(model_config, NeuroManifoldConfig):
            self.model = NeuroManifoldGPT(model_config)
            logger.info("Initialized NeuroManifoldGPT")
        else:
            self.model = GPT(model_config)
            logger.info("Initialized baseline GPT")

        # Log parameter count
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {n_params/1e6:.2f}M")

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None):
        return self.model(x, targets)

    def _compute_loss(self, batch: tuple) -> tuple[torch.Tensor, Dict[str, Any]]:
        """Compute loss and extract info from model output."""
        x, y = batch
        outputs = self.model(x, y)

        # Handle different return signatures
        if len(outputs) == 3:
            logits, loss, info = outputs
        else:
            logits, loss = outputs
            info = {}

        return loss, info

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        loss, info = self._compute_loss(batch)

        # Log metrics
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        # Log perplexity
        ppl = torch.exp(loss.detach()).clamp(max=1000)
        self.log("train/perplexity", ppl, on_step=True, on_epoch=False)

        # Log additional info from model
        for k, v in info.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                self.log(f"train/{k}", v.mean(), on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        loss, info = self._compute_loss(batch)

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

        ppl = torch.exp(loss.detach()).clamp(max=1000)
        self.log("val/perplexity", ppl, sync_dist=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer with cosine LR schedule and warmup.

        Note: We always use manual param grouping with fused=False to avoid
        conflicts with Lightning's gradient clipping in AMP mode.
        """
        # Manual param grouping - separate decay and no-decay params
        decay_params = []
        nodecay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Biases, LayerNorm, embeddings don't get weight decay
            if param.dim() < 2 or "ln" in name or "norm" in name:
                nodecay_params.append(param)
            else:
                decay_params.append(param)

        # IMPORTANT: fused=False to avoid conflict with Lightning's gradient clipping
        # When using AMP (bf16-mixed), fused AdamW does internal gradient unscaling
        # which conflicts with Lightning's gradient clipping mechanism
        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.train_config.weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ],
            lr=self.train_config.learning_rate,
            betas=(self.train_config.beta1, self.train_config.beta2),
            fused=False,  # Required for Lightning gradient clipping compatibility
        )

        # Cosine annealing with warmup
        def lr_lambda(step: int) -> float:
            # Warmup
            if step < self.train_config.warmup_iters:
                return (step + 1) / (self.train_config.warmup_iters + 1)
            # Decay
            if step > self.train_config.lr_decay_iters:
                return self.train_config.min_lr / self.train_config.learning_rate
            # Cosine decay
            decay_ratio = (step - self.train_config.warmup_iters) / (
                self.train_config.lr_decay_iters - self.train_config.warmup_iters
            )
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return (
                self.train_config.min_lr
                + coeff * (self.train_config.learning_rate - self.train_config.min_lr)
            ) / self.train_config.learning_rate

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate tokens autoregressively."""
        self.model.eval()

        for _ in range(max_new_tokens):
            # Crop context if needed
            block_size = getattr(self.model_config, "block_size", 1024)
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]

            # Forward
            outputs = self.model(idx_cond)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            logits = logits[:, -1, :] / temperature

            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        self.model.train()
        return idx


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------
class SampleGenerationCallback(Callback):
    """Generate samples periodically during training."""

    def __init__(
        self,
        sample_interval: int,
        max_tokens: int = 200,
        temperature: float = 1.0,
        top_k: int = 40,
        itos: Optional[Dict[int, str]] = None,
        stoi: Optional[Dict[str, int]] = None,
    ):
        self.sample_interval = sample_interval
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.itos = itos
        self.stoi = stoi

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: NeuroManifoldLitModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        if self.sample_interval <= 0:
            return
        if self.itos is None:
            return

        global_step = trainer.global_step
        if global_step > 0 and global_step % self.sample_interval == 0:
            self._generate_sample(pl_module, trainer)

    def _generate_sample(
        self, pl_module: NeuroManifoldLitModule, trainer: pl.Trainer
    ) -> None:
        """Generate and log a sample."""
        device = pl_module.device

        # Start token
        start_char = "\n"
        if self.stoi and start_char in self.stoi:
            start_idx = self.stoi[start_char]
        else:
            start_idx = 0

        start = torch.tensor([[start_idx]], device=device, dtype=torch.long)

        try:
            out_idx = pl_module.generate(
                start,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
            )

            text = "".join([self.itos[i] for i in out_idx[0].tolist()])
            logger.info(f"\n--- Sample (step {trainer.global_step}) ---\n{text}\n--- End Sample ---")

            # Log to wandb if available
            if trainer.logger and hasattr(trainer.logger, "experiment"):
                try:
                    trainer.logger.experiment.log(
                        {"samples/text": text, "global_step": trainer.global_step}
                    )
                except Exception:
                    pass

        except Exception as e:
            logger.warning(f"Sample generation failed: {e}")


class MFUCallback(Callback):
    """Track Model FLOPs Utilization."""

    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.running_mfu = -1.0

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: NeuroManifoldLitModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        if batch_idx % self.log_interval != 0:
            return

        model = pl_module.model
        if not hasattr(model, "estimate_mfu"):
            return

        # Estimate MFU
        batch_size = batch[0].size(0)
        accum = pl_module.train_config.gradient_accumulation_steps

        # Get time per iteration from trainer
        if hasattr(trainer, "progress_bar_metrics"):
            # Approximate dt from throughput
            dt = 0.1  # fallback
        else:
            dt = 0.1

        try:
            mfu = model.estimate_mfu(batch_size * accum, dt)
            if self.running_mfu < 0:
                self.running_mfu = mfu
            else:
                self.running_mfu = 0.9 * self.running_mfu + 0.1 * mfu

            pl_module.log("train/mfu", self.running_mfu * 100, on_step=True)
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Main Training Function
# -----------------------------------------------------------------------------
def train(config: TrainConfig) -> None:
    """Main training entry point."""
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


# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train NeuroManifoldGPT")
    parser.add_argument("--config", type=str, help="Path to config file")

    # Allow overriding any config value - defaults to None so config file takes precedence
    for f in TrainConfig.__dataclass_fields__:
        field_type = TrainConfig.__dataclass_fields__[f].type
        if field_type == bool:
            parser.add_argument(f"--{f}", type=lambda x: x.lower() == "true", default=None)
        elif field_type == int:
            parser.add_argument(f"--{f}", type=int, default=None)
        elif field_type == float:
            parser.add_argument(f"--{f}", type=float, default=None)
        elif field_type == str:
            if f == "attention":
                parser.add_argument(
                    f"--{f}",
                    type=str,
                    default=None,
                    choices=["standard", "soliton", "sdr", "fast-spectral"],
                    help="Attention mechanism type: standard, soliton, sdr, fast-spectral"
                )
            else:
                parser.add_argument(f"--{f}", type=str, default=None)

    args = parser.parse_args()

    # Start with defaults
    config = TrainConfig()

    # Load config file if provided (overrides defaults)
    if args.config:
        config_globals = {}
        exec(open(args.config).read(), config_globals)
        for k, v in config_globals.items():
            if hasattr(config, k):
                setattr(config, k, v)

    # CLI args override config file (only if explicitly provided)
    for k, v in vars(args).items():
        if k != "config" and v is not None and hasattr(config, k):
            setattr(config, k, v)

    # Run training
    train(config)
