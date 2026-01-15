"""
Training example with component-specific metrics logging.

This script demonstrates how to:
- Integrate ComponentMetricsAggregator into training
- Log component metrics (SDR, FHN, MTP, Memory) to WandB/TensorBoard
- Create a custom PyTorch Lightning callback for metrics logging
- Track component health throughout training

Usage:
    # Train with component metrics logging to WandB
    python examples/train_with_component_metrics.py --wandb_log=True

    # Train with local logging only
    python examples/train_with_component_metrics.py --wandb_log=False

    # Custom configuration
    python examples/train_with_component_metrics.py --n_layer=4 --n_embd=256 --max_steps=1000
"""
import os
import sys

# Handle --help before any imports that require dependencies
if '--help' in sys.argv or '-h' in sys.argv:
    print(__doc__)
    print("\nConfiguration parameters:")
    print("  Training:")
    print("    --max_steps=<int>           Max training steps (default: 1000)")
    print("    --batch_size=<int>          Batch size (default: 16)")
    print("    --learning_rate=<float>     Learning rate (default: 3e-4)")
    print("    --gradient_clip_val=<float> Gradient clipping (default: 1.0)")
    print("")
    print("  Model:")
    print("    --n_layer=<int>             Number of layers (default: 4)")
    print("    --n_head=<int>              Number of attention heads (default: 4)")
    print("    --n_embd=<int>              Embedding dimension (default: 256)")
    print("    --block_size=<int>          Context length (default: 256)")
    print("    --use_sdr=<bool>            Enable SDR encoder (default: True)")
    print("    --dropout=<float>           Dropout rate (default: 0.1)")
    print("")
    print("  Logging:")
    print("    --wandb_log=<bool>          Enable WandB logging (default: False)")
    print("    --wandb_project=<str>       WandB project name (default: 'neuromanifold-metrics')")
    print("    --wandb_run_name=<str>      WandB run name (default: 'component-metrics')")
    print("    --log_component_metrics_every=<int>  Log component metrics every N steps (default: 50)")
    print("")
    print("  Data:")
    print("    --dataset=<str>             Dataset name (default: 'shakespeare_char')")
    print("    --out_dir=<str>             Output directory (default: 'out-metrics')")
    print("")
    print("  Hardware:")
    print("    --devices=<int>             Number of devices (default: 1)")
    print("    --precision=<str>           Training precision: '32', 'bf16-mixed', 'fp16-mixed' (default: 'bf16-mixed')")
    sys.exit(0)

from typing import Dict, Any
import torch
import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neuromanifold_gpt.model.gpt import NeuroManifoldGPT
from neuromanifold_gpt.config.base import NeuroManifoldConfig
from neuromanifold_gpt.evaluation import ComponentMetricsAggregator

# -----------------------------------------------------------------------------
# Custom Callback for Component Metrics Logging
# -----------------------------------------------------------------------------
class ComponentMetricsCallback(Callback):
    """PyTorch Lightning callback that logs component-specific metrics.

    This callback computes and logs SDR, FHN, MTP, and Memory metrics
    during training at specified intervals. Metrics are logged to the
    active logger (WandB, TensorBoard, etc.).
    """

    def __init__(self, log_every_n_steps: int = 50):
        """Initialize the callback.

        Args:
            log_every_n_steps: Log component metrics every N training steps
        """
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.aggregator = ComponentMetricsAggregator()

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Called after each training batch.

        Args:
            trainer: PyTorch Lightning trainer
            pl_module: The LightningModule being trained
            outputs: Outputs from training_step (includes info dict)
            batch: The training batch
            batch_idx: Index of current batch
        """
        # Only log at specified intervals
        if (trainer.global_step + 1) % self.log_every_n_steps != 0:
            return

        # Skip if no info dict in outputs
        if not isinstance(outputs, dict) or 'info' not in outputs:
            return

        info = outputs['info']
        config = pl_module.config

        # Get logits and targets if available
        logits = outputs.get('logits')
        targets = batch.get('labels') if isinstance(batch, dict) else None

        # Compute component metrics
        try:
            metrics = self.aggregator.compute(
                info=info,
                config=config,
                logits=logits,
                targets=targets
            )

            # Log metrics to the trainer's logger
            self._log_metrics(trainer, metrics, prefix='train')

        except Exception as e:
            # Don't crash training if metrics fail
            print(f"Warning: Failed to compute component metrics: {e}")

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called after each validation batch.

        Args:
            trainer: PyTorch Lightning trainer
            pl_module: The LightningModule being validated
            outputs: Outputs from validation_step
            batch: The validation batch
            batch_idx: Index of current batch
            dataloader_idx: Index of dataloader (for multi-loader validation)
        """
        # Only log for first batch of each validation epoch
        if batch_idx != 0:
            return

        # Skip if no info dict in outputs
        if not isinstance(outputs, dict) or 'info' not in outputs:
            return

        info = outputs['info']
        config = pl_module.config

        # Get logits and targets if available
        logits = outputs.get('logits')
        targets = batch.get('labels') if isinstance(batch, dict) else None

        # Compute component metrics
        try:
            metrics = self.aggregator.compute(
                info=info,
                config=config,
                logits=logits,
                targets=targets
            )

            # Log metrics to the trainer's logger
            self._log_metrics(trainer, metrics, prefix='val')

        except Exception as e:
            print(f"Warning: Failed to compute validation component metrics: {e}")

    def _log_metrics(
        self,
        trainer: L.Trainer,
        metrics: Dict[str, Dict[str, float]],
        prefix: str = 'train'
    ) -> None:
        """Log component metrics to the trainer's logger.

        Args:
            trainer: PyTorch Lightning trainer with logger
            metrics: Nested dict of component metrics
            prefix: Metric name prefix ('train' or 'val')
        """
        # Flatten nested metrics dict for logging
        flat_metrics = {}
        for component, component_metrics in metrics.items():
            for metric_name, value in component_metrics.items():
                # Create hierarchical metric name: prefix/component/metric
                full_name = f"{prefix}/{component}/{metric_name}"
                flat_metrics[full_name] = value

        # Log to trainer's logger
        if trainer.logger is not None:
            trainer.logger.log_metrics(flat_metrics, step=trainer.global_step)

# -----------------------------------------------------------------------------
# Enhanced LightningModule with Component Metrics
# -----------------------------------------------------------------------------
class NeuroManifoldWithMetrics(L.LightningModule):
    """Lightning wrapper for NeuroManifoldGPT with component metrics support.

    This module extends the standard training/validation steps to return
    the info dict needed for component metrics logging.
    """

    def __init__(self, config: NeuroManifoldConfig):
        super().__init__()
        self.config = config
        self.model = NeuroManifoldGPT(config)
        self.save_hyperparameters(ignore=['config'])

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        """Forward pass through model."""
        return self.model(input_ids, labels)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Training step with info dict for metrics."""
        input_ids = batch['input_ids']
        labels = batch['labels']

        logits, loss, info = self(input_ids, labels)

        # Log training loss
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)

        # Return outputs including info dict for metrics callback
        return {
            'loss': loss,
            'logits': logits,
            'info': info,
        }

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Validation step with info dict for metrics."""
        input_ids = batch['input_ids']
        labels = batch['labels']

        logits, loss, info = self(input_ids, labels)

        # Log validation loss
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_perplexity', torch.exp(loss), prog_bar=True)

        # Return outputs including info dict for metrics callback
        return {
            'loss': loss,
            'logits': logits,
            'info': info,
        }

    def configure_optimizers(self):
        """Configure optimizer with weight decay groups."""
        # Separate params into decay and no-decay groups
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Biases and norms don't get weight decay
            if param.ndim < 2 or 'ln' in name or 'norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optim_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
        )

        return optimizer

    def on_before_optimizer_step(self, optimizer):
        """Gradient clipping before optimizer step."""
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip,
            )

# -----------------------------------------------------------------------------
# Simple DataModule (using random data for demonstration)
# -----------------------------------------------------------------------------
class DemoDataModule(L.LightningDataModule):
    """Simple DataModule with random data for demonstration.

    In production, replace this with your actual dataset loading logic.
    """

    def __init__(self, vocab_size: int, block_size: int, batch_size: int, num_workers: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str = None):
        """Setup datasets."""
        # In production, load actual data here
        pass

    def train_dataloader(self):
        """Create random training data for demonstration."""
        from torch.utils.data import DataLoader, TensorDataset

        # Generate random tokens
        num_samples = 1000
        input_ids = torch.randint(0, self.vocab_size, (num_samples, self.block_size))
        labels = torch.randint(0, self.vocab_size, (num_samples, self.block_size))

        dataset = TensorDataset(input_ids, labels)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Use 0 for random data
            collate_fn=lambda batch: {
                'input_ids': torch.stack([x[0] for x in batch]),
                'labels': torch.stack([x[1] for x in batch]),
            }
        )

    def val_dataloader(self):
        """Create random validation data for demonstration."""
        from torch.utils.data import DataLoader, TensorDataset

        # Generate random tokens
        num_samples = 100
        input_ids = torch.randint(0, self.vocab_size, (num_samples, self.block_size))
        labels = torch.randint(0, self.vocab_size, (num_samples, self.block_size))

        dataset = TensorDataset(input_ids, labels)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda batch: {
                'input_ids': torch.stack([x[0] for x in batch]),
                'labels': torch.stack([x[1] for x in batch]),
            }
        )

# -----------------------------------------------------------------------------
# Main Training Script
# -----------------------------------------------------------------------------
def main():
    # Configuration
    max_steps = 1000
    batch_size = 16
    learning_rate = 3e-4
    gradient_clip_val = 1.0

    # Model config
    n_layer = 4
    n_head = 4
    n_embd = 256
    block_size = 256
    vocab_size = 50257  # GPT-2 vocab size
    use_sdr = True
    dropout = 0.1

    # Logging
    wandb_log = False
    wandb_project = 'neuromanifold-metrics'
    wandb_run_name = 'component-metrics'
    log_component_metrics_every = 50

    # Data
    dataset = 'shakespeare_char'
    out_dir = 'out-metrics'

    # Hardware
    devices = 1
    precision = 'bf16-mixed'

    # Override from command line
    exec(open('configurator.py').read())

    # Create output directory
    os.makedirs(out_dir, exist_ok=True)

    # Create NeuroManifoldConfig
    config = NeuroManifoldConfig(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        vocab_size=vocab_size,
        use_sdr=use_sdr,
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        grad_clip=gradient_clip_val,
    )

    # Create model
    model = NeuroManifoldWithMetrics(config)

    # Create data module
    data_module = DemoDataModule(
        vocab_size=vocab_size,
        block_size=block_size,
        batch_size=batch_size,
    )

    # Setup logger
    if wandb_log:
        logger = WandbLogger(
            project=wandb_project,
            name=wandb_run_name,
            save_dir=out_dir,
        )
        print(f"Logging to WandB project: {wandb_project}")
    else:
        logger = TensorBoardLogger(
            save_dir=out_dir,
            name='lightning_logs',
        )
        print(f"Logging to TensorBoard in: {out_dir}")

    # Setup callbacks
    callbacks = [
        # Component metrics logging callback
        ComponentMetricsCallback(log_every_n_steps=log_component_metrics_every),

        # Standard checkpoint callback
        ModelCheckpoint(
            dirpath=out_dir,
            filename='neuromanifold-{epoch:02d}-{val_loss:.2f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
        ),

        # Learning rate monitor
        LearningRateMonitor(logging_interval='step'),
    ]

    # Create trainer
    trainer = L.Trainer(
        max_steps=max_steps,
        accelerator='auto',
        devices=devices,
        precision=precision,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=gradient_clip_val,
        log_every_n_steps=10,
        val_check_interval=100,
        enable_progress_bar=True,
    )

    # Print training info
    print("\n" + "="*60)
    print("TRAINING WITH COMPONENT METRICS")
    print("="*60)
    print(f"Model: NeuroManifoldGPT")
    print(f"  - Layers: {n_layer}")
    print(f"  - Heads: {n_head}")
    print(f"  - Embedding dim: {n_embd}")
    print(f"  - Block size: {block_size}")
    print(f"  - SDR enabled: {use_sdr}")
    print(f"\nTraining:")
    print(f"  - Max steps: {max_steps}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Devices: {devices}")
    print(f"  - Precision: {precision}")
    print(f"\nLogging:")
    print(f"  - Component metrics every: {log_component_metrics_every} steps")
    print(f"  - Logger: {'WandB' if wandb_log else 'TensorBoard'}")
    print(f"  - Output dir: {out_dir}")
    print("="*60 + "\n")

    # Start training
    trainer.fit(model, data_module)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Checkpoints saved to: {out_dir}")
    print("\nComponent metrics logged:")
    print("  - SDR metrics (sparsity, entropy, overlap)")
    print("  - FHN wave stability metrics")
    print("  - MTP token prediction accuracy")
    print("  - Memory utilization statistics")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
