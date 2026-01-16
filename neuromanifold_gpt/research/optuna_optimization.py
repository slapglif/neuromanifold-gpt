#!/usr/bin/env python3
"""Optuna hyperparameter optimization for NeuroManifoldGPT.

Provides automated hyperparameter tuning with:
- Configurable search space (learning rate, weight decay, warmup)
- Early stopping via Optuna pruning
- Integration with PyTorch Lightning
- WandB logging support
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from loguru import logger

# Configure loguru
logger.remove()
logger.add(sys.stderr, format="{time:HH:mm:ss} | {level} | {message}")

from neuromanifold_gpt.config.base import NeuroManifoldConfig
from neuromanifold_gpt.training.config import TrainConfig
from neuromanifold_gpt.training.data_modules import StreamingDataModule, TextDataModule
from neuromanifold_gpt.training.lightning_module import NeuroManifoldLitModule
from neuromanifold_gpt.training.callbacks import MFUCallback


@dataclass
class OptunaConfig:
    """Configuration for Optuna hyperparameter optimization.

    Attributes:
        n_trials: Number of optimization trials to run
        study_name: Name for the Optuna study
        storage: Optuna storage backend (None for in-memory)
        pruner: Optuna pruner type ("median", "hyperband", or None)
        sampler: Optuna sampler type ("tpe", "random", or "cmaes")
        n_startup_trials: Number of random trials before using sampler
        timeout: Maximum time for optimization in seconds (None = no limit)
        load_if_exists: Load existing study if it exists
        direction: Optimization direction ("minimize" or "maximize")
    """
    n_trials: int = 20
    study_name: str = "neuromanifold_optuna"
    storage: Optional[str] = None
    pruner: str = "median"
    sampler: str = "tpe"
    n_startup_trials: int = 10
    timeout: Optional[int] = None
    load_if_exists: bool = True
    direction: str = "minimize"


class OptunaTuner:
    """Optuna hyperparameter tuner for NeuroManifoldGPT.

    Integrates Optuna with PyTorch Lightning training to optimize:
    - Learning rate (log-uniform search)
    - Weight decay (uniform search)
    - Warmup iterations (integer search)

    Features:
    - Early stopping via Optuna pruning callbacks
    - WandB integration for trial tracking
    - Validation loss as optimization objective
    - Configurable search space

    Args:
        base_config: Base TrainConfig to use as template
        optuna_config: OptunaConfig with optimization settings
        wandb_log: Enable WandB logging for all trials
        wandb_project: WandB project name
    """

    def __init__(
        self,
        base_config: TrainConfig,
        optuna_config: OptunaConfig,
        wandb_log: bool = True,
        wandb_project: str = "neuromanifold-optuna",
    ):
        self.base_config = base_config
        self.optuna_config = optuna_config
        self.wandb_log = wandb_log
        self.wandb_project = wandb_project
        self.best_params = None
        self.best_value = None

    def _create_study(self) -> optuna.Study:
        """Create or load Optuna study."""
        # Configure pruner
        pruner = None
        if self.optuna_config.pruner == "median":
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=self.optuna_config.n_startup_trials,
                n_warmup_steps=10,
            )
        elif self.optuna_config.pruner == "hyperband":
            pruner = optuna.pruners.HyperbandPruner(
                min_resource=10,
                max_resource=self.base_config.max_iters,
            )

        # Configure sampler
        sampler = None
        if self.optuna_config.sampler == "tpe":
            sampler = optuna.samplers.TPESampler(
                n_startup_trials=self.optuna_config.n_startup_trials
            )
        elif self.optuna_config.sampler == "random":
            sampler = optuna.samplers.RandomSampler()
        elif self.optuna_config.sampler == "cmaes":
            sampler = optuna.samplers.CmaEsSampler(
                n_startup_trials=self.optuna_config.n_startup_trials
            )

        study = optuna.create_study(
            study_name=self.optuna_config.study_name,
            storage=self.optuna_config.storage,
            direction=self.optuna_config.direction,
            pruner=pruner,
            sampler=sampler,
            load_if_exists=self.optuna_config.load_if_exists,
        )

        return study

    def _suggest_hyperparameters(self, trial: optuna.Trial) -> dict:
        """Suggest hyperparameters for a trial.

        Search space:
        - learning_rate: log-uniform [1e-5, 1e-3]
        - weight_decay: uniform [0.0, 0.5]
        - warmup_iters: integer [50, 500]

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of suggested hyperparameters
        """
        hyperparams = {
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-5, 1e-3, log=True
            ),
            "weight_decay": trial.suggest_float(
                "weight_decay", 0.0, 0.5
            ),
            "warmup_iters": trial.suggest_int(
                "warmup_iters", 50, 500
            ),
        }

        return hyperparams

    def _create_config(self, trial: optuna.Trial) -> TrainConfig:
        """Create TrainConfig with trial hyperparameters.

        Args:
            trial: Optuna trial object

        Returns:
            TrainConfig with suggested hyperparameters
        """
        # Get suggested hyperparameters
        hyperparams = self._suggest_hyperparameters(trial)

        # Create config copy with trial-specific values
        from dataclasses import replace
        config = replace(
            self.base_config,
            learning_rate=hyperparams["learning_rate"],
            weight_decay=hyperparams["weight_decay"],
            warmup_iters=hyperparams["warmup_iters"],
            out_dir=os.path.join(
                self.base_config.out_dir, f"trial_{trial.number}"
            ),
            wandb_log=self.wandb_log,
            wandb_project=self.wandb_project,
            wandb_run_name=f"trial_{trial.number}",
        )

        return config

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function.

        Trains model with trial hyperparameters and returns validation loss.

        Args:
            trial: Optuna trial object

        Returns:
            Final validation loss
        """
        # Create trial config
        config = self._create_config(trial)

        logger.info(f"\n=== Trial {trial.number} ===")
        logger.info(f"learning_rate: {config.learning_rate:.6f}")
        logger.info(f"weight_decay: {config.weight_decay:.4f}")
        logger.info(f"warmup_iters: {config.warmup_iters}")

        # Set random seed for reproducibility
        pl.seed_everything(1337 + trial.number)

        # Setup data
        if config.streaming:
            data_module = StreamingDataModule(
                dataset_name=config.dataset,
                block_size=config.block_size,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
            )
        else:
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
            # Training (from trial config)
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            beta1=config.beta1,
            beta2=config.beta2,
            grad_clip=config.grad_clip,
        )

        # Build Lightning module
        lit_module = NeuroManifoldLitModule(
            model_config=model_config,
            train_config=config,
            itos=data_module.itos,
        )

        # Callbacks
        callbacks = [
            LearningRateMonitor(logging_interval="step"),
            MFUCallback(log_interval=config.log_interval),
            PyTorchLightningPruningCallback(trial, monitor="val/loss"),
        ]

        # Optional checkpointing (disabled by default for trials)
        if config.save_checkpoints:
            callbacks.append(
                ModelCheckpoint(
                    dirpath=config.out_dir,
                    filename="ckpt-{step:06d}-{val/loss:.4f}",
                    monitor="val/loss",
                    mode="min",
                    save_top_k=1,
                )
            )

        # Logger
        pl_logger = None
        if self.wandb_log:
            pl_logger = WandbLogger(
                project=self.wandb_project,
                name=config.wandb_run_name,
                save_dir=config.out_dir,
                tags=["optuna", f"trial_{trial.number}"],
            )
            # Log trial hyperparameters to WandB
            pl_logger.experiment.config.update(
                {
                    "trial_number": trial.number,
                    "learning_rate": config.learning_rate,
                    "weight_decay": config.weight_decay,
                    "warmup_iters": config.warmup_iters,
                }
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
            enable_progress_bar=False,  # Disable for cleaner Optuna output
            log_every_n_steps=config.log_interval,
            val_check_interval=config.eval_interval,
            limit_val_batches=config.eval_iters,
            enable_checkpointing=config.save_checkpoints,
        )

        # Train
        try:
            trainer.fit(lit_module, data_module)

            # Get final validation loss
            val_loss = trainer.callback_metrics.get("val/loss")
            if val_loss is None:
                logger.warning("No validation loss found, using large penalty")
                return float("inf")

            final_val_loss = float(val_loss)
            logger.info(f"Trial {trial.number} final val/loss: {final_val_loss:.4f}")

            # Log final validation loss to WandB
            if self.wandb_log and pl_logger is not None:
                pl_logger.experiment.log({
                    "trial/final_val_loss": final_val_loss,
                    "trial/number": trial.number,
                })

            return final_val_loss

        except optuna.TrialPruned:
            logger.info(f"Trial {trial.number} was pruned")
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return float("inf")
        finally:
            # Cleanup
            del lit_module
            del data_module
            torch.cuda.empty_cache()

    def optimize(self) -> optuna.Study:
        """Run hyperparameter optimization.

        Returns:
            Completed Optuna study with optimization results
        """
        logger.info("=== Starting Optuna Optimization ===")
        logger.info(f"Number of trials: {self.optuna_config.n_trials}")
        logger.info(f"Study name: {self.optuna_config.study_name}")
        logger.info(f"Direction: {self.optuna_config.direction}")

        # Create study
        study = self._create_study()

        # Run optimization
        study.optimize(
            self._objective,
            n_trials=self.optuna_config.n_trials,
            timeout=self.optuna_config.timeout,
            show_progress_bar=True,
        )

        # Store best results
        self.best_params = study.best_params
        self.best_value = study.best_value

        # Log results
        logger.info("\n=== Optimization Complete ===")
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best validation loss: {self.best_value:.4f}")
        logger.info("Best hyperparameters:")
        for key, value in self.best_params.items():
            logger.info(f"  {key}: {value}")

        # Print pruned trials
        pruned_trials = study.get_trials(
            deepcopy=False, states=[optuna.trial.TrialState.PRUNED]
        )
        complete_trials = study.get_trials(
            deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]
        )
        logger.info(f"\nStatistics:")
        logger.info(f"  Complete trials: {len(complete_trials)}")
        logger.info(f"  Pruned trials: {len(pruned_trials)}")
        logger.info(f"  Failed trials: {len(study.trials) - len(complete_trials) - len(pruned_trials)}")

        # Log optimization summary to WandB
        if self.wandb_log:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    "optuna/best_trial_number": study.best_trial.number,
                    "optuna/best_val_loss": self.best_value,
                    "optuna/n_complete_trials": len(complete_trials),
                    "optuna/n_pruned_trials": len(pruned_trials),
                    "optuna/n_failed_trials": len(study.trials) - len(complete_trials) - len(pruned_trials),
                })
                # Log best hyperparameters
                for key, value in self.best_params.items():
                    wandb.log({f"optuna/best_{key}": value})

        return study

    def save_best_config(self, output_path: str = "config/optimized_hyperparams.py") -> None:
        """Save best hyperparameters to config file.

        Args:
            output_path: Path to save config file
        """
        if self.best_params is None:
            logger.warning("No optimization results to save. Run optimize() first.")
            return

        logger.info(f"Saving best hyperparameters to {output_path}")

        # Create config directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Generate config file content
        config_content = f'''"""Optimized hyperparameters from Optuna study.

Auto-generated by OptunaTuner on {import_datetime_now()}.
Study: {self.optuna_config.study_name}
Best validation loss: {self.best_value:.4f}
"""

from neuromanifold_gpt.training.config import TrainConfig

def get_config() -> TrainConfig:
    """Get optimized training configuration."""
    config = TrainConfig(
        # Optimized hyperparameters
        learning_rate={self.best_params["learning_rate"]:.6f},
        weight_decay={self.best_params["weight_decay"]:.4f},
        warmup_iters={self.best_params["warmup_iters"]},

        # Base configuration (modify as needed)
        dataset="{self.base_config.dataset}",
        batch_size={self.base_config.batch_size},
        block_size={self.base_config.block_size},
        n_layer={self.base_config.n_layer},
        n_head={self.base_config.n_head},
        n_embd={self.base_config.n_embd},
        max_iters={self.base_config.max_iters},
    )
    return config
'''

        with open(output_path, "w") as f:
            f.write(config_content)

        logger.info(f"Config saved successfully to {output_path}")


def import_datetime_now():
    """Helper to get current datetime string."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter optimization for NeuroManifoldGPT"
    )
    parser.add_argument(
        "--n-trials", type=int, default=20,
        help="Number of optimization trials"
    )
    parser.add_argument(
        "--dataset", type=str, default="shakespeare_char",
        help="Dataset name"
    )
    parser.add_argument(
        "--max-iters", type=int, default=1000,
        help="Maximum training iterations per trial"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Training batch size"
    )
    parser.add_argument(
        "--study-name", type=str, default="neuromanifold_optuna",
        help="Optuna study name"
    )
    parser.add_argument(
        "--wandb-log", action="store_true",
        help="Enable WandB logging"
    )
    parser.add_argument(
        "--wandb-project", type=str, default="neuromanifold-optuna",
        help="WandB project name"
    )
    parser.add_argument(
        "--output-config", type=str, default="config/optimized_hyperparams.py",
        help="Path to save best config"
    )

    args = parser.parse_args()

    # Create base config
    base_config = TrainConfig(
        dataset=args.dataset,
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        out_dir="out-optuna",
        save_checkpoints=False,  # Don't save checkpoints for trials
        eval_interval=200,
        log_interval=50,
    )

    # Create Optuna config
    optuna_config = OptunaConfig(
        n_trials=args.n_trials,
        study_name=args.study_name,
    )

    # Create tuner
    tuner = OptunaTuner(
        base_config=base_config,
        optuna_config=optuna_config,
        wandb_log=args.wandb_log,
        wandb_project=args.wandb_project,
    )

    # Run optimization
    study = tuner.optimize()

    # Save best config
    tuner.save_best_config(args.output_config)


if __name__ == "__main__":
    main()
