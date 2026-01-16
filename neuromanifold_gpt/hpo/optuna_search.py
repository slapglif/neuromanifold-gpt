"""Optuna hyperparameter optimization wrapper with Lightning integration.

This module provides the OptunaHPO class that orchestrates automated
hyperparameter search using Optuna. It integrates with PyTorch Lightning
to train models with different hyperparameter configurations and optimize
for validation loss.

Example usage:
    import yaml
    from neuromanifold_gpt.hpo.optuna_search import OptunaHPO

    # Load configuration
    with open("hpo_config.yaml") as f:
        config = yaml.safe_load(f)

    # Run HPO
    hpo = OptunaHPO(config)
    study = hpo.optimize()

    # Get best trial
    print(f"Best trial: {study.best_trial.params}")
    print(f"Best val_loss: {study.best_value}")
"""

import os
import tempfile
from typing import Any, Dict, Optional

import optuna
import pytorch_lightning as pl
from loguru import logger

from neuromanifold_gpt.hpo.search_space import SearchSpace
from neuromanifold_gpt.training.config import TrainConfig
from neuromanifold_gpt.training.trainer import train


class OptunaHPO:
    """Optuna-based hyperparameter optimization for NeuroManifoldGPT.

    This class wraps Optuna study creation and execution, integrating with
    PyTorch Lightning training. It handles:
    - Study creation with configurable sampler and direction
    - Objective function that trains models with suggested hyperparameters
    - Trial execution and metric reporting
    - Best configuration export

    The HPO process:
    1. Creates an Optuna study with specified sampler (TPE, Random, etc.)
    2. For each trial:
       - Suggests hyperparameters from the search space
       - Creates a TrainConfig with suggested parameters + fixed parameters
       - Trains the model using PyTorch Lightning
       - Returns validation loss to Optuna
    3. Optuna optimizes the search based on trial results

    Attributes:
        config: Full HPO configuration dictionary
        search_space: SearchSpace instance for parameter suggestions
        study_config: Study-specific configuration (name, direction, etc.)
        study: Optuna study instance (created during optimization)

    Args:
        config: Configuration dictionary with keys:
            - search_space: Dictionary of parameters to optimize
            - fixed_params: Dictionary of fixed parameters
            - study: Study configuration (name, direction, n_trials, etc.)
            - pruning (optional): Pruning configuration
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize OptunaHPO with configuration.

        Args:
            config: HPO configuration dictionary loaded from YAML

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self.search_space = SearchSpace(config)
        self.study_config = config.get("study", {})
        self.study: Optional[optuna.Study] = None

        self._validate_study_config()

    def _validate_study_config(self) -> None:
        """Validate study configuration.

        Raises:
            ValueError: If study configuration is invalid
        """
        if "direction" not in self.study_config:
            raise ValueError("study.direction must be specified (minimize or maximize)")

        if self.study_config["direction"] not in ["minimize", "maximize"]:
            raise ValueError(
                f"Invalid direction: {self.study_config['direction']}. "
                f"Must be 'minimize' or 'maximize'"
            )

    def _create_study(self) -> optuna.Study:
        """Create an Optuna study with configured sampler and settings.

        Returns:
            Optuna Study instance
        """
        # Get study configuration
        study_name = self.study_config.get("name", "neuromanifold-hpo")
        direction = self.study_config["direction"]
        storage = self.study_config.get("storage", None)
        sampler_name = self.study_config.get("sampler", "tpe")

        # Create sampler
        if sampler_name == "tpe":
            sampler = optuna.samplers.TPESampler(seed=1337)
        elif sampler_name == "random":
            sampler = optuna.samplers.RandomSampler(seed=1337)
        elif sampler_name == "grid":
            sampler = optuna.samplers.GridSampler()
        elif sampler_name == "cmaes":
            sampler = optuna.samplers.CmaEsSampler(seed=1337)
        else:
            logger.warning(
                f"Unknown sampler '{sampler_name}', defaulting to TPE"
            )
            sampler = optuna.samplers.TPESampler(seed=1337)

        # Create study
        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=sampler,
            storage=storage,
            load_if_exists=storage is not None,
        )

        logger.info(f"Created Optuna study: {study_name}")
        logger.info(f"Direction: {direction}")
        logger.info(f"Sampler: {sampler_name}")
        if storage:
            logger.info(f"Storage: {storage}")

        return study

    def _objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization.

        This function is called for each trial. It:
        1. Suggests hyperparameters using the SearchSpace
        2. Creates a TrainConfig with suggested + fixed parameters
        3. Trains the model using PyTorch Lightning
        4. Returns the validation loss

        Args:
            trial: Optuna trial instance

        Returns:
            Validation loss (float) to minimize/maximize
        """
        # Get suggested parameters
        params = self.search_space.suggest_params(trial)

        logger.info(f"\n{'='*60}")
        logger.info(f"Trial {trial.number} - Testing parameters:")
        for k, v in params.items():
            if k not in self.search_space.get_fixed_param_names():
                logger.info(f"  {k}: {v}")
        logger.info(f"{'='*60}\n")

        # Create TrainConfig from parameters
        train_config = self._create_train_config(params, trial.number)

        # Train model and get validation loss
        try:
            val_loss = self._train_trial(train_config, trial)
            logger.info(f"Trial {trial.number} completed with val_loss: {val_loss:.4f}")
            return val_loss

        except Exception as e:
            logger.error(f"Trial {trial.number} failed with error: {e}")
            # Return a large penalty value for failed trials
            return float('inf') if self.study_config["direction"] == "minimize" else float('-inf')

    def _create_train_config(self, params: Dict[str, Any], trial_number: int) -> TrainConfig:
        """Create a TrainConfig from suggested and fixed parameters.

        Args:
            params: Dictionary of all parameters (suggested + fixed)
            trial_number: Current trial number for unique output directory

        Returns:
            TrainConfig instance
        """
        # Create a unique output directory for this trial
        out_dir = os.path.join(
            params.get("out_dir", "out-hpo"),
            f"trial_{trial_number:04d}"
        )

        # Create TrainConfig with all parameters
        # Start with defaults from TrainConfig
        config_dict = {}

        # Update with parameters from HPO config
        for key, value in params.items():
            # Convert string booleans to actual booleans if needed
            if isinstance(value, str) and value.lower() in ["true", "false"]:
                value = value.lower() == "true"
            config_dict[key] = value

        # Override output directory
        config_dict["out_dir"] = out_dir

        # Ensure checkpoints are disabled during HPO (saves disk space)
        config_dict["save_checkpoints"] = False
        config_dict["save_separate_optimizer"] = False

        # Disable W&B logging during trials (optional, can be enabled)
        if "wandb_log" not in config_dict:
            config_dict["wandb_log"] = False

        # Create TrainConfig instance
        train_config = TrainConfig(**config_dict)

        return train_config

    def _train_trial(self, train_config: TrainConfig, trial: optuna.Trial) -> float:
        """Train a model with the given configuration and return validation loss.

        Args:
            train_config: Training configuration for this trial
            trial: Optuna trial instance

        Returns:
            Final validation loss

        Raises:
            RuntimeError: If training fails
        """
        # Set random seed for reproducibility
        pl.seed_everything(1337 + trial.number)

        # Import here to avoid circular dependency
        from neuromanifold_gpt.training.data_modules import (
            StreamingDataModule,
            TextDataModule,
        )
        from neuromanifold_gpt.training.lightning_module import NeuroManifoldLitModule
        from neuromanifold_gpt.config.base import NeuroManifoldConfig
        from model import GPTConfig, GPT
        import torch
        from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping

        # Setup data module
        if train_config.streaming:
            data_module = StreamingDataModule(
                dataset_name=train_config.dataset,
                block_size=train_config.block_size,
                batch_size=train_config.batch_size,
                num_workers=train_config.num_workers,
            )
        else:
            data_dir = os.path.join("data", train_config.dataset)
            data_module = TextDataModule(
                data_dir=data_dir,
                block_size=train_config.block_size,
                batch_size=train_config.batch_size,
                num_workers=train_config.num_workers,
            )

        data_module.setup()

        # Override vocab_size if specified
        if train_config.vocab_size > 0:
            data_module.vocab_size = train_config.vocab_size

        # Build model config
        if train_config.model_type == "neuromanifold":
            model_config = NeuroManifoldConfig(
                vocab_size=data_module.vocab_size,
                block_size=train_config.block_size,
                n_layer=train_config.n_layer,
                n_heads=train_config.n_head,
                n_embd=train_config.n_embd,
                dropout=train_config.dropout,
                bias=train_config.bias,
                # SDR
                use_sdr=train_config.use_sdr,
                sdr_size=train_config.sdr_size,
                # Manifold
                manifold_dim=train_config.manifold_dim,
                n_eigenvectors=train_config.n_eigenvectors,
                # KAN
                use_kan=train_config.use_kan,
                kan_type=train_config.kan_type,
                kan_wavelet=train_config.kan_wavelet,
                use_fast_wavekan=train_config.use_fast_wavekan,
                kan_num_centers=train_config.kan_num_centers,
                # FHN
                fhn_threshold=train_config.fhn_threshold,
                fhn_tau=train_config.fhn_tau,
                n_fhn_steps=train_config.n_fhn_steps,
                use_fhn_imex=train_config.use_fhn_imex,
                use_fhn_partitioning=train_config.use_fhn_partitioning,
                use_fhn_fused=train_config.use_fhn_fused,
                # mHC
                use_mhc=train_config.use_mhc,
                use_full_mhc=train_config.use_full_mhc,
                mhc_n_streams=train_config.mhc_n_streams,
                # Attention
                use_kaufmann_attention=train_config.use_kaufmann_attention,
                # Speed optimization
                skip_manifold_spectral=train_config.skip_manifold_spectral,
                # Training
                learning_rate=train_config.learning_rate,
                weight_decay=train_config.weight_decay,
                beta1=train_config.beta1,
                beta2=train_config.beta2,
                grad_clip=train_config.grad_clip,
            )
        else:
            model_config = GPTConfig(
                vocab_size=data_module.vocab_size,
                block_size=train_config.block_size,
                n_layer=train_config.n_layer,
                n_head=train_config.n_head,
                n_embd=train_config.n_embd,
                dropout=train_config.dropout,
                bias=train_config.bias,
            )

        # Build Lightning module
        lit_module = NeuroManifoldLitModule(
            model_config=model_config,
            train_config=train_config,
            itos=data_module.itos,
        )

        # Compile model if requested
        if train_config.compile_model:
            logger.info("Compiling model with torch.compile()...")
            lit_module.model = torch.compile(lit_module.model)

        # Setup callbacks
        callbacks = [
            LearningRateMonitor(logging_interval="step"),
        ]

        # Add early stopping if configured
        if train_config.early_stopping_patience > 0:
            callbacks.append(
                EarlyStopping(
                    monitor="val/loss",
                    patience=train_config.early_stopping_patience,
                    mode="min",
                    verbose=False,
                )
            )

        # Create Lightning Trainer
        trainer = pl.Trainer(
            max_steps=train_config.max_iters,
            accelerator="auto",
            devices=train_config.devices,
            precision=train_config.precision,
            gradient_clip_val=train_config.grad_clip,
            accumulate_grad_batches=train_config.gradient_accumulation_steps,
            callbacks=callbacks,
            logger=False,  # Disable logging during HPO
            default_root_dir=train_config.out_dir,
            enable_progress_bar=False,  # Disable progress bar during HPO
            log_every_n_steps=train_config.log_interval,
            val_check_interval=train_config.eval_interval,
            limit_val_batches=train_config.eval_iters,
            enable_checkpointing=False,  # Disable checkpointing during HPO
        )

        # Train
        trainer.fit(lit_module, data_module)

        # Get validation loss from trainer
        # The last logged val/loss is stored in the trainer's callback metrics
        val_loss = trainer.callback_metrics.get("val/loss", float('inf'))

        # Convert to Python float
        if hasattr(val_loss, "item"):
            val_loss = val_loss.item()

        return val_loss

    def optimize(self, n_trials: Optional[int] = None) -> optuna.Study:
        """Run hyperparameter optimization.

        Args:
            n_trials: Number of trials to run (overrides config if provided)

        Returns:
            Completed Optuna Study instance
        """
        # Create study
        self.study = self._create_study()

        # Get number of trials
        if n_trials is None:
            n_trials = self.study_config.get("n_trials", 50)

        logger.info(f"Starting optimization with {n_trials} trials")
        logger.info(f"Search space: {len(self.search_space.get_param_names())} parameters")
        logger.info(f"Fixed parameters: {len(self.search_space.get_fixed_param_names())}")

        # Run optimization
        self.study.optimize(
            self._objective,
            n_trials=n_trials,
            show_progress_bar=True,
        )

        # Log best results
        logger.info(f"\n{'='*60}")
        logger.info("Optimization completed!")
        logger.info(f"Best trial: {self.study.best_trial.number}")
        logger.info(f"Best value: {self.study.best_value:.4f}")
        logger.info("Best parameters:")
        for k, v in self.study.best_trial.params.items():
            logger.info(f"  {k}: {v}")
        logger.info(f"{'='*60}\n")

        return self.study

    def export_best_config(self, output_path: str) -> None:
        """Export the best trial configuration to a Python config file.

        Args:
            output_path: Path to save the configuration file

        Raises:
            RuntimeError: If optimization has not been run yet
        """
        if self.study is None:
            raise RuntimeError("No study available. Run optimize() first.")

        # Get best parameters
        best_params = self.study.best_trial.params.copy()

        # Merge with fixed parameters
        best_params.update(self.search_space.fixed_params)

        # Create config file content
        lines = [
            "# Best hyperparameter configuration from Optuna HPO",
            f"# Trial: {self.study.best_trial.number}",
            f"# Validation loss: {self.study.best_value:.4f}",
            "",
        ]

        # Add all parameters
        for key, value in sorted(best_params.items()):
            if isinstance(value, str):
                lines.append(f'{key} = "{value}"')
            else:
                lines.append(f"{key} = {value}")

        # Write to file
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        logger.info(f"Best configuration exported to: {output_path}")

    def get_study_summary(self) -> Dict[str, Any]:
        """Get a summary of the optimization study.

        Returns:
            Dictionary with study statistics

        Raises:
            RuntimeError: If optimization has not been run yet
        """
        if self.study is None:
            raise RuntimeError("No study available. Run optimize() first.")

        # Get completed trials
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]
        failed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL]

        summary = {
            "n_trials": len(self.study.trials),
            "n_completed": len(completed_trials),
            "n_pruned": len(pruned_trials),
            "n_failed": len(failed_trials),
            "best_trial": self.study.best_trial.number,
            "best_value": self.study.best_value,
            "best_params": self.study.best_trial.params,
        }

        return summary
