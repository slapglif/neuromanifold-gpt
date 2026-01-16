"""
Optimized Hyperparameters Configuration Template

This file stores the best hyperparameters found by Optuna optimization.
It will be automatically updated by the OptunaTuner.save_best_config() method.
"""

import time

# Optimization Metadata
optimization_date = None  # Will be set to timestamp when optimization completes
n_trials = None  # Number of trials run
best_trial_number = None  # Which trial achieved the best result
best_val_loss = None  # Best validation loss achieved

# Optimized Hyperparameters
learning_rate = 6e-4  # Default value, will be optimized
weight_decay = 0.1  # Default value, will be optimized
warmup_iters = 100  # Default value, will be optimized

# Additional training parameters (can be set by optimization)
min_lr = None  # Minimum learning rate for decay
gradient_accumulation_steps = None
max_iters = None
lr_decay_iters = None

# Output configuration
out_dir = 'out-optimized'
wandb_log = True
wandb_project = 'neuromanifold-optuna'
wandb_run_name = 'optimized-' + str(time.time())


def get_config():
    """
    Returns the optimized hyperparameters as a dictionary.

    This function allows programmatic access to the configuration,
    making it easy to load optimized hyperparameters into training scripts.

    Returns:
        dict: Configuration dictionary with all hyperparameters and metadata
    """
    config = {
        # Metadata
        'optimization_date': optimization_date,
        'n_trials': n_trials,
        'best_trial_number': best_trial_number,
        'best_val_loss': best_val_loss,

        # Optimized hyperparameters
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'warmup_iters': warmup_iters,
        'min_lr': min_lr,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'max_iters': max_iters,
        'lr_decay_iters': lr_decay_iters,

        # Output configuration
        'out_dir': out_dir,
        'wandb_log': wandb_log,
        'wandb_project': wandb_project,
        'wandb_run_name': wandb_run_name,
    }
    return config
