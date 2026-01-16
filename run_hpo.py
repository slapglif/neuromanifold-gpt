"""
Automated Hyperparameter Optimization for NeuroManifoldGPT.

Uses Optuna to search over hyperparameters defined in a YAML configuration file.
Integrates with PyTorch Lightning training to evaluate different configurations
and optimize for validation loss.

Supports:
- Configurable search spaces via YAML
- Multiple Optuna samplers (TPE, Random, Grid, CmaEs)
- Early stopping and pruning of unpromising trials
- Best configuration export
- Study persistence via SQLite storage

Usage:
    # Run HPO with example config
    python run_hpo.py --config hpo_config_example.yaml

    # Run with custom number of trials
    python run_hpo.py --config hpo_config_example.yaml --n-trials 100

    # Resume from existing study
    python run_hpo.py --config hpo_config_example.yaml --resume

Example YAML config structure:
    search_space:
      learning_rate:
        type: float
        low: 1e-5
        high: 1e-2
        log: true
      n_layer:
        type: int
        low: 2
        high: 8

    fixed_params:
      dataset: shakespeare_char
      max_iters: 1000

    study:
      name: neuromanifold-hpo
      direction: minimize
      n_trials: 50
      sampler: tpe
"""

import sys
import os

# Allow --help to work without dependency validation
if '--help' not in sys.argv and '-h' not in sys.argv:
    # Validate dependencies before heavy imports to fail fast with clear messages
    import os as _os
    _validation_path = _os.path.join(_os.path.dirname(__file__), 'neuromanifold_gpt', 'config')
    sys.path.insert(0, _validation_path)
    try:
        import validation as _validation_module
        _validation_module.validate_dependencies(verbose=True)
    finally:
        sys.path.pop(0)  # Clean up sys.path

import argparse
from typing import Optional
import yaml

# Import logger and HPO only after validation (or for help)
if '--help' not in sys.argv and '-h' not in sys.argv:
    from loguru import logger
    from neuromanifold_gpt.hpo.optuna_search import OptunaHPO


def load_config(config_path: str) -> dict:
    """Load HPO configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary containing HPO configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is malformed
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    logger.info(f"Loading HPO configuration from: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate config structure
    if 'search_space' not in config:
        raise ValueError("Config must contain 'search_space' section")
    if 'study' not in config:
        raise ValueError("Config must contain 'study' section")

    logger.info(f"Loaded configuration with {len(config.get('search_space', {}))} search parameters")
    logger.info(f"Fixed parameters: {len(config.get('fixed_params', {}))}")

    return config


def main():
    """Main entry point for HPO."""
    parser = argparse.ArgumentParser(
        description="Automated Hyperparameter Optimization for NeuroManifoldGPT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  python run_hpo.py --config hpo_config_example.yaml

  # Run with custom number of trials
  python run_hpo.py --config hpo_config_example.yaml --n-trials 100

  # Resume from existing study (requires storage URL in config)
  python run_hpo.py --config hpo_config_example.yaml --resume

  # Export best config to specific location
  python run_hpo.py --config hpo_config_example.yaml --output-config config/hpo_best.py
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to HPO configuration YAML file"
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Number of trials to run (overrides config file)"
    )

    parser.add_argument(
        "--output-config",
        type=str,
        default="config/hpo_best.py",
        help="Path to save best configuration (default: config/hpo_best.py)"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing study (requires storage URL in config)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    # Print header
    logger.info("=" * 70)
    logger.info("NeuroManifoldGPT - Automated Hyperparameter Optimization")
    logger.info("=" * 70)

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Warn if resuming without storage
    if args.resume and not config.get('study', {}).get('storage'):
        logger.warning(
            "--resume flag set but no storage URL in config. "
            "Study will not be resumed."
        )

    # Create HPO instance
    try:
        hpo = OptunaHPO(config)
    except Exception as e:
        logger.error(f"Failed to create HPO instance: {e}")
        sys.exit(1)

    # Print search space summary
    logger.info("\n" + str(hpo.search_space) + "\n")

    # Run optimization
    try:
        logger.info("Starting hyperparameter optimization...")
        study = hpo.optimize(n_trials=args.n_trials)

    except KeyboardInterrupt:
        logger.warning("\nOptimization interrupted by user")
        if hpo.study is None:
            logger.error("No trials completed. Exiting.")
            sys.exit(1)
        study = hpo.study

    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("Optimization Summary")
    logger.info("=" * 70)

    try:
        summary = hpo.get_study_summary()
        logger.info(f"Total trials: {summary['n_trials']}")
        logger.info(f"Completed: {summary['n_completed']}")
        logger.info(f"Pruned: {summary['n_pruned']}")
        logger.info(f"Failed: {summary['n_failed']}")
        logger.info(f"\nBest trial: #{summary['best_trial']}")
        logger.info(f"Best validation loss: {summary['best_value']:.4f}")
        logger.info("\nBest parameters:")
        for param, value in summary['best_params'].items():
            logger.info(f"  {param}: {value}")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Failed to generate summary: {e}")

    # Export best configuration
    try:
        logger.info(f"\nExporting best configuration to: {args.output_config}")
        hpo.export_best_config(args.output_config)
        logger.info("Best configuration exported successfully")

    except Exception as e:
        logger.error(f"Failed to export configuration: {e}")
        sys.exit(1)

    logger.info("\nHyperparameter optimization complete!")
    logger.info(f"Use the best config with: python train.py --config {args.output_config}")


if __name__ == "__main__":
    main()
