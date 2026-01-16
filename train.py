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

# Import all training components from the neuromanifold_gpt.training package
from neuromanifold_gpt.training import (
    TrainConfig,
    train,
)


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
