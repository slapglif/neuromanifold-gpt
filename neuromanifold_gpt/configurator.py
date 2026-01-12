"""
Command line configurator for NeuroManifoldGPT training.

This script is exec'd by train_nanogpt.py to override default config values.

Usage examples:
    python train_nanogpt.py --batch_size=32 --learning_rate=1e-4
    python train_nanogpt.py config/nano.py  # Load from config file
    python train_nanogpt.py config/nano.py --batch_size=64  # Config file + overrides
"""
import sys


for arg in sys.argv[1:]:
    if "=" in arg:
        # Parse --key=value or key=value
        key, val = arg.lstrip("-").split("=", 1)
        if key in globals():
            # Try to parse as same type as default
            try:
                val_type = type(globals()[key])
                if val_type == bool:
                    # Handle boolean strings
                    globals()[key] = val.lower() in ("true", "1", "yes")
                else:
                    globals()[key] = val_type(val)
            except ValueError:
                globals()[key] = val
    elif arg.endswith(".py"):
        # Load config file
        exec(open(arg).read())
