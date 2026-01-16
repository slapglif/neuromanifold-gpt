"""
Command line configurator for NeuroManifoldGPT training.

This script is exec'd by train_nanogpt.py to override default config values.

Usage examples:
    python train_nanogpt.py --batch_size=32 --learning_rate=1e-4
    python train_nanogpt.py config/nano.py  # Load from config file
    python train_nanogpt.py config/nano.py --batch_size=64  # Config file + overrides
"""
import sys
from ast import literal_eval
from neuromanifold_gpt.errors import ValidationError, ConfigurationError


for arg in sys.argv[1:]:
    if '=' not in arg:
        # assume it's the name of a config file
        if arg.startswith('--'):
            raise ValidationError(
                problem="Invalid config file argument format",
                cause=f"Config file argument '{arg}' cannot start with '--'",
                recovery="Use either: python script.py config_file.py OR python script.py --key=value"
            )
        config_file = arg
        try:
            with open(config_file) as f:
                config_content = f.read()
            exec(config_content)
        except FileNotFoundError:
            raise ValidationError(
                problem=f"Config file not found: {config_file}",
                cause=f"The file '{config_file}' does not exist",
                recovery=f"Check the file path or create the config file at: {config_file}"
            )
        except Exception as e:
            raise ValidationError(
                problem=f"Failed to load config file: {config_file}",
                cause=str(e),
                recovery="Verify the config file has valid Python syntax"
            )
    else:
        # assume it's a --key=value argument
        if not arg.startswith('--'):
            raise ValidationError(
                problem="Invalid override argument format",
                cause=f"Override argument '{arg}' must start with '--'",
                recovery="Use format: --key=value (e.g., --batch_size=32)"
            )
        key, val = arg.split('=', 1)
        key = key[2:]  # strip '--'
        if key in globals():
            try:
                # attempt to eval it (e.g. if bool, number, or etc)
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # if that goes wrong, just use the string
                attempt = val
            # ensure the types match
            if type(attempt) != type(globals()[key]):
                raise ConfigurationError(
                    problem="Configuration type mismatch",
                    cause=f"Cannot override '{key}': expected {type(globals()[key]).__name__}, got {type(attempt).__name__}",
                    recovery=f"Provide a value of type {type(globals()[key]).__name__} (current value: {globals()[key]})"
                )
            globals()[key] = attempt
        else:
            raise ValidationError(
                problem=f"Unknown config key: {key}",
                cause=f"The configuration parameter '{key}' is not defined",
                recovery="Check available config parameters in the config file or use --help"
            )
