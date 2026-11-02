"""Backward compatibility shim for configurator.py pattern.

DEPRECATED: This module provides backward compatibility for scripts that still use the old
exec(open('configurator.py').read()) pattern. New code should use the type-safe load_config()
from neuromanifold_gpt.config.loader instead.

This module emits deprecation warnings to guide users toward the new type-safe configuration
system while maintaining compatibility during the migration period.

Example usage (DEPRECATED):
    from neuromanifold_gpt.config.compat import apply_config_overrides

    # Old pattern (DO NOT USE IN NEW CODE)
    config = {}
    exec(open('configurator.py').read())  # Sets config variables
    config = apply_config_overrides(config, sys.argv[1:])

Recommended alternative:
    from neuromanifold_gpt.config.loader import load_config
    from neuromanifold_gpt.config.training import TrainingConfig

    # New pattern (TYPE-SAFE, RECOMMENDED)
    config = load_config(TrainingConfig, sys.argv[1:])
"""

import sys
import warnings
from ast import literal_eval
from typing import Dict, Any, List, Optional
from neuromanifold_gpt.errors import ValidationError


def apply_config_overrides(
    config_dict: Dict[str, Any],
    args: Optional[List[str]] = None,
    show_deprecation_warning: bool = True
) -> Dict[str, Any]:
    """Apply CLI overrides to a configuration dictionary (DEPRECATED).

    DEPRECATED: This function provides backward compatibility for the old configurator.py
    pattern. It is maintained only for migration purposes and will be removed in a future
    release. Please migrate to the type-safe load_config() function.

    This function takes a dictionary of configuration values (typically from exec()ing
    a configurator.py file) and applies CLI overrides in the format --key=value.

    Args:
        config_dict: Dictionary of configuration values to override
        args: Command-line arguments to parse (defaults to sys.argv[1:])
        show_deprecation_warning: Whether to emit deprecation warning (default: True)

    Returns:
        Updated configuration dictionary with CLI overrides applied

    Raises:
        ValidationError: If arguments are malformed or reference unknown config keys

    Example (DEPRECATED):
        >>> config = {'batch_size': 16, 'learning_rate': 1e-3}
        >>> config = apply_config_overrides(config, ['--batch_size=32'])
        >>> print(config['batch_size'])
        32

    Migration Guide:
        OLD PATTERN:
            config = {}
            exec(open('configurator.py').read())
            config = apply_config_overrides(config, sys.argv[1:])

        NEW PATTERN:
            from neuromanifold_gpt.config.loader import load_config
            from neuromanifold_gpt.config.training import TrainingConfig
            config = load_config(TrainingConfig, sys.argv[1:])
    """
    if show_deprecation_warning:
        warnings.warn(
            "apply_config_overrides() is deprecated and will be removed in a future release. "
            "Please migrate to the type-safe load_config() function from neuromanifold_gpt.config.loader. "
            "See docs/config-migration-guide.md for migration instructions.",
            DeprecationWarning,
            stacklevel=2
        )

    if args is None:
        args = sys.argv[1:]

    # Parse arguments - only process --key=value overrides
    overrides: Dict[str, Any] = {}

    for arg in args:
        if not arg.startswith('--'):
            # Skip non-override arguments (config file names, etc.)
            continue

        if '=' not in arg:
            raise ValidationError(
                problem="Invalid override argument format",
                cause=f"Override argument '{arg}' must be in --key=value format",
                recovery="Use format: --key=value (e.g., --batch_size=32)"
            )

        key, val_str = arg.split('=', 1)
        key = key[2:]  # Remove '--' prefix

        if key not in config_dict:
            # Get available keys for helpful error message
            available = ', '.join(sorted(config_dict.keys()))
            raise ValidationError(
                problem=f"Unknown config key: {key}",
                cause=f"'{key}' is not a valid configuration parameter",
                recovery=f"Use one of: {available}"
            )

        # Parse the value string using literal_eval for type safety
        try:
            parsed_val = literal_eval(val_str)
        except (SyntaxError, ValueError):
            # If literal_eval fails, treat as string
            parsed_val = val_str

        # Get the expected type from the current value
        expected_type = type(config_dict[key])

        # Validate type matches (with some flexibility for numeric types)
        if type(parsed_val) != expected_type:
            # Allow int->float coercion for numeric types
            if expected_type == float and isinstance(parsed_val, int):
                parsed_val = float(parsed_val)
            else:
                raise ValidationError(
                    problem="Configuration type mismatch",
                    cause=f"Cannot override '{key}': expected {expected_type.__name__}, got {type(parsed_val).__name__}",
                    recovery=f"Provide a value of type {expected_type.__name__} (current value: {config_dict[key]})"
                )

        overrides[key] = parsed_val

    # Apply overrides to the config dict
    if overrides:
        print("Applying CLI overrides (using deprecated compat layer):")
        for key, val in overrides.items():
            print(f"  Overriding: {key} = {val}")
            config_dict[key] = val

    return config_dict


def load_config_from_file(
    config_file_path: str,
    args: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Load config from a file using exec() pattern (DEPRECATED).

    DEPRECATED: This function uses exec() to load configuration files, which is a security
    risk and lacks type safety. It is provided only for backward compatibility during
    migration. Please migrate to the type-safe load_config() function.

    Args:
        config_file_path: Path to the configurator.py file to execute
        args: Command-line arguments to apply as overrides

    Returns:
        Dictionary of configuration values with CLI overrides applied

    Raises:
        ValidationError: If the config file cannot be loaded or parsed
        FileNotFoundError: If the config file does not exist

    Example (DEPRECATED):
        >>> config = load_config_from_file('configurator.py', ['--batch_size=32'])

    Migration Guide:
        Instead of executing config files, use preset modules:

        OLD:
            config = load_config_from_file('config/train_nano.py')

        NEW:
            from neuromanifold_gpt.config.loader import load_config
            from neuromanifold_gpt.config.training import TrainingConfig
            config = load_config(TrainingConfig, ['neuromanifold_gpt.config.presets.nano'])
    """
    warnings.warn(
        "load_config_from_file() uses exec() which is a security risk and lacks type safety. "
        "Please migrate to the type-safe load_config() function from neuromanifold_gpt.config.loader. "
        "See docs/config-migration-guide.md for migration instructions.",
        DeprecationWarning,
        stacklevel=2
    )

    try:
        with open(config_file_path, 'r') as f:
            config_code = f.read()
    except FileNotFoundError:
        raise ValidationError(
            problem=f"Config file not found: {config_file_path}",
            cause="The specified configurator file does not exist",
            recovery="Verify the file path or migrate to using preset modules with load_config()"
        )

    # Execute the config file in a namespace
    config_namespace: Dict[str, Any] = {}
    try:
        exec(config_code, config_namespace)
    except Exception as e:
        raise ValidationError(
            problem=f"Failed to execute config file: {config_file_path}",
            cause=str(e),
            recovery="Check the config file for syntax errors or migrate to the type-safe config system"
        )

    # Extract config variables (exclude builtins and modules)
    config_dict = {
        key: val for key, val in config_namespace.items()
        if not key.startswith('_') and not callable(val)
    }

    # Apply CLI overrides
    return apply_config_overrides(config_dict, args, show_deprecation_warning=False)
