"""Type-safe configuration loader with CLI override support.

This module provides a type-safe replacement for the unsafe exec pattern used in configurator.py.
Instead of executing arbitrary Python files, it safely imports config modules and uses dataclass
fields for validation and type checking.

Example usage:
    from neuromanifold_gpt.config.loader import load_config
    from neuromanifold_gpt.config.training import TrainingConfig

    # Load with CLI overrides
    config = load_config(TrainingConfig, sys.argv[1:])

    # Load from preset file
    config = load_config(TrainingConfig, ['neuromanifold_gpt.config.presets.nano', '--batch_size=32'])
"""

import sys
import importlib
from ast import literal_eval
from dataclasses import fields, MISSING
from typing import TypeVar, Type, List, Dict, Any, Optional
from neuromanifold_gpt.errors import ValidationError, ConfigurationError


T = TypeVar('T')


def load_config(
    config_class: Type[T],
    args: Optional[List[str]] = None,
    show_help: bool = True
) -> T:
    """Load configuration with type-safe CLI overrides.

    This function provides a type-safe alternative to executing configurator.py files.
    It loads configuration from dataclass defaults, optionally imports a preset module,
    and applies CLI overrides with type validation.

    Args:
        config_class: The dataclass type to instantiate (e.g., TrainingConfig)
        args: Command-line arguments to parse (defaults to sys.argv[1:])
        show_help: Whether to show help and exit when --help is passed

    Returns:
        An instance of config_class with all overrides applied

    Raises:
        ValidationError: If arguments are malformed or types don't match
        ConfigurationError: If config values are invalid

    Example:
        >>> from neuromanifold_gpt.config.training import TrainingConfig
        >>> config = load_config(TrainingConfig, ['--batch_size=32', '--learning_rate=1e-4'])
        >>> print(config.batch_size)
        32
    """
    if args is None:
        args = sys.argv[1:]

    # Handle --help
    if show_help and '--help' in args:
        _print_help(config_class)
        sys.exit(0)

    # Parse arguments into config_file and overrides
    config_module_name = None
    overrides: Dict[str, Any] = {}

    for arg in args:
        if '=' not in arg:
            # Assume it's the name of a config module to import
            if arg.startswith('--'):
                raise ValidationError(
                    problem="Invalid config module argument format",
                    cause=f"Config module argument '{arg}' cannot start with '--'",
                    recovery="Use either: python script.py module.path.to.config OR python script.py --key=value"
                )
            config_module_name = arg
        else:
            # Assume it's a --key=value override
            if not arg.startswith('--'):
                raise ValidationError(
                    problem="Invalid override argument format",
                    cause=f"Override argument '{arg}' must start with '--'",
                    recovery="Use format: --key=value (e.g., --batch_size=32)"
                )
            key, val = arg.split('=', 1)
            key = key[2:]  # Remove '--' prefix
            overrides[key] = val

    # Start with default config
    config_dict = _get_default_config(config_class)

    # Load preset config module if specified
    if config_module_name:
        print(f"Loading config from {config_module_name}:")
        preset_values = _load_config_module(config_module_name)
        print(f"  Loaded {len(preset_values)} settings from preset")
        config_dict.update(preset_values)

    # Apply CLI overrides with type validation
    if overrides:
        print("Applying CLI overrides:")
    for key, val_str in overrides.items():
        if key not in config_dict:
            # Get available keys for helpful error message
            available = ', '.join(sorted(config_dict.keys()))
            raise ValidationError(
                problem=f"Unknown config key: {key}",
                cause=f"'{key}' is not a valid configuration parameter for {config_class.__name__}",
                recovery=f"Use one of: {available}"
            )

        # Get the expected type from the default value
        expected_type = type(config_dict[key])

        # Parse the value string
        try:
            parsed_val = literal_eval(val_str)
        except (SyntaxError, ValueError):
            # If literal_eval fails, treat as string
            parsed_val = val_str

        # Validate type matches
        if type(parsed_val) != expected_type:
            raise ValidationError(
                problem="Configuration type mismatch",
                cause=f"Cannot override '{key}': expected {expected_type.__name__}, got {type(parsed_val).__name__}",
                recovery=f"Provide a value of type {expected_type.__name__} (current value: {config_dict[key]})"
            )

        print(f"  Overriding: {key} = {parsed_val}")
        config_dict[key] = parsed_val

    # Instantiate the config dataclass
    try:
        return config_class(**config_dict)
    except TypeError as e:
        raise ConfigurationError(
            problem=f"Failed to create {config_class.__name__}",
            cause=str(e),
            recovery="Check that all required fields are provided and types are correct"
        )


def _get_default_config(config_class: Type[T]) -> Dict[str, Any]:
    """Extract default values from a dataclass.

    Args:
        config_class: The dataclass type to extract defaults from

    Returns:
        Dictionary mapping field names to default values
    """
    config_dict = {}
    for field in fields(config_class):
        # Skip fields without defaults (should be rare for configs)
        if field.default is not MISSING:
            config_dict[field.name] = field.default
        elif field.default_factory is not MISSING:
            config_dict[field.name] = field.default_factory()

    return config_dict


def _load_config_module(module_name: str) -> Dict[str, Any]:
    """Import a config module and extract configuration values.

    This safely imports a Python module (instead of exec()ing it) and extracts
    all module-level variables that don't start with underscore.

    Args:
        module_name: Dotted module path (e.g., 'neuromanifold_gpt.config.presets.nano')
                    or file path (e.g., 'config/train_gpt2.py')

    Returns:
        Dictionary of configuration values from the module

    Raises:
        ValidationError: If the module cannot be imported
    """
    try:
        # Try importing as a module path first
        module = importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError):
        # Try as a file path
        if module_name.endswith('.py'):
            # Convert file path to module name
            # e.g., 'config/train_gpt2.py' -> 'config.train_gpt2'
            module_path = module_name[:-3].replace('/', '.')
            try:
                module = importlib.import_module(module_path)
            except (ImportError, ModuleNotFoundError) as e:
                raise ValidationError(
                    problem=f"Cannot load config module: {module_name}",
                    cause=str(e),
                    recovery="Provide a valid module path (e.g., 'neuromanifold_gpt.config.presets.nano') or file path"
                )
        else:
            raise ValidationError(
                problem=f"Cannot load config module: {module_name}",
                cause="Module not found",
                recovery="Provide a valid module path (e.g., 'neuromanifold_gpt.config.presets.nano') or file path"
            )

    # Extract all module-level variables (except private ones)
    config_values = {}
    for name in dir(module):
        if not name.startswith('_'):
            value = getattr(module, name)
            # Skip imported modules and classes
            if not callable(value) or isinstance(value, type):
                # Skip types/classes but keep primitive values
                if not isinstance(value, type):
                    config_values[name] = value

    return config_values


def _print_help(config_class: Type) -> None:
    """Print help message for config loading.

    Args:
        config_class: The config dataclass to show help for
    """
    print(f"""
Type-Safe Configuration Loader - {config_class.__name__}

Usage:
    python script.py [config_module] [--key=value ...]

Examples:
    python script.py neuromanifold_gpt.config.presets.nano
    python script.py --batch_size=32 --learning_rate=1e-4
    python script.py neuromanifold_gpt.config.presets.nano --batch_size=32

Available Configuration Options for {config_class.__name__}:
""")

    # Show all available fields with their defaults
    for field in fields(config_class):
        default_str = ""
        if field.default is not MISSING:
            default_str = f" (default: {field.default})"
        elif field.default_factory is not MISSING:
            default_str = " (default: <factory>)"

        # Get type hint as string
        type_hint = field.type
        if hasattr(type_hint, '__name__'):
            type_name = type_hint.__name__
        else:
            type_name = str(type_hint)

        print(f"    --{field.name}=<{type_name}>{default_str}")

    print("""
Note: Boolean values should be 'True' or 'False' (case-sensitive)
      Strings, numbers, and other Python literals are parsed automatically
""")
