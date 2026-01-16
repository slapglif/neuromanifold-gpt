"""Rich-formatted argparse for NeuroManifoldGPT CLI scripts.

This module provides a unified, beautiful help system for all CLI scripts,
replacing the exec(configurator.py) pattern with proper argparse while
maintaining backward compatibility with config file overrides.

Features:
- Rich-formatted colored help output
- Grouped arguments (I/O, Model, Training, Hardware, etc.)
- Config file override support (first positional argument)
- CLI override support (--key=value flags)
- Type coercion and validation
- Example usage sections

Usage:
    from neuromanifold_gpt.cli.help_formatter import create_parser_from_defaults

    # Define defaults
    defaults = {
        'out_dir': 'out',
        'batch_size': 64,
        'learning_rate': 3e-4,
        # ... more defaults
    }

    # Create parser with groups
    parser = create_parser_from_defaults(
        defaults=defaults,
        description="Train NeuroManifoldGPT model",
        groups={
            'I/O': ['out_dir', 'data_dir'],
            'Training': ['batch_size', 'learning_rate'],
        },
        examples=[
            "python train.py --batch_size=32",
            "python train.py config/shakespeare.py --learning_rate=1e-4",
        ]
    )

    args = parser.parse_args()
    # args contains all values with config overrides applied
"""

import sys
import argparse
from typing import Any, Dict, List, Optional, Tuple, Union
from typing import TextIO
from ast import literal_eval
from pathlib import Path
from dataclasses import fields, is_dataclass, MISSING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Optional loguru import - fall back to standard logging if not available
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


class RichHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Custom argparse formatter with rich-formatted output.

    This formatter provides:
    - Colored help text sections
    - Better spacing and readability
    - Integration with rich.console for terminal output

    Note: This is used internally by RichArgumentParser.
    Users should use create_parser_from_defaults() instead.
    """

    def __init__(self, prog: str, **kwargs: Any) -> None:
        super().__init__(prog, max_help_position=40, width=100, **kwargs)


class RichArgumentParser(argparse.ArgumentParser):
    """ArgumentParser with rich-formatted help output.

    Extends argparse.ArgumentParser to provide beautiful, colored help
    output using rich.console. Automatically formats argument groups
    with colors and adds example sections.

    Attributes:
        examples: List of example usage strings to display in help
        _rich_console: Rich console for colored output
    """

    def __init__(
        self,
        *args,
        examples: Optional[List[str]] = None,
        **kwargs
    ):
        """Initialize parser with rich formatting.

        Args:
            *args: Positional arguments for ArgumentParser
            examples: Optional list of example usage strings
            **kwargs: Keyword arguments for ArgumentParser
        """
        # Set custom formatter
        kwargs.setdefault('formatter_class', RichHelpFormatter)
        super().__init__(*args, **kwargs)

        self.examples = examples or []
        self._rich_console = Console()

    def format_help(self) -> str:
        """Format help message with rich styling.

        Returns:
            Formatted help string with ANSI color codes
        """
        # Get standard help text
        help_text = super().format_help()

        # Add examples section if provided
        if self.examples:
            help_text += "\n[bold cyan]Examples:[/bold cyan]\n"
            for example in self.examples:
                help_text += f"  [dim]$[/dim] [green]{example}[/green]\n"

        return help_text

    def print_help(self, file: Optional[TextIO] = None) -> None:
        """Print help with rich formatting to terminal.

        Args:
            file: Output file (default: stdout)
        """
        if file is None:
            # Use rich console for colored output
            help_text = self.format_help()

            # Remove markup tags for plain argparse sections
            # and add rich markup for our custom sections
            lines = help_text.split('\n')
            formatted_lines = []

            for line in lines:
                # Highlight section headers
                if line and not line.startswith(' ') and line.endswith(':'):
                    formatted_lines.append(f"[bold cyan]{line}[/bold cyan]")
                # Highlight argument names
                elif line.strip().startswith('-'):
                    formatted_lines.append(f"[yellow]{line}[/yellow]")
                else:
                    formatted_lines.append(line)

            self._rich_console.print('\n'.join(formatted_lines))
        else:
            # Fallback to standard output if file is specified
            print(self.format_help(), file=file)


def create_parser_from_defaults(
    defaults: Dict[str, Any],
    description: str,
    groups: Optional[Dict[str, List[str]]] = None,
    examples: Optional[List[str]] = None,
    config_file_help: str = "Path to config file (overrides defaults)",
) -> RichArgumentParser:
    """Create rich argparse parser from default values.

    This is the main entry point for creating CLI parsers. It automatically
    generates argument flags from a dictionary of defaults, supporting:
    - Type inference from default values
    - Grouped arguments (I/O, Model, Training, etc.)
    - Config file as first positional argument
    - CLI overrides via --key=value

    Args:
        defaults: Dictionary of parameter names to default values
        description: Program description for help text
        groups: Optional dict mapping group names to lists of param names
                Example: {'I/O': ['out_dir', 'data_dir'], 'Training': ['lr']}
        examples: Optional list of example usage strings
        config_file_help: Help text for config file argument

    Returns:
        RichArgumentParser instance configured with all arguments

    Example:
        >>> defaults = {'batch_size': 64, 'learning_rate': 3e-4}
        >>> parser = create_parser_from_defaults(
        ...     defaults=defaults,
        ...     description="Train model",
        ...     groups={'Training': ['batch_size', 'learning_rate']},
        ...     examples=["python train.py --batch_size=32"]
        ... )
        >>> args = parser.parse_args()
    """
    parser = RichArgumentParser(
        description=description,
        examples=examples,
        epilog="Note: Config file overrides defaults, CLI flags override config file.",
    )

    # Add optional config file as first positional argument
    parser.add_argument(
        'config',
        nargs='?',
        default=None,
        help=config_file_help,
    )

    # If groups are provided, organize arguments into groups
    if groups:
        # Create argument groups
        group_objects = {}
        for group_name, param_names in groups.items():
            group_objects[group_name] = parser.add_argument_group(
                f"[{group_name}]",
                description=f"{group_name} parameters"
            )

        # Add arguments to their groups
        for group_name, param_names in groups.items():
            group = group_objects[group_name]
            for param_name in param_names:
                if param_name in defaults:
                    _add_argument_from_default(
                        group, param_name, defaults[param_name]
                    )

        # Add ungrouped arguments to a separate "Other" group
        grouped_params = set()
        for param_names in groups.values():
            grouped_params.update(param_names)

        ungrouped_params = set(defaults.keys()) - grouped_params
        if ungrouped_params:
            other_group = parser.add_argument_group(
                "[Other]",
                description="Other parameters"
            )
            for param_name in sorted(ungrouped_params):
                _add_argument_from_default(
                    other_group, param_name, defaults[param_name]
                )
    else:
        # No groups, add all arguments directly
        for param_name, default_value in defaults.items():
            _add_argument_from_default(parser, param_name, default_value)

    return parser


def _add_argument_from_default(
    parser_or_group: Union[argparse.ArgumentParser, argparse._ArgumentGroup],
    param_name: str,
    default_value: Union[str, int, float, bool],
) -> None:
    """Add an argument to parser/group inferred from default value.

    Args:
        parser_or_group: ArgumentParser or argument group
        param_name: Parameter name (converted to --param-name)
        default_value: Default value (type is inferred from this)
    """
    # Infer type from default value
    param_type = type(default_value)

    # Convert underscores to hyphens for CLI flag
    flag_name = f"--{param_name}"

    # Special handling for booleans (use store_true/store_false)
    if isinstance(default_value, bool):
        parser_or_group.add_argument(
            flag_name,
            action='store_true' if not default_value else 'store_false',
            default=default_value,
            help=f"(default: {default_value})"
        )
    else:
        # For other types, use type inference
        parser_or_group.add_argument(
            flag_name,
            type=param_type,
            default=default_value,
            help=f"(default: {default_value})"
        )


def parse_args_with_config_override(
    parser: RichArgumentParser,
    args: Optional[List[str]] = None,
) -> argparse.Namespace:
    """Parse arguments with config file override support.

    This function mimics the behavior of configurator.py:
    1. If first argument is a file path, exec it to get overrides
    2. Apply CLI --key=value overrides on top

    Args:
        parser: RichArgumentParser instance
        args: Optional argument list (default: sys.argv[1:])

    Returns:
        Namespace with parsed arguments and overrides applied

    Example:
        >>> parser = create_parser_from_defaults(...)
        >>> args = parse_args_with_config_override(parser)
    """
    if args is None:
        args = sys.argv[1:]

    # Parse arguments normally first
    parsed = parser.parse_args(args)

    # If config file is provided, load and override
    if parsed.config:
        config_path = Path(parsed.config)
        if config_path.exists():
            logger.info(f"Loading config from {config_path}")

            # Read and exec config file (like configurator.py)
            config_globals = {}
            with open(config_path) as f:
                config_code = f.read()
                exec(config_code, config_globals)

            # Override parsed values with config values
            for key, value in config_globals.items():
                if hasattr(parsed, key) and not key.startswith('_'):
                    setattr(parsed, key, value)
                    logger.debug(f"Config override: {key} = {value}")

    return parsed


def create_parser_from_globals(
    globals_dict: Dict[str, Any],
    description: str,
    groups: Optional[Dict[str, List[str]]] = None,
    examples: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
) -> RichArgumentParser:
    """Create parser from a globals() dict (for backward compatibility).

    This is a convenience wrapper for scripts that currently use:
        exec(open('configurator.py').read())

    It extracts simple typed values (str, int, float, bool) from globals
    and creates a parser from them.

    Args:
        globals_dict: globals() dictionary from calling script
        description: Program description
        groups: Optional argument groups
        examples: Optional usage examples
        exclude: Optional list of variable names to exclude

    Returns:
        RichArgumentParser instance

    Example:
        >>> # In your script (replacing configurator.py):
        >>> parser = create_parser_from_globals(
        ...     globals(),
        ...     description="Sample from model",
        ...     groups={'Sampling': ['num_samples', 'temperature']},
        ...     exclude=['os', 'sys', 'torch']  # Exclude imports
        ... )
    """
    exclude = set(exclude or [])

    # Extract simple typed values from globals
    defaults = {}
    for key, value in globals_dict.items():
        # Skip private, excluded, and complex types
        if (
            key.startswith('_') or
            key in exclude or
            not isinstance(value, (str, int, float, bool))
        ):
            continue

        defaults[key] = value

    return create_parser_from_defaults(
        defaults=defaults,
        description=description,
        groups=groups,
        examples=examples,
    )


def dataclass_to_defaults(dataclass_type: type) -> Dict[str, Any]:
    """Extract default values from a dataclass into a dictionary.

    Converts a dataclass definition into a flat dictionary of parameter
    names to default values, suitable for passing to create_parser_from_defaults().

    This enables the pattern:
        @dataclass
        class TrainConfig:
            batch_size: int = 64
            learning_rate: float = 3e-4
            ...

        defaults = dataclass_to_defaults(TrainConfig)
        parser = create_parser_from_defaults(defaults, ...)

    Args:
        dataclass_type: A dataclass type (not an instance)

    Returns:
        Dictionary mapping field names to their default values

    Raises:
        TypeError: If dataclass_type is not a dataclass

    Example:
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Config:
        ...     batch_size: int = 64
        ...     lr: float = 1e-3
        >>> defaults = dataclass_to_defaults(Config)
        >>> defaults
        {'batch_size': 64, 'lr': 0.001}
    """
    if not is_dataclass(dataclass_type):
        raise TypeError(f"{dataclass_type} is not a dataclass")

    defaults = {}
    for field in fields(dataclass_type):
        # Check if field has a default value
        if field.default is not MISSING:
            # Has explicit default value
            defaults[field.name] = field.default
        elif field.default_factory is not MISSING:
            # Has default_factory (e.g., field(default_factory=list))
            defaults[field.name] = field.default_factory()
        # Skip fields without defaults (required fields)

    return defaults


def create_parser_from_dataclass(
    dataclass_type: type,
    description: str,
    groups: Optional[Dict[str, List[str]]] = None,
    examples: Optional[List[str]] = None,
    config_file_help: str = "Path to config file (overrides defaults)",
) -> RichArgumentParser:
    """Create rich argparse parser directly from a dataclass.

    This is a convenience wrapper that combines dataclass_to_defaults()
    and create_parser_from_defaults() into a single call.

    Replaces the manual pattern in train.py:
        for f in TrainConfig.__dataclass_fields__:
            field_type = TrainConfig.__dataclass_fields__[f].type
            if field_type == bool:
                parser.add_argument(f"--{f}", type=lambda x: x.lower() == "true")
            ...

    With a simple:
        parser = create_parser_from_dataclass(
            TrainConfig,
            description="Train model",
            groups={'Training': ['batch_size', 'learning_rate']}
        )

    Args:
        dataclass_type: A dataclass type defining configuration
        description: Program description for help text
        groups: Optional dict mapping group names to lists of param names
        examples: Optional list of example usage strings
        config_file_help: Help text for config file argument

    Returns:
        RichArgumentParser instance configured with all dataclass fields

    Raises:
        TypeError: If dataclass_type is not a dataclass

    Example:
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Config:
        ...     batch_size: int = 64
        ...     lr: float = 1e-3
        >>> parser = create_parser_from_dataclass(
        ...     Config,
        ...     description="Train model",
        ...     groups={'Training': ['batch_size', 'lr']}
        ... )
    """
    defaults = dataclass_to_defaults(dataclass_type)

    return create_parser_from_defaults(
        defaults=defaults,
        description=description,
        groups=groups,
        examples=examples,
        config_file_help=config_file_help,
    )
