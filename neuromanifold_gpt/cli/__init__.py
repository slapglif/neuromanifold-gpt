"""CLI utilities for NeuroManifoldGPT.

This module provides rich-formatted command-line interfaces with:
- Colored help output via rich.console
- Grouped argument sections (I/O, Model, Training, Hardware)
- Config file override support (compatible with configurator.py)
- Example usage sections
"""

from neuromanifold_gpt.cli.help_formatter import (
    RichHelpFormatter,
    create_parser_from_defaults,
)

__all__ = [
    "RichHelpFormatter",
    "create_parser_from_defaults",
]
