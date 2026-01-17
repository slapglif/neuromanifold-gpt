"""
Unified logging module with rich-formatted output.

Integrates loguru's structured logging with rich's formatting capabilities
for consistent, beautiful console output across all scripts.

Usage:
    from neuromanifold_gpt.utils.logging import get_logger

    logger = get_logger(__name__)
    logger.info("Starting training")
    logger.metric("loss", 0.42)
    logger.progress("Training", 50, 100)
    logger.section("Evaluation Results")
"""

import os
import sys
from typing import Any, Dict, Optional

from loguru import logger as loguru_logger
from rich.console import Console, RenderableType
from rich.theme import Theme

# Initialize rich console with custom theme
_console = Console(
    theme=Theme(
        {
            "info": "cyan",
            "warning": "yellow",
            "error": "bold red",
            "metric": "bold green",
            "progress": "yellow",
            "section": "bold blue",
        }
    )
)


class RichLogger:
    """
    Logger that combines loguru with rich formatting.

    Provides standard logging methods (info, warning, error, debug) plus
    specialized methods for metrics, progress tracking, and section headers.
    """

    def __init__(self, name: str):
        """
        Initialize logger with a name.

        Args:
            name: Logger name (typically __name__ or module name)
        """
        self.name = name
        self._logger = loguru_logger

    def info(self, message: str, **kwargs: Any) -> None:
        """
        Log info message.

        Args:
            message: Log message
            **kwargs: Additional context to include
        """
        self._logger.info(f"[{self.name}] {message}", **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """
        Log warning message.

        Args:
            message: Log message
            **kwargs: Additional context to include
        """
        self._logger.warning(f"[{self.name}] {message}", **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """
        Log error message.

        Args:
            message: Log message
            **kwargs: Additional context to include
        """
        self._logger.error(f"[{self.name}] {message}", **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        """
        Log debug message.

        Args:
            message: Log message
            **kwargs: Additional context to include
        """
        self._logger.debug(f"[{self.name}] {message}", **kwargs)

    def metric(self, name: str, value: float, unit: str = "", **kwargs: Any) -> None:
        """
        Log a metric with structured output and rich formatting.

        Args:
            name: Metric name
            value: Metric value
            unit: Optional unit string (e.g., "ms", "%", "tokens/s")
            **kwargs: Additional metadata
        """
        unit_str = f" {unit}" if unit else ""
        formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)

        _console.print(
            f"[metric]METRIC[/metric] {name}: [bold green]{formatted_value}{unit_str}[/bold green]"
        )

        # Also log to loguru for file/structured logging
        self._logger.info(f"METRIC {name}={formatted_value}{unit_str}", **kwargs)

    def progress(self, task: str, current: int, total: int) -> None:
        """
        Log progress information with percentage.

        Args:
            task: Task name/description
            current: Current step/iteration
            total: Total steps/iterations
        """
        percentage = (current / total) * 100 if total > 0 else 0

        _console.print(
            f"[progress]PROGRESS[/progress] {task}: "
            f"[bold]{current}/{total}[/bold] ([bold cyan]{percentage:.1f}%[/bold cyan])"
        )

        # Also log to loguru
        self._logger.info(f"PROGRESS {task} {current}/{total} ({percentage:.1f}%)")

    def section(self, title: str) -> None:
        """
        Print a visual section break with title.

        Args:
            title: Section title
        """
        _console.rule(f"[section]{title}[/section]")

        # Also log to loguru for file output
        self._logger.info(f"=== {title} ===")

    def bind(self, **kwargs: Any) -> "RichLogger":
        """
        Bind context to logger (for structured logging).

        Args:
            **kwargs: Context key-value pairs

        Returns:
            Self for chaining
        """
        self._logger = self._logger.bind(**kwargs)
        return self

    def table(self, renderable: RenderableType) -> None:
        """
        Print a Rich renderable object (e.g., Table, Panel, etc.).

        Args:
            renderable: Any Rich renderable object (Table, Panel, etc.)
        """
        _console.print(renderable)


def get_logger(name: str) -> RichLogger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (typically __name__ or module name)

    Returns:
        Configured RichLogger instance with rich formatting

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting process")
        >>> logger.metric("accuracy", 0.95, unit="%")
    """
    return RichLogger(name)


def configure_logging(
    level: Optional[str] = None,
    format: Optional[str] = None,
    sink: Any = sys.stderr,
    theme: Optional[Dict[str, str]] = None,
) -> None:
    """
    Configure global logging settings.

    Supports environment variables for easy configuration:
    - LOG_LEVEL: Set log level (DEBUG, INFO, WARNING, ERROR)
    - LOG_FORMAT: Set custom format template

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
               Defaults to LOG_LEVEL env var or "INFO"
        format: Optional custom format string for loguru.
                Defaults to LOG_FORMAT env var or default template
        sink: Output sink (default: stderr)
        theme: Optional custom theme dict for rich console.
               Keys: info, warning, error, metric, progress, section

    Example:
        >>> # Use defaults
        >>> configure_logging()

        >>> # Set custom level
        >>> configure_logging(level="DEBUG")

        >>> # Use environment variable
        >>> import os
        >>> os.environ['LOG_LEVEL'] = 'DEBUG'
        >>> configure_logging()

        >>> # Custom theme
        >>> configure_logging(theme={
        ...     "metric": "bold magenta",
        ...     "progress": "cyan"
        ... })
    """
    global _console

    # Get level from parameter, env var, or default
    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO")

    # Get format from parameter, env var, or default
    if format is None:
        format = os.environ.get("LOG_FORMAT")
        if format is None:
            # Default format with time, level, and message
            format = (
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<level>{message}</level>"
            )

    # Configure rich theme
    if theme is not None:
        # Merge with defaults
        default_theme = {
            "info": "cyan",
            "warning": "yellow",
            "error": "bold red",
            "metric": "bold green",
            "progress": "yellow",
            "section": "bold blue",
        }
        default_theme.update(theme)
        _console = Console(theme=Theme(default_theme))

    # Remove default loguru handler
    loguru_logger.remove()

    # Add configured handler
    loguru_logger.add(
        sink,
        format=format,
        level=level,
        colorize=True,
    )


# Configure default logging on module import
configure_logging()
