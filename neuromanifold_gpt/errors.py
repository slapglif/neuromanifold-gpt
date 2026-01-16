"""Structured error handling with rich-formatted panels.

This module provides custom exception classes that display helpful error messages
with problem descriptions, causes, and recovery suggestions using rich panels.

Example:
    raise ConfigurationError(
        problem="n_embd must be divisible by n_heads",
        cause=f"n_embd={config.n_embd} is not divisible by n_heads={config.n_heads}",
        recovery="Set n_embd to a multiple of n_heads (e.g., n_embd=384, n_heads=8)"
    )
"""

from typing import Optional
from rich.console import Console
from rich.panel import Panel


console = Console(stderr=True)


class NeuroManifoldError(Exception):
    """Base exception class for NeuroManifoldGPT errors with rich formatting.

    All custom errors inherit from this base class, which provides consistent
    rich-formatted error panels with problem, cause, and recovery information.

    Attributes:
        problem: A concise description of what went wrong
        cause: Explanation of why the error occurred
        recovery: Actionable steps to fix the issue
        context: Optional additional context (e.g., related config values)
    """

    def __init__(
        self,
        problem: str,
        cause: Optional[str] = None,
        recovery: Optional[str] = None,
        context: Optional[str] = None,
    ):
        """Initialize a NeuroManifoldError with structured error information.

        Args:
            problem: A concise description of what went wrong
            cause: Explanation of why the error occurred (optional)
            recovery: Actionable steps to fix the issue (optional)
            context: Optional additional context information
        """
        self.problem = problem
        self.cause = cause
        self.recovery = recovery
        self.context = context

        # Build the error message
        message_parts = [f"[bold red]Problem:[/bold red] {problem}"]

        if cause:
            message_parts.append(f"\n[bold yellow]Cause:[/bold yellow] {cause}")

        if recovery:
            message_parts.append(f"\n[bold green]Recovery:[/bold green] {recovery}")

        if context:
            message_parts.append(f"\n[bold blue]Context:[/bold blue] {context}")

        self.message = "\n".join(message_parts)

        # Display the error panel
        self._display_error()

        # Call parent constructor with plain text for standard error handling
        super().__init__(problem)

    def _display_error(self):
        """Display the error message as a rich panel."""
        panel = Panel(
            self.message,
            title=f"[bold red]{self.__class__.__name__}[/bold red]",
            border_style="red",
            expand=False,
        )
        console.print(panel)


class ConfigurationError(NeuroManifoldError):
    """Error raised for invalid configuration values.

    This error is raised when configuration parameters are invalid, inconsistent,
    or violate architectural constraints (e.g., divisibility requirements,
    incompatible feature combinations).

    Example:
        raise ConfigurationError(
            problem="n_embd must be divisible by n_heads",
            cause="n_embd=100 is not divisible by n_heads=8",
            recovery="Use n_embd=128 (or any multiple of 8) with n_heads=8"
        )
    """
    pass


class ModelError(NeuroManifoldError):
    """Error raised for model loading, initialization, or architecture issues.

    This error is raised when there are problems with model operations such as:
    - Loading pretrained weights
    - Model architecture mismatches
    - Invalid model states
    - Missing required model components

    Example:
        raise ModelError(
            problem="Cannot load pretrained model 'invalid-model'",
            cause="Model type must be one of: gpt2, gpt2-medium, gpt2-large, gpt2-xl",
            recovery="Specify a valid model type or use --init_from=scratch"
        )
    """
    pass


class ValidationError(NeuroManifoldError):
    """Error raised for input validation failures.

    This error is raised when input data, arguments, or parameters fail validation
    checks (e.g., wrong types, out of range values, invalid formats).

    Example:
        raise ValidationError(
            problem="Invalid argument format",
            cause="Arguments cannot start with '--' when using config file syntax",
            recovery="Use either: python script.py config_name OR python script.py --flag=value"
        )
    """
    pass


class RuntimeError(NeuroManifoldError):
    """Error raised for runtime issues during model execution.

    This error is raised when there are problems during model forward pass,
    training, or inference that are not related to configuration or input validation.

    Example:
        raise RuntimeError(
            problem="Imagination module not available",
            cause="Model was initialized without imagination support",
            recovery="Set use_imagination=True in config to enable this feature"
        )
    """
    pass


class CheckpointError(NeuroManifoldError):
    """Error raised for checkpoint loading failures.

    This error is raised when there are problems loading or restoring model
    checkpoints, such as:
    - Missing checkpoint files
    - Corrupted checkpoint data
    - Version mismatches between saved and current model
    - Missing required checkpoint keys

    Example:
        raise CheckpointError(
            problem="Failed to load checkpoint from 'out/model.pt'",
            cause="Checkpoint file does not exist or is corrupted",
            recovery="Verify the checkpoint path and ensure the file is valid"
        )
    """
    pass


class DataError(NeuroManifoldError):
    """Error raised for data loading and format issues.

    This error is raised when there are problems with data operations such as:
    - Loading training or validation data
    - Invalid or corrupted data file formats
    - Data encoding or decoding issues
    - Unexpected data structures or schemas
    - Missing required data files

    Example:
        raise DataError(
            problem="Failed to load training data from 'data/train.bin'",
            cause="Data file is corrupted or has invalid format",
            recovery="Regenerate the data file using prepare.py or verify the data source"
        )
    """
    pass


class MemoryError(NeuroManifoldError):
    """Error raised for GPU memory allocation and management issues.

    This error is raised when there are problems with GPU memory operations such as:
    - Out of memory (OOM) errors during model loading or training
    - Insufficient GPU memory for batch size or model size
    - Memory allocation failures
    - Memory fragmentation issues
    - CUDA memory errors

    Example:
        raise MemoryError(
            problem="Insufficient GPU memory for batch size",
            cause="Batch size of 64 requires 12GB but only 8GB available",
            recovery="Reduce batch_size to 32 or use gradient accumulation"
        )
    """
    pass
