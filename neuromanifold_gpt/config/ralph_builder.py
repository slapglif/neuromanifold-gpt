"""Ralph Loop configuration builder.

This module provides a builder pattern for creating RalphBaseConfig instances
with delta-based overrides. This allows for:
- Method chaining for fluent configuration
- Type-safe overrides of base configuration
- Clear documentation of what changed from defaults
- Easy composition of configuration variants

The builder pattern enables defining configs by specifying only the parameters
that differ from RalphBaseConfig defaults, eliminating duplication and making
configs more maintainable.

Example - Basic usage:
    >>> from neuromanifold_gpt.config.ralph_builder import RalphConfigBuilder
    >>>
    >>> builder = RalphConfigBuilder()
    >>> config = builder.with_overrides(
    ...     batch_size=32,
    ...     n_layer=4,
    ...     use_kan=True
    ... ).build()
    >>> config.batch_size
    32
    >>> config.n_layer
    4

Example - Custom experiment config:
    >>> config = RalphConfigBuilder().with_overrides(
    ...     # Model architecture
    ...     n_layer=8,
    ...     n_embd=512,
    ...     n_head=8,
    ...
    ...     # NeuroManifold features
    ...     use_kan=True,
    ...     kan_type="faster",
    ...     use_mhc=True,
    ...     mhc_n_streams=4,
    ...
    ...     # Training
    ...     max_iters=5000,
    ...     learning_rate=1e-3,
    ...     batch_size=32,
    ...
    ...     # Output
    ...     out_dir="out-custom-experiment"
    ... ).build()
    >>> # All unspecified params inherit from RalphBaseConfig defaults
    >>> config.dataset  # "shakespeare_char" (inherited)
    'shakespeare_char'
    >>> config.precision  # "bf16-mixed" (inherited)
    'bf16-mixed'

Example - Method chaining:
    >>> builder = RalphConfigBuilder()
    >>> builder.with_overrides(batch_size=32, n_layer=4)
    <RalphConfigBuilder object>
    >>> builder.with_overrides(learning_rate=1e-3)  # Can chain multiple calls
    <RalphConfigBuilder object>
    >>> config = builder.build()

See also:
    - neuromanifold_gpt.config.ralph_base: Base configuration with all defaults
    - neuromanifold_gpt.config.ralph_configs: Registry of existing iterations
    - examples/ralph_config_usage.py: Complete usage examples
"""

from typing import Any

from .ralph_base import RalphBaseConfig


class RalphConfigBuilder:
    """Builder for RalphBaseConfig with delta-based overrides.

    This builder starts with the default RalphBaseConfig values and allows
    selective overrides through method chaining.

    Example:
        >>> builder = RalphConfigBuilder()
        >>> config = builder.with_overrides(
        ...     batch_size=32,
        ...     n_layer=4,
        ...     learning_rate=1e-3
        ... ).build()

    Attributes:
        _overrides: Dictionary of configuration overrides to apply.
    """

    def __init__(self) -> None:
        """Initialize builder with empty overrides."""
        self._overrides: dict[str, Any] = {}

    def with_overrides(self, **kwargs: Any) -> "RalphConfigBuilder":
        """Add configuration overrides.

        This method allows fluent configuration by accepting any number of
        keyword arguments that correspond to RalphBaseConfig fields.

        Args:
            **kwargs: Configuration fields to override. Keys must match
                      RalphBaseConfig field names.

        Returns:
            Self for method chaining.

        Example:
            >>> builder = RalphConfigBuilder()
            >>> builder.with_overrides(batch_size=32, n_layer=4)
            >>> builder.with_overrides(learning_rate=1e-3)  # Can chain
        """
        self._overrides.update(kwargs)
        return self

    def build(self) -> RalphBaseConfig:
        """Build the final configuration with overrides applied.

        Creates a RalphBaseConfig instance with the accumulated overrides.
        The base config's __post_init__ validation will run on the result.

        Returns:
            RalphBaseConfig instance with overrides applied.

        Raises:
            ValueError: If overrides result in invalid configuration
                       (e.g., n_embd not divisible by n_head).
        """
        return RalphBaseConfig(**self._overrides)
