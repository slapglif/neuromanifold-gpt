"""Ralph Loop configuration builder.

This module provides a builder pattern for creating RalphBaseConfig instances
with delta-based overrides. This allows for:
- Method chaining for fluent configuration
- Type-safe overrides of base configuration
- Clear documentation of what changed from defaults
- Easy composition of configuration variants

Example:
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
"""

from typing import Any

from neuromanifold_gpt.config.ralph_base import RalphBaseConfig


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
