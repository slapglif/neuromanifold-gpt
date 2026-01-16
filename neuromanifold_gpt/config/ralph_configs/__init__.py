"""Ralph Loop iteration configurations.

This module provides a composition-based config system for Ralph Loop experiments,
replacing the previous 73 ralph_iter*.py files with a DRY approach.

The Ralph Loop is a rapid iteration framework for NeuroManifold GPT experiments
with tight constraints: val_loss < 1.5, training_time < 100s on consumer GPUs.

Usage - Load existing iteration:
    >>> from neuromanifold_gpt.config.ralph_configs import get_ralph_config
    >>> config = get_ralph_config(1)
    >>> config.batch_size
    32
    >>> config.n_layer
    2
    >>> config.max_iters
    500

Usage - List all iterations:
    >>> from neuromanifold_gpt.config.ralph_configs import list_ralph_iterations
    >>> iterations = list_ralph_iterations()
    >>> len(iterations)
    73
    >>> iterations[:5]  # First 5 iterations
    [1, 2, 3, 4, 5]

Usage - Compare configurations:
    >>> config1 = get_ralph_config(1)   # Tiny model for speed
    >>> config2 = get_ralph_config(10)  # Balanced model
    >>> config1.n_layer, config2.n_layer
    (2, 4)
    >>> config1.max_iters, config2.max_iters
    (500, 1000)

Usage - Create custom config:
    >>> from neuromanifold_gpt.config.ralph_builder import RalphConfigBuilder
    >>> custom = RalphConfigBuilder().with_overrides(
    ...     n_layer=4,
    ...     use_kan=True,
    ...     max_iters=2000
    ... ).build()

The new system:
- Defines a RalphBaseConfig with all common parameters
- Uses RalphConfigBuilder for delta-based overrides
- Stores iterations in a registry (only deltas specified)
- Reduces ~4380 lines of duplication to ~300 lines (92% reduction)

Benefits:
- ✓ DRY: Single source of truth for defaults
- ✓ Type-safe: Full dataclass validation and IDE support
- ✓ Discoverable: Clear documentation of all 60+ parameters
- ✓ Composable: Easy to create config variants
- ✓ Maintainable: Changes propagate automatically
- ✓ Testable: Built-in validation in __post_init__

See also:
    - neuromanifold_gpt.config.ralph_base: Base configuration dataclass
    - neuromanifold_gpt.config.ralph_builder: Builder pattern for config composition
    - neuromanifold_gpt.config.ralph_configs.registry: Iteration registry implementation
    - examples/ralph_config_usage.py: Complete usage examples
    - docs/ralph-config-system.md: Comprehensive documentation
"""

from .registry import (
    get_ralph_config,
    list_ralph_iterations,
)

__all__ = [
    "get_ralph_config",
    "list_ralph_iterations",
]
