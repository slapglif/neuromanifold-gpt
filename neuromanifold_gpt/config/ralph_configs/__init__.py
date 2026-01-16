"""Ralph Loop iteration configurations.

This module provides a composition-based config system for Ralph Loop experiments,
replacing the previous 73 ralph_iter*.py files with a DRY approach.

Usage:
    >>> from neuromanifold_gpt.config.ralph_configs import get_ralph_config
    >>> config = get_ralph_config(1)
    >>> config.batch_size
    32

The new system:
- Defines a RalphBaseConfig with all common parameters
- Uses RalphConfigBuilder for delta-based overrides
- Stores iterations in a registry (only deltas specified)
- Reduces ~4380 lines of duplication to ~300 lines

Migration status:
- Phase 1: Base config system created (subtask-1-1, 1-2, 1-3) ✓
- Phase 2: Migration from old ralph_iter*.py files (subtask-2-1, 2-2) - IN PROGRESS
- Phase 3: Archive old files (subtask-3-1, 3-2) - PENDING
- Phase 4: Documentation (subtask-4-1, 4-2, 4-3) - PENDING

See also:
    - neuromanifold_gpt.config.ralph_base: Base configuration dataclass
    - neuromanifold_gpt.config.ralph_builder: Builder pattern for config composition
    - neuromanifold_gpt.config.ralph_configs.registry: Iteration registry implementation
"""

from neuromanifold_gpt.config.ralph_configs.registry import (
    get_ralph_config,
    list_ralph_iterations,
)

__all__ = [
    "get_ralph_config",
    "list_ralph_iterations",
]
