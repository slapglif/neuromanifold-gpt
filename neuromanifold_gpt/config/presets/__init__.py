"""Configuration presets for common training scenarios.

This module provides type-safe configuration presets for various use cases:

- nano: Fast experimentation and testing (~1M parameters)
- shakespeare_char: Character-level modeling on small datasets
- small: GPT-2 small scale (~124M parameters)
- medium: GPT-2 medium scale (~350M parameters)
- reasoning: System 2 reasoning components enabled

Usage:
    from neuromanifold_gpt.config.presets import get_nano_config

    config = get_nano_config()
    # Use config for training...
"""

from neuromanifold_gpt.config.presets.nano import get_nano_config
from neuromanifold_gpt.config.presets.shakespeare_char import get_shakespeare_char_config
from neuromanifold_gpt.config.presets.small import get_small_config
from neuromanifold_gpt.config.presets.medium import get_medium_config
from neuromanifold_gpt.config.presets.reasoning import get_reasoning_config

__all__ = [
    'get_nano_config',
    'get_shakespeare_char_config',
    'get_small_config',
    'get_medium_config',
    'get_reasoning_config',
]
