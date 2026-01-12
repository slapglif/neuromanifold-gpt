"""Configuration module for NeuroManifoldGPT.

Exports:
    NeuroManifoldConfig: Full configuration dataclass with all hyperparameters
    NeuroManifoldConfigNano: Preset for small/fast experimentation

Directory Structure:
    base.py: Core configuration dataclasses
    presets/: Model size presets (nano, small, medium, shakespeare_char)

Example:
    from neuromanifold_gpt.config import NeuroManifoldConfig

    # Custom config
    config = NeuroManifoldConfig(
        vocab_size=50257,
        n_layer=6,
        n_head=6,
        n_embd=384,
        block_size=256,
    )

    # Or use preset
    from neuromanifold_gpt.config import NeuroManifoldConfigNano
    config = NeuroManifoldConfigNano()
"""

from .base import NeuroManifoldConfig, NeuroManifoldConfigNano

__all__ = [
    "NeuroManifoldConfig",
    "NeuroManifoldConfigNano",
]
