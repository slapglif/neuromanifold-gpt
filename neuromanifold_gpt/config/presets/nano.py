"""Nano preset for fast experimentation and testing.

This preset provides a minimal configuration suitable for:
- Quick iteration during development
- Unit testing
- Resource-constrained environments
- Debugging model components

The configuration uses the NeuroManifoldConfigNano dataclass with
appropriate settings for fast training (~1M parameters, trains in
minutes on a single GPU).
"""

from neuromanifold_gpt.config.base import NeuroManifoldConfig


def get_nano_config() -> NeuroManifoldConfig:
    """Get the nano configuration for fast experimentation.

    Returns:
        NeuroManifoldConfig: Configuration with nano preset values.
    """
    return NeuroManifoldConfig(
        # Reduced model size for fast iteration
        block_size=256,
        n_layer=4,
        n_heads=4,
        n_embd=128,

        # Reduced manifold dimensions
        manifold_dim=32,
        n_eigenvectors=16,
        sdr_size=1024,

        # Higher learning rate for small model
        learning_rate=1e-3,
    )
