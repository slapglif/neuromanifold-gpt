"""Shakespeare character-level training preset.

This preset provides a configuration suitable for:
- Quick sanity checks on character-level data
- Fast iteration on smaller datasets
- Testing model components on simple tasks

The configuration uses a smaller model appropriate for character-level
modeling, with reduced block size and faster training iterations.
"""

from neuromanifold_gpt.config.base import NeuroManifoldConfig


def get_shakespeare_char_config() -> NeuroManifoldConfig:
    """Get the Shakespeare character-level configuration.

    Returns:
        NeuroManifoldConfig: Configuration with Shakespeare char preset values.
    """
    return NeuroManifoldConfig(
        # Smaller model for char-level
        n_layer=6,
        n_heads=6,
        n_embd=384,
        block_size=256,

        # SDR and manifold settings
        sdr_size=1024,
        manifold_dim=64,
        n_eigenvectors=32,

        # Faster training with higher learning rate
        learning_rate=1e-3,
    )
