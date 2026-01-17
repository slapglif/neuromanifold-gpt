"""Small preset - similar to GPT-2 small (124M parameters).

This preset provides a configuration suitable for:
- Single-GPU training on consumer hardware
- Production-quality models with reasonable compute
- Scaling up from nano/debug to serious training

The configuration uses approximately 124M parameters, similar to GPT-2 small,
making it trainable on a single GPU with gradient accumulation.
"""

from neuromanifold_gpt.config.base import NeuroManifoldConfig


def get_small_config() -> NeuroManifoldConfig:
    """Get the small configuration (GPT-2 small scale).

    Returns:
        NeuroManifoldConfig: Configuration with small preset values.
    """
    return NeuroManifoldConfig(
        # GPT-2 small scale
        n_layer=12,
        n_heads=12,
        n_embd=768,
        block_size=1024,

        # SDR and manifold settings
        sdr_size=2048,
        manifold_dim=128,
        n_eigenvectors=64,

        # Training configuration
        learning_rate=6e-4,
    )
