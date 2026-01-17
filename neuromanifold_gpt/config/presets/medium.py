"""Medium preset - similar to GPT-2 medium (350M parameters).

This preset provides a configuration suitable for:
- High-quality model training with substantial compute
- Multi-GPU or high-VRAM single GPU setups (~24GB VRAM)
- Production deployments requiring stronger performance

The configuration uses approximately 350M parameters, similar to GPT-2 medium,
requiring more substantial hardware but delivering better results.
"""

from neuromanifold_gpt.config.base import NeuroManifoldConfig


def get_medium_config() -> NeuroManifoldConfig:
    """Get the medium configuration (GPT-2 medium scale).

    Returns:
        NeuroManifoldConfig: Configuration with medium preset values.
    """
    return NeuroManifoldConfig(
        # GPT-2 medium scale
        n_layer=24,
        n_heads=16,
        n_embd=1024,
        block_size=1024,

        # SDR and manifold settings
        sdr_size=4096,
        manifold_dim=256,
        n_eigenvectors=128,

        # Training configuration
        learning_rate=3e-4,
    )
