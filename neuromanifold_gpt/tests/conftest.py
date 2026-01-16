# neuromanifold_gpt/tests/conftest.py
"""Shared pytest fixtures for neuromanifold_gpt tests."""
import pytest
from neuromanifold_gpt.config import NeuroManifoldConfigNano
from neuromanifold_gpt.config.block_config import NeuroManifoldBlockConfig


@pytest.fixture
def nano_config():
    """Fixture providing NeuroManifoldConfigNano instance.

    Returns:
        NeuroManifoldConfigNano: Nano preset configuration for testing.
    """
    return NeuroManifoldConfigNano()


@pytest.fixture
def block_config():
    """Fixture providing NeuroManifoldBlockConfig instance.

    Returns:
        NeuroManifoldBlockConfig: Standard block configuration for testing.
    """
    return NeuroManifoldBlockConfig(
        sdr_size=2048,
        embed_dim=384,
        manifold_dim=64,
        n_eigenvectors=32,
        n_heads=8
    )
