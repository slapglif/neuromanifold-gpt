# neuromanifold_gpt/tests/conftest.py
"""Shared pytest fixtures for neuromanifold_gpt tests."""
import pytest
import torch
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


@pytest.fixture
def sample_tensor_2d():
    """Fixture providing 2D tensor for testing.

    Returns:
        torch.Tensor: 2D tensor of shape (20, 64).
    """
    return torch.randn(20, 64)


@pytest.fixture
def sample_tensor_3d():
    """Fixture providing 3D tensor for testing.

    Returns:
        torch.Tensor: 3D tensor of shape (2, 20, 64) for batch processing.
    """
    return torch.randn(2, 20, 64)


@pytest.fixture
def sample_embeddings():
    """Fixture providing embedding tensor for testing.

    Returns:
        torch.Tensor: Embedding tensor of shape (2, 20, 128).
    """
    return torch.randn(2, 20, 128)


@pytest.fixture
def sample_sdr():
    """Fixture providing sparse distributed representation for testing.

    Returns:
        torch.Tensor: SDR tensor of shape (2, 20, 2048).
    """
    return torch.randn(2, 20, 2048)


@pytest.fixture
def sample_tokens():
    """Fixture providing token IDs for testing.

    Returns:
        torch.Tensor: Token ID tensor of shape (2, 20) with values in [0, 1000).
    """
    return torch.randint(0, 1000, (2, 20))


@pytest.fixture
def device():
    """Fixture providing appropriate device for testing.

    Returns:
        str: 'cuda' if GPU is available, 'cpu' otherwise.
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def seed():
    """Fixture for setting random seed for reproducibility.

    Sets torch random seed to 42 when called.
    """
    torch.manual_seed(42)
