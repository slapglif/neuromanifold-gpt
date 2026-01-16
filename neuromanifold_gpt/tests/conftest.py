"""Shared pytest fixtures and configuration for neuromanifold_gpt tests.

Includes config loader hacks to bypass torch dependencies where possible,
and standard fixtures for model testing.
"""
import sys
import importlib.util
from pathlib import Path
import pytest
import torch
from neuromanifold_gpt.config import NeuroManifoldConfigNano
from neuromanifold_gpt.config.block_config import NeuroManifoldBlockConfig
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT

# -----------------------------------------------------------------------------
# Config Loader Hacks (from PR #32)
# -----------------------------------------------------------------------------
# Get the project root directory
_conftest_dir = Path(__file__).parent
_project_root = _conftest_dir.parent.parent

def _load_module_direct(module_name, file_path):
    """Load a module directly from file path without triggering package imports."""
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Pre-load modules to bypass torch dependencies if needed
try:
    _load_module_direct(
        'neuromanifold_gpt.errors',
        _project_root / 'neuromanifold_gpt' / 'errors.py'
    )
    _load_module_direct(
        'neuromanifold_gpt.config.training',
        _project_root / 'neuromanifold_gpt' / 'config' / 'training.py'
    )
    _load_module_direct(
        'neuromanifold_gpt.config.loader',
        _project_root / 'neuromanifold_gpt' / 'config' / 'loader.py'
    )
except Exception as e:
    print(f"Warning: Failed to pre-load config modules: {e}")

# -----------------------------------------------------------------------------
# Shared Fixtures (from Master)
# -----------------------------------------------------------------------------

@pytest.fixture
def nano_config():
    """Fixture providing NeuroManifoldConfigNano instance.

    Returns:
        NeuroManifoldConfigNano: Nano preset configuration for testing.
    """
    return NeuroManifoldConfigNano()


@pytest.fixture
def gpt_model(nano_config):
    """Fixture providing NeuroManifoldGPT instance with nano_config.

    Args:
        nano_config: NeuroManifoldConfigNano fixture.

    Returns:
        NeuroManifoldGPT: GPT model instance configured with nano preset.
    """
    return NeuroManifoldGPT(nano_config)


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
def tokens_1x10(nano_config):
    """Single batch, short sequence token fixture.

    Args:
        nano_config: NeuroManifoldConfigNano fixture.

    Returns:
        torch.Tensor: Token ID tensor of shape (1, 10) with values in [0, vocab_size).
    """
    return torch.randint(0, nano_config.vocab_size, (1, 10))


@pytest.fixture
def tokens_1x20(nano_config):
    """Single batch, medium sequence token fixture.

    Args:
        nano_config: NeuroManifoldConfigNano fixture.

    Returns:
        torch.Tensor: Token ID tensor of shape (1, 20) with values in [0, vocab_size).
    """
    return torch.randint(0, nano_config.vocab_size, (1, 20))


@pytest.fixture
def tokens_2x20(nano_config):
    """Batch of 2, medium sequence token fixture.

    Args:
        nano_config: NeuroManifoldConfigNano fixture.

    Returns:
        torch.Tensor: Token ID tensor of shape (2, 20) with values in [0, vocab_size).
    """
    return torch.randint(0, nano_config.vocab_size, (2, 20))


@pytest.fixture
def tokens_2x50(nano_config):
    """Batch of 2, longer sequence token fixture.

    Args:
        nano_config: NeuroManifoldConfigNano fixture.

    Returns:
        torch.Tensor: Token ID tensor of shape (2, 50) with values in [0, vocab_size).
    """
    return torch.randint(0, nano_config.vocab_size, (2, 50))


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