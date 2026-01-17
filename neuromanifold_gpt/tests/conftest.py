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

@pytest.fixture(autouse=True)
def reset_random_seed():
    """Autouse fixture to reset random seed before each test.

    This fixture runs automatically before every test to ensure reproducibility
    by setting torch random seed to 42. Tests no longer need inline
    torch.manual_seed() calls.
    """
    torch.manual_seed(42)


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
def sample_tensor_2d_grad():
    """Fixture providing 2D tensor with gradient enabled for testing.

    Returns:
        torch.Tensor: 2D tensor of shape (20, 64) with requires_grad=True.
    """
    return torch.randn(20, 64, requires_grad=True)


@pytest.fixture
def sample_tensor_3d_grad():
    """Fixture providing 3D tensor with gradient enabled for testing.

    Returns:
        torch.Tensor: 3D tensor of shape (2, 20, 64) with requires_grad=True.
    """
    return torch.randn(2, 20, 64, requires_grad=True)


@pytest.fixture
def sample_embeddings_grad():
    """Fixture providing embedding tensor with gradient enabled for testing.

    Returns:
        torch.Tensor: Embedding tensor of shape (2, 20, 128) with requires_grad=True.
    """
    return torch.randn(2, 20, 128, requires_grad=True)


@pytest.fixture
def sample_sdr_grad():
    """Fixture providing SDR tensor with gradient enabled for testing.

    Returns:
        torch.Tensor: SDR tensor of shape (2, 20, 2048) with requires_grad=True.
    """
    return torch.randn(2, 20, 2048, requires_grad=True)


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


# -----------------------------------------------------------------------------
# SDR Test Helper Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sdr_zero_with_active():
    """Fixture providing zero SDR with first 40 bits active.

    Common pattern for SDR operations testing. Creates a 2048-dimensional
    zero tensor with the first 40 bits set to 1.

    Returns:
        torch.Tensor: SDR tensor of shape (2048,) with first 40 bits set to 1.
    """
    sdr = torch.zeros(2048)
    sdr[:40] = 1
    return sdr


@pytest.fixture
def sdr_zero_with_active_offset():
    """Fixture providing zero SDR with 40 active bits at offset 20.

    Creates a 2048-dimensional zero tensor with bits 20-59 set to 1.
    Useful for testing overlap and intersection operations.

    Returns:
        torch.Tensor: SDR tensor of shape (2048,) with bits 20-59 set to 1.
    """
    sdr = torch.zeros(2048)
    sdr[20:60] = 1
    return sdr


@pytest.fixture
def sdr_zero_with_active_disjoint():
    """Fixture providing zero SDR with 40 active bits at offset 100.

    Creates a 2048-dimensional zero tensor with bits 100-139 set to 1.
    Useful for testing disjoint SDR operations with zero overlap.

    Returns:
        torch.Tensor: SDR tensor of shape (2048,) with bits 100-139 set to 1.
    """
    sdr = torch.zeros(2048)
    sdr[100:140] = 1
    return sdr


@pytest.fixture
def sdr_zero_empty():
    """Fixture providing empty SDR with no active bits.

    Creates a 2048-dimensional zero tensor with all bits set to 0.
    Useful for testing edge cases and empty SDR operations.

    Returns:
        torch.Tensor: SDR tensor of shape (2048,) with all bits set to 0.
    """
    return torch.zeros(2048)


@pytest.fixture
def sdr_zero_batched():
    """Fixture providing batched zero SDR with first 40 bits active.

    Creates a batched 2048-dimensional zero tensor with the first 40 bits
    set to 1 across all batch dimensions. Useful for testing batched operations.

    Returns:
        torch.Tensor: SDR tensor of shape (2, 3, 2048) with first 40 bits set to 1.
    """
    sdr = torch.zeros(2, 3, 2048)
    sdr[..., :40] = 1
    return sdr