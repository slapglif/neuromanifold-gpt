"""NeuroManifoldGPT - Novel transformer with FHN attention and SDR memory.

A neural language model combining:
- FitzHugh-Nagumo dynamics for attention (excitable neural media)
- Manifold projections for geometric embedding
- Sparse Distributed Representations (SDR) for memory
- Kolmogorov-Arnold Networks (KAN) for feed-forward layers

Example:
    from neuromanifold_gpt import NeuroManifoldGPT, NeuroManifoldConfig

    config = NeuroManifoldConfig(vocab_size=50257, n_layer=6, n_head=6, n_embd=384)
    model = NeuroManifoldGPT(config)
    logits, loss, info = model(input_ids, labels)
"""

__version__ = "0.1.0"

# Core model and config
from neuromanifold_gpt.config import NeuroManifoldConfig, NeuroManifoldConfigNano
from neuromanifold_gpt.model import (
    NeuroManifoldGPT,
    NeuroManifoldBlock,
    FHNAttention,
    ManifoldProjection,
    SDREngramMemory,
    SDROperations,
    SemanticFoldingEncoder,
    SpectralDecomposition,
)

# Training module
# NOTE: Temporarily commented out to avoid import issues with old train.py
# The new training modules are in neuromanifold_gpt.training package
# from neuromanifold_gpt.train import NeuroManifoldLightning

__all__ = [
    # Version
    "__version__",
    # Config
    "NeuroManifoldConfig",
    "NeuroManifoldConfigNano",
    # Core model
    "NeuroManifoldGPT",
    "NeuroManifoldBlock",
    # Attention
    "FHNAttention",
    # Components
    "ManifoldProjection",
    "SDREngramMemory",
    "SDROperations",
    "SemanticFoldingEncoder",
    "SpectralDecomposition",
    # Training
    # "NeuroManifoldLightning",  # Old training module - use neuromanifold_gpt.training instead
]
