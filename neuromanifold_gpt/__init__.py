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

# Lazy imports to allow utils.logging to be imported without torch
def __getattr__(name):
    """Lazy import to avoid importing torch when not needed."""
    if name == "NeuroManifoldConfig":
        from neuromanifold_gpt.config import NeuroManifoldConfig
        return NeuroManifoldConfig
    elif name == "NeuroManifoldConfigNano":
        from neuromanifold_gpt.config import NeuroManifoldConfigNano
        return NeuroManifoldConfigNano
    elif name == "NeuroManifoldGPT":
        from neuromanifold_gpt.model import NeuroManifoldGPT
        return NeuroManifoldGPT
    elif name == "NeuroManifoldBlock":
        from neuromanifold_gpt.model import NeuroManifoldBlock
        return NeuroManifoldBlock
    elif name == "FHNAttention":
        from neuromanifold_gpt.model import FHNAttention
        return FHNAttention
    elif name == "ManifoldProjection":
        from neuromanifold_gpt.model import ManifoldProjection
        return ManifoldProjection
    elif name == "SDREngramMemory":
        from neuromanifold_gpt.model import SDREngramMemory
        return SDREngramMemory
    elif name == "SDROperations":
        from neuromanifold_gpt.model import SDROperations
        return SDROperations
    elif name == "SemanticFoldingEncoder":
        from neuromanifold_gpt.model import SemanticFoldingEncoder
        return SemanticFoldingEncoder
    elif name == "SpectralDecomposition":
        from neuromanifold_gpt.model import SpectralDecomposition
        return SpectralDecomposition
    elif name == "NeuroManifoldLightning":
        from neuromanifold_gpt.train import NeuroManifoldLightning
        return NeuroManifoldLightning
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

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
    "NeuroManifoldLightning",
]
