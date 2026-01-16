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

# Lazy imports to avoid loading heavy dependencies (torch, lightning) when only
# importing CLI utilities (e.g., for --help). This allows CLI scripts to display
# help without requiring full environment setup.
def __getattr__(name):
    """Lazy import heavy dependencies only when accessed."""
    if name in ("NeuroManifoldConfig", "NeuroManifoldConfigNano"):
        from neuromanifold_gpt.config import NeuroManifoldConfig, NeuroManifoldConfigNano
        return locals()[name]
    elif name in (
        "NeuroManifoldGPT",
        "NeuroManifoldBlock",
        "FHNAttention",
        "ManifoldProjection",
        "SDREngramMemory",
        "SDROperations",
        "SemanticFoldingEncoder",
        "SpectralDecomposition",
    ):
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
        return locals()[name]
    elif name == "NeuroManifoldLightning":
        from neuromanifold_gpt.train import NeuroManifoldLightning
        return NeuroManifoldLightning
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
