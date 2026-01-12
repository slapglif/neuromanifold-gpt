"""NeuroManifoldGPT model components.

This module exports the core building blocks:

Main Model:
    NeuroManifoldGPT: Complete language model
    NeuroManifoldBlock: Single transformer block

Attention:
    FHNAttention: FitzHugh-Nagumo dynamics attention
    FHNDynamics: Core FHN neural dynamics
    KnotAttention: Topological knot-based attention
    KaufmannAttention: Combined FHN + Knot reaction-diffusion attention

Manifold/Geometry:
    ManifoldProjection: Projects embeddings onto manifold
    SpectralDecomposition: Graph Laplacian spectral analysis

SDR/Memory:
    SDROperations: Sparse Distributed Representation operations
    SDREngramMemory: Breadcrumb-based memory retrieval
    SemanticFoldingEncoder: SDR-based semantic encoding

Hyper-Connections:
    HyperConnections: Manifold-constrained residual streams (mHC)
    Residual: Simple residual connection fallback
"""

# Main model
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT
from neuromanifold_gpt.model.block import NeuroManifoldBlock

# Attention mechanisms
from neuromanifold_gpt.model.attention.fhn import FHNAttention, FHNDynamics
from neuromanifold_gpt.model.attention.knot import KnotAttention
from neuromanifold_gpt.model.attention.kaufmann import KaufmannAttention

# Manifold and geometry
from neuromanifold_gpt.model.manifold import ManifoldProjection
from neuromanifold_gpt.model.spectral import SpectralDecomposition

# SDR and memory
from neuromanifold_gpt.model.sdr_ops import SDROperations
from neuromanifold_gpt.model.memory.engram import SDREngramMemory
from neuromanifold_gpt.model.semantic_folding import SemanticFoldingEncoder
from neuromanifold_gpt.model.semantic_retina import SemanticRetina
from neuromanifold_gpt.model.context_encoder import ContextEncoder

# Hyper-connections (mHC)
from neuromanifold_gpt.model.mhc import HyperConnections, Residual

__all__ = [
    # Main model
    "NeuroManifoldGPT",
    "NeuroManifoldBlock",
    # Attention
    "FHNAttention",
    "FHNDynamics",
    "KnotAttention",
    "KaufmannAttention",
    # Manifold/Geometry
    "ManifoldProjection",
    "SpectralDecomposition",
    # SDR/Memory
    "SDROperations",
    "SDREngramMemory",
    "SemanticFoldingEncoder",
    "SemanticRetina",
    "ContextEncoder",
    # Hyper-connections
    "HyperConnections",
    "Residual",
]
