# neuromanifold_gpt/model/topology/__init__.py
"""Topological Features for NeuroManifoldGPT.

Exports:
    BraidGroup: Braid group mathematical structure
    BraidEncoder: Neural encoder producing braid group representations
    JonesApproximator: Neural network approximating Jones polynomial invariants
    TopologicalHead: Loss computation head using topological invariants

Topological methods capture structural invariants in sequences that are
preserved under continuous deformations. They are ideal for modeling:
- Syntactic dependencies (crossed vs nested)
- Hierarchical structure in language
- Long-range correlations immune to local perturbations

The topology module implements:
- Braid groups: Mathematical structures encoding strand crossings
- Jones polynomial: Knot invariant computable from braid representations
- Topological head: Loss function encouraging topologically consistent outputs

Key concepts:
- Braid group B_n: Group of n strands with crossings as generators
- Artin generators: Elementary crossings sigma_i (strand i over strand i+1)
- Jones polynomial: Invariant distinguishing non-equivalent knot types
- Topological loss: Regularizer enforcing structural consistency

These components provide topological inductive biases that capture
syntactic structure as deformation-invariant properties, complementing
the wave dynamics from the soliton physics module.

Usage:
    from neuromanifold_gpt.model.topology import BraidEncoder

    # Encode sequence into braid group representation:
    >>> encoder = BraidEncoder(embed_dim=384, n_strands=4)
    >>> x = torch.randn(2, 32, 384)
    >>> rep = encoder(x)

    # Compute Jones polynomial approximation:
    from neuromanifold_gpt.model.topology import JonesApproximator
    >>> jones = JonesApproximator(embed_dim=384)
    >>> x = torch.randn(2, 32, 384)
    >>> poly, info = jones(x)

    # Use topological head for loss computation:
    from neuromanifold_gpt.model.topology import TopologicalHead
    >>> head = TopologicalHead(embed_dim=384)
    >>> loss, info = head(x)
"""

# Braid group representations
from neuromanifold_gpt.model.topology.braid import (
    BraidConfig,
    BraidCrossing,
    BraidEncoder,
    BraidGroup,
    TemperleyLiebAlgebra,
)

# Jones polynomial approximation
from neuromanifold_gpt.model.topology.jones_polynomial import (
    JonesApproximator,
    JonesConfig,
    JonesEvaluator,
    JonesLoss,
    KauffmanBracketNetwork,
)

# Topological head for loss computation
from neuromanifold_gpt.model.topology.topological_head import (
    TopologicalHead,
    TopologicalHeadConfig,
)

__all__ = [
    "BraidGroup",
    "BraidEncoder",
    "BraidCrossing",
    "BraidConfig",
    "TemperleyLiebAlgebra",
    "JonesApproximator",
    "JonesConfig",
    "JonesEvaluator",
    "JonesLoss",
    "KauffmanBracketNetwork",
    "TopologicalHead",
    "TopologicalHeadConfig",
]
