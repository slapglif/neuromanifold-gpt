# neuromanifold_gpt/model/attention/__init__.py
"""Attention mechanisms for NeuroManifoldGPT.

Exports:
    FHNAttention: FitzHugh-Nagumo neural dynamics attention
    FHNDynamics: Core excitable neural medium dynamics
    KnotAttention: Topological knot-theory based attention
    KaufmannAttention: Combined FHN + Knot reaction-diffusion system

The attention mechanisms implement biologically-inspired neural dynamics
rather than standard softmax attention, enabling wave-like information
propagation across the token sequence.
"""

from neuromanifold_gpt.model.attention.fhn import FHNAttention, FHNDynamics
from neuromanifold_gpt.model.attention.knot import KnotAttention
from neuromanifold_gpt.model.attention.kaufmann import KaufmannAttention

__all__ = [
    "FHNAttention",
    "FHNDynamics",
    "KnotAttention",
    "KaufmannAttention",
]
