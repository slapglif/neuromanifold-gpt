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

Performance Optimization:
FHNAttention uses Flash Attention fusion by default (use_flash_fhn_fusion=True)
for 2-4x speedup. This optimizes FHN modulation by:
- Using PyTorch's scaled_dot_product_attention (Flash Attention kernel)
- Computing output variance as a cheap FHN stimulus proxy
- Modulating attention output directly instead of attention weights

The fusion approach maintains FHN's biologically-inspired dynamics while
preserving Flash Attention's memory efficiency and kernel fusion benefits.
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
