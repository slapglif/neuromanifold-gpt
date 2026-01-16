# neuromanifold_gpt/model/attention/__init__.py
"""Attention mechanisms for NeuroManifoldGPT.

Exports:
    StandardAttention: Standard causal self-attention (baseline)
    FHNAttention: FitzHugh-Nagumo neural dynamics attention
    FHNDynamics: Core excitable neural medium dynamics
    KnotAttention: Topological knot-theory based attention
    KaufmannAttention: Combined FHN + Knot reaction-diffusion system
    MultiHeadLatentAttention: DeepSeek-style KV cache compression attention
    RMSNorm: Root Mean Square Layer Normalization

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

from neuromanifold_gpt.model.attention.standard import StandardAttention
from neuromanifold_gpt.model.attention.fhn import FHNAttention, FHNDynamics
from neuromanifold_gpt.model.attention.knot import KnotAttention
from neuromanifold_gpt.model.attention.kaufmann import KaufmannAttention
from neuromanifold_gpt.model.attention.mla import MultiHeadLatentAttention, RMSNorm


def get_attention_class(attention_type: str):
    """Get attention class by type string.

    Args:
        attention_type: Attention mechanism type string
            - "standard": Standard causal self-attention (baseline)
            - "fhn": FitzHugh-Nagumo neural dynamics attention
            - "knot": Topological knot-theory based attention
            - "kaufmann": Combined FHN + Knot reaction-diffusion system
            - "mla": DeepSeek-style KV cache compression attention

    Returns:
        Attention class constructor

    Raises:
        ValueError: If attention_type is unknown
    """
    if attention_type == "standard":
        return StandardAttention
    elif attention_type == "fhn":
        return FHNAttention
    elif attention_type == "knot":
        return KnotAttention
    elif attention_type == "kaufmann":
        return KaufmannAttention
    elif attention_type == "mla":
        return MultiHeadLatentAttention
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")


__all__ = [
    "StandardAttention",
    "FHNAttention",
    "FHNDynamics",
    "KnotAttention",
    "KaufmannAttention",
    "MultiHeadLatentAttention",
    "RMSNorm",
    "get_attention_class",
]
