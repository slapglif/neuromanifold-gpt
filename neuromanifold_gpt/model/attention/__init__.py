# neuromanifold_gpt/model/attention/__init__.py
"""Attention mechanisms for NeuroManifoldGPT.

This module provides a registry-based attention selection system. Use the
`attention_type` parameter in NeuroManifoldConfig to select the mechanism:

  - "standard": Standard causal self-attention (baseline)
  - "fhn": FitzHugh-Nagumo neural dynamics attention (default)
  - "knot": Topological knot-theory based attention
  - "kaufmann": Combined FHN + Knot reaction-diffusion system
  - "mla": DeepSeek-style KV cache compression attention

  DEPRECATED: Boolean flags use_kaufmann_attention and use_knot_attention
  are deprecated. Use attention_type instead.

Exports:
    StandardAttention: Standard causal self-attention (baseline)
    FHNAttention: FitzHugh-Nagumo neural dynamics attention
    FHNDynamics: Core excitable neural medium dynamics
    KnotAttention: Topological knot-theory based attention
    KaufmannAttention: Combined FHN + Knot reaction-diffusion system
    RMSNorm: Root Mean Square Layer Normalization
    get_attention_class: Registry function for attention type selection

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

Memory-Efficient Long Sequence Training:
For sequences longer than 2048 tokens, FHNAttention automatically uses chunked
processing to reduce memory from O(T²) to O(chunk_size²). This enables training
on sequences up to 8192+ tokens without OOM errors.

Configuration via NeuroManifoldConfig:
  config = NeuroManifoldConfig(
      attention_type="fhn",
      fhn_chunk_size=512,  # Chunk size for long sequences (default: 512)
      n_fhn_steps=2,       # Enable FHN dynamics (0 = Flash Attention only)
  )

Chunk Size Guidelines:
  - chunk_size=256: Limited GPU memory (8GB or less)
  - chunk_size=512: Good balance for most GPUs (default)
  - chunk_size=1024: High-memory GPUs with very long sequences (8192+)

Expected Memory Reduction:
  - Sequence length 2048: ~30-40% memory savings
  - Sequence length 4096: ~50-60% memory savings
  - Sequence length 8192: ~70-80% memory savings

Benchmark results: See neuromanifold_gpt/benchmarks/bench_fhn_chunked_memory.py
"""

from neuromanifold_gpt.model.attention.standard import StandardAttention
from neuromanifold_gpt.model.attention.fhn import FHNAttention, FHNDynamics
from neuromanifold_gpt.model.attention.knot import KnotAttention
from neuromanifold_gpt.model.attention.kaufmann import KaufmannAttention
# NOTE: MultiHeadLatentAttention not yet implemented in mla.py
from neuromanifold_gpt.model.attention.mla import RMSNorm


def get_attention_class(attention_type: str):
    """Get attention class by type string.

    Args:
        attention_type: Attention mechanism type string
            - "standard": Standard causal self-attention (baseline)
            - "fhn": FitzHugh-Nagumo neural dynamics attention
            - "knot": Topological knot-theory based attention
            - "kaufmann": Combined FHN + Knot reaction-diffusion system
            - "mla": DeepSeek-style KV cache compression attention
            - "soliton": Alias for "fhn" (excitable wave dynamics)
            - "sdr": Alias for "knot" (SDR memory with knot attention)
            - "fast-spectral": Alias for "fhn" (uses spectral basis)

    Returns:
        Attention class constructor

    Raises:
        ValueError: If attention_type is unknown
    """
    # Handle aliases
    if attention_type == "soliton":
        attention_type = "fhn"
    elif attention_type == "sdr":
        attention_type = "knot"
    elif attention_type == "fast-spectral":
        attention_type = "fhn"

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
    # "MultiHeadLatentAttention",  # Not yet implemented
    "RMSNorm",
    "get_attention_class",
]
