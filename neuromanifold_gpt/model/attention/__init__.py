# neuromanifold_gpt/model/attention/__init__.py
"""Attention mechanisms for NeuroManifoldGPT.

This module provides a registry-based attention selection system with automatic
backend selection for optimal GPU compatibility. Use the `attention_type` and
`attention_backend` parameters in NeuroManifoldConfig to select the mechanism:

Attention Types:
  - "standard": Standard causal self-attention (baseline)
  - "fhn": FitzHugh-Nagumo neural dynamics attention (default)
  - "knot": Topological knot-theory based attention
  - "kaufmann": Combined FHN + Knot reaction-diffusion system
  - "mla": DeepSeek-style KV cache compression attention

Attention Backends:
  - "auto": Automatically detect best backend for current GPU (recommended)
  - "flash": Flash Attention 2 (fastest, requires Ampere+ GPU)
  - "xformers": xformers memory-efficient attention (Volta+ GPU)
  - "triton": Triton custom kernels (Volta+ GPU)
  - "pytorch": PyTorch native scaled_dot_product_attention
  - "manual": Standard PyTorch implementation (CPU compatible)

  DEPRECATED: Boolean flags use_kaufmann_attention and use_knot_attention
  are deprecated. Use attention_type instead.

Exports:
    StandardAttention: Standard causal self-attention (baseline)
    FHNAttention: FitzHugh-Nagumo neural dynamics attention
    FHNDynamics: Core excitable neural medium dynamics
    KnotAttention: Topological knot-theory based attention
    KaufmannAttention: Combined FHN + Knot reaction-diffusion system
    RMSNorm: Root Mean Square Layer Normalization
    get_attention_class: Registry function for attention type and backend selection

The attention mechanisms implement biologically-inspired neural dynamics
rather than standard softmax attention, enabling wave-like information
propagation across the token sequence.

Backend Selection:
When attention_backend='auto', the registry automatically selects the optimal
backend based on GPU compute capability:
- Ampere+ (SM 8.0+): Flash Attention 2 (RTX 30xx/40xx, A100, H100)
- Volta+ (SM 7.0+): xformers or Triton (RTX 20xx, V100, T4)
- CPU or older GPUs: Manual PyTorch implementation

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
# NOTE: MultiHeadLatentAttention not yet implemented in mla.py
from neuromanifold_gpt.model.attention.mla import RMSNorm

from typing import Optional


def get_attention_class(attention_type: str, backend: Optional[str] = None):
    """Get attention class by type string with backend-aware selection.

    This function serves as a registry for attention mechanisms, supporting
    automatic backend selection based on GPU capabilities when backend='auto'.

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
        backend: Optional backend selection string
            - "auto": Automatically detect best backend for current GPU
            - "flash": Flash Attention 2 (requires Ampere+ GPU)
            - "xformers": xformers memory-efficient attention (Volta+)
            - "triton": Triton custom kernels (Volta+)
            - "pytorch": PyTorch native scaled_dot_product_attention
            - "manual": Standard PyTorch implementation (CPU compatible)
            - None: Use attention class defaults (backward compatible)

    Returns:
        Attention class constructor

    Raises:
        ValueError: If attention_type is unknown

    Notes:
        - When backend='auto', uses GPU detection to select optimal backend
        - The returned attention class will internally handle backend selection
        - Backend parameter is advisory; actual backend used depends on availability
        - For FHN attention, backend controls the FHN dynamics computation method
    """
    # Resolve 'auto' backend to optimal backend for current GPU
    if backend == "auto":
        from neuromanifold_gpt.utils.gpu_detection import get_optimal_attention_backend
        resolved_backend = get_optimal_attention_backend()
        # Note: The attention class will use this backend internally
        # For now, we just resolve it for logging/debugging purposes
        # The actual backend configuration happens at block/model level
    elif backend is not None:
        resolved_backend = backend
    else:
        resolved_backend = None

    # Handle aliases
    if attention_type == "soliton":
        attention_type = "fhn"
    elif attention_type == "sdr":
        attention_type = "knot"
    elif attention_type == "fast-spectral":
        attention_type = "fhn"

    # Select attention class based on type
    if attention_type == "standard":
        attn_cls = StandardAttention
    elif attention_type == "fhn":
        attn_cls = FHNAttention
    elif attention_type == "knot":
        attn_cls = KnotAttention
    elif attention_type == "kaufmann":
        attn_cls = KaufmannAttention
    elif attention_type == "mla":
        attn_cls = MultiHeadLatentAttention
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")

    # Store resolved backend as class attribute for debugging
    # This allows inspection of which backend was auto-selected
    if resolved_backend is not None:
        attn_cls._resolved_backend = resolved_backend

    return attn_cls


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
