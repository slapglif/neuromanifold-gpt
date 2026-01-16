# neuromanifold_gpt/model/fno/__init__.py
"""Fourier Neural Operators for NeuroManifoldGPT.

Exports:
    SpectralConv1d: 1D spectral convolution layer (core FNO operation)
    SpectralConv2d: 2D spectral convolution for image/grid data
    FNOBlock: Single Fourier Neural Operator block
    FNOEncoder: Stack of FNO blocks for feature extraction
    MultimodalFNOEncoder: FNO-based encoder for multimodal inputs
    FNOConfig: Configuration dataclass for FNO hyperparameters

Fourier Neural Operators (FNO) learn mappings between function spaces by
operating in the frequency domain. They are ideal for modeling:
- Resolution-invariant representations
- Global pattern recognition (frequency-domain processing)
- Continuous function interpolation
- Physics-informed inductive biases via spectral methods

The FNO module implements:
- SpectralConv: Core operation multiplying Fourier coefficients by learnable weights
- FNOBlock: Complete layer combining spectral convolution with local operations
- MultimodalFNOEncoder: Unified encoder for different input modalities (bytes, audio, images)

Key concepts:
- Fourier transform: Decomposes signals into frequency components
- Mode truncation: Keep only k lowest frequency modes for efficiency
- Spectral convolution: Pointwise multiplication in frequency domain = global convolution
- Resolution invariance: Same weights work at any discretization level

The spectral convolution operation:
    F^(-1)(W * F(x))
where F is FFT, W are learnable complex weights, and F^(-1) is inverse FFT.

This enables O(N log N) global operations (vs O(N^2) for attention) while
maintaining universal approximation properties for continuous operators.

FNO complements the wave-based architecture by:
1. Processing continuous inputs (audio, images) in native frequency space
2. Providing resolution-independent representations
3. Efficient global receptive field via spectral methods
4. Natural integration with existing spectral decomposition in NeuroManifoldGPT

Usage:
    from neuromanifold_gpt.model.fno import SpectralConv1d

    # Core spectral convolution:
    >>> sc = SpectralConv1d(in_channels=64, out_channels=64, modes=16)
    >>> x = torch.randn(2, 64, 128)  # (batch, channels, length)
    >>> y = sc(x)  # Same shape, globally mixed via FFT

    # Complete FNO block:
    from neuromanifold_gpt.model.fno import FNOBlock
    >>> fno = FNOBlock(embed_dim=384, modes=32)
    >>> x = torch.randn(2, 32, 384)  # (batch, seq_len, embed_dim)
    >>> y = fno(x)

    # Multimodal encoding:
    from neuromanifold_gpt.model.fno import MultimodalFNOEncoder
    >>> enc = MultimodalFNOEncoder(vocab_size=256, embed_dim=384)
    >>> bytes_input = torch.randint(0, 256, (2, 128))
    >>> x = enc(bytes_input)  # (2, 128, 384) continuous embeddings
"""

# Spectral convolution layers (core FNO operation)
from neuromanifold_gpt.model.fno.spectral_conv import (
    SpectralConv,
    SpectralConv1d,
    SpectralConv2d,
    SpectralConvNd,
    SpectralConvConfig,
)

# Note: Additional imports will be added as modules are implemented in subsequent subtasks.
# Future imports:
# from neuromanifold_gpt.model.fno.fourier_operator import (
#     FNOBlock,
#     FNOEncoder,
#     FNOConfig,
# )
# from neuromanifold_gpt.model.fno.multimodal_encoder import (
#     MultimodalFNOEncoder,
#     MultimodalConfig,
# )

__all__ = [
    # Spectral convolution layers
    "SpectralConv",
    "SpectralConv1d",
    "SpectralConv2d",
    "SpectralConvNd",
    "SpectralConvConfig",
]
