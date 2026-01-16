# neuromanifold_gpt/model/fno/multimodal_encoder.py
"""
Multimodal Input Encoder with Fourier Neural Operators.

Implements a unified encoder for processing multiple input modalities
(bytes, audio, images) using FNO-based spectral processing. The encoder
handles:

1. Byte sequences: Raw byte-level inputs (vocab_size=256)
2. Audio signals: Continuous waveforms sampled at arbitrary rates
3. Images: 2D spatial data (height x width)

Key features:
- Resolution-invariant processing via FNO's spectral convolution
- Learnable modality embeddings for distinguishing input types
- Sinusoidal position encoding for sequence positions
- Spectral preprocessing for continuous signals

The architecture:
1. Modality-specific embedding (bytes -> embed, audio -> project, image -> patch)
2. Position encoding (sinusoidal + learned)
3. FNO stack for spectral feature extraction
4. Optional cross-modal attention for multimodal fusion

This enables byte-level language modeling while supporting seamless
integration of continuous modalities through a unified latent space.

Reference:
- Li et al. "Fourier Neural Operator for Parametric PDEs" (2020)
- Peebles & Xie "Scalable Diffusion Models with Transformers" (DiT, 2023)
- Yu et al. "Scaling Autoregressive Models for Content-Rich Text-to-Image Generation"
"""

from dataclasses import dataclass
from typing import Optional, Literal
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuromanifold_gpt.model.fno.fourier_operator import FNOBlock, FNOEncoder


@dataclass
class MultimodalConfig:
    """Configuration for multimodal FNO encoder."""

    # Common parameters
    vocab_size: int = 256  # Byte-level vocabulary
    embed_dim: int = 384  # Embedding dimension
    max_seq_len: int = 2048  # Maximum sequence length for position encoding

    # FNO parameters
    fno_layers: int = 4  # Number of FNO blocks
    fno_modes: int = 32  # Fourier modes to retain
    dropout: float = 0.1  # Dropout probability

    # Position encoding
    use_learned_pos: bool = True  # Use learned position embeddings
    use_sinusoidal_pos: bool = True  # Add sinusoidal position encoding

    # Modality-specific
    n_modalities: int = 3  # Number of supported modalities
    modality_embed_dim: int = 32  # Dimension for modality type embedding

    # Audio-specific
    audio_channels: int = 1  # Mono or stereo
    audio_sample_rate: int = 16000  # Sample rate for audio

    # Image-specific
    image_channels: int = 3  # RGB channels
    patch_size: int = 16  # Patch size for image tokenization


def sinusoidal_position_encoding(
    seq_len: int,
    embed_dim: int,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generate sinusoidal position encodings.

    Uses sine/cosine functions at different frequencies to encode position:
    PE(pos, 2i) = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    Args:
        seq_len: Sequence length
        embed_dim: Embedding dimension
        device: Device for output tensor
        dtype: Data type for output tensor

    Returns:
        Position encoding of shape (seq_len, embed_dim)
    """
    if device is None:
        device = torch.device('cpu')

    position = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2, device=device, dtype=dtype)
        * (-math.log(10000.0) / embed_dim)
    )

    pe = torch.zeros(seq_len, embed_dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[:embed_dim // 2])  # Handle odd embed_dim

    return pe


class ByteEmbedding(nn.Module):
    """
    Byte-level embedding with optional positional encoding.

    Embeds discrete byte tokens (0-255) into continuous vectors,
    adding positional information for sequence modeling.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        embed_dim: int = 384,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        use_learned_pos: bool = True,
        use_sinusoidal_pos: bool = True,
    ):
        """
        Initialize byte embedding layer.

        Args:
            vocab_size: Size of vocabulary (256 for bytes)
            embed_dim: Output embedding dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            use_learned_pos: Add learned position embeddings
            use_sinusoidal_pos: Add sinusoidal position encoding
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.use_learned_pos = use_learned_pos
        self.use_sinusoidal_pos = use_sinusoidal_pos

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, embed_dim)

        # Position embeddings
        if use_learned_pos:
            self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        else:
            self.register_parameter('pos_embed', None)

        # Sinusoidal encoding (precomputed buffer)
        if use_sinusoidal_pos:
            self.register_buffer(
                'sinusoidal_pe',
                sinusoidal_position_encoding(max_seq_len, embed_dim),
            )
        else:
            self.register_buffer('sinusoidal_pe', None)

        # Scaling factor for embedding
        self.scale = math.sqrt(embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Layer norm for output stabilization
        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        if self.pos_embed is not None:
            nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Embed byte sequence.

        Args:
            x: Input tensor of shape (B, T) with byte values [0, vocab_size)
            positions: Optional position indices (B, T). If None, uses 0, 1, ..., T-1

        Returns:
            Embedded tensor of shape (B, T, embed_dim)
        """
        B, T = x.shape

        # Token embedding: (B, T) -> (B, T, D)
        embedded = self.token_embed(x) * self.scale

        # Add positional information
        if positions is None:
            positions = torch.arange(T, device=x.device)

        if self.use_learned_pos and self.pos_embed is not None:
            # Clamp positions to valid range
            pos_clamped = positions.clamp(0, self.max_seq_len - 1)
            if positions.dim() == 1:
                pos_clamped = pos_clamped.unsqueeze(0).expand(B, -1)
            embedded = embedded + self.pos_embed(pos_clamped)

        if self.use_sinusoidal_pos and self.sinusoidal_pe is not None:
            # Add sinusoidal encoding
            seq_len = min(T, self.max_seq_len)
            embedded[:, :seq_len] = embedded[:, :seq_len] + self.sinusoidal_pe[:seq_len]

        # Normalize and dropout
        embedded = self.norm(embedded)
        embedded = self.dropout(embedded)

        return embedded

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.vocab_size}, "
            f"embed_dim={self.embed_dim}, "
            f"max_seq_len={self.max_seq_len}"
        )


class AudioEncoder(nn.Module):
    """
    Audio signal encoder using spectral processing.

    Processes continuous audio waveforms by:
    1. Short-time Fourier transform (STFT) for time-frequency representation
    2. 1D convolution for local feature extraction
    3. Projection to embedding dimension
    """

    def __init__(
        self,
        embed_dim: int = 384,
        n_channels: int = 1,
        n_fft: int = 512,
        hop_length: int = 128,
        dropout: float = 0.1,
    ):
        """
        Initialize audio encoder.

        Args:
            embed_dim: Output embedding dimension
            n_channels: Number of audio channels (1=mono, 2=stereo)
            n_fft: FFT size for STFT
            hop_length: Hop length for STFT (determines output length)
            dropout: Dropout probability
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.n_channels = n_channels
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Frequency bins from STFT
        self.n_freq_bins = n_fft // 2 + 1

        # Project frequency bins to intermediate dimension
        self.freq_proj = nn.Linear(self.n_freq_bins * n_channels, embed_dim)

        # Local feature extraction
        self.conv = nn.Conv1d(
            embed_dim, embed_dim,
            kernel_size=3, padding=1, groups=1,
        )

        # Layer norm and dropout
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        """
        Encode audio waveform.

        Args:
            x: Audio waveform of shape (B, n_samples) or (B, C, n_samples)
            sample_rate: Audio sample rate (for documentation, not used currently)

        Returns:
            Encoded features of shape (B, T', embed_dim) where T' depends on hop_length
        """
        # Handle channel dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, n_samples)

        B, C, N = x.shape

        # STFT for each channel
        # Using torch.stft with return_complex=True for modern PyTorch
        spectrograms = []
        for c in range(C):
            spec = torch.stft(
                x[:, c],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                return_complex=True,
            )
            # Take magnitude: (B, n_freq, n_time)
            spec_mag = spec.abs()
            spectrograms.append(spec_mag)

        # Concatenate channels: (B, n_freq * C, n_time)
        combined = torch.cat(spectrograms, dim=1)

        # Transpose for linear: (B, n_time, n_freq * C)
        combined = combined.transpose(1, 2)

        # Project to embedding dimension: (B, n_time, embed_dim)
        embedded = self.freq_proj(combined)

        # Local convolution: transpose, convolve, transpose back
        embedded = embedded.transpose(1, 2)  # (B, embed_dim, n_time)
        embedded = self.conv(embedded)
        embedded = embedded.transpose(1, 2)  # (B, n_time, embed_dim)

        # Normalize and dropout
        embedded = self.norm(embedded)
        embedded = self.dropout(embedded)

        return embedded


class ImagePatchEncoder(nn.Module):
    """
    Image encoder using patch embedding.

    Divides image into non-overlapping patches and projects each
    to embedding dimension, similar to ViT.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        image_channels: int = 3,
        patch_size: int = 16,
        dropout: float = 0.1,
    ):
        """
        Initialize image patch encoder.

        Args:
            embed_dim: Output embedding dimension
            image_channels: Number of image channels (3 for RGB)
            patch_size: Size of each patch (assumes square patches)
            dropout: Dropout probability
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.image_channels = image_channels
        self.patch_size = patch_size

        # Patch embedding via convolution
        self.patch_embed = nn.Conv2d(
            image_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

        # Layer norm and dropout
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image into patch sequence.

        Args:
            x: Image tensor of shape (B, C, H, W)

        Returns:
            Patch embeddings of shape (B, n_patches, embed_dim)
            where n_patches = (H // patch_size) * (W // patch_size)
        """
        B, C, H, W = x.shape

        # Patch embedding: (B, C, H, W) -> (B, embed_dim, H', W')
        patches = self.patch_embed(x)

        # Flatten spatial dimensions: (B, embed_dim, H', W') -> (B, embed_dim, n_patches)
        patches = patches.flatten(2)

        # Transpose: (B, n_patches, embed_dim)
        patches = patches.transpose(1, 2)

        # Normalize and dropout
        patches = self.norm(patches)
        patches = self.dropout(patches)

        return patches


class MultimodalFNOEncoder(nn.Module):
    """
    Unified Multimodal Encoder using Fourier Neural Operators.

    Processes multiple input modalities through a shared FNO backbone:
    1. Modality-specific embedding (bytes, audio, images)
    2. Modality type encoding (learned embedding per modality)
    3. FNO stack for spectral feature extraction
    4. Output projection and normalization

    The FNO layers provide:
    - Resolution-invariant processing (same weights work at any sequence length)
    - Global receptive field via spectral convolution
    - Efficient O(N log N) computation

    Example:
        >>> encoder = MultimodalFNOEncoder(vocab_size=256, embed_dim=384)
        >>> bytes_input = torch.randint(0, 256, (2, 128))
        >>> output = encoder(bytes_input)
        >>> assert output.shape == (2, 128, 384)
    """

    def __init__(
        self,
        vocab_size: int = 256,
        embed_dim: int = 384,
        max_seq_len: int = 2048,
        fno_layers: int = 4,
        fno_modes: int = 32,
        dropout: float = 0.1,
        use_learned_pos: bool = True,
        use_sinusoidal_pos: bool = True,
        n_modalities: int = 3,
        audio_n_fft: int = 512,
        audio_hop_length: int = 128,
        image_patch_size: int = 16,
        image_channels: int = 3,
    ):
        """
        Initialize multimodal FNO encoder.

        Args:
            vocab_size: Vocabulary size for byte embedding (256 for bytes)
            embed_dim: Embedding dimension
            max_seq_len: Maximum sequence length
            fno_layers: Number of FNO blocks
            fno_modes: Number of Fourier modes to retain
            dropout: Dropout probability
            use_learned_pos: Use learned position embeddings
            use_sinusoidal_pos: Use sinusoidal position encoding
            n_modalities: Number of supported modalities
            audio_n_fft: FFT size for audio STFT
            audio_hop_length: Hop length for audio STFT
            image_patch_size: Patch size for image tokenization
            image_channels: Number of image channels
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.n_modalities = n_modalities

        # Modality type embedding
        self.modality_embed = nn.Embedding(n_modalities, embed_dim)

        # Byte embedding (primary modality for language)
        self.byte_embed = ByteEmbedding(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            max_seq_len=max_seq_len,
            dropout=dropout,
            use_learned_pos=use_learned_pos,
            use_sinusoidal_pos=use_sinusoidal_pos,
        )

        # Audio encoder
        self.audio_encoder = AudioEncoder(
            embed_dim=embed_dim,
            n_fft=audio_n_fft,
            hop_length=audio_hop_length,
            dropout=dropout,
        )

        # Image encoder
        self.image_encoder = ImagePatchEncoder(
            embed_dim=embed_dim,
            image_channels=image_channels,
            patch_size=image_patch_size,
            dropout=dropout,
        )

        # FNO backbone for spectral processing
        self.fno_encoder = FNOEncoder(
            embed_dim=embed_dim,
            n_layers=fno_layers,
            modes=fno_modes,
            dropout=dropout,
            activation='gelu',
            residual=True,
            prenorm=True,
        )

        # Output projection (optional refinement)
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

        # Final layer norm
        self.final_norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize modality embedding weights."""
        nn.init.normal_(self.modality_embed.weight, mean=0.0, std=0.02)

        # Initialize output projection
        for module in self.output_proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        modality: Literal['bytes', 'audio', 'image'] = 'bytes',
        positions: Optional[torch.Tensor] = None,
        return_fno_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input through multimodal FNO.

        Args:
            x: Input tensor. Shape depends on modality:
               - bytes: (B, T) with integer values [0, vocab_size)
               - audio: (B, n_samples) or (B, C, n_samples)
               - image: (B, C, H, W)
            modality: Input modality type ('bytes', 'audio', 'image')
            positions: Optional position indices for bytes modality
            return_fno_features: If True, also return pre-projection FNO features

        Returns:
            Encoded tensor of shape (B, T, embed_dim)
            If return_fno_features=True: tuple of (output, fno_features)
        """
        # Modality-specific embedding
        if modality == 'bytes':
            modality_idx = 0
            embedded = self.byte_embed(x, positions)
        elif modality == 'audio':
            modality_idx = 1
            embedded = self.audio_encoder(x)
        elif modality == 'image':
            modality_idx = 2
            embedded = self.image_encoder(x)
        else:
            raise ValueError(f"Unknown modality: {modality}. Expected 'bytes', 'audio', or 'image'")

        B, T, D = embedded.shape

        # Add modality type embedding
        modality_type = torch.tensor([modality_idx], device=x.device)
        modality_emb = self.modality_embed(modality_type)  # (1, D)
        embedded = embedded + modality_emb.unsqueeze(0)  # Broadcast across batch and sequence

        # Apply FNO backbone for spectral processing
        fno_features = self.fno_encoder(embedded)

        # Output projection with residual
        output = fno_features + self.output_proj(fno_features)

        # Final normalization
        output = self.final_norm(output)

        if return_fno_features:
            return output, fno_features
        return output

    def encode_bytes(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Convenience method for encoding byte sequences.

        Args:
            x: Byte tensor of shape (B, T)
            positions: Optional position indices

        Returns:
            Encoded tensor of shape (B, T, embed_dim)
        """
        return self.forward(x, modality='bytes', positions=positions)

    def encode_audio(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convenience method for encoding audio.

        Args:
            x: Audio tensor of shape (B, n_samples) or (B, C, n_samples)

        Returns:
            Encoded tensor of shape (B, T', embed_dim)
        """
        return self.forward(x, modality='audio')

    def encode_image(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convenience method for encoding images.

        Args:
            x: Image tensor of shape (B, C, H, W)

        Returns:
            Encoded tensor of shape (B, n_patches, embed_dim)
        """
        return self.forward(x, modality='image')

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters.

        Args:
            non_embedding: If True, exclude embedding parameters

        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.byte_embed.token_embed.weight.numel()
            if self.byte_embed.pos_embed is not None:
                n_params -= self.byte_embed.pos_embed.weight.numel()
            n_params -= self.modality_embed.weight.numel()
        return n_params

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.vocab_size}, "
            f"embed_dim={self.embed_dim}, "
            f"max_seq_len={self.max_seq_len}, "
            f"n_modalities={self.n_modalities}"
        )


class MultimodalFusionEncoder(nn.Module):
    """
    Extended multimodal encoder with cross-modal attention fusion.

    Builds on MultimodalFNOEncoder by adding cross-attention between
    different modalities for richer multimodal understanding.

    This is useful when processing inputs that combine multiple modalities
    (e.g., text describing an image, audio with transcript).
    """

    def __init__(
        self,
        vocab_size: int = 256,
        embed_dim: int = 384,
        n_heads: int = 8,
        fno_layers: int = 4,
        fno_modes: int = 32,
        dropout: float = 0.1,
    ):
        """
        Initialize multimodal fusion encoder.

        Args:
            vocab_size: Vocabulary size
            embed_dim: Embedding dimension
            n_heads: Number of attention heads for cross-modal attention
            fno_layers: Number of FNO layers
            fno_modes: Number of Fourier modes
            dropout: Dropout probability
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads

        # Base multimodal encoder
        self.base_encoder = MultimodalFNOEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            fno_layers=fno_layers,
            fno_modes=fno_modes,
            dropout=dropout,
        )

        # Cross-modal attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query_input: torch.Tensor,
        context_input: torch.Tensor,
        query_modality: Literal['bytes', 'audio', 'image'] = 'bytes',
        context_modality: Literal['bytes', 'audio', 'image'] = 'image',
    ) -> torch.Tensor:
        """
        Encode with cross-modal attention.

        Args:
            query_input: Primary input tensor
            context_input: Context input tensor from another modality
            query_modality: Modality of query input
            context_modality: Modality of context input

        Returns:
            Fused representation of shape (B, T_query, embed_dim)
        """
        # Encode both modalities
        query_encoded = self.base_encoder(query_input, modality=query_modality)
        context_encoded = self.base_encoder(context_input, modality=context_modality)

        # Cross-modal attention: query attends to context
        attended, _ = self.cross_attn(
            query=query_encoded,
            key=context_encoded,
            value=context_encoded,
        )

        # Add & norm
        fused = self.norm1(query_encoded + attended)

        # MLP
        fused = self.norm2(fused + self.fusion_mlp(fused))

        return fused
