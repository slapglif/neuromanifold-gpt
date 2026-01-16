# neuromanifold_gpt/model/fno/encoder.py
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

Reference:
- Li et al. "Fourier Neural Operator for Parametric PDEs" (2020)
- Peebles & Xie "Scalable Diffusion Models with Transformers" (DiT, 2023)
"""

from typing import Optional, Literal

import torch
import torch.nn as nn

from neuromanifold_gpt.model.fno.byte_embedding import ByteEmbedding
from neuromanifold_gpt.model.fno.audio_encoder import AudioEncoder
from neuromanifold_gpt.model.fno.image_encoder import ImagePatchEncoder
from neuromanifold_gpt.model.fno.fourier_operator import FNOEncoder


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
