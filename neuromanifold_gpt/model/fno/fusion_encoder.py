# neuromanifold_gpt/model/fno/fusion_encoder.py
"""
Multimodal Fusion Encoder with Cross-Modal Attention.

Extends the base multimodal encoder with cross-attention between
different modalities for richer multimodal understanding.

This is useful when processing inputs that combine multiple modalities
(e.g., text describing an image, audio with transcript). The fusion
encoder allows one modality to attend to another, enabling:

1. Image captioning: byte tokens attend to image patches
2. Audio transcription: byte tokens attend to audio frames
3. Visual question answering: text query attends to image context
4. Multimodal retrieval: cross-modal similarity computation

Architecture:
1. Encode query and context modalities separately via base encoder
2. Cross-modal attention: query attends to context
3. Residual connection and layer normalization
4. Feed-forward MLP for feature fusion
5. Final layer normalization

Example:
    >>> encoder = MultimodalFusionEncoder(vocab_size=256, embed_dim=384)
    >>> text = torch.randint(0, 256, (2, 64))  # Byte tokens
    >>> image = torch.randn(2, 3, 224, 224)    # RGB image
    >>> output = encoder(text, image, 'bytes', 'image')
    >>> assert output.shape == (2, 64, 384)

Reference:
- Vaswani et al. "Attention Is All You Need" (2017)
- Lu et al. "ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations" (2019)
"""

from typing import Literal

import torch
import torch.nn as nn

from neuromanifold_gpt.model.fno.multimodal_encoder import MultimodalFNOEncoder


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
