# neuromanifold_gpt/model/fno/image_encoder.py
"""
Image Encoder using Patch Embedding.

Implements image encoding through patch-based tokenization,
similar to Vision Transformer (ViT) approach. The encoder
divides images into non-overlapping patches and projects
each patch to an embedding dimension.

Key features:
- Non-overlapping patch extraction via strided convolution
- Resolution-flexible processing (works with any image size)
- LayerNorm for output stabilization
- Compatible with transformer-based architectures

The architecture:
1. Patch embedding: Conv2d with kernel_size=stride=patch_size
2. Spatial flattening: (H', W') -> sequence of patches
3. Layer normalization and dropout

This enables images to be processed as sequences of patch tokens,
suitable for transformer models or FNO-based processing.

Reference:
- Dosovitskiy et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (ViT, 2021)
- Peebles & Xie "Scalable Diffusion Models with Transformers" (DiT, 2023)
"""

import torch
import torch.nn as nn


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
