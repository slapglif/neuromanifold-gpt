# neuromanifold_gpt/model/fno/config.py
"""
Configuration dataclasses for multimodal FNO encoder.

Defines configuration parameters for the Fourier Neural Operator-based
multimodal encoder, including:
- Common embedding parameters (vocab size, dimensions)
- FNO architecture parameters (layers, modes)
- Position encoding settings
- Modality-specific configurations (audio, image)
"""

from dataclasses import dataclass


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
