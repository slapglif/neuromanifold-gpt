# neuromanifold_gpt/model/fno/audio_encoder.py
"""
Audio Signal Encoder with Spectral Processing.

Implements an encoder for processing continuous audio waveforms using
spectral analysis. The encoder handles:

1. Raw audio waveforms sampled at arbitrary rates
2. Multi-channel audio (mono or stereo)
3. Time-frequency representations via STFT

Key features:
- Short-Time Fourier Transform (STFT) for spectral decomposition
- Local feature extraction via 1D convolution
- Resolution-invariant through hop_length parameter

The architecture:
1. STFT transform to time-frequency domain
2. Magnitude spectrum extraction
3. Linear projection to embedding space
4. 1D convolution for local temporal patterns
5. Normalization and dropout

This enables processing of continuous audio signals into a latent
representation suitable for multimodal integration.

Reference:
- Griffin & Lim "Signal Estimation from Modified Short-Time Fourier Transform" (1984)
- van den Oord et al. "WaveNet: A Generative Model for Raw Audio" (2016)
"""

import torch
import torch.nn as nn


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
