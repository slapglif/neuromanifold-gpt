# neuromanifold_gpt/model/embeddings/spectral.py
"""
Spectral Token Embedding - Wave-based input encoding.

Inspired by:
- AFNO (Adaptive Fourier Neural Operator) token mixer
- Cortical waves paper: "Transformers and cortical waves: encoders for pulling in context"
- Physical intuition: Universe processes information as waves, not discrete tokens

Key idea: Map tokens to wave functions in frequency space, then process
through Fourier domain mixing. This naturally aligns with FHN dynamics
which propagate soliton waves.

Token → Fourier Embedding → Spectral Mixing → Wave Function Output

CAUSALITY WARNING:
    AFNOTokenMixer performs FFT over the ENTIRE sequence, which BREAKS
    autoregressive causality. Output at position t depends on all positions
    including t+1, t+2, etc. This module is NOT SUITABLE for autoregressive
    language modeling unless:
    1. Used only at embedding stage before causal attention
    2. Modified to use causal convolutions or windowed FFT
    3. Used in bidirectional/encoder-only contexts

    SpectralTokenEmbedding is causal (per-token processing).
    WaveInputEncoder with use_afno_mixer=False is causal.

STATUS: NOT INTEGRATED - This module is experimental and not currently
    used in NeuroManifoldGPT. The model uses standard embeddings.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange


class SpectralTokenEmbedding(nn.Module):
    """Wave-based token embedding via spectral decomposition.

    Instead of lookup table → dense vector, we:
    1. Embed tokens into frequency modes (Fourier basis)
    2. Apply learnable spectral mixing
    3. Output wave function representation

    This creates continuous, smooth embeddings that naturally
    support interference patterns (like attention).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        n_modes: int = 32,  # Number of Fourier modes to keep
        mode_init: str = "random",  # "random" or "frequency"
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_modes = min(n_modes, embed_dim // 2)

        # Token to frequency coefficients (real + imaginary)
        # Each token gets a set of complex Fourier coefficients
        self.freq_real = nn.Embedding(vocab_size, self.n_modes)
        self.freq_imag = nn.Embedding(vocab_size, self.n_modes)

        # Learnable frequency weights (soft-thresholding like AFNO)
        self.mode_weights = nn.Parameter(torch.ones(self.n_modes))

        # Project from frequency domain to embedding space
        self.freq_to_embed = nn.Linear(self.n_modes * 2, embed_dim)

        # Phase offset per mode (learnable)
        self.phase_offset = nn.Parameter(torch.zeros(self.n_modes))

        self._init_weights(mode_init)

    def _init_weights(self, mode_init: str):
        if mode_init == "frequency":
            # Initialize to approximate Fourier basis
            for i in range(self.n_modes):
                freq = (i + 1) * math.pi / self.n_modes
                self.freq_real.weight.data[:, i] = (
                    torch.cos(freq * torch.arange(self.vocab_size).float()) * 0.1
                )
                self.freq_imag.weight.data[:, i] = (
                    torch.sin(freq * torch.arange(self.vocab_size).float()) * 0.1
                )
        else:
            nn.init.normal_(self.freq_real.weight, std=0.1)
            nn.init.normal_(self.freq_imag.weight, std=0.1)

        nn.init.xavier_uniform_(self.freq_to_embed.weight)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed tokens as wave functions.

        Args:
            tokens: (B, T) token indices

        Returns:
            embeddings: (B, T, embed_dim) wave function embeddings
        """
        B, T = tokens.shape

        # Get frequency coefficients for each token
        real = self.freq_real(tokens)  # (B, T, n_modes)
        imag = self.freq_imag(tokens)  # (B, T, n_modes)

        # Apply soft-thresholding on modes (like AFNO sparsification)
        weights = F.softplus(self.mode_weights)
        real = real * weights
        imag = imag * weights

        # Add phase offset (allows learning phase relationships)
        phase = self.phase_offset.unsqueeze(0).unsqueeze(0)  # (1, 1, n_modes)
        real_rotated = real * torch.cos(phase) - imag * torch.sin(phase)
        imag_rotated = real * torch.sin(phase) + imag * torch.cos(phase)

        # Concatenate real and imaginary for projection
        freq_features = torch.cat(
            [real_rotated, imag_rotated], dim=-1
        )  # (B, T, 2*n_modes)

        # Project to embedding space
        embeddings = self.freq_to_embed(freq_features)  # (B, T, embed_dim)

        return embeddings


class AFNOTokenMixer(nn.Module):
    """AFNO-style token mixing in Fourier domain.

    Applies global mixing via FFT → weight → IFFT.
    This is like attention but operates in frequency space.

    Key advantages:
    - O(T log T) complexity vs O(T²) for attention
    - Naturally captures periodic patterns
    - Aligns with wave-based processing paradigm

    CAUSALITY WARNING:
        This module BREAKS autoregressive causality! The FFT is computed
        over the entire sequence dimension, so output at position t depends
        on ALL positions in the sequence (past AND future).

        DO NOT use in autoregressive language models without modification.
        For causal AFNO, consider:
        - Causal convolutions instead of FFT
        - Windowed/local FFT
        - Using only at embedding layer before causal attention
    """

    def __init__(
        self,
        embed_dim: int,
        n_modes: int = 32,
        mlp_ratio: float = 4.0,
        sparsity_threshold: float = 0.01,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_modes = n_modes
        self.sparsity_threshold = sparsity_threshold

        # Complex weights for each frequency mode
        # Block-diagonal structure for efficiency
        self.block_size = embed_dim // 8  # 8 blocks
        self.n_blocks = 8

        # Per-mode learnable complex weights (real and imaginary)
        self.weights_real = nn.Parameter(
            torch.randn(self.n_blocks, n_modes, self.block_size, self.block_size) * 0.02
        )
        self.weights_imag = nn.Parameter(
            torch.randn(self.n_blocks, n_modes, self.block_size, self.block_size) * 0.02
        )

        # Soft-thresholding parameters
        self.softshrink = nn.Softshrink(sparsity_threshold)

        # MLP for non-linear processing after frequency mixing
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply AFNO token mixing.

        Args:
            x: (B, T, D) input embeddings

        Returns:
            mixed: (B, T, D) mixed embeddings
        """
        B, T, D = x.shape
        residual = x

        # Reshape for block-diagonal processing
        x = rearrange(
            x, "b t (n d) -> b n t d", n=self.n_blocks
        )  # (B, n_blocks, T, block_size)

        # FFT along sequence dimension
        x_ft = torch.fft.rfft(
            x, dim=2, norm="ortho"
        )  # (B, n_blocks, T//2+1, block_size)
        freq_len = x_ft.shape[2]

        # Only process up to n_modes frequencies
        n_modes = min(self.n_modes, freq_len)

        # Apply complex multiplication in frequency domain (VECTORIZED)
        # W_real * x_real - W_imag * x_imag + i(W_real * x_imag + W_imag * x_real)
        out_ft = torch.zeros_like(x_ft)

        # Extract first n_modes for batched processing
        x_modes = x_ft[:, :, :n_modes, :]  # (B, n_blocks, n_modes, block_size)
        x_real = x_modes.real
        x_imag = x_modes.imag

        # Weights: (n_blocks, n_modes, block_size, block_size)
        W_real = self.weights_real[:, :n_modes, :, :]
        W_imag = self.weights_imag[:, :n_modes, :, :]

        # Batched complex multiplication via einsum
        # n=n_blocks, m=n_modes, i,j=block_size, b=batch
        out_real = einsum(W_real, x_real, "n m i j, b n m j -> b n m i") - einsum(
            W_imag, x_imag, "n m i j, b n m j -> b n m i"
        )
        out_imag = einsum(W_real, x_imag, "n m i j, b n m j -> b n m i") + einsum(
            W_imag, x_real, "n m i j, b n m j -> b n m i"
        )

        # Apply soft-thresholding for sparsity
        out_real = self.softshrink(out_real)
        out_imag = self.softshrink(out_imag)

        # Write back to out_ft
        out_ft[:, :, :n_modes, :] = torch.complex(out_real, out_imag)

        # IFFT back to sequence domain
        x = torch.fft.irfft(
            out_ft, n=T, dim=2, norm="ortho"
        )  # (B, n_blocks, T, block_size)

        # Reshape back
        x = rearrange(x, "b n t d -> b t (n d)")  # (B, T, D)

        # Add residual and apply MLP
        x = x + residual
        x = self.norm(x)
        x = x + self.mlp(x)

        return x


class WaveInputEncoder(nn.Module):
    """Complete wave-based input encoder.

    Combines:
    1. SpectralTokenEmbedding: Tokens → wave functions
    2. AFNOTokenMixer: Global mixing in Fourier domain
    3. Positional encoding via phase modulation

    This creates input representations that are:
    - Continuous and smooth (wave functions)
    - Globally aware (Fourier mixing)
    - Naturally compatible with FHN dynamics
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_seq_len: int = 1024,
        n_modes: int = 32,
        use_afno_mixer: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Spectral token embedding
        self.token_embed = SpectralTokenEmbedding(
            vocab_size, embed_dim, n_modes=n_modes
        )

        # Position as phase modulation
        self.pos_phases = nn.Parameter(torch.randn(max_seq_len, embed_dim) * 0.02)

        # Optional AFNO mixer for pre-mixing
        self.use_afno_mixer = use_afno_mixer
        if use_afno_mixer:
            self.afno = AFNOTokenMixer(embed_dim, n_modes=n_modes)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Encode tokens as wave functions.

        Args:
            tokens: (B, T) token indices

        Returns:
            wave_embed: (B, T, D) wave function embeddings
        """
        B, T = tokens.shape

        # Get spectral embeddings
        x = self.token_embed(tokens)  # (B, T, D)

        # Add positional phase modulation
        pos_phase = self.pos_phases[:T]  # (T, D)
        # Modulate as phase: x * cos(pos) + x.roll * sin(pos)
        x = x * torch.cos(pos_phase) + x.roll(1, dims=-1) * torch.sin(pos_phase)

        # Optional AFNO mixing
        if self.use_afno_mixer:
            x = self.afno(x)

        x = self.norm(x)

        return x
