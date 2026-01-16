# neuromanifold_gpt/tests/test_fno.py
"""
Comprehensive tests for Fourier Neural Operator (FNO) components.

Tests cover:
- SpectralConvConfig: Configuration dataclass
- SpectralConv1d: 1D spectral convolution (core FNO operation)
- SpectralConv2d: 2D spectral convolution for images/grids
- SpectralConvNd: N-dimensional generalized spectral convolution
- FNOConfig: Configuration for FNO modules
- FNOBlock: Single FNO block with spectral + local paths
- FNOEncoder: Stack of FNO blocks
- FNOWithMLP: FNO block with feed-forward MLP
- MultimodalConfig: Configuration for multimodal encoder
- ByteEmbedding: Byte-level embedding with positional encoding
- AudioEncoder: STFT-based audio encoding
- ImagePatchEncoder: ViT-style patch embedding
- MultimodalFNOEncoder: Unified multimodal encoder with FNO
- MultimodalFusionEncoder: Cross-modal attention fusion
- sinusoidal_position_encoding: Position encoding utility
"""

import pytest
import torch
import torch.nn as nn
import math

from neuromanifold_gpt.model.fno import (
    # Spectral convolution
    SpectralConv,
    SpectralConv1d,
    SpectralConv2d,
    SpectralConvNd,
    SpectralConvConfig,
    # FNO blocks
    FNOBlock,
    FNOEncoder,
    FNOConfig,
    FNOWithMLP,
    # Multimodal encoder
    MultimodalFNOEncoder,
    MultimodalConfig,
    MultimodalFusionEncoder,
    ByteEmbedding,
    AudioEncoder,
    ImagePatchEncoder,
    sinusoidal_position_encoding,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def device():
    """Return CPU device for testing."""
    return torch.device("cpu")


@pytest.fixture
def dtype():
    """Return default dtype for testing."""
    return torch.float32


@pytest.fixture
def batch_size():
    """Standard batch size for tests."""
    return 2


@pytest.fixture
def seq_len():
    """Standard sequence length for tests."""
    return 32


@pytest.fixture
def embed_dim():
    """Standard embedding dimension for tests."""
    return 64


@pytest.fixture
def channels():
    """Standard number of channels for tests."""
    return 32


@pytest.fixture
def modes():
    """Standard number of Fourier modes for tests."""
    return 8


# ==============================================================================
# SpectralConvConfig Tests
# ==============================================================================

class TestSpectralConvConfig:
    """Tests for SpectralConvConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SpectralConvConfig()
        assert config.in_channels == 64
        assert config.out_channels == 64
        assert config.modes == 16
        assert config.modes2 == 16
        assert config.scale == 1.0
        assert config.bias is True
        assert config.dropout == 0.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = SpectralConvConfig(
            in_channels=32,
            out_channels=64,
            modes=8,
            modes2=8,
            scale=0.5,
            bias=False,
            dropout=0.1,
        )
        assert config.in_channels == 32
        assert config.out_channels == 64
        assert config.modes == 8
        assert config.modes2 == 8
        assert config.scale == 0.5
        assert config.bias is False
        assert config.dropout == 0.1

    def test_config_is_dataclass(self):
        """Test that SpectralConvConfig is a proper dataclass."""
        from dataclasses import is_dataclass
        assert is_dataclass(SpectralConvConfig)


# ==============================================================================
# SpectralConv1d Tests
# ==============================================================================

class TestSpectralConv1d:
    """Tests for SpectralConv1d."""

    def test_instantiation(self, channels, modes):
        """Test basic instantiation."""
        sc = SpectralConv1d(in_channels=channels, out_channels=channels, modes=modes)
        assert sc.in_channels == channels
        assert sc.out_channels == channels
        assert sc.modes == modes

    def test_instantiation_with_params(self):
        """Test instantiation with custom parameters."""
        sc = SpectralConv1d(
            in_channels=32,
            out_channels=64,
            modes=16,
            scale=0.5,
            bias=False,
            dropout=0.2,
        )
        assert sc.in_channels == 32
        assert sc.out_channels == 64
        assert sc.modes == 16
        assert sc.scale_init == 0.5
        assert sc.bias is None
        assert isinstance(sc.dropout, nn.Dropout)

    def test_forward_shape_same_channels(self, batch_size, channels, modes, device, dtype):
        """Test forward pass output shape with same in/out channels."""
        sc = SpectralConv1d(in_channels=channels, out_channels=channels, modes=modes)
        x = torch.randn(batch_size, channels, 64, device=device, dtype=dtype)
        y = sc(x)
        assert y.shape == x.shape

    def test_forward_shape_different_channels(self, batch_size, device, dtype):
        """Test forward pass with different in/out channels."""
        sc = SpectralConv1d(in_channels=32, out_channels=64, modes=8)
        x = torch.randn(batch_size, 32, 64, device=device, dtype=dtype)
        y = sc(x)
        assert y.shape == (batch_size, 64, 64)

    def test_forward_short_sequence(self, batch_size, channels, device, dtype):
        """Test forward pass with sequence shorter than modes."""
        sc = SpectralConv1d(in_channels=channels, out_channels=channels, modes=16)
        # Sequence length 8 is shorter than modes 16
        x = torch.randn(batch_size, channels, 8, device=device, dtype=dtype)
        y = sc(x)
        assert y.shape == x.shape

    def test_complex_multiply(self, channels, modes, device, dtype):
        """Test complex multiplication operation."""
        sc = SpectralConv1d(in_channels=channels, out_channels=channels, modes=modes)

        # Create complex input
        x_hat = torch.randn(2, channels, modes, device=device, dtype=dtype) + \
                1j * torch.randn(2, channels, modes, device=device, dtype=dtype)

        out = sc.complex_multiply(x_hat, sc.weight_real, sc.weight_imag)

        assert out.shape == (2, channels, modes)
        assert out.is_complex()

    def test_weight_shapes(self, channels, modes):
        """Test that weight tensors have correct shapes."""
        sc = SpectralConv1d(in_channels=channels, out_channels=channels, modes=modes)
        assert sc.weight_real.shape == (channels, channels, modes)
        assert sc.weight_imag.shape == (channels, channels, modes)

    def test_bias_shape(self, channels):
        """Test bias tensor shape when enabled."""
        sc = SpectralConv1d(in_channels=channels, out_channels=channels, modes=8, bias=True)
        assert sc.bias is not None
        assert sc.bias.shape == (channels,)

    def test_no_bias(self, channels):
        """Test that bias is None when disabled."""
        sc = SpectralConv1d(in_channels=channels, out_channels=channels, modes=8, bias=False)
        assert sc.bias is None

    def test_extra_repr(self, channels, modes):
        """Test string representation."""
        sc = SpectralConv1d(in_channels=channels, out_channels=channels, modes=modes)
        repr_str = sc.extra_repr()
        assert f'in_channels={channels}' in repr_str
        assert f'out_channels={channels}' in repr_str
        assert f'modes={modes}' in repr_str


# ==============================================================================
# SpectralConv2d Tests
# ==============================================================================

class TestSpectralConv2d:
    """Tests for SpectralConv2d."""

    def test_instantiation(self):
        """Test basic instantiation."""
        sc = SpectralConv2d(in_channels=3, out_channels=64, modes1=12, modes2=12)
        assert sc.in_channels == 3
        assert sc.out_channels == 64
        assert sc.modes1 == 12
        assert sc.modes2 == 12

    def test_forward_shape(self, batch_size, device, dtype):
        """Test forward pass output shape."""
        sc = SpectralConv2d(in_channels=3, out_channels=64, modes1=8, modes2=8)
        x = torch.randn(batch_size, 3, 32, 32, device=device, dtype=dtype)
        y = sc(x)
        assert y.shape == (batch_size, 64, 32, 32)

    def test_forward_same_channels(self, batch_size, device, dtype):
        """Test forward pass with same in/out channels."""
        sc = SpectralConv2d(in_channels=32, out_channels=32, modes1=8, modes2=8)
        x = torch.randn(batch_size, 32, 16, 16, device=device, dtype=dtype)
        y = sc(x)
        assert y.shape == x.shape

    def test_forward_small_image(self, batch_size, device, dtype):
        """Test forward pass with image smaller than modes."""
        sc = SpectralConv2d(in_channels=3, out_channels=64, modes1=16, modes2=16)
        # Image 8x8 is smaller than modes 16
        x = torch.randn(batch_size, 3, 8, 8, device=device, dtype=dtype)
        y = sc(x)
        assert y.shape == (batch_size, 64, 8, 8)

    def test_weight_shapes(self):
        """Test that weight tensors have correct shapes."""
        sc = SpectralConv2d(in_channels=3, out_channels=64, modes1=12, modes2=12)
        assert sc.weight1_real.shape == (3, 64, 12, 12)
        assert sc.weight1_imag.shape == (3, 64, 12, 12)
        assert sc.weight2_real.shape == (3, 64, 12, 12)
        assert sc.weight2_imag.shape == (3, 64, 12, 12)

    def test_extra_repr(self):
        """Test string representation."""
        sc = SpectralConv2d(in_channels=3, out_channels=64, modes1=12, modes2=8)
        repr_str = sc.extra_repr()
        assert 'in_channels=3' in repr_str
        assert 'out_channels=64' in repr_str
        assert '(12, 8)' in repr_str


# ==============================================================================
# SpectralConvNd Tests
# ==============================================================================

class TestSpectralConvNd:
    """Tests for SpectralConvNd."""

    def test_instantiation_1d(self):
        """Test 1D instantiation."""
        sc = SpectralConvNd(in_channels=32, out_channels=32, modes=(8,))
        assert sc.in_channels == 32
        assert sc.out_channels == 32
        assert sc.modes == (8,)
        assert sc.ndim == 1

    def test_instantiation_2d(self):
        """Test 2D instantiation."""
        sc = SpectralConvNd(in_channels=32, out_channels=64, modes=(8, 8))
        assert sc.ndim == 2

    def test_instantiation_3d(self):
        """Test 3D instantiation."""
        sc = SpectralConvNd(in_channels=16, out_channels=32, modes=(4, 4, 4))
        assert sc.ndim == 3

    def test_forward_1d(self, batch_size, device, dtype):
        """Test forward pass for 1D."""
        sc = SpectralConvNd(in_channels=32, out_channels=32, modes=(8,))
        x = torch.randn(batch_size, 32, 32, device=device, dtype=dtype)
        y = sc(x)
        assert y.shape == x.shape

    def test_forward_2d(self, batch_size, device, dtype):
        """Test forward pass for 2D."""
        sc = SpectralConvNd(in_channels=3, out_channels=64, modes=(8, 8))
        x = torch.randn(batch_size, 3, 16, 16, device=device, dtype=dtype)
        y = sc(x)
        assert y.shape == (batch_size, 64, 16, 16)


# ==============================================================================
# FNOConfig Tests
# ==============================================================================

class TestFNOConfig:
    """Tests for FNOConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = FNOConfig()
        assert config.embed_dim == 384
        assert config.modes == 32
        assert config.n_layers == 4
        assert config.activation == "gelu"
        assert config.dropout == 0.1
        assert config.residual is True
        assert config.prenorm is True
        assert config.local_expansion == 4
        assert config.bias is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = FNOConfig(
            embed_dim=256,
            modes=16,
            n_layers=6,
            activation="silu",
            dropout=0.2,
            residual=False,
            prenorm=False,
        )
        assert config.embed_dim == 256
        assert config.modes == 16
        assert config.n_layers == 6
        assert config.activation == "silu"
        assert config.dropout == 0.2
        assert config.residual is False
        assert config.prenorm is False

    def test_config_is_dataclass(self):
        """Test that FNOConfig is a proper dataclass."""
        from dataclasses import is_dataclass
        assert is_dataclass(FNOConfig)


# ==============================================================================
# FNOBlock Tests
# ==============================================================================

class TestFNOBlock:
    """Tests for FNOBlock."""

    def test_instantiation(self, embed_dim, modes):
        """Test basic instantiation."""
        fno = FNOBlock(embed_dim=embed_dim, modes=modes)
        assert fno.embed_dim == embed_dim
        assert fno.modes == modes

    def test_instantiation_with_params(self):
        """Test instantiation with custom parameters."""
        fno = FNOBlock(
            embed_dim=256,
            modes=16,
            dropout=0.2,
            activation="silu",
            residual=False,
            prenorm=False,
        )
        assert fno.embed_dim == 256
        assert fno.modes == 16
        assert fno.residual is False
        assert fno.prenorm is False

    def test_forward_shape(self, batch_size, seq_len, embed_dim, modes, device, dtype):
        """Test forward pass output shape."""
        fno = FNOBlock(embed_dim=embed_dim, modes=modes)
        x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
        y = fno(x)
        assert y.shape == x.shape

    def test_forward_return_spectral(self, batch_size, seq_len, embed_dim, modes, device, dtype):
        """Test forward pass with return_spectral=True."""
        fno = FNOBlock(embed_dim=embed_dim, modes=modes)
        x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
        y, spectral = fno(x, return_spectral=True)
        assert y.shape == x.shape
        assert spectral.shape == x.shape

    def test_residual_connection(self, batch_size, seq_len, embed_dim, modes, device, dtype):
        """Test that residual connection works."""
        fno_res = FNOBlock(embed_dim=embed_dim, modes=modes, residual=True)
        fno_no_res = FNOBlock(embed_dim=embed_dim, modes=modes, residual=False)

        x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
        y_res = fno_res(x)
        y_no_res = fno_no_res(x)

        # Both should produce valid outputs
        assert y_res.shape == x.shape
        assert y_no_res.shape == x.shape

    def test_prenorm_vs_postnorm(self, batch_size, seq_len, embed_dim, modes, device, dtype):
        """Test pre-normalization vs post-normalization."""
        fno_pre = FNOBlock(embed_dim=embed_dim, modes=modes, prenorm=True)
        fno_post = FNOBlock(embed_dim=embed_dim, modes=modes, prenorm=False)

        x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
        y_pre = fno_pre(x)
        y_post = fno_post(x)

        assert y_pre.shape == x.shape
        assert y_post.shape == x.shape

    def test_different_activations(self, batch_size, seq_len, embed_dim, modes, device, dtype):
        """Test different activation functions."""
        activations = ["gelu", "relu", "silu", "tanh"]
        x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)

        for act_name in activations:
            fno = FNOBlock(embed_dim=embed_dim, modes=modes, activation=act_name)
            y = fno(x)
            assert y.shape == x.shape

    def test_dropout(self, batch_size, seq_len, embed_dim, modes, device, dtype):
        """Test dropout is applied during training."""
        fno = FNOBlock(embed_dim=embed_dim, modes=modes, dropout=0.5)
        fno.train()

        x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
        y1 = fno(x)
        y2 = fno(x)

        # Both should have same shape
        assert y1.shape == x.shape
        assert y2.shape == x.shape

    def test_extra_repr(self, embed_dim, modes):
        """Test string representation."""
        fno = FNOBlock(embed_dim=embed_dim, modes=modes)
        repr_str = fno.extra_repr()
        assert f'embed_dim={embed_dim}' in repr_str
        assert f'modes={modes}' in repr_str


# ==============================================================================
# FNOEncoder Tests
# ==============================================================================

class TestFNOEncoder:
    """Tests for FNOEncoder."""

    def test_instantiation(self, embed_dim, modes):
        """Test basic instantiation."""
        enc = FNOEncoder(embed_dim=embed_dim, n_layers=4, modes=modes)
        assert enc.embed_dim == embed_dim
        assert enc.n_layers == 4
        assert enc.modes == modes
        assert len(enc.blocks) == 4

    def test_forward_shape(self, batch_size, seq_len, embed_dim, modes, device, dtype):
        """Test forward pass output shape."""
        enc = FNOEncoder(embed_dim=embed_dim, n_layers=2, modes=modes)
        x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
        y = enc(x)
        assert y.shape == x.shape

    def test_forward_return_all_layers(self, batch_size, seq_len, embed_dim, modes, device, dtype):
        """Test forward with return_all_layers=True."""
        n_layers = 3
        enc = FNOEncoder(embed_dim=embed_dim, n_layers=n_layers, modes=modes)
        x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
        y, layer_outputs = enc(x, return_all_layers=True)

        assert y.shape == x.shape
        assert len(layer_outputs) == n_layers
        for layer_out in layer_outputs:
            assert layer_out.shape == x.shape

    def test_single_layer(self, batch_size, seq_len, embed_dim, modes, device, dtype):
        """Test encoder with single layer."""
        enc = FNOEncoder(embed_dim=embed_dim, n_layers=1, modes=modes)
        x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
        y = enc(x)
        assert y.shape == x.shape

    def test_many_layers(self, batch_size, seq_len, embed_dim, modes, device, dtype):
        """Test encoder with many layers."""
        enc = FNOEncoder(embed_dim=embed_dim, n_layers=8, modes=modes)
        x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
        y = enc(x)
        assert y.shape == x.shape

    def test_extra_repr(self, embed_dim, modes):
        """Test string representation."""
        enc = FNOEncoder(embed_dim=embed_dim, n_layers=4, modes=modes)
        repr_str = enc.extra_repr()
        assert f'embed_dim={embed_dim}' in repr_str
        assert 'n_layers=4' in repr_str
        assert f'modes={modes}' in repr_str


# ==============================================================================
# FNOWithMLP Tests
# ==============================================================================

class TestFNOWithMLP:
    """Tests for FNOWithMLP."""

    def test_instantiation(self, embed_dim, modes):
        """Test basic instantiation."""
        block = FNOWithMLP(embed_dim=embed_dim, modes=modes)
        assert block.embed_dim == embed_dim
        assert block.modes == modes

    def test_instantiation_with_params(self):
        """Test instantiation with custom parameters."""
        block = FNOWithMLP(
            embed_dim=256,
            modes=16,
            mlp_ratio=2,
            dropout=0.2,
            activation="silu",
        )
        assert block.embed_dim == 256
        assert block.modes == 16
        assert block.mlp_ratio == 2

    def test_forward_shape(self, batch_size, seq_len, embed_dim, modes, device, dtype):
        """Test forward pass output shape."""
        block = FNOWithMLP(embed_dim=embed_dim, modes=modes)
        x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
        y = block(x)
        assert y.shape == x.shape

    def test_mlp_expansion(self, batch_size, seq_len, device, dtype):
        """Test different MLP expansion ratios."""
        embed_dim = 64
        for mlp_ratio in [1, 2, 4, 8]:
            block = FNOWithMLP(embed_dim=embed_dim, modes=8, mlp_ratio=mlp_ratio)
            x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
            y = block(x)
            assert y.shape == x.shape

    def test_extra_repr(self, embed_dim, modes):
        """Test string representation."""
        block = FNOWithMLP(embed_dim=embed_dim, modes=modes, mlp_ratio=4)
        repr_str = block.extra_repr()
        assert f'embed_dim={embed_dim}' in repr_str
        assert f'modes={modes}' in repr_str
        assert 'mlp_ratio=4' in repr_str


# ==============================================================================
# MultimodalConfig Tests
# ==============================================================================

class TestMultimodalConfig:
    """Tests for MultimodalConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = MultimodalConfig()
        assert config.vocab_size == 256
        assert config.embed_dim == 384
        assert config.max_seq_len == 2048
        assert config.fno_layers == 4
        assert config.fno_modes == 32
        assert config.dropout == 0.1
        assert config.use_learned_pos is True
        assert config.use_sinusoidal_pos is True
        assert config.n_modalities == 3

    def test_custom_values(self):
        """Test custom configuration values."""
        config = MultimodalConfig(
            vocab_size=512,
            embed_dim=256,
            max_seq_len=1024,
            fno_layers=2,
            fno_modes=16,
        )
        assert config.vocab_size == 512
        assert config.embed_dim == 256
        assert config.max_seq_len == 1024
        assert config.fno_layers == 2
        assert config.fno_modes == 16

    def test_config_is_dataclass(self):
        """Test that MultimodalConfig is a proper dataclass."""
        from dataclasses import is_dataclass
        assert is_dataclass(MultimodalConfig)


# ==============================================================================
# sinusoidal_position_encoding Tests
# ==============================================================================

class TestSinusoidalPositionEncoding:
    """Tests for sinusoidal_position_encoding function."""

    def test_output_shape(self):
        """Test output shape."""
        pe = sinusoidal_position_encoding(seq_len=100, embed_dim=64)
        assert pe.shape == (100, 64)

    def test_output_device(self, device):
        """Test output device."""
        pe = sinusoidal_position_encoding(seq_len=100, embed_dim=64, device=device)
        assert pe.device == device

    def test_output_dtype(self, dtype):
        """Test output dtype."""
        pe = sinusoidal_position_encoding(seq_len=100, embed_dim=64, dtype=dtype)
        assert pe.dtype == dtype

    def test_different_positions_differ(self):
        """Test that different positions have different encodings."""
        pe = sinusoidal_position_encoding(seq_len=100, embed_dim=64)
        # Different positions should have different encodings
        assert not torch.allclose(pe[0], pe[1])
        assert not torch.allclose(pe[0], pe[50])

    def test_odd_embed_dim(self):
        """Test with odd embedding dimension."""
        pe = sinusoidal_position_encoding(seq_len=100, embed_dim=65)
        assert pe.shape == (100, 65)


# ==============================================================================
# ByteEmbedding Tests
# ==============================================================================

class TestByteEmbedding:
    """Tests for ByteEmbedding."""

    def test_instantiation(self, embed_dim):
        """Test basic instantiation."""
        emb = ByteEmbedding(vocab_size=256, embed_dim=embed_dim)
        assert emb.vocab_size == 256
        assert emb.embed_dim == embed_dim

    def test_instantiation_with_params(self):
        """Test instantiation with custom parameters."""
        emb = ByteEmbedding(
            vocab_size=512,
            embed_dim=256,
            max_seq_len=1024,
            dropout=0.2,
            use_learned_pos=False,
            use_sinusoidal_pos=False,
        )
        assert emb.vocab_size == 512
        assert emb.embed_dim == 256
        assert emb.max_seq_len == 1024
        assert emb.use_learned_pos is False
        assert emb.use_sinusoidal_pos is False
        assert emb.pos_embed is None

    def test_forward_shape(self, batch_size, seq_len, embed_dim, device):
        """Test forward pass output shape."""
        emb = ByteEmbedding(vocab_size=256, embed_dim=embed_dim, max_seq_len=512)
        x = torch.randint(0, 256, (batch_size, seq_len), device=device)
        y = emb(x)
        assert y.shape == (batch_size, seq_len, embed_dim)

    def test_forward_with_positions(self, batch_size, seq_len, embed_dim, device):
        """Test forward pass with custom positions."""
        emb = ByteEmbedding(vocab_size=256, embed_dim=embed_dim, max_seq_len=512)
        x = torch.randint(0, 256, (batch_size, seq_len), device=device)
        positions = torch.arange(seq_len, device=device)
        y = emb(x, positions=positions)
        assert y.shape == (batch_size, seq_len, embed_dim)

    def test_learned_pos_embedding(self, embed_dim):
        """Test learned position embedding."""
        emb = ByteEmbedding(
            vocab_size=256,
            embed_dim=embed_dim,
            use_learned_pos=True,
            use_sinusoidal_pos=False,
        )
        assert emb.pos_embed is not None
        assert emb.pos_embed.weight.shape == (emb.max_seq_len, embed_dim)

    def test_sinusoidal_pos_encoding(self, embed_dim):
        """Test sinusoidal position encoding."""
        emb = ByteEmbedding(
            vocab_size=256,
            embed_dim=embed_dim,
            use_learned_pos=False,
            use_sinusoidal_pos=True,
        )
        assert emb.sinusoidal_pe is not None
        assert emb.sinusoidal_pe.shape == (emb.max_seq_len, embed_dim)

    def test_extra_repr(self, embed_dim):
        """Test string representation."""
        emb = ByteEmbedding(vocab_size=256, embed_dim=embed_dim)
        repr_str = emb.extra_repr()
        assert 'vocab_size=256' in repr_str
        assert f'embed_dim={embed_dim}' in repr_str


# ==============================================================================
# AudioEncoder Tests
# ==============================================================================

class TestAudioEncoder:
    """Tests for AudioEncoder."""

    def test_instantiation(self, embed_dim):
        """Test basic instantiation."""
        enc = AudioEncoder(embed_dim=embed_dim)
        assert enc.embed_dim == embed_dim

    def test_instantiation_with_params(self):
        """Test instantiation with custom parameters."""
        enc = AudioEncoder(
            embed_dim=256,
            n_channels=2,
            n_fft=1024,
            hop_length=256,
            dropout=0.2,
        )
        assert enc.embed_dim == 256
        assert enc.n_channels == 2
        assert enc.n_fft == 1024
        assert enc.hop_length == 256

    def test_forward_shape_mono(self, batch_size, embed_dim, device, dtype):
        """Test forward pass with mono audio."""
        enc = AudioEncoder(embed_dim=embed_dim, n_fft=256, hop_length=64)
        # Mono audio: (batch, n_samples)
        x = torch.randn(batch_size, 1024, device=device, dtype=dtype)
        y = enc(x)
        assert y.dim() == 3
        assert y.shape[0] == batch_size
        assert y.shape[2] == embed_dim

    def test_forward_shape_stereo(self, batch_size, embed_dim, device, dtype):
        """Test forward pass with stereo audio."""
        enc = AudioEncoder(embed_dim=embed_dim, n_channels=2, n_fft=256, hop_length=64)
        # Stereo audio: (batch, channels, n_samples)
        x = torch.randn(batch_size, 2, 1024, device=device, dtype=dtype)
        y = enc(x)
        assert y.dim() == 3
        assert y.shape[0] == batch_size
        assert y.shape[2] == embed_dim

    def test_forward_shape_with_channel_dim(self, batch_size, embed_dim, device, dtype):
        """Test forward pass with explicit channel dimension."""
        enc = AudioEncoder(embed_dim=embed_dim, n_channels=1, n_fft=256, hop_length=64)
        x = torch.randn(batch_size, 1, 1024, device=device, dtype=dtype)
        y = enc(x)
        assert y.dim() == 3
        assert y.shape[2] == embed_dim


# ==============================================================================
# ImagePatchEncoder Tests
# ==============================================================================

class TestImagePatchEncoder:
    """Tests for ImagePatchEncoder."""

    def test_instantiation(self, embed_dim):
        """Test basic instantiation."""
        enc = ImagePatchEncoder(embed_dim=embed_dim)
        assert enc.embed_dim == embed_dim

    def test_instantiation_with_params(self):
        """Test instantiation with custom parameters."""
        enc = ImagePatchEncoder(
            embed_dim=256,
            image_channels=1,
            patch_size=8,
            dropout=0.2,
        )
        assert enc.embed_dim == 256
        assert enc.image_channels == 1
        assert enc.patch_size == 8

    def test_forward_shape(self, batch_size, embed_dim, device, dtype):
        """Test forward pass output shape."""
        enc = ImagePatchEncoder(embed_dim=embed_dim, patch_size=8)
        # Image: (batch, channels, height, width)
        x = torch.randn(batch_size, 3, 32, 32, device=device, dtype=dtype)
        y = enc(x)
        # n_patches = (32/8) * (32/8) = 16
        assert y.shape == (batch_size, 16, embed_dim)

    def test_forward_different_image_sizes(self, batch_size, embed_dim, device, dtype):
        """Test forward pass with different image sizes."""
        enc = ImagePatchEncoder(embed_dim=embed_dim, patch_size=8)

        # 64x64 image -> 64 patches
        x1 = torch.randn(batch_size, 3, 64, 64, device=device, dtype=dtype)
        y1 = enc(x1)
        assert y1.shape == (batch_size, 64, embed_dim)

        # 48x32 image -> 24 patches
        x2 = torch.randn(batch_size, 3, 48, 32, device=device, dtype=dtype)
        y2 = enc(x2)
        assert y2.shape == (batch_size, 24, embed_dim)

    def test_forward_grayscale(self, batch_size, embed_dim, device, dtype):
        """Test forward pass with grayscale images."""
        enc = ImagePatchEncoder(embed_dim=embed_dim, image_channels=1, patch_size=8)
        x = torch.randn(batch_size, 1, 32, 32, device=device, dtype=dtype)
        y = enc(x)
        assert y.shape == (batch_size, 16, embed_dim)


# ==============================================================================
# MultimodalFNOEncoder Tests
# ==============================================================================

class TestMultimodalFNOEncoder:
    """Tests for MultimodalFNOEncoder."""

    def test_instantiation(self, embed_dim):
        """Test basic instantiation."""
        enc = MultimodalFNOEncoder(vocab_size=256, embed_dim=embed_dim)
        assert enc.vocab_size == 256
        assert enc.embed_dim == embed_dim

    def test_instantiation_with_params(self):
        """Test instantiation with custom parameters."""
        enc = MultimodalFNOEncoder(
            vocab_size=512,
            embed_dim=256,
            max_seq_len=1024,
            fno_layers=2,
            fno_modes=16,
            dropout=0.2,
        )
        assert enc.vocab_size == 512
        assert enc.embed_dim == 256
        assert enc.max_seq_len == 1024

    def test_forward_bytes_shape(self, batch_size, seq_len, embed_dim, device):
        """Test forward pass with bytes input."""
        enc = MultimodalFNOEncoder(vocab_size=256, embed_dim=embed_dim, fno_layers=1, fno_modes=8)
        x = torch.randint(0, 256, (batch_size, seq_len), device=device)
        y = enc(x, modality='bytes')
        assert y.shape == (batch_size, seq_len, embed_dim)

    def test_forward_bytes_default(self, batch_size, seq_len, embed_dim, device):
        """Test that default modality is bytes."""
        enc = MultimodalFNOEncoder(vocab_size=256, embed_dim=embed_dim, fno_layers=1, fno_modes=8)
        x = torch.randint(0, 256, (batch_size, seq_len), device=device)
        y = enc(x)  # Default modality
        assert y.shape == (batch_size, seq_len, embed_dim)

    def test_forward_return_fno_features(self, batch_size, seq_len, embed_dim, device):
        """Test forward with return_fno_features=True."""
        enc = MultimodalFNOEncoder(vocab_size=256, embed_dim=embed_dim, fno_layers=1, fno_modes=8)
        x = torch.randint(0, 256, (batch_size, seq_len), device=device)
        y, fno_features = enc(x, return_fno_features=True)
        assert y.shape == (batch_size, seq_len, embed_dim)
        assert fno_features.shape == (batch_size, seq_len, embed_dim)

    def test_encode_bytes_convenience(self, batch_size, seq_len, embed_dim, device):
        """Test encode_bytes convenience method."""
        enc = MultimodalFNOEncoder(vocab_size=256, embed_dim=embed_dim, fno_layers=1, fno_modes=8)
        x = torch.randint(0, 256, (batch_size, seq_len), device=device)
        y = enc.encode_bytes(x)
        assert y.shape == (batch_size, seq_len, embed_dim)

    def test_encode_image_convenience(self, batch_size, embed_dim, device, dtype):
        """Test encode_image convenience method."""
        enc = MultimodalFNOEncoder(
            vocab_size=256,
            embed_dim=embed_dim,
            fno_layers=1,
            fno_modes=8,
            image_patch_size=8,
        )
        x = torch.randn(batch_size, 3, 32, 32, device=device, dtype=dtype)
        y = enc.encode_image(x)
        # n_patches = 16
        assert y.shape == (batch_size, 16, embed_dim)

    def test_invalid_modality(self, batch_size, seq_len, embed_dim, device):
        """Test that invalid modality raises error."""
        enc = MultimodalFNOEncoder(vocab_size=256, embed_dim=embed_dim, fno_layers=1, fno_modes=8)
        x = torch.randint(0, 256, (batch_size, seq_len), device=device)
        with pytest.raises(ValueError, match="Unknown modality"):
            enc(x, modality='video')

    def test_get_num_params(self, embed_dim):
        """Test get_num_params method."""
        enc = MultimodalFNOEncoder(vocab_size=256, embed_dim=embed_dim, fno_layers=1, fno_modes=8)
        n_params_all = enc.get_num_params(non_embedding=False)
        n_params_non_emb = enc.get_num_params(non_embedding=True)
        assert n_params_all > n_params_non_emb
        assert n_params_non_emb > 0

    def test_extra_repr(self, embed_dim):
        """Test string representation."""
        enc = MultimodalFNOEncoder(vocab_size=256, embed_dim=embed_dim)
        repr_str = enc.extra_repr()
        assert 'vocab_size=256' in repr_str
        assert f'embed_dim={embed_dim}' in repr_str


# ==============================================================================
# MultimodalFusionEncoder Tests
# ==============================================================================

class TestMultimodalFusionEncoder:
    """Tests for MultimodalFusionEncoder."""

    def test_instantiation(self, embed_dim):
        """Test basic instantiation."""
        enc = MultimodalFusionEncoder(vocab_size=256, embed_dim=embed_dim)
        assert enc.embed_dim == embed_dim

    def test_instantiation_with_params(self):
        """Test instantiation with custom parameters."""
        enc = MultimodalFusionEncoder(
            vocab_size=512,
            embed_dim=256,
            n_heads=4,
            fno_layers=2,
            fno_modes=16,
            dropout=0.2,
        )
        assert enc.embed_dim == 256
        assert enc.n_heads == 4

    def test_forward_bytes_to_image(self, batch_size, seq_len, embed_dim, device, dtype):
        """Test cross-modal fusion: bytes query, image context."""
        enc = MultimodalFusionEncoder(
            vocab_size=256,
            embed_dim=embed_dim,
            n_heads=4,
            fno_layers=1,
            fno_modes=8,
        )

        query = torch.randint(0, 256, (batch_size, seq_len), device=device)
        context = torch.randn(batch_size, 3, 32, 32, device=device, dtype=dtype)

        y = enc(
            query_input=query,
            context_input=context,
            query_modality='bytes',
            context_modality='image',
        )
        assert y.shape == (batch_size, seq_len, embed_dim)


# ==============================================================================
# Edge Cases and Integration Tests
# ==============================================================================

class TestFNOEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_zero_input_spectral_conv(self, device, dtype):
        """Test SpectralConv1d with zero input."""
        sc = SpectralConv1d(in_channels=32, out_channels=32, modes=8)
        x = torch.zeros(2, 32, 64, device=device, dtype=dtype)
        y = sc(x)
        assert torch.isfinite(y).all()

    def test_zero_input_fno_block(self, device, dtype):
        """Test FNOBlock with zero input."""
        fno = FNOBlock(embed_dim=64, modes=8)
        x = torch.zeros(2, 32, 64, device=device, dtype=dtype)
        y = fno(x)
        assert torch.isfinite(y).all()

    def test_large_input_stability(self, device, dtype):
        """Test numerical stability with large inputs."""
        fno = FNOBlock(embed_dim=64, modes=8)
        x = torch.randn(2, 32, 64, device=device, dtype=dtype) * 100
        y = fno(x)
        assert torch.isfinite(y).all()

    def test_batch_size_one(self, device, dtype):
        """Test with batch size of 1."""
        fno = FNOBlock(embed_dim=64, modes=8)
        x = torch.randn(1, 32, 64, device=device, dtype=dtype)
        y = fno(x)
        assert y.shape == x.shape

    def test_sequence_length_one(self, device, dtype):
        """Test with sequence length of 1."""
        fno = FNOBlock(embed_dim=64, modes=8)
        x = torch.randn(2, 1, 64, device=device, dtype=dtype)
        y = fno(x)
        assert y.shape == x.shape

    def test_sequence_length_power_of_two(self, device, dtype):
        """Test with power-of-two sequence length for optimal FFT."""
        fno = FNOBlock(embed_dim=64, modes=16)
        for seq_len in [16, 32, 64, 128]:
            x = torch.randn(2, seq_len, 64, device=device, dtype=dtype)
            y = fno(x)
            assert y.shape == x.shape

    def test_sequence_length_non_power_of_two(self, device, dtype):
        """Test with non-power-of-two sequence length."""
        fno = FNOBlock(embed_dim=64, modes=8)
        for seq_len in [17, 33, 65, 100]:
            x = torch.randn(2, seq_len, 64, device=device, dtype=dtype)
            y = fno(x)
            assert y.shape == x.shape


# ==============================================================================
# Gradient Flow Tests
# ==============================================================================

class TestFNOGradientFlow:
    """Tests for gradient flow through FNO components."""

    def test_gradient_flow_spectral_conv1d(self, device, dtype):
        """Test gradients flow through SpectralConv1d."""
        sc = SpectralConv1d(in_channels=32, out_channels=32, modes=8)
        x = torch.randn(2, 32, 64, device=device, dtype=dtype, requires_grad=True)
        y = sc(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_gradient_flow_spectral_conv2d(self, device, dtype):
        """Test gradients flow through SpectralConv2d."""
        sc = SpectralConv2d(in_channels=3, out_channels=32, modes1=4, modes2=4)
        x = torch.randn(2, 3, 16, 16, device=device, dtype=dtype, requires_grad=True)
        y = sc(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_gradient_flow_fno_block(self, device, dtype):
        """Test gradients flow through FNOBlock."""
        fno = FNOBlock(embed_dim=64, modes=8)
        x = torch.randn(2, 32, 64, device=device, dtype=dtype, requires_grad=True)
        y = fno(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_gradient_flow_fno_encoder(self, device, dtype):
        """Test gradients flow through FNOEncoder."""
        enc = FNOEncoder(embed_dim=64, n_layers=2, modes=8)
        x = torch.randn(2, 32, 64, device=device, dtype=dtype, requires_grad=True)
        y = enc(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_gradient_flow_multimodal_encoder(self, device):
        """Test gradients flow through MultimodalFNOEncoder."""
        enc = MultimodalFNOEncoder(vocab_size=256, embed_dim=64, fno_layers=1, fno_modes=8)
        x = torch.randint(0, 256, (2, 32), device=device)

        # Get embedding with gradients
        enc.train()
        y = enc(x)
        loss = y.sum()
        loss.backward()

        # Check that model parameters used for bytes have gradients
        # Note: audio_encoder and image_encoder params won't have gradients
        # since we're only using bytes modality
        for name, param in enc.named_parameters():
            if param.requires_grad:
                # Skip audio/image encoders - they're not used for bytes modality
                if 'audio_encoder' in name or 'image_encoder' in name:
                    continue
                assert param.grad is not None, f"No gradient for {name}"


# ==============================================================================
# Learnable Parameter Tests
# ==============================================================================

class TestFNOLearnableParameters:
    """Tests for learnable parameters in FNO components."""

    def test_spectral_conv1d_learnable_params(self):
        """Test SpectralConv1d has learnable parameters."""
        sc = SpectralConv1d(in_channels=32, out_channels=32, modes=8)
        params = list(sc.parameters())
        assert len(params) > 0
        assert any('weight_real' in name for name, _ in sc.named_parameters())
        assert any('weight_imag' in name for name, _ in sc.named_parameters())

    def test_fno_block_learnable_params(self):
        """Test FNOBlock has learnable parameters."""
        fno = FNOBlock(embed_dim=64, modes=8)
        params = list(fno.parameters())
        assert len(params) > 0
        assert any('spectral_conv' in name for name, _ in fno.named_parameters())
        assert any('local_linear' in name for name, _ in fno.named_parameters())
        assert any('out_proj' in name for name, _ in fno.named_parameters())

    def test_fno_encoder_learnable_params(self):
        """Test FNOEncoder has learnable parameters."""
        enc = FNOEncoder(embed_dim=64, n_layers=2, modes=8)
        params = list(enc.parameters())
        assert len(params) > 0
        assert any('blocks' in name for name, _ in enc.named_parameters())
        assert any('final_norm' in name for name, _ in enc.named_parameters())

    def test_multimodal_encoder_learnable_params(self):
        """Test MultimodalFNOEncoder has learnable parameters."""
        enc = MultimodalFNOEncoder(vocab_size=256, embed_dim=64, fno_layers=1, fno_modes=8)
        params = list(enc.parameters())
        assert len(params) > 0
        assert any('byte_embed' in name for name, _ in enc.named_parameters())
        assert any('fno_encoder' in name for name, _ in enc.named_parameters())
        assert any('modality_embed' in name for name, _ in enc.named_parameters())


# ==============================================================================
# SpectralConv alias test
# ==============================================================================

class TestSpectralConvAlias:
    """Test SpectralConv alias for SpectralConv1d."""

    def test_spectral_conv_is_spectral_conv1d(self):
        """Test that SpectralConv is an alias for SpectralConv1d."""
        assert SpectralConv is SpectralConv1d

    def test_spectral_conv_works(self, batch_size, device, dtype):
        """Test that SpectralConv works as expected."""
        sc = SpectralConv(in_channels=32, out_channels=32, modes=8)
        x = torch.randn(batch_size, 32, 64, device=device, dtype=dtype)
        y = sc(x)
        assert y.shape == x.shape


# ==============================================================================
# Run tests if executed directly
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
