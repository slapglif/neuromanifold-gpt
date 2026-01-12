# neuromanifold_gpt/tests/test_spectral.py
"""Tests for SpectralDecomposition module."""
import pytest
import torch
from neuromanifold_gpt.model.spectral import SpectralDecomposition, FastSpectralAttention


def test_spectral_output_shapes():
    """Check all output tensor shapes."""
    spectral = SpectralDecomposition(manifold_dim=64, n_eigenvectors=16)
    coords = torch.randn(2, 20, 64)

    basis, freqs, ortho_loss = spectral(coords)

    assert basis.shape == (2, 20, 16)
    assert freqs.shape == (2, 16)
    assert ortho_loss.shape == ()  # Scalar loss for orthogonality regularization


def test_spectral_basis_normalized():
    """Spectral basis should sum to 1 per position (softmax)."""
    spectral = SpectralDecomposition(manifold_dim=64, n_eigenvectors=16)
    coords = torch.randn(1, 20, 64)

    basis, _, _ = spectral(coords)

    # With softmax, each row sums to 1
    row_sums = basis[0].sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(20), atol=1e-5)


def test_spectral_frequencies_positive():
    """Learned frequencies should be positive (abs applied)."""
    spectral = SpectralDecomposition(manifold_dim=64, n_eigenvectors=16)
    coords = torch.randn(1, 20, 64)

    _, freqs, _ = spectral(coords)

    assert (freqs >= 0).all()


def test_ortho_regularization():
    """Orthogonality loss should encourage orthonormal basis."""
    spectral = SpectralDecomposition(manifold_dim=64, n_eigenvectors=16, ortho_weight=0.01)
    coords = torch.randn(1, 20, 64)

    basis, _, ortho_loss = spectral(coords)

    # Ortho loss should be non-negative (Frobenius norm squared)
    assert ortho_loss >= 0
    # Ortho loss should be small but not zero (learned basis isn't perfectly orthogonal)
    assert ortho_loss < 1.0  # Reasonable upper bound with weight=0.01


def test_spectral_gradient_flow():
    """Gradients should flow through spectral decomposition."""
    spectral = SpectralDecomposition(manifold_dim=64, n_eigenvectors=16)
    coords = torch.randn(2, 20, 64, requires_grad=True)

    basis, freqs, _ = spectral(coords)
    loss = basis.sum() + freqs.sum()
    loss.backward()

    assert coords.grad is not None
    assert not torch.isnan(coords.grad).any()


def test_random_features_mode():
    """Test non-learned random Fourier features mode."""
    spectral = SpectralDecomposition(
        manifold_dim=64, n_eigenvectors=16, use_learned_basis=False
    )
    coords = torch.randn(2, 20, 64)

    basis, freqs, _ = spectral(coords)

    assert basis.shape == (2, 20, 16)
    assert freqs.shape == (2, 16)
    assert not torch.isnan(basis).any()


class TestFastSpectralAttention:
    """Tests for FastSpectralAttention module."""

    def test_output_shape(self):
        """Output should match input shape."""
        attn = FastSpectralAttention(embed_dim=128, n_eigenvectors=16, n_heads=4)
        x = torch.randn(2, 20, 128)
        spectral_basis = torch.randn(2, 20, 16)
        spectral_freqs = torch.randn(2, 16)

        out = attn(x, spectral_basis, spectral_freqs)

        assert out.shape == (2, 20, 128)

    def test_gradient_flow(self):
        """Gradients should flow through attention."""
        attn = FastSpectralAttention(embed_dim=128, n_eigenvectors=16, n_heads=4)
        x = torch.randn(2, 20, 128, requires_grad=True)
        spectral_basis = torch.randn(2, 20, 16)
        spectral_freqs = torch.randn(2, 16)

        out = attn(x, spectral_basis, spectral_freqs)
        out.sum().backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_numerical_stability(self):
        """Output should not contain NaN or Inf."""
        attn = FastSpectralAttention(embed_dim=128, n_eigenvectors=16, n_heads=4)
        x = torch.randn(2, 20, 128)
        spectral_basis = torch.randn(2, 20, 16)
        spectral_freqs = torch.randn(2, 16)

        out = attn(x, spectral_basis, spectral_freqs)

        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
