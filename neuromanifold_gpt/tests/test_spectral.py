# neuromanifold_gpt/tests/test_spectral.py
"""Tests for SpectralDecomposition module."""
import pytest
import torch
from neuromanifold_gpt.model.spectral import SpectralDecomposition


def test_spectral_output_shapes():
    """Check all output tensor shapes."""
    spectral = SpectralDecomposition(manifold_dim=64, n_eigenvectors=16)
    coords = torch.randn(2, 20, 64)

    basis, freqs, laplacian = spectral(coords)

    assert basis.shape == (2, 20, 16)
    assert freqs.shape == (2, 16)
    assert laplacian.shape == (2, 20, 20)


def test_eigenvectors_orthonormal():
    """Eigenvectors should be orthonormal."""
    spectral = SpectralDecomposition(manifold_dim=64, n_eigenvectors=16)
    coords = torch.randn(1, 20, 64)

    basis, _, _ = spectral(coords)

    # V^T V should be identity
    vtv = basis[0].T @ basis[0]
    eye = torch.eye(16)
    assert torch.allclose(vtv, eye, atol=1e-4)


def test_eigenvalues_sorted():
    """Eigenvalues should be sorted ascending (smallest = smoothest)."""
    spectral = SpectralDecomposition(manifold_dim=64, n_eigenvectors=16)
    coords = torch.randn(1, 20, 64)

    _, freqs, _ = spectral(coords)

    # Should be sorted
    assert (freqs[0, 1:] >= freqs[0, :-1]).all()


def test_laplacian_symmetric():
    """Normalized Laplacian should be symmetric."""
    spectral = SpectralDecomposition(manifold_dim=64, n_eigenvectors=16)
    coords = torch.randn(1, 20, 64)

    _, _, laplacian = spectral(coords)

    assert torch.allclose(laplacian, laplacian.transpose(1, 2), atol=1e-5)
