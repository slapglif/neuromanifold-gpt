# neuromanifold_gpt/model/spectral.py
"""
Spectral Decomposition for O(n) Attention.

Decomposes manifold graph into eigenvectors of the Laplacian.
Low-frequency eigenvectors = global structure
High-frequency eigenvectors = local details
"""
import torch
import torch.nn as nn


class SpectralDecomposition(nn.Module):
    """
    Compute spectral basis from manifold coordinates.

    The eigenvectors provide an orthonormal basis that captures
    manifold structure at different scales.
    """

    def __init__(self, manifold_dim: int, n_eigenvectors: int = 32, sigma: float = 1.0):
        super().__init__()
        self.manifold_dim = manifold_dim
        self.n_eig = n_eigenvectors
        self.sigma = nn.Parameter(torch.tensor(sigma))

    def forward(
        self,
        coords: torch.Tensor,
        metric: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute spectral decomposition.

        Args:
            coords: (B, T, manifold_dim)
            metric: optional (B, T, D, D) Riemannian metric

        Returns:
            spectral_basis: (B, T, n_eig) eigenvector values
            spectral_freqs: (B, n_eig) eigenvalues
            laplacian: (B, T, T) normalized Laplacian
        """
        B, T, D = coords.shape

        # Compute distances
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        dist_sq = (diff ** 2).sum(dim=-1)

        # Gaussian adjacency
        sigma_sq = self.sigma.abs() ** 2 + 1e-6
        adjacency = torch.exp(-dist_sq / (2 * sigma_sq))

        # Graph Laplacian
        degree = adjacency.sum(dim=-1)
        laplacian = torch.diag_embed(degree) - adjacency

        # Normalized Laplacian
        d_inv_sqrt = torch.diag_embed(1.0 / (degree.sqrt() + 1e-6))
        laplacian_norm = d_inv_sqrt @ laplacian @ d_inv_sqrt

        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian_norm)

        # Skip constant eigenvector (eigenvalue ≈ 0)
        k = min(self.n_eig, T - 1)
        spectral_basis = eigenvectors[..., 1:k+1]
        spectral_freqs = eigenvalues[..., 1:k+1]

        return spectral_basis, spectral_freqs, laplacian_norm
