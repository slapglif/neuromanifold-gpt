"""Spectral decomposition evaluation metrics.

This module provides metrics for evaluating spectral basis quality:
- Eigenvalue statistics: Distribution of learned frequency/eigenvalue estimates
- Basis statistics: Quality and properties of learned spectral basis
- Orthogonality metrics: How well the basis approximates orthonormal eigenvectors

These metrics help monitor whether the spectral decomposition maintains
numerical stability and produces meaningful spectral representations.

Reference: neuromanifold_gpt/model/spectral.py (SpectralDecomposition)
"""

from typing import Any, Dict

import torch


class SpectralMetrics:
    """Compute evaluation metrics for spectral decomposition.

    Analyzes the spectral basis and eigenvalue characteristics from
    the model's info dict returned by forward pass.
    """

    @staticmethod
    def compute_eigenvalue_statistics(spectral_freqs: torch.Tensor) -> Dict[str, float]:
        """Compute statistics of learned eigenvalues/frequencies.

        The spectral frequencies represent learned eigenvalue estimates
        that should capture the spectrum of the graph Laplacian.

        Args:
            spectral_freqs: Frequency/eigenvalue tensor of shape (B, n_eig)

        Returns:
            Dictionary with eigenvalue statistics:
                - eigenvalue_mean: Average eigenvalue across all modes
                - eigenvalue_std: Standard deviation of eigenvalues
                - eigenvalue_min: Minimum eigenvalue
                - eigenvalue_max: Maximum eigenvalue
                - eigenvalue_range: Range of eigenvalues (max - min)
        """
        return {
            "eigenvalue_mean": spectral_freqs.mean().item(),
            "eigenvalue_std": spectral_freqs.std().item(),
            "eigenvalue_min": spectral_freqs.min().item(),
            "eigenvalue_max": spectral_freqs.max().item(),
            "eigenvalue_range": (spectral_freqs.max() - spectral_freqs.min()).item(),
        }

    @staticmethod
    def compute_basis_statistics(spectral_basis: torch.Tensor) -> Dict[str, float]:
        """Compute statistics of learned spectral basis.

        The spectral basis should be well-normalized and have reasonable
        magnitudes to avoid numerical issues in downstream attention.

        Args:
            spectral_basis: Spectral coefficients tensor of shape (B, T, n_eig)

        Returns:
            Dictionary with basis statistics:
                - basis_mean: Average basis coefficient
                - basis_std: Standard deviation of coefficients
                - basis_abs_mean: Mean absolute coefficient (activity level)
                - basis_min: Minimum coefficient
                - basis_max: Maximum coefficient
        """
        return {
            "basis_mean": spectral_basis.mean().item(),
            "basis_std": spectral_basis.std().item(),
            "basis_abs_mean": spectral_basis.abs().mean().item(),
            "basis_min": spectral_basis.min().item(),
            "basis_max": spectral_basis.max().item(),
        }

    @staticmethod
    def compute_orthogonality_metrics(
        spectral_basis: torch.Tensor,
        ortho_loss: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute orthogonality quality metrics.

        The learned basis should approximate an orthonormal basis (Φᵀ Φ ≈ I).
        Lower orthogonality loss indicates better approximation of eigenvectors.

        Args:
            spectral_basis: Spectral coefficients tensor of shape (B, T, n_eig)
            ortho_loss: Orthogonality regularization loss (scalar)

        Returns:
            Dictionary with orthogonality metrics:
                - ortho_loss: Orthogonality regularization loss
                - basis_norm_mean: Average L2 norm of basis vectors
                - basis_norm_std: Std deviation of basis vector norms
        """
        metrics = {}

        # Extract ortho loss
        if ortho_loss.dim() == 0:
            metrics["ortho_loss"] = ortho_loss.item()
        else:
            metrics["ortho_loss"] = ortho_loss.mean().item()

        # Compute basis vector norms along sequence dimension
        # Shape: (B, T, n_eig) -> compute norm over T for each (B, n_eig)
        basis_norms = torch.norm(spectral_basis, dim=1)  # (B, n_eig)
        metrics["basis_norm_mean"] = basis_norms.mean().item()
        metrics["basis_norm_std"] = basis_norms.std().item()

        return metrics

    @staticmethod
    def compute_all(info: Dict[str, Any]) -> Dict[str, float]:
        """Compute all spectral decomposition metrics from model info dict.

        Args:
            info: Model forward pass info dict containing:
                - 'spectral_basis': (B, T, n_eig) spectral coefficients
                - 'spectral_freqs': (B, n_eig) eigenvalue estimates
                - 'ortho_loss': scalar orthogonality loss

        Returns:
            Dictionary with all spectral metrics:
                - eigenvalue_mean: Average eigenvalue
                - eigenvalue_std: Standard deviation of eigenvalues
                - eigenvalue_min: Minimum eigenvalue
                - eigenvalue_max: Maximum eigenvalue
                - eigenvalue_range: Range of eigenvalues
                - basis_mean: Average basis coefficient
                - basis_std: Standard deviation of basis
                - basis_abs_mean: Mean absolute coefficient
                - basis_min: Minimum coefficient
                - basis_max: Maximum coefficient
                - ortho_loss: Orthogonality loss
                - basis_norm_mean: Average basis vector norm
                - basis_norm_std: Std deviation of norms
        """
        metrics = {}

        # Extract tensors from info dict
        spectral_basis = info.get("spectral_basis")
        spectral_freqs = info.get("spectral_freqs")
        ortho_loss = info.get("ortho_loss")

        # Compute eigenvalue statistics
        if spectral_freqs is not None:
            # Handle scalar tensors (convert to 1D for consistent processing)
            if spectral_freqs.dim() == 0:
                spectral_freqs = spectral_freqs.unsqueeze(0)

            metrics.update(
                SpectralMetrics.compute_eigenvalue_statistics(spectral_freqs)
            )

        # Compute basis statistics
        if spectral_basis is not None:
            # Handle scalar tensors
            if spectral_basis.dim() == 0:
                spectral_basis = spectral_basis.unsqueeze(0).unsqueeze(0)
            elif spectral_basis.dim() == 1:
                spectral_basis = spectral_basis.unsqueeze(0)

            metrics.update(SpectralMetrics.compute_basis_statistics(spectral_basis))

            # Compute orthogonality metrics (requires both basis and ortho_loss)
            if ortho_loss is not None:
                metrics.update(
                    SpectralMetrics.compute_orthogonality_metrics(
                        spectral_basis, ortho_loss
                    )
                )

        return metrics
