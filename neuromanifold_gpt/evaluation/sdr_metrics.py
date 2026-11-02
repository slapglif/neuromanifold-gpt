"""SDR (Sparse Distributed Representation) evaluation metrics.

This module provides metrics for evaluating SDR quality and behavior:
- Sparsity: Fraction of active bits (target ~2% for semantic folding)
- Entropy: Information content of the SDR distribution
- Overlap statistics: Pairwise overlap analysis for semantic similarity assessment

These metrics help monitor whether SDRs maintain proper sparsity constraints
and whether semantic relationships are encoded via bit overlap as expected.
"""
from typing import Dict

import torch


class SDRMetrics:
    """Compute evaluation metrics for SDR representations.

    All methods work on tensors of any shape, operating on the last dimension.
    SDRs are represented as float tensors with values in {0.0, 1.0}.
    """

    @staticmethod
    def compute_sparsity(sdr: torch.Tensor) -> torch.Tensor:
        """Compute sparsity of an SDR (fraction of active bits).

        Target sparsity for semantic folding is ~2% (40/2048).

        Args:
            sdr: Binary tensor of shape (..., sdr_dim)

        Returns:
            Sparsity ratio (active bits / total bits) of shape (...)
        """
        return sdr.sum(dim=-1) / sdr.shape[-1]

    @staticmethod
    def compute_entropy(sdr: torch.Tensor, n_active: int) -> torch.Tensor:
        """Compute normalized entropy of SDR distribution.

        Entropy measures how evenly distributed the active bits are.
        - High entropy: active bits well distributed
        - Low entropy: active bits concentrated in few positions

        Returns normalized entropy in [0, 1] where 1 is maximum entropy.

        Args:
            sdr: Binary tensor of shape (..., sdr_dim)
            n_active: Expected number of active bits (for normalization)

        Returns:
            Normalized entropy of shape (...)
        """
        # Compute probability distribution over bit positions
        # For binary SDRs, we compute the entropy of the active bit distribution
        # across the sdr dimension

        # Add small epsilon to avoid log(0)
        eps = 1e-10

        # Treat SDR as a probability distribution (normalize to sum to 1)
        # For each SDR in the batch, compute entropy
        probs = sdr / (sdr.sum(dim=-1, keepdim=True) + eps)

        # Shannon entropy: -sum(p * log(p))
        # Only compute for active bits (probs > 0)
        log_probs = torch.where(
            probs > 0, torch.log(probs + eps), torch.zeros_like(probs)
        )
        entropy = -(probs * log_probs).sum(dim=-1)

        # Normalize by maximum possible entropy for n_active uniform bits
        # Max entropy = log(sdr_dim) when all positions equally likely
        # But with n_active constraint, max entropy = log(n_active)
        max_entropy = torch.log(torch.tensor(n_active, dtype=torch.float32))
        normalized_entropy = entropy / (max_entropy + eps)

        return normalized_entropy

    @staticmethod
    def compute_overlap_statistics(
        sdr: torch.Tensor, n_active: int
    ) -> Dict[str, float]:
        """Compute pairwise overlap statistics across a batch of SDRs.

        Overlap = number of shared active bits between two SDRs.
        This measures semantic similarity in SDR representations.

        Args:
            sdr: Binary tensor of shape (batch, seq_len, sdr_dim)
            n_active: Expected number of active bits (for normalization)

        Returns:
            Dictionary with overlap statistics:
                - overlap_mean: Average pairwise overlap
                - overlap_std: Standard deviation of overlaps
                - overlap_min: Minimum overlap observed
                - overlap_max: Maximum overlap observed
                - overlap_mean_norm: Mean overlap normalized by n_active
        """
        # Flatten batch and sequence dimensions for pairwise comparison
        flat_sdr = sdr.reshape(-1, sdr.shape[-1])  # (batch*seq_len, sdr_dim)
        n_sdrs = flat_sdr.shape[0]

        if n_sdrs < 2:
            # Need at least 2 SDRs for overlap statistics
            return {
                "overlap_mean": 0.0,
                "overlap_std": 0.0,
                "overlap_min": 0.0,
                "overlap_max": 0.0,
                "overlap_mean_norm": 0.0,
            }

        # Compute pairwise overlaps using matrix multiplication
        # overlap[i,j] = sum(sdr[i] * sdr[j]) = count of shared active bits
        overlaps = torch.matmul(flat_sdr, flat_sdr.t())  # (n_sdrs, n_sdrs)

        # Extract upper triangle (excluding diagonal) for unique pairs
        mask = torch.triu(torch.ones_like(overlaps), diagonal=1).bool()
        unique_overlaps = overlaps[mask]

        return {
            "overlap_mean": unique_overlaps.mean().item(),
            "overlap_std": unique_overlaps.std().item(),
            "overlap_min": unique_overlaps.min().item(),
            "overlap_max": unique_overlaps.max().item(),
            "overlap_mean_norm": (unique_overlaps.mean() / n_active).item(),
        }

    @staticmethod
    def compute_all(sdr: torch.Tensor, n_active: int) -> Dict[str, float]:
        """Compute all SDR metrics.

        Args:
            sdr: Binary tensor of shape (batch, seq_len, sdr_dim)
            n_active: Expected number of active bits

        Returns:
            Dictionary with all metrics:
                - sparsity: Mean sparsity across batch
                - sparsity_std: Standard deviation of sparsity
                - entropy: Mean normalized entropy
                - entropy_std: Standard deviation of entropy
                - overlap_mean: Average pairwise overlap
                - overlap_std: Standard deviation of overlaps
                - overlap_min: Minimum overlap observed
                - overlap_max: Maximum overlap observed
                - overlap_mean_norm: Mean overlap normalized by n_active
        """
        # Compute sparsity across all SDRs
        sparsity = SDRMetrics.compute_sparsity(sdr)

        # Compute entropy across all SDRs
        entropy = SDRMetrics.compute_entropy(sdr, n_active)

        # Compute overlap statistics
        overlap_stats = SDRMetrics.compute_overlap_statistics(sdr, n_active)

        # Aggregate metrics
        metrics = {
            "sparsity": sparsity.mean().item(),
            "sparsity_std": sparsity.std().item(),
            "entropy": entropy.mean().item(),
            "entropy_std": entropy.std().item(),
            **overlap_stats,
        }

        return metrics
