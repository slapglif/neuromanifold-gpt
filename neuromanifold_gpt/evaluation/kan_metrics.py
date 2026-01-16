"""KAN (Kolmogorov-Arnold Network) evaluation metrics.

This module provides metrics for evaluating KAN layer behavior:
- Activation statistics: Mean, variance, and range of layer activations
- Grid utilization: How evenly the basis functions are utilized
- Spline statistics: Distribution and behavior of learnable spline weights

These metrics help monitor whether KAN layers maintain stable activation
patterns and whether the basis functions are being effectively utilized
for function approximation.

Reference: KAN layers use learnable spline basis functions instead of
fixed activation functions, enabling more expressive function approximation.
"""

import torch
from typing import Dict, Any


class KANMetrics:
    """Compute evaluation metrics for KAN layer dynamics.

    Analyzes KAN activations and basis function utilization from
    the model's info dict returned by forward pass.
    """

    @staticmethod
    def compute_activation_statistics(kan_activations: torch.Tensor) -> Dict[str, float]:
        """Compute statistics of KAN layer activations.

        KAN activations should remain bounded and stable during training.
        Large activation magnitudes or high variance may indicate
        numerical instability or poor initialization.

        Args:
            kan_activations: Activation tensor of any shape

        Returns:
            Dictionary with activation statistics:
                - activation_mean: Average activation value
                - activation_std: Standard deviation of activations
                - activation_min: Minimum activation observed
                - activation_max: Maximum activation observed
                - activation_range: Range of activations (max - min)
                - activation_abs_mean: Mean absolute activation (activity level)
        """
        return {
            'activation_mean': kan_activations.mean().item(),
            'activation_std': kan_activations.std().item(),
            'activation_min': kan_activations.min().item(),
            'activation_max': kan_activations.max().item(),
            'activation_range': (kan_activations.max() - kan_activations.min()).item(),
            'activation_abs_mean': kan_activations.abs().mean().item(),
        }

    @staticmethod
    def compute_grid_utilization(basis_output: torch.Tensor) -> Dict[str, float]:
        """Compute grid utilization statistics for basis functions.

        KAN uses multiple basis functions (typically B-splines) on a grid.
        Good utilization means:
        - Basis functions are evenly used (not concentrated in few)
        - Variance across basis dimension indicates diversity

        The last dimension of basis_output represents different basis functions.

        Args:
            basis_output: Basis function outputs of shape (..., n_basis)

        Returns:
            Dictionary with grid utilization metrics:
                - grid_utilization_mean: Average basis output magnitude
                - grid_utilization_std: Std deviation across all basis outputs
                - grid_basis_variance: Variance across basis dimension (diversity)
                - grid_basis_mean_abs: Mean absolute basis output
        """
        # Compute variance across the basis dimension to measure diversity
        # Higher variance = more diverse basis utilization
        basis_variance = basis_output.var(dim=-1).mean().item()

        return {
            'grid_utilization_mean': basis_output.mean().item(),
            'grid_utilization_std': basis_output.std().item(),
            'grid_basis_variance': basis_variance,
            'grid_basis_mean_abs': basis_output.abs().mean().item(),
        }

    @staticmethod
    def compute_spline_statistics(spline_weights: torch.Tensor) -> Dict[str, float]:
        """Compute statistics of learnable spline weights.

        Spline weights control the shape of basis functions in KAN.
        Monitoring these helps ensure:
        - Weights remain bounded (not exploding)
        - Learning is occurring (weights changing from initialization)

        Args:
            spline_weights: Spline weight tensor of any shape

        Returns:
            Dictionary with spline weight statistics:
                - spline_weight_mean: Average spline weight value
                - spline_weight_std: Standard deviation of weights
                - spline_weight_min: Minimum weight observed
                - spline_weight_max: Maximum weight observed
                - spline_weight_abs_mean: Mean absolute weight magnitude
        """
        return {
            'spline_weight_mean': spline_weights.mean().item(),
            'spline_weight_std': spline_weights.std().item(),
            'spline_weight_min': spline_weights.min().item(),
            'spline_weight_max': spline_weights.max().item(),
            'spline_weight_abs_mean': spline_weights.abs().mean().item(),
        }

    @staticmethod
    def compute_all(info: Dict[str, Any]) -> Dict[str, float]:
        """Compute all KAN metrics from model info dict.

        Args:
            info: Model forward pass info dict containing:
                - 'kan_activations': Tensor of KAN layer activations
                - 'kan_basis_output': Tensor of basis function outputs
                - 'kan_spline_weights': Optional tensor of spline weights

        Returns:
            Dictionary with all KAN metrics:
                - activation_mean: Average activation value
                - activation_std: Standard deviation of activations
                - activation_min: Minimum activation
                - activation_max: Maximum activation
                - activation_range: Range of activations
                - activation_abs_mean: Mean absolute activation
                - grid_utilization_mean: Average basis output
                - grid_utilization_std: Std of basis outputs
                - grid_basis_variance: Variance across basis dimension
                - grid_basis_mean_abs: Mean absolute basis output
                - spline_weight_mean: Average spline weight (if available)
                - spline_weight_std: Std of spline weights (if available)
                - spline_weight_min: Minimum spline weight (if available)
                - spline_weight_max: Maximum spline weight (if available)
                - spline_weight_abs_mean: Mean absolute weight (if available)
        """
        metrics = {}

        # Extract tensors from info dict
        kan_activations = info.get('kan_activations')
        basis_output = info.get('kan_basis_output')
        spline_weights = info.get('kan_spline_weights')

        # Compute activation statistics
        if kan_activations is not None:
            # Handle scalar tensors (convert to 1D for consistent processing)
            if kan_activations.dim() == 0:
                kan_activations = kan_activations.unsqueeze(0)

            metrics.update(KANMetrics.compute_activation_statistics(kan_activations))

        # Compute grid utilization statistics
        if basis_output is not None:
            # Handle scalar tensors
            if basis_output.dim() == 0:
                basis_output = basis_output.unsqueeze(0)

            metrics.update(KANMetrics.compute_grid_utilization(basis_output))

        # Compute spline weight statistics (optional)
        if spline_weights is not None:
            # Handle scalar tensors
            if spline_weights.dim() == 0:
                spline_weights = spline_weights.unsqueeze(0)

            metrics.update(KANMetrics.compute_spline_statistics(spline_weights))

        return metrics
