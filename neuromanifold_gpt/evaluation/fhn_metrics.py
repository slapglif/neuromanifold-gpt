"""FHN (FitzHugh-Nagumo) wave stability evaluation metrics.

This module provides metrics for evaluating FHN excitable wave dynamics:
- Wave state statistics: Mean and variance of membrane potential (v)
- Pulse width analysis: Distribution of wave propagation widths
- Stability indicators: Wave oscillation and convergence metrics

These metrics help monitor whether FHN dynamics maintain stable soliton-like
wave propagation without numerical instability or degradation.

Reference: neuromanifold_gpt/model/attention/fhn.py
"""

import torch
from typing import Dict, Any


class FHNMetrics:
    """Compute evaluation metrics for FHN wave dynamics.

    Analyzes the FHN state and wave propagation characteristics from
    the model's info dict returned by forward pass.
    """

    @staticmethod
    def compute_state_statistics(fhn_state: torch.Tensor) -> Dict[str, float]:
        """Compute statistics of FHN membrane potential state.

        The FHN state (v variable) should remain bounded in [-3, 3] for
        numerical stability (see FHNDynamics.forward clamp).

        Args:
            fhn_state: Membrane potential tensor of any shape

        Returns:
            Dictionary with state statistics:
                - fhn_state_mean: Average membrane potential
                - fhn_state_std: Standard deviation of potential
                - fhn_state_min: Minimum potential observed
                - fhn_state_max: Maximum potential observed
                - fhn_state_range: Range of potentials (max - min)
        """
        return {
            'fhn_state_mean': fhn_state.mean().item(),
            'fhn_state_std': fhn_state.std().item(),
            'fhn_state_min': fhn_state.min().item(),
            'fhn_state_max': fhn_state.max().item(),
            'fhn_state_range': (fhn_state.max() - fhn_state.min()).item(),
        }

    @staticmethod
    def compute_pulse_width_statistics(pulse_widths: torch.Tensor) -> Dict[str, float]:
        """Compute statistics of FHN pulse widths.

        Pulse widths control wave propagation duration. The base value is
        typically around 4 (see FHNAttention.pulse_width_base), with content-
        dependent modulation via pulse_width_net.

        Args:
            pulse_widths: Pulse width tensor of any shape

        Returns:
            Dictionary with pulse width statistics:
                - pulse_width_mean: Average pulse width
                - pulse_width_std: Standard deviation of widths
                - pulse_width_min: Minimum width observed
                - pulse_width_max: Maximum width observed
        """
        return {
            'pulse_width_mean': pulse_widths.mean().item(),
            'pulse_width_std': pulse_widths.std().item(),
            'pulse_width_min': pulse_widths.min().item(),
            'pulse_width_max': pulse_widths.max().item(),
        }

    @staticmethod
    def compute_wave_stability(fhn_state: torch.Tensor) -> Dict[str, float]:
        """Compute wave stability indicators.

        Stable soliton waves should have:
        - Bounded oscillations (not growing exponentially)
        - Smooth gradients (no sharp discontinuities)

        Args:
            fhn_state: Membrane potential tensor

        Returns:
            Dictionary with stability metrics:
                - fhn_stability_bounded: 1.0 if all states in [-3, 3], else 0.0
                - fhn_state_abs_mean: Mean absolute potential (activity level)
        """
        # Check if states are within numerical bounds
        is_bounded = ((fhn_state >= -3.0) & (fhn_state <= 3.0)).all().float().item()

        return {
            'fhn_stability_bounded': is_bounded,
            'fhn_state_abs_mean': fhn_state.abs().mean().item(),
        }

    @staticmethod
    def compute_all(info: Dict[str, Any]) -> Dict[str, float]:
        """Compute all FHN wave stability metrics from model info dict.

        Args:
            info: Model forward pass info dict containing:
                - 'pulse_widths': Tensor of pulse widths
                - 'fhn_state': Tensor of membrane potentials

        Returns:
            Dictionary with all FHN metrics:
                - fhn_state_mean: Average membrane potential
                - fhn_state_std: Standard deviation
                - fhn_state_min: Minimum potential
                - fhn_state_max: Maximum potential
                - fhn_state_range: Range of potentials
                - pulse_width_mean: Average pulse width
                - pulse_width_std: Standard deviation of widths
                - pulse_width_min: Minimum width
                - pulse_width_max: Maximum width
                - fhn_stability_bounded: Stability indicator
                - fhn_state_abs_mean: Mean absolute activity
        """
        metrics = {}

        # Extract tensors from info dict
        fhn_state = info.get('fhn_state')
        pulse_widths = info.get('pulse_widths')

        # Compute state statistics
        if fhn_state is not None:
            # Handle scalar tensors (convert to 1D for consistent processing)
            if fhn_state.dim() == 0:
                fhn_state = fhn_state.unsqueeze(0)

            metrics.update(FHNMetrics.compute_state_statistics(fhn_state))
            metrics.update(FHNMetrics.compute_wave_stability(fhn_state))

        # Compute pulse width statistics
        if pulse_widths is not None:
            # Handle scalar tensors
            if pulse_widths.dim() == 0:
                pulse_widths = pulse_widths.unsqueeze(0)

            metrics.update(FHNMetrics.compute_pulse_width_statistics(pulse_widths))

        return metrics
