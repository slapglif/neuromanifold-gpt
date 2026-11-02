"""Component-specific metrics aggregator.

This module provides the main ComponentMetricsAggregator class that processes
the model's info dict from forward() and computes all component-specific metrics:
- SDR metrics (sparsity, entropy, overlap)
- FHN wave stability metrics
- MTP token prediction accuracy (if available)
- Memory utilization metrics

Usage:
    from neuromanifold_gpt.evaluation.component_metrics import ComponentMetricsAggregator
    from neuromanifold_gpt.model.gpt import NeuroManifoldGPT

    model = NeuroManifoldGPT(config)
    logits, loss, info = model(tokens, targets)

    aggregator = ComponentMetricsAggregator()
    metrics = aggregator.compute(info, config)

    # Access component-specific metrics
    print(metrics['sdr']['sparsity'])
    print(metrics['fhn']['fhn_state_mean'])
    print(metrics['memory']['memory_size'])
"""

from typing import Any, Dict, List, Optional

import torch

from neuromanifold_gpt.evaluation.fhn_metrics import FHNMetrics
from neuromanifold_gpt.evaluation.memory_metrics import MemoryMetrics
from neuromanifold_gpt.evaluation.mtp_metrics import MTPMetrics
from neuromanifold_gpt.evaluation.sdr_metrics import SDRMetrics


class ComponentMetricsAggregator:
    """Aggregates component-specific metrics from model info dict.

    This class processes the info dict returned by NeuroManifoldGPT.forward()
    and computes metrics for each component:

    - SDR: Semantic folding encoder metrics (sparsity, entropy, overlap)
    - FHN: Wave stability metrics (state statistics, pulse widths)
    - MTP: Multi-token prediction accuracy (if logits provided)
    - Memory: Engram memory utilization statistics

    The compute() method returns a nested dictionary with metrics grouped by
    component type, making it easy to log or analyze specific subsystems.
    """

    def __init__(self):
        """Initialize the metrics aggregator.

        No state is maintained between calls to compute().
        """
        pass

    def compute(
        self,
        info: Dict[str, Any],
        config: Any,
        logits: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Compute all component-specific metrics from model info dict.

        Args:
            info: Model forward pass info dict containing:
                - 'sdr': SDR representations (batch, seq_len, sdr_size)
                - 'block_infos': List of per-block info dicts with FHN data
                - 'memory_stats': Memory utilization statistics
                - (optional) 'mtp_logits': Multi-token prediction logits
            config: NeuroManifoldConfig instance with model hyperparameters
            logits: Optional main prediction logits for MTP metrics
                Shape: (batch, seq_len, vocab_size)
            targets: Optional target tokens for MTP accuracy computation
                Shape: (batch, seq_len)

        Returns:
            Dictionary mapping component names to metric dictionaries:
                {
                    'sdr': {
                        'sparsity': float,
                        'entropy': float,
                        'overlap_mean': float,
                        ...
                    },
                    'fhn': {
                        'fhn_state_mean': float,
                        'pulse_width_mean': float,
                        'fhn_stability_bounded': float,
                        ...
                    },
                    'mtp': {
                        'accuracy': float,
                        'top_1_accuracy': float,
                        'confidence_mean': float,
                        ...
                    },
                    'memory': {
                        'memory_size': float,
                        'memory_utilization': float,
                        'total_size': float,
                        ...
                    }
                }
        """
        metrics = {}

        # SDR Metrics
        if config.use_sdr and "sdr" in info and info["sdr"] is not None:
            sdr = info["sdr"]
            sdr_metrics = SDRMetrics.compute_all(sdr, n_active=config.sdr_n_active)
            metrics["sdr"] = sdr_metrics

        # FHN Metrics - aggregate from block_infos
        if "block_infos" in info and len(info["block_infos"]) > 0:
            fhn_metrics = self._aggregate_fhn_metrics(info["block_infos"])
            if fhn_metrics:  # Only add if we found FHN data
                metrics["fhn"] = fhn_metrics

        # MTP Metrics - compute from logits if provided
        if logits is not None and targets is not None:
            # Check if MTP is enabled
            use_mtp = getattr(config, "use_mtp", False)
            if use_mtp:
                # For multi-token prediction, we could have multiple heads
                # For now, compute metrics from main logits
                # (per-depth accuracy would require separate logits per depth)
                mtp_metrics = MTPMetrics.compute_all(logits, targets)
                metrics["mtp"] = mtp_metrics

        # Memory Metrics
        if "memory_stats" in info:
            memory_stats = info["memory_stats"]
            memory_metrics = MemoryMetrics.compute_all(memory_stats)
            metrics["memory"] = memory_metrics

        return metrics

    def _aggregate_fhn_metrics(
        self, block_infos: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Aggregate FHN metrics from multiple transformer blocks.

        Each block may contain FHN state and pulse width information.
        We aggregate these across all blocks by computing mean statistics.

        Args:
            block_infos: List of info dicts from each transformer block

        Returns:
            Dictionary with aggregated FHN metrics across all blocks.
            Returns empty dict if no FHN data is found.
        """
        # Collect FHN states and pulse widths from all blocks
        all_fhn_states: List[torch.Tensor] = []
        all_pulse_widths: List[torch.Tensor] = []

        for block_info in block_infos:
            if "fhn_state" in block_info and block_info["fhn_state"] is not None:
                fhn_state = block_info["fhn_state"]
                # Handle scalar tensors
                if isinstance(fhn_state, torch.Tensor):
                    if fhn_state.dim() == 0:
                        fhn_state = fhn_state.unsqueeze(0)
                    all_fhn_states.append(fhn_state)
                else:
                    # Scalar value
                    all_fhn_states.append(torch.tensor([fhn_state]))

            if "pulse_widths" in block_info and block_info["pulse_widths"] is not None:
                pulse_widths = block_info["pulse_widths"]
                # Handle scalar tensors
                if isinstance(pulse_widths, torch.Tensor):
                    if pulse_widths.dim() == 0:
                        pulse_widths = pulse_widths.unsqueeze(0)
                    all_pulse_widths.append(pulse_widths)
                else:
                    # Scalar value
                    all_pulse_widths.append(torch.tensor([pulse_widths]))

        # If no FHN data found, return empty dict
        if not all_fhn_states and not all_pulse_widths:
            return {}

        # Concatenate all collected tensors
        aggregated_info: Dict[str, torch.Tensor] = {}

        if all_fhn_states:
            # Flatten all states into a single tensor for aggregate statistics
            combined_fhn_states = torch.cat([s.flatten() for s in all_fhn_states])
            aggregated_info["fhn_state"] = combined_fhn_states

        if all_pulse_widths:
            # Flatten all pulse widths into a single tensor
            combined_pulse_widths = torch.cat([p.flatten() for p in all_pulse_widths])
            aggregated_info["pulse_widths"] = combined_pulse_widths

        # Compute metrics using FHNMetrics.compute_all
        fhn_metrics = FHNMetrics.compute_all(aggregated_info)

        return fhn_metrics
