"""NeuroManifoldGPT evaluation metrics.

This module provides component-specific evaluation metrics for monitoring
the health and behavior of individual model components during training
and evaluation.

Metric Categories:

SDR Metrics:
    SDRMetrics: Sparsity, entropy, and overlap statistics for SDR representations

FHN Metrics:
    FHNMetrics: Wave stability and pulse width statistics for FHN dynamics

MTP Metrics:
    MTPMetrics: Token prediction accuracy metrics (not just loss) for MTP heads

Memory Metrics:
    MemoryMetrics: Memory utilization and retrieval statistics

Aggregator:
    ComponentMetricsAggregator: Main class for computing all metrics from model info dict
"""

# Component-specific metrics
from neuromanifold_gpt.evaluation.sdr_metrics import SDRMetrics
from neuromanifold_gpt.evaluation.fhn_metrics import FHNMetrics
from neuromanifold_gpt.evaluation.mtp_metrics import MTPMetrics
from neuromanifold_gpt.evaluation.memory_metrics import MemoryMetrics

# Metrics aggregator
from neuromanifold_gpt.evaluation.component_metrics import ComponentMetricsAggregator

__all__ = [
    # Component metrics
    "SDRMetrics",
    "FHNMetrics",
    "MTPMetrics",
    "MemoryMetrics",
    # Aggregator
    "ComponentMetricsAggregator",
]
