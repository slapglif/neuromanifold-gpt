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

KAN Metrics:
    KANMetrics: B-spline basis statistics and activation patterns for KAN layers

Spectral Metrics:
    SpectralMetrics: Spectral filtering statistics and frequency domain behavior

Aggregator:
    ComponentMetricsAggregator: Main class for computing all metrics from model info dict
"""

# Component-specific metrics
# Metrics aggregator
from neuromanifold_gpt.evaluation.component_metrics import ComponentMetricsAggregator
from neuromanifold_gpt.evaluation.fhn_metrics import FHNMetrics
from neuromanifold_gpt.evaluation.kan_metrics import KANMetrics
from neuromanifold_gpt.evaluation.memory_metrics import MemoryMetrics
from neuromanifold_gpt.evaluation.mtp_metrics import MTPMetrics
from neuromanifold_gpt.evaluation.sdr_metrics import SDRMetrics
from neuromanifold_gpt.evaluation.spectral_metrics import SpectralMetrics

__all__ = [
    # Component metrics
    "SDRMetrics",
    "FHNMetrics",
    "MTPMetrics",
    "MemoryMetrics",
    "KANMetrics",
    "SpectralMetrics",
    # Aggregator
    "ComponentMetricsAggregator",
]
