"""Training callbacks for NeuroManifoldGPT.

This module provides callbacks for training observability and health monitoring,
including real-time metrics tracking, anomaly detection, and rich console dashboards.

Exports:
    TrainingHealthCallback: Comprehensive training health monitoring callback

Example:
    from neuromanifold_gpt.callbacks import TrainingHealthCallback

    callback = TrainingHealthCallback(log_interval=100)
    # Use in training loop
"""

from neuromanifold_gpt.callbacks.training_health import TrainingHealthCallback

__all__ = ['TrainingHealthCallback']
