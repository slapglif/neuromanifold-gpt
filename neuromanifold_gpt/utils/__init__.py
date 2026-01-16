"""Utility modules for NeuroManifoldGPT.

This package contains utility functions and classes for logging,
checkpoint selection, profiling, and other common tasks.
"""

from neuromanifold_gpt.utils.checkpoints import (
    select_checkpoint,
    find_best_checkpoint,
    list_checkpoints,
    export_checkpoints_metadata,
)
from neuromanifold_gpt.utils.logging import get_logger, configure_logging, RichLogger
from neuromanifold_gpt.utils.helpers import exists, default
from neuromanifold_gpt.utils.sinkhorn import sinkhorn_log
from neuromanifold_gpt.utils.stream_ops import get_expand_reduce_stream_functions

__all__ = [
    "select_checkpoint",
    "find_best_checkpoint",
    "list_checkpoints",
    "export_checkpoints_metadata",
    "get_logger",
    "configure_logging",
    "RichLogger",
    "exists",
    "default",
    "sinkhorn_log",
    "get_expand_reduce_stream_functions",
]