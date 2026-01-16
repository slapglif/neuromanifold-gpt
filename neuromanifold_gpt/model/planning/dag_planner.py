"""DAG Planner Module - Stub implementation."""

import torch
import torch.nn as nn


class ForcedDAGPlanner(nn.Module):
    """
    Stub implementation of forced DAG planner.

    This is a placeholder to allow the package to import correctly.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass-through forward method."""
        return x
