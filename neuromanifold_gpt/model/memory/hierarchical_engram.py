"""
Hierarchical Engram Memory - Stub implementation.

This is a placeholder for future L1/L2/L3 tiered memory system.
Currently not implemented but imported by gpt.py for future use.
"""
import torch
import torch.nn as nn


class HierarchicalEngramMemory(nn.Module):
    """
    Stub for HierarchicalEngramMemory.

    This will be implemented in a future update to provide
    L1/L2/L3 tiered memory consolidation.
    """
    def __init__(self, sdr_size, n_active, content_dim,
                 l1_capacity=64, l2_capacity=512, l3_capacity=4096):
        super().__init__()
        self.sdr_size = sdr_size
        self.n_active = n_active
        self.content_dim = content_dim

    def forward(self, sdr, content):
        """Stub forward pass - returns input unchanged."""
        return content
