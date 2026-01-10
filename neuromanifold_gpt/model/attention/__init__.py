# neuromanifold_gpt/model/attention/__init__.py
"""Attention mechanisms for NeuroManifoldGPT."""

from neuromanifold_gpt.model.attention.soliton import SolitonAttention, SolitonDynamics

__all__ = ["SolitonAttention", "SolitonDynamics"]
