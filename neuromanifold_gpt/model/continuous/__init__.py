# neuromanifold_gpt/model/continuous/__init__.py
"""Continuous Generation for NeuroManifoldGPT.

Exports:
    RectifiedFlowScheduler: Flow Matching scheduler (2026 industry standard)
    KANVelocityField: KAN-based score network (2026 scientific standard)
    ContinuousOutputHead: Unified continuous generation head
    ContinuousOutputConfig: Configuration for continuous generation
    
    SACPolicy: Soft Actor-Critic with KAN backbone
    SACConfig: Config for SAC
    DDPGPolicy: Deep Deterministic Policy Gradient with KAN backbone
    DDPGConfig: Config for DDPG

This module implements "Rectified Flow" (Flow Matching) for generation and 
RL policies (SAC/DDPG) for continuous control of the semantic manifold.
"""

from .flow_scheduler import RectifiedFlowScheduler, FlowConfig
from .velocity_field import KANVelocityField
from .output_head import ContinuousOutputHead, ContinuousOutputConfig
from .sac_policy import SACPolicy, SACConfig
from .ddpg_policy import DDPGPolicy, DDPGConfig

__all__ = [
    "RectifiedFlowScheduler",
    "FlowConfig",
    "KANVelocityField",
    "ContinuousOutputHead",
    "ContinuousOutputConfig",
    "SACPolicy",
    "SACConfig",
    "DDPGPolicy",
    "DDPGConfig",
]
