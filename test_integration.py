#!/usr/bin/env python3
"""Integration test for System 2 components."""

from neuromanifold_gpt.config import NeuroManifoldConfig
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT
import torch

# Create config with all System 2 components enabled
config = NeuroManifoldConfig(
    n_layer=2,
    n_head=4,
    n_embd=128,
    use_dag_planner=True,
    use_hierarchical_memory=True,
    use_imagination=True
)

# Instantiate model
model = NeuroManifoldGPT(config)

print('All System 2 components loaded successfully')
