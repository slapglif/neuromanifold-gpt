"""
System 2 Reasoning Mixin - Encapsulates optional System 2 components.

This mixin provides initialization and management for System 2 reasoning
components that can be optionally enabled in NeuroManifoldGPT:
- Hybrid Reasoning (symbolic-neural reasoning)
- DAG Planner (structured task decomposition)
- Hierarchical Memory (L1/L2/L3 tiered memory)
- Imagination (counterfactual exploration)

Design: Extracted from gpt.py to improve separation of concerns and testability.
"""
import torch
import torch.nn as nn

from neuromanifold_gpt.model.hybrid_reasoning import HybridReasoningModule
from neuromanifold_gpt.model.planning.dag_planner import ForcedDAGPlanner
from neuromanifold_gpt.model.memory.hierarchical_engram import HierarchicalEngramMemory
from neuromanifold_gpt.model.imagination import ConsistencyImaginationModule


class SystemTwoReasoningMixin:
    """
    Mixin for System 2 reasoning components.

    Provides conditional initialization for advanced reasoning capabilities.
    Classes inheriting this mixin must be nn.Module subclasses and should call
    _init_system_two_components(config) during initialization.
    """

    def _init_system_two_components(self, config):
        """
        Initialize System 2 reasoning components based on config flags.

        Args:
            config: NeuroManifoldConfig with optional System 2 flags:
                - use_hybrid_reasoning: Enable hybrid symbolic-neural reasoning
                - use_dag_planner: Enable DAG-based task planning
                - use_hierarchical_memory: Enable L1/L2/L3 tiered memory
                - use_imagination: Enable counterfactual imagination
        """
        # HybridReasoningModule - Symbolic-neural reasoning
        self.use_hybrid_reasoning = getattr(config, 'use_hybrid_reasoning', False)
        if self.use_hybrid_reasoning:
            self.hybrid_reasoning = HybridReasoningModule(
                embed_dim=config.n_embd,
                n_thinking_layers=getattr(config, 'n_thinking_layers', 2),
                n_heads=config.n_heads,
                dropout=config.dropout,
                use_e7_prior=True,
                thinking_threshold=getattr(config, 'thinking_threshold', 0.5),
            )

        # ForcedDAGPlanner - Decompose tasks into DAGs for systematic reasoning
        self.use_dag_planner = getattr(config, 'use_dag_planner', False)
        if self.use_dag_planner:
            self.dag_planner = ForcedDAGPlanner(
                embed_dim=config.n_embd,
                manifold_dim=config.manifold_dim,
                max_nodes=getattr(config, 'dag_max_nodes', 32),
                min_nodes=getattr(config, 'dag_min_nodes', 3),
            )

        # HierarchicalEngramMemory - L1/L2/L3 tiered memory (optional upgrade)
        self.use_hierarchical_memory = getattr(config, 'use_hierarchical_memory', False)
        if self.use_hierarchical_memory:
            self.hierarchical_memory = HierarchicalEngramMemory(
                sdr_size=config.sdr_size,
                n_active=config.sdr_n_active,
                content_dim=config.n_embd,
                l1_capacity=getattr(config, 'hierarchical_l1_capacity', 64),
                l2_capacity=getattr(config, 'hierarchical_l2_capacity', 512),
                l3_capacity=getattr(config, 'hierarchical_l3_capacity', 4096),
            )

        # ConsistencyImaginationModule - Counterfactual exploration
        self.use_imagination = getattr(config, 'use_imagination', False)
        if self.use_imagination:
            self.imagination = ConsistencyImaginationModule(
                embed_dim=config.n_embd,
                manifold_dim=config.manifold_dim,
                n_imagination_steps=getattr(config, 'imagination_steps', 4),
            )
            self.imagination_n_alternatives = getattr(config, 'imagination_n_alternatives', 4)

    def _apply_hybrid_reasoning(self, x):
        """
        Apply hybrid reasoning to input tensor if enabled.

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)

        Returns:
            tuple: (output_tensor, hybrid_info_dict)
                - output_tensor: Processed tensor (same shape as input)
                - hybrid_info_dict: Dictionary with reasoning metadata (empty if disabled)
        """
        hybrid_info = {}
        if self.use_hybrid_reasoning:
            # Get E7 tier if available (set by trainer based on curriculum)
            e7_tier = getattr(self, '_e7_tier', None)
            x, hybrid_info = self.hybrid_reasoning(x, e7_tier=e7_tier)
        return x, hybrid_info

    def _apply_dag_planner(self, x):
        """
        Apply DAG planner to input tensor if enabled.

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)

        Returns:
            dict: DAG planner output info containing:
                - node_embeddings: Planned task node representations
                - adj_matrix: Task dependency adjacency matrix
                - surface_area: Geometric complexity measure
                - complexities: Per-node complexity estimates
                Empty dict if DAG planner is disabled.
        """
        dag_info = {}
        if self.use_dag_planner:
            dag_output = self.dag_planner(x, deterministic=not self.training)
            dag_info = {
                'node_embeddings': dag_output['node_embeddings'],
                'adj_matrix': dag_output['adj_matrix'],
                'surface_area': dag_output['surface_area'],
                'complexities': dag_output['complexities'],
            }
        return dag_info
