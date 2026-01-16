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

    def imagine_alternatives(
        self,
        idx: torch.Tensor,
        goal_tokens: torch.Tensor | None = None,
        n_alternatives: int = 4,
    ) -> dict[str, torch.Tensor]:
        """
        Use imagination module to explore alternative reasoning paths.

        Args:
            idx: Input token IDs (B, T)
            goal_tokens: Optional goal tokens to optimize toward (B, T_goal)
            n_alternatives: Number of alternatives to generate

        Returns:
            dict with alternatives, scores, and best selection
        """
        if not self.use_imagination:
            raise RuntimeError("Imagination module not enabled. Set use_imagination=True in config.")

        # Get hidden states
        B, T = idx.shape
        device = idx.device

        # Forward pass to get representations
        logits, _, info = self(idx)

        # Get final hidden states before lm_head
        x = self.ln_f(info.get('x', logits))  # Approximate from logits if needed

        # If goal tokens provided, encode them
        goal_emb = None
        if goal_tokens is not None:
            goal_logits, _, _ = self(goal_tokens)
            goal_emb = goal_logits.mean(dim=1)  # (B, D) pooled goal

        # Use imagination
        if goal_emb is not None:
            result = self.imagination(x, goal=goal_emb, n_alternatives=n_alternatives)
        else:
            result = self.imagination(x, n_alternatives=n_alternatives)

        return result

    def plan_task(
        self,
        idx: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Use DAG planner to decompose a task into subtasks.

        Args:
            idx: Input token IDs (B, T)

        Returns:
            dict with node_embeddings, adj_matrix, execution_order, etc.
        """
        if not self.use_dag_planner:
            raise RuntimeError("DAG planner not enabled. Set use_dag_planner=True in config.")

        # Forward pass to get representations
        logits, _, info = self(idx)

        # DAG info is already computed in forward
        dag_info = info.get('dag_planner', {})

        # Add execution order
        if 'adj_matrix' in dag_info and 'node_mask' in dag_info:
            # Get execution order for first example
            exec_order = self.dag_planner.get_execution_order(
                dag_info['adj_matrix'][0],
                dag_info['node_mask'][0] if 'node_mask' in dag_info else torch.ones(dag_info['adj_matrix'].shape[1], dtype=torch.bool),
            )
            dag_info['execution_order'] = exec_order

        return dag_info
