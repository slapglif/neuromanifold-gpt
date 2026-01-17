"""Reasoning preset - System 2 reasoning components enabled.

This preset provides a configuration suitable for:
- Deliberate, systematic reasoning tasks
- Multi-step problem solving
- Tasks requiring planning and counterfactual exploration
- Advanced reasoning with memory hierarchies

The configuration enables all System 2 reasoning components:
- ForcedDAGPlanner: Decomposes tasks into DAGs for systematic reasoning
- HierarchicalEngramMemory: L1/L2/L3 tiered memory system
- ConsistencyImaginationModule: Counterfactual exploration
- Multi-Token Prediction: Predicting multiple future tokens
"""

from neuromanifold_gpt.config.base import NeuroManifoldConfig


def get_reasoning_config() -> NeuroManifoldConfig:
    """Get the reasoning configuration with System 2 components enabled.

    Returns:
        NeuroManifoldConfig: Configuration with reasoning preset values.
    """
    return NeuroManifoldConfig(
        # ========================================
        # Model Configuration
        # ========================================
        n_layer=12,
        n_heads=12,
        n_embd=768,
        block_size=1024,

        # SDR and manifold settings
        manifold_dim=128,
        n_eigenvectors=64,

        # ========================================
        # System 2 Reasoning Components
        # ========================================

        # ForcedDAGPlanner - Decomposes tasks into DAGs for systematic reasoning
        use_dag_planner=True,
        dag_max_nodes=32,
        dag_min_nodes=3,

        # HierarchicalEngramMemory - L1/L2/L3 tiered memory system
        use_hierarchical_memory=True,
        hierarchical_l1_capacity=64,
        hierarchical_l2_capacity=512,
        hierarchical_l3_capacity=4096,

        # ConsistencyImaginationModule - Counterfactual exploration
        use_imagination=True,
        imagination_steps=4,
        imagination_n_alternatives=4,

        # Multi-Token Prediction - Predicting multiple future tokens
        use_mtp=True,
        mtp_n_predict=4,
        mtp_loss_weight=0.1,

        # Training configuration
        learning_rate=3e-4,
    )
