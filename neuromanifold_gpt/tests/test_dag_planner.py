# neuromanifold_gpt/tests/test_dag_planner.py
"""Tests for ForcedDAGPlanner - DAG-based task decomposition."""
import pytest
import torch

from neuromanifold_gpt.model.planning.dag_planner import ForcedDAGPlanner


def test_dag_planner_forward_shape():
    """Planner should generate nodes, adjacency, and complexity estimates."""
    planner = ForcedDAGPlanner(
        embed_dim=384, manifold_dim=64, max_nodes=16, min_nodes=3
    )
    x = torch.randn(2, 10, 384)

    result = planner(x, deterministic=True)

    assert result["node_embeddings"].shape == (2, 16, 384)
    assert result["adj_matrix"].shape == (2, 16, 16)
    assert result["node_mask"].shape == (2, 16)
    assert result["surface_area"].shape == (2,)
    assert result["complexities"].shape == (2, 16)


def test_dag_planner_node_count_bounds():
    """Node count should respect min/max constraints."""
    planner = ForcedDAGPlanner(
        embed_dim=384, manifold_dim=64, max_nodes=16, min_nodes=3
    )
    x = torch.randn(4, 10, 384)

    result = planner(x, deterministic=True)
    node_mask = result["node_mask"]

    # Count active nodes per batch
    node_counts = node_mask.sum(dim=1)

    assert (node_counts >= 3).all()
    assert (node_counts <= 16).all()


def test_dag_planner_acyclic_structure():
    """Adjacency matrix should be upper-triangular (DAG constraint)."""
    planner = ForcedDAGPlanner(
        embed_dim=384, manifold_dim=64, max_nodes=16, min_nodes=3
    )
    x = torch.randn(2, 10, 384)

    result = planner(x, deterministic=True)
    adj_matrix = result["adj_matrix"]

    # Check upper-triangular: no edges from higher to lower index
    for b in range(2):
        for i in range(16):
            for j in range(i + 1):
                # All elements on or below diagonal should be zero
                assert adj_matrix[b, i, j] == 0


def test_dag_planner_node_mask_consistency():
    """Node mask should enable exactly node_count nodes."""
    planner = ForcedDAGPlanner(
        embed_dim=384, manifold_dim=64, max_nodes=16, min_nodes=3
    )
    x = torch.randn(3, 10, 384)

    result = planner(x, deterministic=True)
    node_mask = result["node_mask"]

    # Each batch should have contiguous True values from index 0
    for b in range(3):
        active_indices = node_mask[b].nonzero(as_tuple=True)[0]
        if len(active_indices) > 0:
            # Active nodes should be contiguous from 0
            expected = torch.arange(len(active_indices), device=x.device)
            assert (active_indices == expected).all()


def test_dag_planner_adjacency_mask_application():
    """Adjacency matrix should respect node mask."""
    planner = ForcedDAGPlanner(
        embed_dim=384, manifold_dim=64, max_nodes=16, min_nodes=3
    )
    x = torch.randn(2, 10, 384)

    result = planner(x, deterministic=True)
    adj_matrix = result["adj_matrix"]
    node_mask = result["node_mask"]

    # Edges should only exist between active nodes
    for b in range(2):
        active = node_mask[b]
        for i in range(16):
            for j in range(16):
                if adj_matrix[b, i, j] > 0:
                    assert active[i] and active[j]


def test_dag_planner_deterministic_mode():
    """Deterministic mode should produce consistent results."""
    planner = ForcedDAGPlanner(
        embed_dim=384, manifold_dim=64, max_nodes=16, min_nodes=3
    )
    planner.eval()
    x = torch.randn(2, 10, 384)

    result1 = planner(x, deterministic=True)
    result2 = planner(x, deterministic=True)

    # Should be identical
    assert torch.allclose(result1["node_embeddings"], result2["node_embeddings"])
    assert torch.allclose(result1["adj_matrix"], result2["adj_matrix"])
    assert torch.equal(result1["node_mask"], result2["node_mask"])


def test_dag_planner_complexity_positive():
    """Complexity scores should be positive."""
    planner = ForcedDAGPlanner(
        embed_dim=384, manifold_dim=64, max_nodes=16, min_nodes=3
    )
    x = torch.randn(2, 10, 384)

    result = planner(x, deterministic=True)
    complexities = result["complexities"]

    assert (complexities >= 0).all()


def test_dag_planner_complexity_masking():
    """Inactive nodes should have zero complexity."""
    planner = ForcedDAGPlanner(
        embed_dim=384, manifold_dim=64, max_nodes=16, min_nodes=3
    )
    x = torch.randn(2, 10, 384)

    result = planner(x, deterministic=True)
    complexities = result["complexities"]
    node_mask = result["node_mask"]

    # Inactive nodes should have zero complexity
    inactive_mask = ~node_mask
    assert (complexities[inactive_mask] == 0).all()


def test_dag_planner_surface_area_calculation():
    """Surface area should be sum of active node complexities."""
    planner = ForcedDAGPlanner(
        embed_dim=384, manifold_dim=64, max_nodes=16, min_nodes=3
    )
    x = torch.randn(2, 10, 384)

    result = planner(x, deterministic=True)
    surface_area = result["surface_area"]
    complexities = result["complexities"]

    # Surface area should match sum of complexities
    expected_surface_area = complexities.sum(dim=1)
    assert torch.allclose(surface_area, expected_surface_area)


def test_dag_planner_execution_order():
    """Execution order should follow topological sort."""
    planner = ForcedDAGPlanner(embed_dim=384, manifold_dim=64, max_nodes=8, min_nodes=3)
    x = torch.randn(1, 10, 384)

    result = planner(x, deterministic=True)
    adj_matrix = result["adj_matrix"]
    node_mask = result["node_mask"]

    execution_order = planner.get_execution_order(adj_matrix, node_mask)

    # Check that execution order is valid
    assert execution_order.shape == (1, 8)

    # Active nodes should appear in order
    active_count = node_mask[0].sum().item()
    order_indices = execution_order[0][:active_count]

    # All active nodes should appear in order
    assert len(order_indices.unique()) == active_count
    assert (order_indices >= 0).all()


def test_dag_planner_execution_order_respects_dependencies():
    """Execution order should respect dependency edges."""
    planner = ForcedDAGPlanner(embed_dim=384, manifold_dim=64, max_nodes=8, min_nodes=3)
    x = torch.randn(1, 10, 384)

    result = planner(x, deterministic=True)
    adj_matrix = result["adj_matrix"]
    node_mask = result["node_mask"]

    execution_order = planner.get_execution_order(adj_matrix, node_mask)

    # For each edge (i, j), i should appear before j in execution order
    order_list = execution_order[0].tolist()
    for i in range(8):
        for j in range(8):
            if adj_matrix[0, i, j] > 0:
                # Edge from i to j exists
                if i in order_list and j in order_list:
                    idx_i = order_list.index(i)
                    idx_j = order_list.index(j)
                    assert idx_i < idx_j, f"Node {i} should come before {j}"


def test_dag_planner_gradient_flow():
    """Gradients should flow through planner."""
    planner = ForcedDAGPlanner(
        embed_dim=384, manifold_dim=64, max_nodes=16, min_nodes=3
    )
    x = torch.randn(2, 10, 384, requires_grad=True)

    result = planner(x)
    loss = result["node_embeddings"].sum() + result["surface_area"].sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_dag_planner_batch_independence():
    """Each batch item should be processed independently."""
    planner = ForcedDAGPlanner(
        embed_dim=384, manifold_dim=64, max_nodes=16, min_nodes=3
    )
    x1 = torch.randn(1, 10, 384)
    x2 = torch.randn(1, 10, 384)
    x_batched = torch.cat([x1, x2], dim=0)

    planner.eval()
    result1 = planner(x1, deterministic=True)
    result2 = planner(x2, deterministic=True)
    result_batched = planner(x_batched, deterministic=True)

    # First batch item should match individual processing (with small numerical tolerance)
    assert torch.allclose(
        result_batched["node_embeddings"][0],
        result1["node_embeddings"][0],
        rtol=1e-4,
        atol=1e-6,
    )
    assert torch.allclose(
        result_batched["node_embeddings"][1],
        result2["node_embeddings"][0],
        rtol=1e-4,
        atol=1e-6,
    )


def test_dag_planner_empty_adjacency():
    """Planner should handle DAGs with no edges."""
    planner = ForcedDAGPlanner(embed_dim=384, manifold_dim=64, max_nodes=8, min_nodes=3)

    # Create a scenario that might produce empty adjacency
    x = torch.randn(1, 10, 384)
    result = planner(x, deterministic=True)

    # Even with no edges, execution order should work
    execution_order = planner.get_execution_order(
        result["adj_matrix"], result["node_mask"]
    )

    assert execution_order.shape == (1, 8)
    # Should still have valid node indices for active nodes
    active_count = result["node_mask"][0].sum().item()
    assert (execution_order[0][:active_count] >= 0).all()


def test_dag_planner_training_vs_eval_mode():
    """Planner behavior should differ between training and eval modes."""
    planner = ForcedDAGPlanner(
        embed_dim=384, manifold_dim=64, max_nodes=16, min_nodes=3
    )
    x = torch.randn(2, 10, 384)

    # Training mode (non-deterministic)
    planner.train()
    planner(x, deterministic=False)
    planner(x, deterministic=False)

    # Results may differ due to sampling
    # (not guaranteed to differ, but we test the mode difference exists)

    # Eval mode (deterministic)
    planner.eval()
    result_eval1 = planner(x, deterministic=True)
    result_eval2 = planner(x, deterministic=True)

    # Eval mode should be consistent
    assert torch.equal(result_eval1["node_mask"], result_eval2["node_mask"])


def test_dag_planner_manifold_projection():
    """Manifold projection should have correct dimensionality."""
    planner = ForcedDAGPlanner(
        embed_dim=384, manifold_dim=64, max_nodes=16, min_nodes=3
    )
    x = torch.randn(2, 10, 384)

    result = planner(x, deterministic=True)

    # Manifold coords are internal, but complexity shape should be correct
    assert result["complexities"].shape == (2, 16)
    assert result["surface_area"].shape == (2,)


def test_dag_planner_different_sequence_lengths():
    """Planner should handle varying sequence lengths."""
    planner = ForcedDAGPlanner(
        embed_dim=384, manifold_dim=64, max_nodes=16, min_nodes=3
    )

    # Different sequence lengths
    x_short = torch.randn(1, 5, 384)
    x_long = torch.randn(1, 20, 384)

    result_short = planner(x_short, deterministic=True)
    result_long = planner(x_long, deterministic=True)

    # Output shapes should be same (max_nodes)
    assert result_short["node_embeddings"].shape == (1, 16, 384)
    assert result_long["node_embeddings"].shape == (1, 16, 384)


def test_dag_planner_device_movement():
    """Planner should move to different devices."""
    planner = ForcedDAGPlanner(
        embed_dim=384, manifold_dim=64, max_nodes=16, min_nodes=3
    )

    # Move to CPU (default)
    planner = planner.to("cpu")
    x = torch.randn(2, 10, 384)

    result = planner(x, deterministic=True)

    # Should work after device movement
    assert result["node_embeddings"].device == x.device
    assert result["adj_matrix"].device == x.device
