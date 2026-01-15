"""
ForcedDAGPlanner - System 2 reasoning via task decomposition.

Decomposes complex tasks into Directed Acyclic Graphs (DAGs) for
systematic multi-step reasoning. Inspired by human deliberate planning.

Key insight: Complex reasoning requires breaking problems into subtasks.
The DAG structure enforces logical dependencies while preventing circular
reasoning. This is the "System 2" thinking complement to fast "System 1"
associative reasoning.

Architecture:
1. Task Encoder: Embeds input as initial task representation
2. Node Generator: Predicts N subtask nodes via learned manifold projection
3. Edge Predictor: Computes pairwise dependencies (adjacency matrix)
4. DAG Enforcer: Masks edges to ensure acyclicity via triangular constraint
5. Complexity Estimator: Predicts surface area (info content) of each node

The manifold projection enables geometric reasoning about task structure:
nearby points in manifold space represent related subtasks, and geodesic
distances approximate dependency strength.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ForcedDAGPlanner(nn.Module):
    """
    Decomposes tasks into DAGs for structured reasoning.

    Takes embeddings and produces a graph of reasoning steps with
    enforced acyclic structure. Each node represents a subtask,
    edges represent dependencies, and topological ordering gives
    execution sequence.

    This enables "thinking before answering" - the model can plan
    multiple steps ahead rather than generating tokens reactively.

    Args:
        embed_dim: Dimension of input embeddings
        manifold_dim: Dimension for manifold projection (default 64)
        max_nodes: Maximum number of nodes in DAG (default 32)
        min_nodes: Minimum number of nodes to generate (default 3)
        dropout: Dropout probability (default 0.0)

    Example:
        >>> planner = ForcedDAGPlanner(embed_dim=384, manifold_dim=64)
        >>> x = torch.randn(2, 10, 384)  # (batch, seq_len, embed_dim)
        >>> output = planner(x, deterministic=False)
        >>> print(output.keys())
        dict_keys(['node_embeddings', 'adj_matrix', 'surface_area',
                   'complexities', 'node_mask'])
        >>> # Get execution order
        >>> order = planner.get_execution_order(
        ...     output['adj_matrix'][0],
        ...     output['node_mask'][0]
        ... )
    """

    def __init__(
        self,
        embed_dim: int,
        manifold_dim: int = 64,
        max_nodes: int = 32,
        min_nodes: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.manifold_dim = manifold_dim
        self.max_nodes = max_nodes
        self.min_nodes = min_nodes
        self.dropout = dropout

        # Task encoder: aggregate sequence into single task representation
        # Uses attention pooling to focus on most relevant tokens
        self.task_query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.task_attn = nn.MultiheadAttention(
            embed_dim, num_heads=8, dropout=dropout, batch_first=True
        )

        # Node generator: task -> N node embeddings
        # Projects to manifold space for geometric reasoning
        self.manifold_proj = nn.Linear(embed_dim, manifold_dim)
        self.node_generator = nn.Sequential(
            nn.Linear(manifold_dim, manifold_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(manifold_dim * 2, max_nodes * manifold_dim),
        )

        # Edge predictor: compute pairwise node affinities
        # Uses bilinear form for efficiency: e_ij = node_i^T W node_j
        self.edge_predictor = nn.Bilinear(manifold_dim, manifold_dim, 1)

        # Node importance: predict which nodes are active
        # Allows variable-size DAGs (min_nodes to max_nodes)
        self.node_importance = nn.Linear(manifold_dim, 1)

        # Complexity estimator: predict surface area (information content)
        # Inspired by holographic principle - complexity ~ surface area
        self.complexity_predictor = nn.Sequential(
            nn.Linear(manifold_dim, manifold_dim // 2),
            nn.GELU(),
            nn.Linear(manifold_dim // 2, 1),
        )

        # Project manifold nodes back to embedding space
        self.node_proj = nn.Linear(manifold_dim, embed_dim)

        # Layer norm for stability
        self.norm = nn.LayerNorm(manifold_dim)

    def forward(
        self, x: torch.Tensor, deterministic: bool = False
    ) -> dict[str, torch.Tensor]:
        """
        Decompose input into task DAG.

        Args:
            x: Input embeddings (batch_size, seq_len, embed_dim)
            deterministic: If True, use argmax for node selection (eval mode)

        Returns:
            Dictionary containing:
                - node_embeddings: (batch, max_nodes, embed_dim) node representations
                - adj_matrix: (batch, max_nodes, max_nodes) adjacency matrix (DAG)
                - surface_area: (batch,) total surface area (sum of complexities)
                - complexities: (batch, max_nodes) per-node complexity scores
                - node_mask: (batch, max_nodes) boolean mask of active nodes
        """
        batch_size, seq_len, _ = x.shape

        # 1. Encode task: aggregate sequence via attention pooling
        task_query = self.task_query.expand(batch_size, -1, -1)
        task_repr, _ = self.task_attn(
            task_query, x, x, need_weights=False
        )  # (batch, 1, embed_dim)
        task_repr = task_repr.squeeze(1)  # (batch, embed_dim)

        # 2. Project to manifold space
        task_manifold = self.manifold_proj(task_repr)  # (batch, manifold_dim)
        task_manifold = self.norm(task_manifold)

        # 3. Generate nodes in manifold space
        node_flat = self.node_generator(task_manifold)  # (batch, max_nodes * manifold_dim)
        node_manifold = node_flat.view(
            batch_size, self.max_nodes, self.manifold_dim
        )  # (batch, max_nodes, manifold_dim)
        node_manifold = self.norm(node_manifold)

        # 4. Predict node importance (which nodes are active)
        importance_logits = self.node_importance(node_manifold).squeeze(
            -1
        )  # (batch, max_nodes)

        # Sample or select nodes (top-k)
        if deterministic:
            # Eval mode: select top-k nodes
            # Ensure at least min_nodes
            k = max(self.min_nodes, min(self.max_nodes, seq_len // 2 + 1))
            _, top_indices = torch.topk(importance_logits, k=k, dim=-1)
            node_mask = torch.zeros_like(importance_logits, dtype=torch.bool)
            node_mask.scatter_(1, top_indices, True)
        else:
            # Training mode: use Gumbel-Softmax for differentiability
            # Encourage min_nodes by adding bias
            importance_logits_biased = importance_logits.clone()
            importance_logits_biased[:, : self.min_nodes] += 2.0

            node_probs = torch.sigmoid(importance_logits_biased)
            # Sample with Gumbel trick
            gumbel_noise = -torch.log(
                -torch.log(torch.rand_like(node_probs) + 1e-10) + 1e-10
            )
            node_mask = (node_probs + gumbel_noise * 0.1) > 0.5

            # Ensure at least min_nodes are active
            active_count = node_mask.sum(dim=-1, keepdim=True)
            if (active_count < self.min_nodes).any():
                # Force top min_nodes to be active
                _, top_min = torch.topk(
                    importance_logits, k=self.min_nodes, dim=-1
                )
                node_mask.scatter_(1, top_min, True)

        # 5. Predict edge affinities (dependencies between nodes)
        # Compute all pairwise affinities using bilinear form
        # e_ij = node_i^T W node_j
        edge_logits = torch.zeros(
            batch_size, self.max_nodes, self.max_nodes, device=x.device
        )
        for i in range(self.max_nodes):
            for j in range(self.max_nodes):
                edge_logits[:, i, j] = self.edge_predictor(
                    node_manifold[:, i], node_manifold[:, j]
                ).squeeze(-1)

        # 6. Enforce DAG structure: mask upper triangle (i < j means i depends on j)
        # This ensures acyclicity - execution flows from high indices to low
        dag_mask = torch.triu(
            torch.ones(self.max_nodes, self.max_nodes, device=x.device), diagonal=1
        )
        dag_mask = dag_mask.unsqueeze(0).expand(batch_size, -1, -1)

        # Apply DAG mask and node mask
        adj_matrix = torch.sigmoid(edge_logits) * dag_mask
        # Mask out edges involving inactive nodes
        node_mask_2d = node_mask.unsqueeze(1) & node_mask.unsqueeze(
            2
        )  # (batch, max_nodes, max_nodes)
        adj_matrix = adj_matrix * node_mask_2d.float()

        # Apply threshold for sparsity
        adj_matrix = (adj_matrix > 0.3).float()

        # 7. Predict complexity (surface area) for each node
        complexities = self.complexity_predictor(node_manifold).squeeze(
            -1
        )  # (batch, max_nodes)
        complexities = F.softplus(complexities)  # Ensure positive
        complexities = complexities * node_mask.float()  # Mask inactive nodes

        # Total surface area (sum of node complexities)
        surface_area = complexities.sum(dim=-1)  # (batch,)

        # 8. Project nodes back to embedding space for downstream use
        node_embeddings = self.node_proj(node_manifold)  # (batch, max_nodes, embed_dim)

        return {
            "node_embeddings": node_embeddings,
            "adj_matrix": adj_matrix,
            "surface_area": surface_area,
            "complexities": complexities,
            "node_mask": node_mask,
        }

    def get_execution_order(
        self, adj_matrix: torch.Tensor, node_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute topological ordering of DAG (Kahn's algorithm).

        Returns the order in which nodes should be executed, respecting
        dependency constraints. Nodes with no dependencies come first.

        Args:
            adj_matrix: (max_nodes, max_nodes) adjacency matrix
                       adj[i,j]=1 means i depends on j (j must execute before i)
            node_mask: (max_nodes,) boolean mask of active nodes

        Returns:
            execution_order: (num_active_nodes,) indices in execution order
        """
        # Extract active nodes
        active_indices = torch.where(node_mask)[0]
        if len(active_indices) == 0:
            return torch.tensor([], dtype=torch.long, device=adj_matrix.device)

        # Build subgraph with only active nodes
        sub_adj = adj_matrix[active_indices][:, active_indices]  # (num_active, num_active)
        num_active = len(active_indices)

        # Compute in-degree for each node (number of dependencies)
        in_degree = sub_adj.sum(dim=0)  # (num_active,)

        # Kahn's algorithm for topological sort
        execution_order = []
        queue = torch.where(in_degree == 0)[0].tolist()

        while queue:
            # Process node with no remaining dependencies
            node = queue.pop(0)
            execution_order.append(active_indices[node].item())

            # Remove this node's outgoing edges and update in-degrees
            for neighbor in range(num_active):
                if sub_adj[neighbor, node] > 0:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

        # Convert to tensor
        execution_order = torch.tensor(
            execution_order, dtype=torch.long, device=adj_matrix.device
        )

        # If we didn't process all nodes, the graph had a cycle (shouldn't happen)
        # Return partial order
        if len(execution_order) < num_active:
            # Add remaining nodes in arbitrary order
            remaining = set(range(num_active)) - set(
                [i for i, idx in enumerate(active_indices) if idx.item() in execution_order]
            )
            execution_order = torch.cat(
                [
                    execution_order,
                    torch.tensor(
                        [active_indices[i].item() for i in remaining],
                        dtype=torch.long,
                        device=adj_matrix.device,
                    ),
                ]
            )

        return execution_order
