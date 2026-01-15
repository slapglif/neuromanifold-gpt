"""
Forced DAG Planner for System 2 Reasoning.

Decomposes complex tasks into a directed acyclic graph (DAG) of subtasks.
This enables hierarchical planning and counterfactual reasoning.

Key properties:
- Node embeddings: Learned representations of subtask nodes
- Adjacency matrix: DAG structure enforcing causal dependencies
- Surface area: Topological complexity measure (Betti numbers)
- Task complexities: Per-node difficulty estimates

Reference: Planning as inference in hierarchical task networks
"""
import torch
import torch.nn as nn
from einops import einsum


class ForcedDAGPlanner(nn.Module):
    """
    Plan tasks as directed acyclic graphs on learned manifold.

    Projects sequence embeddings into a DAG structure where nodes represent
    subtasks and edges represent dependencies. The DAG is "forced" to be acyclic
    through a learned upper-triangular structure.

    Args:
        embed_dim: Dimension of input embeddings (typically 384)
        manifold_dim: Dimension of manifold for task representation (typically 64)
        max_nodes: Maximum number of nodes in DAG (typically 32)
        min_nodes: Minimum number of nodes in DAG (typically 3)
        hidden_dim: Hidden layer dimension (default: embed_dim)
    """

    def __init__(
        self,
        embed_dim: int,
        manifold_dim: int,
        max_nodes: int = 32,
        min_nodes: int = 3,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.manifold_dim = manifold_dim
        self.max_nodes = max_nodes
        self.min_nodes = min_nodes
        hidden_dim = hidden_dim or embed_dim

        # Sequence -> node embeddings
        # Compress sequence to fixed number of nodes
        self.node_encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, max_nodes * manifold_dim),
        )

        # Node embeddings -> adjacency matrix (upper triangular for DAG)
        # Project to max_nodes^2 logits, mask to upper triangular
        self.adj_net = nn.Sequential(
            nn.Linear(max_nodes * manifold_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, max_nodes * max_nodes),
        )

        # Surface area estimation (topological complexity via Betti numbers)
        # Simple proxy: count of connected components and cycles
        self.surface_net = nn.Sequential(
            nn.Linear(max_nodes * manifold_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Per-node complexity scores
        self.complexity_net = nn.Sequential(
            nn.Linear(manifold_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        # Upper triangular mask for DAG (no self-loops, no backward edges)
        # Shape: (max_nodes, max_nodes)
        # mask[i,j] = 1 if i < j (edge from i to j allowed), 0 otherwise
        mask = torch.triu(torch.ones(max_nodes, max_nodes), diagonal=1)
        self.register_buffer("dag_mask", mask)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        # Node encoder: Kaiming initialization
        for m in self.node_encoder:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Adjacency net: small initialization (sparse graphs initially)
        for m in self.adj_net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Surface and complexity nets: Xavier initialization
        for net in [self.surface_net, self.complexity_net]:
            for m in net:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Plan task decomposition as DAG.

        Args:
            x: (B, T, embed_dim) sequence embeddings

        Returns:
            dict with keys:
                node_embeddings: (B, max_nodes, manifold_dim) node representations
                adj_matrix: (B, max_nodes, max_nodes) DAG adjacency (upper triangular)
                surface_area: (B,) topological complexity measure
                complexities: (B, max_nodes) per-node difficulty scores
        """
        B, T, D = x.shape

        # Pool sequence to single vector for planning
        # Simple mean pooling (can be upgraded to attention pooling)
        x_pooled = x.mean(dim=1)  # (B, embed_dim)

        # Generate node embeddings
        node_flat = self.node_encoder(x_pooled)  # (B, max_nodes * manifold_dim)
        node_embeddings = node_flat.view(B, self.max_nodes, self.manifold_dim)

        # Generate adjacency matrix logits
        adj_logits = self.adj_net(node_flat)  # (B, max_nodes^2)
        adj_logits = adj_logits.view(B, self.max_nodes, self.max_nodes)

        # Apply DAG mask (upper triangular) and sigmoid
        # This forces acyclic structure: edges only from lower to higher indices
        adj_matrix = torch.sigmoid(adj_logits) * self.dag_mask.unsqueeze(0)

        # Estimate surface area (topological complexity)
        surface_area = self.surface_net(node_flat).squeeze(-1)  # (B,)
        surface_area = torch.relu(surface_area)  # Non-negative

        # Per-node complexity scores
        complexities = self.complexity_net(node_embeddings).squeeze(-1)  # (B, max_nodes)
        complexities = torch.relu(complexities)  # Non-negative

        return {
            "node_embeddings": node_embeddings,
            "adj_matrix": adj_matrix,
            "surface_area": surface_area,
            "complexities": complexities,
        }


class DAGExecutor(nn.Module):
    """
    Execute DAG plan via topological traversal.

    Given a DAG from ForcedDAGPlanner, execute nodes in topological order
    and propagate information along edges.

    This is a stub for future implementation.
    """

    def __init__(self, embed_dim: int, manifold_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.manifold_dim = manifold_dim

        # Placeholder: node update function
        self.node_update = nn.Linear(manifold_dim, manifold_dim)

    def forward(
        self, node_embeddings: torch.Tensor, adj_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Execute DAG in topological order.

        Args:
            node_embeddings: (B, N, manifold_dim)
            adj_matrix: (B, N, N) adjacency matrix

        Returns:
            updated_embeddings: (B, N, manifold_dim)
        """
        # Stub: simple linear transformation
        # TODO: Implement proper topological traversal and message passing
        return self.node_update(node_embeddings)
