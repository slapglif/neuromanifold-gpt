"""
ForcedDAGPlanner - Decomposes tasks into DAGs for systematic reasoning.

Inspired by System 2 thinking: break complex problems into structured
subproblems with explicit dependencies. The planner generates a DAG
where nodes are reasoning steps and edges are dependencies.

Key components:
- Node generation: Extract meaningful reasoning steps from input
- Adjacency prediction: Learn which steps depend on which others
- DAG enforcement: Ensure acyclic structure via upper-triangular masking
- Execution ordering: Topological sort for sequential execution
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ForcedDAGPlanner(nn.Module):
    """
    Generate DAG decomposition of reasoning tasks.

    Takes input embeddings and produces a structured plan as a DAG,
    where each node is a reasoning step and edges represent dependencies.
    Enforces valid DAG structure via architectural constraints.

    Args:
        embed_dim: Dimension of input embeddings
        manifold_dim: Dimension of manifold projection (for complexity estimation)
        max_nodes: Maximum number of planning nodes
        min_nodes: Minimum number of planning nodes
        hidden_dim: Hidden layer dimension (default: embed_dim * 2)
        dropout: Dropout rate (default: 0.0)
    """

    def __init__(
        self,
        embed_dim: int,
        manifold_dim: int = 64,
        max_nodes: int = 32,
        min_nodes: int = 3,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.manifold_dim = manifold_dim
        self.max_nodes = max_nodes
        self.min_nodes = min_nodes
        hidden_dim = hidden_dim or embed_dim * 2

        # Node count predictor: estimate problem complexity
        self.node_count_predictor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Node generator: extract reasoning steps
        self.node_generator = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max_nodes * embed_dim),
        )

        # Adjacency predictor: learn dependencies between nodes
        # Takes concatenated node pairs and predicts edge probability
        self.adjacency_predictor = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Manifold projection for complexity estimation
        self.manifold_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, manifold_dim),
        )

        # Complexity estimator: predict difficulty of each node
        self.complexity_net = nn.Sequential(
            nn.Linear(manifold_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        # Initialize node count predictor to output around mid-range
        with torch.no_grad():
            avg_nodes = (self.min_nodes + self.max_nodes) / 2
            if hasattr(self.node_count_predictor[-1], 'bias'):
                self.node_count_predictor[-1].bias.fill_(avg_nodes)

        # Initialize adjacency predictor to be sparse (fewer edges initially)
        for m in self.adjacency_predictor:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, -1.0)  # Bias toward sparse

    def forward(
        self,
        x: torch.Tensor,
        deterministic: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Generate DAG plan from input embeddings.

        Args:
            x: (B, T, embed_dim) input embeddings
            deterministic: If True, use argmax instead of sampling

        Returns:
            Dictionary with:
                - node_embeddings: (B, max_nodes, embed_dim) reasoning step embeddings
                - adj_matrix: (B, max_nodes, max_nodes) adjacency matrix (0/1)
                - node_mask: (B, max_nodes) which nodes are active
                - surface_area: (B,) estimated problem surface area
                - complexities: (B, max_nodes) complexity score per node
        """
        B, T, D = x.shape

        # 1. Predict number of nodes needed (problem complexity)
        # Pool sequence for global context
        x_pooled = x.mean(dim=1)  # (B, embed_dim)

        node_count_logit = self.node_count_predictor(x_pooled).squeeze(-1)  # (B,)
        node_count_continuous = torch.sigmoid(node_count_logit) * (self.max_nodes - self.min_nodes) + self.min_nodes

        if deterministic:
            node_count = node_count_continuous.round().long()
        else:
            # Add small noise during training
            node_count = (node_count_continuous + 0.1 * torch.randn_like(node_count_continuous)).round().long()

        node_count = torch.clamp(node_count, self.min_nodes, self.max_nodes)

        # 2. Generate node embeddings (reasoning steps)
        node_features = self.node_generator(x_pooled)  # (B, max_nodes * embed_dim)
        node_embeddings = node_features.view(B, self.max_nodes, self.embed_dim)

        # 3. Create node mask based on predicted count
        node_mask = torch.zeros(B, self.max_nodes, device=x.device, dtype=torch.bool)
        for b in range(B):
            node_mask[b, :node_count[b]] = True

        # 4. Predict adjacency matrix (dependencies)
        # For each pair (i, j), predict if there's an edge from i to j
        adj_matrix = torch.zeros(B, self.max_nodes, self.max_nodes, device=x.device)

        for i in range(self.max_nodes):
            for j in range(self.max_nodes):
                if i >= j:
                    # Enforce DAG constraint: only edges from lower to higher index
                    # This guarantees acyclic structure
                    continue

                # Concatenate node embeddings
                pair = torch.cat([node_embeddings[:, i], node_embeddings[:, j]], dim=-1)  # (B, 2*D)
                edge_logit = self.adjacency_predictor(pair).squeeze(-1)  # (B,)

                if deterministic:
                    edge = (torch.sigmoid(edge_logit) > 0.5).float()
                else:
                    # Gumbel-softmax for differentiable sampling
                    edge = torch.sigmoid(edge_logit)
                    if self.training:
                        # Bernoulli sampling during training
                        edge = (torch.rand_like(edge) < edge).float()

                adj_matrix[:, i, j] = edge

        # Apply node mask to adjacency matrix
        adj_matrix = adj_matrix * node_mask.unsqueeze(1).float() * node_mask.unsqueeze(2).float()

        # 5. Estimate complexity via manifold projection
        manifold_coords = self.manifold_proj(node_embeddings)  # (B, max_nodes, manifold_dim)
        complexities = self.complexity_net(manifold_coords).squeeze(-1)  # (B, max_nodes)
        complexities = F.softplus(complexities)  # Ensure positive

        # Mask out inactive nodes
        complexities = complexities * node_mask.float()

        # 6. Compute surface area (total problem complexity)
        surface_area = complexities.sum(dim=1)  # (B,)

        return {
            "node_embeddings": node_embeddings,
            "adj_matrix": adj_matrix,
            "node_mask": node_mask,
            "surface_area": surface_area,
            "complexities": complexities,
        }

    def get_execution_order(
        self,
        adj_matrix: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute topological ordering for DAG execution.

        Uses Kahn's algorithm: repeatedly remove nodes with no incoming edges.

        Args:
            adj_matrix: (B, max_nodes, max_nodes) adjacency matrix
            node_mask: (B, max_nodes) which nodes are active

        Returns:
            execution_order: (B, max_nodes) indices in topological order
                            (-1 for invalid/inactive nodes)
        """
        B, N, _ = adj_matrix.shape
        device = adj_matrix.device

        execution_order = torch.full((B, N), -1, dtype=torch.long, device=device)

        # Process each batch item separately (no efficient batch topo sort)
        for b in range(B):
            adj = adj_matrix[b].clone()
            mask = node_mask[b].clone()
            order = []

            # Count incoming edges for each node
            in_degree = adj.sum(dim=0)  # (N,)

            # Find nodes with no incoming edges
            available = (in_degree == 0) & mask

            while available.any():
                # Pick first available node (deterministic)
                node = available.nonzero(as_tuple=True)[0][0].item()
                order.append(node)

                # Remove this node
                mask[node] = False
                available[node] = False

                # Update in-degrees
                for j in range(N):
                    if adj[node, j] > 0:
                        in_degree[j] -= 1

                # Update available nodes
                available = (in_degree == 0) & mask

            # Fill execution order
            for i, node_idx in enumerate(order):
                execution_order[b, i] = node_idx

        return execution_order
