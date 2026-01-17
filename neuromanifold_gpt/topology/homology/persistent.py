from typing import List

import torch
import torch.nn as nn


class PersistentHomology(nn.Module):
    """
    Persistent Homology for manifold smoothness regularization.
    """

    def __init__(self, max_dimension: int = 2):
        super().__init__()
        self.max_dimension = max_dimension

    def compute_distance_matrix(self, embeddings: torch.Tensor) -> torch.Tensor:
        batch, seq_len, embed_dim = embeddings.shape

        embeddings_flat = embeddings.view(-1, embed_dim)
        dist = torch.cdist(embeddings_flat, embeddings_flat)

        return dist

    def compute_betti_numbers(
        self, distance_matrix: torch.Tensor, threshold: float
    ) -> List[int]:
        adj_matrix = (distance_matrix < threshold).float()

        betti_0 = self._compute_connected_components(adj_matrix)

        return [betti_0]

    def _compute_connected_components(self, adj_matrix: torch.Tensor) -> int:
        n = adj_matrix.size(0)
        visited = torch.zeros(n, dtype=torch.bool, device=adj_matrix.device)
        num_components = 0

        for i in range(n):
            if not visited[i]:
                self._dfs(adj_matrix, visited, i)
                num_components += 1

        return num_components

    def _dfs(self, adj_matrix: torch.Tensor, visited: torch.Tensor, node: int):
        visited[node] = True
        neighbors = torch.where(adj_matrix[node] > 0)[0]
        for neighbor in neighbors:
            if not visited[neighbor]:
                self._dfs(adj_matrix, visited, neighbor.item())

    def compute_smoothness_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        dist_matrix = self.compute_distance_matrix(embeddings)

        mean_dist = dist_matrix.mean()
        std_dist = dist_matrix.std()

        smoothness_loss = std_dist / (mean_dist + 1e-8)

        return smoothness_loss

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.compute_smoothness_loss(embeddings)
