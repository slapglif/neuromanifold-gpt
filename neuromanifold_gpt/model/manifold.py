"""
Manifold Learning Layer.

Projects SDRs onto a learned Riemannian manifold with local metric tensors.
The manifold captures intrinsic geometry of the semantic space.

Key properties:
- Coordinates: Low-dimensional (e.g., 64-dim from 2048-bit SDRs)
- Metric G(x): Position-dependent, always positive definite via G = A^T A + εI
- Geodesic distance: Uses midpoint metric for pairwise distances
"""
import torch
import torch.nn as nn
from einops import einsum


class ManifoldProjection(nn.Module):
    """
    Project SDRs to low-dimensional manifold with learned Riemannian metric.

    The manifold projection transforms high-dimensional sparse representations
    into a lower-dimensional space while preserving semantic geometry through
    a learned Riemannian metric tensor.

    Args:
        sdr_size: Dimension of input SDR (typically 2048)
        manifold_dim: Dimension of manifold coordinates (typically 64)
        hidden_dim: Hidden layer dimension (default: sdr_size // 2)
    """

    def __init__(self, sdr_size: int, manifold_dim: int, hidden_dim: int | None = None):
        super().__init__()
        self.sdr_size = sdr_size
        self.manifold_dim = manifold_dim
        hidden_dim = hidden_dim or sdr_size // 2

        # SDR -> manifold coordinates
        self.encoder = nn.Sequential(
            nn.Linear(sdr_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, manifold_dim),
        )

        # Local metric tensor G(x)
        # Maps coordinates to a matrix that will be made positive definite
        self.metric_net = nn.Sequential(
            nn.Linear(manifold_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, manifold_dim * manifold_dim),
        )

        # Small epsilon for numerical stability
        self._eps = 1e-4

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize encoder to near-identity or random orthogonal?
        # Initialize metric net to output near zero (so A is small, G is approx Identity)
        # This makes the manifold locally Euclidean at initialization
        for m in self.metric_net:
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

        # Encoder: Kaiming normal
        for m in self.encoder:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, sdr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Project SDRs to manifold.

        Args:
            sdr: (B, T, sdr_size) input SDR vectors

        Returns:
            coords: (B, T, manifold_dim) manifold coordinates
            metric: (B, T, manifold_dim, manifold_dim) positive definite metric tensor
        """
        # Project SDRs to manifold coordinates
        coords = self.encoder(sdr)

        # Compute local metric tensor
        metric_flat = self.metric_net(coords)
        B, T, _ = coords.shape
        A = metric_flat.view(B, T, self.manifold_dim, self.manifold_dim)

        # G = A^T A + εI ensures positive definite
        # Using einsum: A^T A means contracting on the last dimension of A^T and A
        metric = einsum(A, A, "b t i j, b t k j -> b t i k")

        # Add εI for numerical stability
        eye = torch.eye(self.manifold_dim, device=metric.device, dtype=metric.dtype)
        metric = metric + self._eps * eye

        return coords, metric

    def geodesic_distance(
        self, coords: torch.Tensor, metric: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute geodesic distances using Riemannian metric.

        Uses the midpoint approximation for the metric:
            d²(i,j) = (x_i - x_j)^T G_mid (x_i - x_j)
            where G_mid = (G_i + G_j) / 2

        This is a first-order approximation to the true geodesic distance,
        which would require integrating along the geodesic path.

        Args:
            coords: (B, T, D) manifold coordinates
            metric: (B, T, D, D) metric tensors at each point

        Returns:
            dist_sq: (B, T, T) pairwise squared geodesic distances
        """
        B, T, D = coords.shape

        # Pairwise differences: x_i - x_j
        # (B, T, 1, D) - (B, 1, T, D) -> (B, T, T, D)
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)

        # Midpoint metric: G_mid = (G_i + G_j) / 2
        # metric_i: (B, T, 1, D, D)
        # metric_j: (B, 1, T, D, D)
        metric_i = metric.unsqueeze(2)
        metric_j = metric.unsqueeze(1)
        metric_mid = (metric_i + metric_j) / 2  # (B, T, T, D, D)

        # d² = diff^T @ G @ diff
        # First: G @ diff -> (B, T, T, D)
        temp = einsum(metric_mid, diff, "b i j d e, b i j e -> b i j d")
        # Then: diff^T @ (G @ diff) -> (B, T, T)
        dist_sq = einsum(diff, temp, "b i j d, b i j d -> b i j")

        return dist_sq
