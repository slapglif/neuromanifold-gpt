# neuromanifold_gpt/model/topology/topological_head.py
"""
Topological Head for Loss Computation.

The TopologicalHead computes auxiliary loss terms based on topological invariants
derived from the Jones polynomial and braid group representations. This provides
regularization that encourages the model to learn representations with consistent
topological structure.

Loss components:
1. Polynomial smoothness: Encourages smooth Jones polynomial coefficients
2. Braid consistency: Encourages stable braid group representations
3. Topological regularity: Penalizes extreme topological complexity
4. Temporal coherence: Encourages topologically similar nearby tokens

The head can be added to any transformer-like model to provide topological
inductive biases without modifying the main architecture.

Example:
    >>> th = TopologicalHead(embed_dim=384)
    >>> x = torch.randn(2, 32, 384)
    >>> loss, info = th(x)
    >>> loss.dim()  # Scalar loss
    0

Reference:
- Jones, V.F.R. "A polynomial invariant for knots via von Neumann algebras" (1985)
- Birman, J.S. "Braids, Links, and Mapping Class Groups" (1974)
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .braid import BraidEncoder
from .jones_polynomial import JonesApproximator


@dataclass
class TopologicalHeadConfig:
    """Configuration for TopologicalHead."""

    embed_dim: int = 384
    n_strands: int = 4
    n_coefficients: int = 8
    # Loss component weights
    smoothness_weight: float = 1.0
    consistency_weight: float = 0.5
    regularity_weight: float = 0.1
    coherence_weight: float = 0.5
    # Feature flags
    use_jones: bool = True
    use_braid: bool = True
    use_temporal_coherence: bool = True
    # Hyperparameters
    smoothness_order: int = 2  # Order of finite differences for smoothness
    coherence_window: int = 3  # Window size for temporal coherence
    dropout: float = 0.0


class TopologicalHead(nn.Module):
    """
    Head module for computing topological loss terms.

    Combines braid group representations and Jones polynomial approximations
    to compute auxiliary losses that encourage topologically consistent
    representations.

    The loss components are:

    1. **Polynomial Smoothness Loss**: Penalizes large jumps in polynomial
       coefficients, encouraging smooth topological features.
       L_smooth = ||diff^k(coefficients)||^2

    2. **Braid Consistency Loss**: Encourages the braid representation to
       be close to valid braid group elements (stable under small perturbations).
       L_consist = ||rep - normalize(rep)||^2

    3. **Topological Regularity Loss**: Penalizes extreme values in topological
       invariants to prevent degenerate solutions.
       L_reg = ||invariants||_1

    4. **Temporal Coherence Loss**: Encourages nearby tokens to have similar
       topological features (local consistency).
       L_coh = mean(||poly_t - poly_{t+1}||^2)

    Total loss: L = w1*L_smooth + w2*L_consist + w3*L_reg + w4*L_coh

    Example:
        >>> head = TopologicalHead(384)
        >>> x = torch.randn(2, 32, 384)
        >>> loss, info = head(x)
        >>> assert loss.dim() == 0  # Scalar
        >>> assert 'smoothness_loss' in info
    """

    def __init__(
        self,
        embed_dim: int,
        n_strands: int = 4,
        n_coefficients: int = 8,
        smoothness_weight: float = 1.0,
        consistency_weight: float = 0.5,
        regularity_weight: float = 0.1,
        coherence_weight: float = 0.5,
        use_jones: bool = True,
        use_braid: bool = True,
        use_temporal_coherence: bool = True,
        smoothness_order: int = 2,
        coherence_window: int = 3,
        dropout: float = 0.0,
    ):
        """
        Initialize TopologicalHead.

        Args:
            embed_dim: Input embedding dimension
            n_strands: Number of braid strands
            n_coefficients: Number of polynomial coefficients
            smoothness_weight: Weight for smoothness loss
            consistency_weight: Weight for consistency loss
            regularity_weight: Weight for regularity loss
            coherence_weight: Weight for temporal coherence loss
            use_jones: Enable Jones polynomial computation
            use_braid: Enable braid representation computation
            use_temporal_coherence: Enable temporal coherence loss
            smoothness_order: Order of finite differences for smoothness
            coherence_window: Window size for temporal coherence
            dropout: Dropout probability
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.n_strands = n_strands
        self.n_coefficients = n_coefficients

        # Loss weights
        self.smoothness_weight = smoothness_weight
        self.consistency_weight = consistency_weight
        self.regularity_weight = regularity_weight
        self.coherence_weight = coherence_weight

        # Feature flags
        self.use_jones = use_jones
        self.use_braid = use_braid
        self.use_temporal_coherence = use_temporal_coherence

        # Hyperparameters
        self.smoothness_order = smoothness_order
        self.coherence_window = coherence_window

        # Braid encoder for representation computation
        if use_braid:
            self.braid_encoder = BraidEncoder(
                embed_dim=embed_dim,
                n_strands=n_strands,
                use_reduced=True,
                dropout=dropout,
            )
            self.rep_dim = n_strands - 1

        # Jones polynomial approximator
        if use_jones:
            self.jones = JonesApproximator(
                embed_dim=embed_dim,
                n_strands=n_strands,
                n_coefficients=n_coefficients,
                dropout=dropout,
            )

        # Per-position feature extractor for temporal coherence
        if use_temporal_coherence:
            self.position_encoder = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, n_coefficients * 2),
            )

        # Learnable temperature for loss scaling (scalar)
        self.log_temperature = nn.Parameter(torch.tensor(0.0))

    def compute_smoothness_loss(
        self,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute smoothness loss for polynomial coefficients.

        Uses finite differences to measure smoothness of coefficients.
        Lower-order differences penalize more, encouraging smooth polynomials.

        Args:
            coefficients: Polynomial coefficients (B, n_coef)

        Returns:
            Scalar smoothness loss
        """
        B = coefficients.shape[0]

        # Compute finite differences up to smoothness_order
        diff = coefficients
        loss = torch.tensor(0.0, device=coefficients.device)

        for k in range(1, self.smoothness_order + 1):
            # k-th order finite difference
            if diff.shape[-1] > 1:
                diff = diff[..., 1:] - diff[..., :-1]
                # Weight by 1/k! to balance different orders
                weight = 1.0 / max(1, k)
                loss = loss + weight * diff.pow(2).mean()

        return loss

    def compute_consistency_loss(
        self,
        braid_rep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute consistency loss for braid representation.

        Encourages the representation to be a valid braid group element
        by penalizing deviation from normalized matrices.

        Args:
            braid_rep: Braid representation (B, rep_dim, rep_dim)

        Returns:
            Scalar consistency loss
        """
        B = braid_rep.shape[0]

        # Compute matrix norm
        rep_norm = torch.linalg.matrix_norm(braid_rep, ord='fro')

        # Normalize representation
        rep_normalized = braid_rep / (rep_norm.unsqueeze(-1).unsqueeze(-1) + 1e-8)

        # Consistency: distance from normalized version
        # This encourages representations close to unit-norm matrices
        consistency = (braid_rep - rep_normalized * rep_norm.unsqueeze(-1).unsqueeze(-1)).pow(2)

        return consistency.mean()

    def compute_regularity_loss(
        self,
        invariants: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute regularity loss for topological invariants.

        Penalizes extreme values to prevent degenerate solutions
        and encourage well-behaved invariants.

        Args:
            invariants: Topological invariants (B, dim)

        Returns:
            Scalar regularity loss
        """
        # L1 regularization for sparsity
        l1_reg = invariants.abs().mean()

        # Soft penalty for very large values (beyond 3 std)
        std = invariants.std() + 1e-8
        extreme_penalty = F.relu(invariants.abs() - 3 * std).mean()

        return l1_reg + 0.1 * extreme_penalty

    def compute_coherence_loss(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute temporal coherence loss.

        Encourages nearby positions to have similar topological features,
        providing local consistency without global uniformity.

        Args:
            x: Input tensor (B, T, D)

        Returns:
            Scalar coherence loss
        """
        B, T, D = x.shape

        if T < 2:
            return torch.tensor(0.0, device=x.device)

        # Compute per-position topological features
        pos_features = self.position_encoder(x)  # (B, T, n_coef*2)

        # Compute pairwise distances within window
        loss = torch.tensor(0.0, device=x.device)
        count = 0

        for offset in range(1, min(self.coherence_window + 1, T)):
            # Features at position t and t+offset
            feat_t = pos_features[:, :-offset]  # (B, T-offset, n_coef*2)
            feat_t_off = pos_features[:, offset:]  # (B, T-offset, n_coef*2)

            # Distance penalty (weighted by offset)
            weight = 1.0 / offset  # Closer positions weighted more
            dist = (feat_t - feat_t_off).pow(2).mean()
            loss = loss + weight * dist
            count += 1

        if count > 0:
            loss = loss / count

        return loss

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute topological loss from input sequence.

        Args:
            x: Input tensor of shape (B, T, D)
            mask: Optional attention mask of shape (B, T)

        Returns:
            Tuple of (loss, info) where:
            - loss: Scalar loss value
            - info: Dictionary with component losses and statistics
        """
        B, T, D = x.shape
        device = x.device

        info = {}
        total_loss = torch.tensor(0.0, device=device)

        # Temperature scaling
        temperature = torch.exp(self.log_temperature)

        # Jones polynomial computation
        if self.use_jones:
            poly_features, jones_info = self.jones(x, mask=mask)

            # Smoothness loss on polynomial coefficients
            smoothness_loss = self.compute_smoothness_loss(poly_features)
            total_loss = total_loss + self.smoothness_weight * smoothness_loss
            info['smoothness_loss'] = smoothness_loss.item()

            # Regularity loss on invariants
            regularity_loss = self.compute_regularity_loss(poly_features)
            total_loss = total_loss + self.regularity_weight * regularity_loss
            info['regularity_loss'] = regularity_loss.item()

            # Store Jones info
            info['poly_norm'] = poly_features.norm(dim=-1).mean().item()
            info.update({f'jones_{k}': v for k, v in jones_info.items() if isinstance(v, (int, float))})

        # Braid representation computation
        if self.use_braid:
            braid_rep, braid_info = self.braid_encoder(x, mask=mask, return_info=True)

            # Consistency loss on braid representation
            consistency_loss = self.compute_consistency_loss(braid_rep)
            total_loss = total_loss + self.consistency_weight * consistency_loss
            info['consistency_loss'] = consistency_loss.item()

            # Store braid info
            info['braid_norm'] = braid_rep.norm(dim=(-2, -1)).mean().item()
            info['dominant_generator'] = braid_info['dominant_generator'].float().mean().item()

        # Temporal coherence loss
        if self.use_temporal_coherence:
            coherence_loss = self.compute_coherence_loss(x)
            total_loss = total_loss + self.coherence_weight * coherence_loss
            info['coherence_loss'] = coherence_loss.item()

        # Apply temperature scaling
        total_loss = total_loss / temperature

        # Store summary statistics
        info['total_loss'] = total_loss.item()
        info['temperature'] = temperature.item()

        return total_loss, info

    def get_topological_features(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Extract topological features without computing loss.

        Useful for analysis and visualization of learned representations.

        Args:
            x: Input tensor of shape (B, T, D)
            mask: Optional attention mask of shape (B, T)

        Returns:
            Dictionary with topological features:
            - 'jones_poly': Jones polynomial features (B, output_dim)
            - 'braid_rep': Braid representation matrix (B, rep_dim, rep_dim)
            - 'position_features': Per-position features (B, T, n_coef*2)
        """
        features = {}

        if self.use_jones:
            poly_features, _ = self.jones(x, mask=mask)
            features['jones_poly'] = poly_features

        if self.use_braid:
            braid_rep = self.braid_encoder(x, mask=mask, return_info=False)
            features['braid_rep'] = braid_rep

        if self.use_temporal_coherence:
            pos_features = self.position_encoder(x)
            features['position_features'] = pos_features

        return features

    def topological_distance(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute topological distance between two sequences.

        Combines Jones polynomial distance and braid representation
        distance for a comprehensive topological metric.

        Args:
            x1: First sequence (B, T1, D)
            x2: Second sequence (B, T2, D)

        Returns:
            Distance of shape (B,)
        """
        # Get topological features
        feat1 = self.get_topological_features(x1)
        feat2 = self.get_topological_features(x2)

        distance = torch.tensor(0.0, device=x1.device)

        if self.use_jones and 'jones_poly' in feat1 and 'jones_poly' in feat2:
            jones_dist = (feat1['jones_poly'] - feat2['jones_poly']).norm(dim=-1)
            distance = distance + jones_dist

        if self.use_braid and 'braid_rep' in feat1 and 'braid_rep' in feat2:
            braid_dist = (feat1['braid_rep'] - feat2['braid_rep']).norm(dim=(-2, -1))
            distance = distance + braid_dist

        return distance

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, "
            f"n_strands={self.n_strands}, "
            f"n_coefficients={self.n_coefficients}, "
            f"jones={self.use_jones}, "
            f"braid={self.use_braid}, "
            f"coherence={self.use_temporal_coherence}"
        )
