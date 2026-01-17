# neuromanifold_gpt/model/topology/jones_polynomial.py
"""
Jones Polynomial Approximation Network.

The Jones polynomial V(t) is a powerful knot invariant that captures topological
information about braids and knots. This module provides neural network
approximations for computing Jones polynomial-like invariants from sequences.

Key concepts:
- Jones polynomial: V: Knots -> Z[t^{1/2}, t^{-1/2}] is a Laurent polynomial
- Kauffman bracket: <.> : Diagrams -> Z[A, A^{-1}] computes via skein relations
- Connection: V(t) = (-A^3)^{-w(L)} <L> where w(L) is writhe

For language modeling, Jones polynomial invariants capture:
- Topological complexity of syntactic dependencies
- Crossing number (minimum crossings in any diagram)
- Structural invariants preserved under continuous deformation

The approximation network learns to:
1. Encode sequences into braid representations
2. Approximate Kauffman bracket evaluations
3. Output polynomial coefficients or sampled values

Reference:
- Jones, V.F.R. "A polynomial invariant for knots via von Neumann algebras" (1985)
- Kauffman, L.H. "State models and the Jones polynomial" (1987)
- Witten, E. "Quantum field theory and the Jones polynomial" (1989)
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .braid import BraidEncoder, TemperleyLiebAlgebra


@dataclass
class JonesConfig:
    """Configuration for Jones polynomial approximation."""

    embed_dim: int = 384
    n_strands: int = 4
    n_coefficients: int = 8  # Number of polynomial coefficients to predict
    use_kauffman_bracket: bool = True  # Use Kauffman bracket computation path
    use_temperley_lieb: bool = True  # Use TL algebra representation
    t_values: tuple = (-1.0, 0.5, 1.0, 2.0)  # Evaluation points for V(t)
    dropout: float = 0.0


class KauffmanBracketNetwork(nn.Module):
    """
    Neural network computing Kauffman bracket approximations.

    The Kauffman bracket <D> of a diagram D is computed via skein relations:
    - <empty> = 1
    - <D union O> = delta * <D>  (O = unknot, delta = -A^2 - A^{-2})
    - <crossing+> = A<0-resolution> + A^{-1}<1-resolution>

    This network learns to approximate bracket values from braid representations.

    Example:
        >>> kb = KauffmanBracketNetwork(embed_dim=384)
        >>> braid_rep = torch.randn(2, 3, 3)  # From BraidEncoder
        >>> bracket = kb(braid_rep)
        >>> bracket.shape
        torch.Size([2, 8])  # Complex coefficients (re, im pairs)
    """

    def __init__(
        self,
        embed_dim: int,
        rep_dim: int = 3,
        n_coefficients: int = 8,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ):
        """
        Initialize Kauffman bracket network.

        Args:
            embed_dim: Input embedding dimension (for auxiliary features)
            rep_dim: Dimension of braid representation matrices
            n_coefficients: Number of bracket coefficients to output
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.rep_dim = rep_dim
        self.n_coefficients = n_coefficients

        # Process braid representation matrix
        # Flatten rep_dim x rep_dim -> rep_dim^2
        flat_dim = rep_dim * rep_dim

        # MLP for computing bracket from representation
        self.bracket_mlp = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_coefficients * 2),  # Complex coefficients
        )

        # Layer normalization for stability
        self.input_norm = nn.LayerNorm(flat_dim)

        # Learnable scale for output
        self.output_scale = nn.Parameter(torch.ones(1))

    def forward(
        self,
        braid_rep: torch.Tensor,
        return_complex: bool = False,
    ) -> torch.Tensor:
        """
        Compute Kauffman bracket approximation from braid representation.

        Args:
            braid_rep: Braid representation matrix of shape (B, rep_dim, rep_dim)
            return_complex: If True, return as complex tensor

        Returns:
            Bracket coefficients of shape (B, n_coefficients) or (B, n_coefficients) complex
        """
        B = braid_rep.shape[0]

        # Flatten representation matrix
        rep_flat = braid_rep.view(B, -1)

        # Normalize
        rep_flat = self.input_norm(rep_flat)

        # Compute bracket coefficients
        bracket_raw = self.bracket_mlp(rep_flat)

        # Scale output
        bracket_raw = bracket_raw * self.output_scale

        if return_complex:
            # Reshape to complex: (B, n_coefficients, 2) -> complex (B, n_coefficients)
            bracket = bracket_raw.view(B, self.n_coefficients, 2)
            return torch.complex(bracket[..., 0], bracket[..., 1])
        else:
            return bracket_raw.view(B, self.n_coefficients * 2)

    def extra_repr(self) -> str:
        return f"rep_dim={self.rep_dim}, n_coefficients={self.n_coefficients}"


class JonesEvaluator(nn.Module):
    """
    Evaluates Jones polynomial at specific parameter values.

    Given polynomial coefficients, evaluates V(t) at multiple t values
    to produce a feature vector of polynomial samples.

    The Jones polynomial is related to the Kauffman bracket by:
    V(t) = (-A^3)^{-w(L)} <L>  where t = A^{-4}

    This network learns to produce consistent evaluations.
    """

    def __init__(
        self,
        n_coefficients: int = 8,
        t_values: tuple = (-1.0, 0.5, 1.0, 2.0),
        hidden_dim: int = 64,
    ):
        """
        Initialize Jones evaluator.

        Args:
            n_coefficients: Number of input polynomial coefficients
            t_values: Evaluation points for V(t)
            hidden_dim: Hidden dimension for refinement network
        """
        super().__init__()

        self.n_coefficients = n_coefficients
        self.t_values = t_values
        self.n_evaluations = len(t_values)

        # Register t values as buffer
        self.register_buffer("t_buffer", torch.tensor(list(t_values)))

        # Build Vandermonde-like matrix for polynomial evaluation
        # Each row is [1, t, t^2, ..., t^{n-1}]
        powers = torch.arange(n_coefficients // 2).float()  # Use half (real part)
        vandermonde = torch.stack(
            [t**powers for t in t_values]
        )  # (n_evals, n_coefficients//2)
        self.register_buffer("vandermonde", vandermonde)

        # Refinement network for learned adjustments
        self.refine = nn.Sequential(
            nn.Linear(self.n_evaluations, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.n_evaluations),
        )

        # Output projection
        self.out_proj = nn.Linear(self.n_evaluations, self.n_evaluations)

    def forward(
        self,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate polynomial at configured t values.

        Args:
            coefficients: Polynomial coefficients of shape (B, n_coefficients * 2)
                         or (B, n_coefficients) if complex

        Returns:
            Evaluations of shape (B, n_evaluations)
        """
        B = coefficients.shape[0]

        # Handle complex vs real input
        if coefficients.is_complex():
            coef_real = coefficients.real
        else:
            # Split into real/imaginary and use real part
            coef_split = coefficients.view(B, -1, 2)
            coef_real = coef_split[..., 0]

        # Ensure correct shape for matmul
        if coef_real.shape[-1] != self.vandermonde.shape[-1]:
            # Truncate or pad coefficients
            target_len = self.vandermonde.shape[-1]
            if coef_real.shape[-1] > target_len:
                coef_real = coef_real[..., :target_len]
            else:
                pad_len = target_len - coef_real.shape[-1]
                coef_real = F.pad(coef_real, (0, pad_len), value=0)

        # Polynomial evaluation: (B, n_coef//2) @ (n_coef//2, n_evals) -> (B, n_evals)
        evaluations = torch.matmul(coef_real, self.vandermonde.T)

        # Apply learned refinement
        evaluations = evaluations + self.refine(evaluations)

        # Final projection
        evaluations = self.out_proj(evaluations)

        return evaluations

    def extra_repr(self) -> str:
        return f"n_coefficients={self.n_coefficients}, t_values={self.t_values}"


class JonesApproximator(nn.Module):
    """
    Neural network approximating Jones polynomial invariants.

    Combines braid encoding, Kauffman bracket computation, and polynomial
    evaluation into a unified module for computing topological invariants
    from sequences.

    Architecture:
    1. BraidEncoder: Sequence -> Braid representation matrix
    2. KauffmanBracketNetwork: Braid rep -> Bracket coefficients
    3. JonesEvaluator: Coefficients -> V(t) evaluations

    The output is a topological feature vector capturing:
    - Polynomial coefficients (structural complexity)
    - Sampled evaluations (invariant comparisons)
    - Writhe (signed crossing number)

    Example:
        >>> ja = JonesApproximator(embed_dim=384)
        >>> x = torch.randn(2, 32, 384)  # (batch, seq, dim)
        >>> poly, info = ja(x)
        >>> poly.shape
        torch.Size([2, 20])  # Combined features
    """

    def __init__(
        self,
        embed_dim: int,
        n_strands: int = 4,
        n_coefficients: int = 8,
        t_values: tuple = (-1.0, 0.5, 1.0, 2.0),
        hidden_dim: int = 128,
        dropout: float = 0.0,
        use_kauffman_bracket: bool = True,
        use_temperley_lieb: bool = True,
    ):
        """
        Initialize Jones polynomial approximator.

        Args:
            embed_dim: Input embedding dimension
            n_strands: Number of braid strands
            n_coefficients: Number of polynomial coefficients
            t_values: Evaluation points for V(t)
            hidden_dim: Hidden dimension for networks
            dropout: Dropout probability
            use_kauffman_bracket: Enable Kauffman bracket path
            use_temperley_lieb: Enable Temperley-Lieb algebra features
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.n_strands = n_strands
        self.n_coefficients = n_coefficients
        self.t_values = t_values
        self.use_kauffman_bracket = use_kauffman_bracket
        self.use_temperley_lieb = use_temperley_lieb

        # Representation dimension (reduced Burau)
        self.rep_dim = n_strands - 1

        # Braid encoder: sequence -> braid representation
        self.braid_encoder = BraidEncoder(
            embed_dim=embed_dim,
            n_strands=n_strands,
            use_reduced=True,
            dropout=dropout,
        )

        # Kauffman bracket network
        if use_kauffman_bracket:
            self.bracket_net = KauffmanBracketNetwork(
                embed_dim=embed_dim,
                rep_dim=self.rep_dim,
                n_coefficients=n_coefficients,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )

        # Jones evaluator
        self.evaluator = JonesEvaluator(
            n_coefficients=n_coefficients,
            t_values=t_values,
            hidden_dim=hidden_dim // 2,
        )

        # Temperley-Lieb algebra for additional invariants
        if use_temperley_lieb:
            self.tl_algebra = TemperleyLiebAlgebra(n=n_strands)
            # Network to process TL features
            tl_dim = self.tl_algebra.dim
            self.tl_processor = nn.Sequential(
                nn.Linear(self.rep_dim * self.rep_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, tl_dim),
            )

        # Writhe predictor: predicts signed crossing number
        self.writhe_head = nn.Sequential(
            nn.Linear(self.rep_dim * self.rep_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Output dimension calculation
        self.n_evaluations = len(t_values)
        bracket_dim = n_coefficients * 2 if use_kauffman_bracket else 0
        tl_out_dim = self.tl_algebra.dim if use_temperley_lieb else 0
        self.output_dim = (
            self.n_evaluations + bracket_dim + tl_out_dim + 1
        )  # +1 for writhe

        # Final projection to unified output
        self.output_proj = nn.Linear(self.output_dim, self.output_dim)

        # Layer norm for output stability
        self.output_norm = nn.LayerNorm(self.output_dim)

    def compute_invariants(
        self,
        braid_rep: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute all topological invariants from braid representation.

        Args:
            braid_rep: Braid representation of shape (B, rep_dim, rep_dim)

        Returns:
            Tuple of (invariants, info_dict)
        """
        B = braid_rep.shape[0]
        info = {}
        features = []

        # Flatten representation for processing
        rep_flat = braid_rep.view(B, -1)

        # Kauffman bracket coefficients
        if self.use_kauffman_bracket:
            bracket = self.bracket_net(braid_rep)  # (B, n_coef * 2)
            features.append(bracket)

            # Evaluate polynomial at t values
            evaluations = self.evaluator(bracket)  # (B, n_evals)
            features.append(evaluations)

            info["bracket_norm"] = bracket.norm(dim=-1).mean()
            info["poly_evals"] = evaluations.detach()
        else:
            # Direct evaluation from representation
            direct_coef = rep_flat[:, : self.n_coefficients * 2]
            if direct_coef.shape[-1] < self.n_coefficients * 2:
                direct_coef = F.pad(
                    direct_coef, (0, self.n_coefficients * 2 - direct_coef.shape[-1])
                )
            evaluations = self.evaluator(direct_coef)
            features.append(evaluations)

        # Temperley-Lieb features
        if self.use_temperley_lieb:
            tl_features = self.tl_processor(rep_flat)  # (B, tl_dim)
            features.append(tl_features)
            info["tl_norm"] = tl_features.norm(dim=-1).mean()

        # Writhe (signed crossing number)
        writhe = self.writhe_head(rep_flat)  # (B, 1)
        features.append(writhe)
        info["writhe"] = writhe.squeeze(-1).detach()

        # Concatenate all features
        invariants = torch.cat(features, dim=-1)

        return invariants, info

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute Jones polynomial approximation from input sequence.

        Args:
            x: Input tensor of shape (B, T, D)
            mask: Optional attention mask of shape (B, T)

        Returns:
            Tuple of (polynomial_features, info_dict) where:
            - polynomial_features: Tensor of shape (B, output_dim)
            - info_dict: Dictionary with invariant information
        """
        B, T, D = x.shape

        # Encode sequence to braid representation
        braid_rep, braid_info = self.braid_encoder(x, mask=mask, return_info=True)

        # Compute topological invariants
        invariants, invariant_info = self.compute_invariants(braid_rep)

        # Project and normalize output
        output = self.output_proj(invariants)
        output = self.output_norm(output)

        # Compile info dictionary
        info = {
            "braid_norm": braid_rep.norm(dim=(-2, -1)).mean(),
            **invariant_info,
        }

        return output, info

    def compute_jones_distance(
        self,
        poly1: torch.Tensor,
        poly2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute distance between Jones polynomial features.

        This provides a topologically-aware distance metric.

        Args:
            poly1: First polynomial features (B, output_dim)
            poly2: Second polynomial features (B, output_dim)

        Returns:
            Distance of shape (B,)
        """
        # L2 distance in polynomial space
        return (poly1 - poly2).norm(dim=-1)

    def topological_similarity(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute topological similarity between two sequences.

        Args:
            x1: First sequence (B, T1, D)
            x2: Second sequence (B, T2, D)

        Returns:
            Similarity score of shape (B,)
        """
        poly1, _ = self(x1)
        poly2, _ = self(x2)

        # Cosine similarity in polynomial space
        sim = F.cosine_similarity(poly1, poly2, dim=-1)
        return sim

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, "
            f"n_strands={self.n_strands}, "
            f"n_coefficients={self.n_coefficients}, "
            f"output_dim={self.output_dim}, "
            f"bracket={self.use_kauffman_bracket}, "
            f"tl={self.use_temperley_lieb}"
        )


class JonesLoss(nn.Module):
    """
    Loss function based on Jones polynomial invariants.

    Provides regularization encouraging topological consistency:
    - Invariance: Similar inputs should have similar polynomials
    - Distinctiveness: Different topological types should be separated
    - Conservation: Polynomial structure should be preserved

    Can be combined with standard language modeling losses.
    """

    def __init__(
        self,
        embed_dim: int,
        margin: float = 0.5,
        invariance_weight: float = 1.0,
        distinctiveness_weight: float = 0.5,
    ):
        """
        Initialize Jones loss.

        Args:
            embed_dim: Embedding dimension
            margin: Margin for triplet-like losses
            invariance_weight: Weight for invariance term
            distinctiveness_weight: Weight for distinctiveness term
        """
        super().__init__()

        self.margin = margin
        self.invariance_weight = invariance_weight
        self.distinctiveness_weight = distinctiveness_weight

        # Jones approximator for computing invariants
        self.jones = JonesApproximator(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        x_aug: Optional[torch.Tensor] = None,
        x_neg: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute Jones polynomial loss.

        Args:
            x: Input sequence (B, T, D)
            x_aug: Optional augmented version (should have same topology)
            x_neg: Optional negative sample (different topology)

        Returns:
            Tuple of (loss, info_dict)
        """
        # Compute polynomial features
        poly, info = self.jones(x)

        loss = torch.tensor(0.0, device=x.device)

        # Invariance loss: augmented should be close
        if x_aug is not None:
            poly_aug, _ = self.jones(x_aug)
            inv_loss = (poly - poly_aug).pow(2).mean()
            loss = loss + self.invariance_weight * inv_loss
            info["invariance_loss"] = inv_loss.item()

        # Distinctiveness loss: negatives should be far
        if x_neg is not None:
            poly_neg, _ = self.jones(x_neg)
            dist = (poly - poly_neg).norm(dim=-1)
            dist_loss = F.relu(self.margin - dist).mean()
            loss = loss + self.distinctiveness_weight * dist_loss
            info["distinctiveness_loss"] = dist_loss.item()

        # Regularization: encourage sparse polynomial coefficients
        reg_loss = poly.abs().mean() * 0.01
        loss = loss + reg_loss
        info["reg_loss"] = reg_loss.item()

        info["total_loss"] = loss.item()

        return loss, info

    def extra_repr(self) -> str:
        return f"margin={self.margin}, inv={self.invariance_weight}, dist={self.distinctiveness_weight}"
