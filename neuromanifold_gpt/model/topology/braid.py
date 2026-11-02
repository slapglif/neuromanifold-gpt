# neuromanifold_gpt/model/topology/braid.py
"""
Braid Group Representations for Topological Language Features.

Braid groups B_n capture the topology of n intertwined strands:
- Artin generators σ_i represent strand i crossing over strand i+1
- Relations: σ_i σ_j = σ_j σ_i for |i-j| > 1 (far commutativity)
- Braid relation: σ_i σ_{i+1} σ_i = σ_{i+1} σ_i σ_{i+1} (Yang-Baxter)

For language modeling, braid groups capture:
- Syntactic dependencies (crossed vs nested structures)
- Long-range correlations as topological invariants
- Word order permutations as braid elements

Key components:
- BraidGroup: Mathematical structure with Artin generators and representations
- BraidEncoder: Neural network producing braid group representations
- BurauRepresentation: Matrix representation for computing invariants

The Burau representation provides a homomorphism from B_n to GL_n(Z[t, t^-1]),
enabling computation of topological invariants like the Alexander polynomial.

Reference:
- Birman, J. S. "Braids, Links, and Mapping Class Groups"
- Kassel & Turaev "Braid Groups"
- Lawrence, R. "Representations of Braid Groups" (arXiv:math/9903006)
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BraidConfig:
    """Configuration for braid group modules."""

    n_strands: int = 4
    use_burau: bool = True
    use_reduced: bool = True  # Use reduced Burau representation
    t_param: float = -1.0  # Burau parameter (t=-1 gives integer matrices)
    dropout: float = 0.0


class BraidGroup:
    """
    Mathematical structure representing the braid group B_n.

    The braid group B_n on n strands has generators σ_1, ..., σ_{n-1}
    subject to the relations:
    - σ_i σ_j = σ_j σ_i for |i-j| > 1 (far generators commute)
    - σ_i σ_{i+1} σ_i = σ_{i+1} σ_i σ_{i+1} (braid/Yang-Baxter relation)

    This class provides matrix representations (Burau) for computing
    topological invariants and encoding braid structure in neural networks.

    Example:
        >>> bg = BraidGroup(n_strands=4)
        >>> sigma_1 = bg.generator_matrix(0)  # First generator σ_1
        >>> sigma_2 = bg.generator_matrix(1)  # Second generator σ_2
        >>> # Verify braid relation: σ_1 σ_2 σ_1 = σ_2 σ_1 σ_2
        >>> lhs = sigma_1 @ sigma_2 @ sigma_1
        >>> rhs = sigma_2 @ sigma_1 @ sigma_2
        >>> torch.allclose(lhs, rhs)
        True
    """

    def __init__(
        self,
        n_strands: int = 4,
        t_param: float = -1.0,
        use_reduced: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize braid group B_n.

        Args:
            n_strands: Number of strands n (group is B_n)
            t_param: Parameter for Burau representation. t=-1 gives integers.
            use_reduced: Use reduced (n-1) x (n-1) Burau representation
            device: Device for matrices (defaults to CPU)
            dtype: Data type for matrices
        """
        self.n_strands = n_strands
        self.n_generators = n_strands - 1  # Generators σ_1, ..., σ_{n-1}
        self.t_param = t_param
        self.use_reduced = use_reduced
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        # Matrix dimension for representation
        self.rep_dim = n_strands - 1 if use_reduced else n_strands

        # Cache generator matrices
        self._generators = None
        self._inverse_generators = None

    def generator_matrix(self, i: int, inverse: bool = False) -> torch.Tensor:
        """
        Get Burau representation matrix for generator σ_i (or σ_i^{-1}).

        The unreduced Burau representation sends σ_i to the n×n matrix:
        - Identity except for 2×2 block at rows/cols i, i+1
        - Block is [[1-t, t], [1, 0]] for σ_i
        - Block is [[0, 1], [1/t, (t-1)/t]] for σ_i^{-1}

        The reduced representation is (n-1)×(n-1), quotient by a simple module.

        Args:
            i: Generator index (0-indexed, so σ_{i+1} in standard notation)
            inverse: If True, return inverse generator matrix

        Returns:
            Matrix representation of shape (rep_dim, rep_dim)
        """
        if i < 0 or i >= self.n_generators:
            raise ValueError(
                f"Generator index {i} out of range [0, {self.n_generators})"
            )

        t = self.t_param

        if self.use_reduced:
            # Reduced Burau representation: (n-1) x (n-1) matrices
            matrix = torch.eye(self.rep_dim, device=self.device, dtype=self.dtype)

            if not inverse:
                # σ_i acts on reduced representation
                if i < self.rep_dim:
                    matrix[i, i] = -t
                if i > 0:
                    matrix[i, i - 1] = 1.0
                if i < self.rep_dim - 1:
                    matrix[i, i + 1] = t
            else:
                # σ_i^{-1}
                if i < self.rep_dim:
                    matrix[i, i] = -1.0 / t
                if i > 0:
                    matrix[i, i - 1] = 1.0 / t
                if i < self.rep_dim - 1:
                    matrix[i, i + 1] = 1.0
        else:
            # Full Burau representation: n x n matrices
            matrix = torch.eye(self.n_strands, device=self.device, dtype=self.dtype)

            if not inverse:
                # Standard Burau generator
                matrix[i, i] = 1.0 - t
                matrix[i, i + 1] = t
                matrix[i + 1, i] = 1.0
                matrix[i + 1, i + 1] = 0.0
            else:
                # Inverse generator
                matrix[i, i] = 0.0
                matrix[i, i + 1] = 1.0
                matrix[i + 1, i] = 1.0 / t
                matrix[i + 1, i + 1] = (t - 1.0) / t

        return matrix

    def get_all_generators(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get all generator matrices and their inverses.

        Returns:
            Tuple of (generators, inverse_generators) each of shape
            (n_generators, rep_dim, rep_dim)
        """
        if self._generators is None:
            generators = torch.stack(
                [
                    self.generator_matrix(i, inverse=False)
                    for i in range(self.n_generators)
                ]
            )
            inverses = torch.stack(
                [
                    self.generator_matrix(i, inverse=True)
                    for i in range(self.n_generators)
                ]
            )
            self._generators = generators
            self._inverse_generators = inverses

        return self._generators, self._inverse_generators

    def word_to_matrix(
        self,
        word: list[tuple[int, int]],
    ) -> torch.Tensor:
        """
        Convert a braid word to its matrix representation.

        A braid word is a sequence of (generator_index, power) pairs.
        Positive power means σ_i, negative means σ_i^{-1}.

        Args:
            word: List of (generator_index, power) tuples

        Returns:
            Product matrix of shape (rep_dim, rep_dim)
        """
        result = torch.eye(self.rep_dim, device=self.device, dtype=self.dtype)

        for gen_idx, power in word:
            gen_matrix = self.generator_matrix(gen_idx, inverse=(power < 0))
            for _ in range(abs(power)):
                result = result @ gen_matrix

        return result

    def random_braid_word(self, length: int) -> list[tuple[int, int]]:
        """
        Generate a random braid word.

        Args:
            length: Number of generators in the word

        Returns:
            List of (generator_index, power) tuples
        """
        import random

        word = []
        for _ in range(length):
            gen = random.randint(0, self.n_generators - 1)
            power = random.choice([-1, 1])
            word.append((gen, power))
        return word

    def verify_braid_relation(self) -> bool:
        """
        Verify that the Yang-Baxter braid relation holds.

        σ_i σ_{i+1} σ_i = σ_{i+1} σ_i σ_{i+1}

        Returns:
            True if relation holds (up to numerical tolerance)
        """
        if self.n_generators < 2:
            return True  # No adjacent generators to test

        for i in range(self.n_generators - 1):
            sigma_i = self.generator_matrix(i)
            sigma_ip1 = self.generator_matrix(i + 1)

            lhs = sigma_i @ sigma_ip1 @ sigma_i
            rhs = sigma_ip1 @ sigma_i @ sigma_ip1

            if not torch.allclose(lhs, rhs, atol=1e-5):
                return False

        return True

    def __repr__(self) -> str:
        return f"BraidGroup(n_strands={self.n_strands}, t={self.t_param}, reduced={self.use_reduced})"


class BraidEncoder(nn.Module):
    """
    Neural encoder producing braid group representations from sequences.

    Maps input sequences to braid group elements, capturing the topological
    structure of dependencies. The encoder learns to:
    - Identify crossing patterns (which strands interact)
    - Compute crossing signs (over/under crossings)
    - Aggregate into braid group representation matrix

    Architecture:
    1. Project input to crossing logits (which generators to use)
    2. Compute soft generator selection weights
    3. Linearly combine Burau matrices
    4. Optional: learn continuous deformations of braid structure

    The output representation can be used to:
    - Compute Jones polynomial via Temperley-Lieb algebra
    - Extract topological invariants of syntactic structure
    - Regularize attention to respect braid relations

    Example:
        >>> encoder = BraidEncoder(embed_dim=384, n_strands=4)
        >>> x = torch.randn(2, 32, 384)  # (batch, seq, dim)
        >>> rep = encoder(x)  # Braid representation
        >>> rep.shape
        torch.Size([2, 3, 3])  # (batch, rep_dim, rep_dim) with reduced=True
    """

    def __init__(
        self,
        embed_dim: int,
        n_strands: int = 4,
        n_crossings: int = 8,
        t_param: float = -1.0,
        use_reduced: bool = True,
        use_attention: bool = True,
        dropout: float = 0.0,
    ):
        """
        Initialize braid encoder.

        Args:
            embed_dim: Input embedding dimension
            n_strands: Number of braid strands (determines group B_n)
            n_crossings: Number of crossings to predict per sequence position
            t_param: Burau representation parameter
            use_reduced: Use reduced Burau representation
            use_attention: Use attention to weight positions
            dropout: Dropout probability
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.n_strands = n_strands
        self.n_generators = n_strands - 1
        self.n_crossings = n_crossings
        self.t_param = t_param
        self.use_reduced = use_reduced
        self.use_attention = use_attention

        self.rep_dim = n_strands - 1 if use_reduced else n_strands

        # Create braid group for generator matrices
        self.braid_group = BraidGroup(
            n_strands=n_strands,
            t_param=t_param,
            use_reduced=use_reduced,
        )

        # Project input to crossing probabilities
        # For each position, predict which generators are active and their signs
        # Output: 2 * n_generators (positive and negative for each generator)
        self.crossing_proj = nn.Linear(embed_dim, 2 * self.n_generators)

        # Optional: attention weights for aggregating over positions
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.GELU(),
                nn.Linear(embed_dim // 4, 1),
            )

        # Optional: learnable refinement of generator matrices
        # This allows the network to learn continuous deformations
        self.use_learnable_generators = False
        if self.use_learnable_generators:
            self.generator_refine = nn.Parameter(
                torch.zeros(2 * self.n_generators, self.rep_dim, self.rep_dim)
            )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Layer norm for output stabilization
        self.output_norm = nn.LayerNorm(self.rep_dim * self.rep_dim)

        # Register generator matrices as buffers
        self._register_generator_buffers()

    def _register_generator_buffers(self):
        """Register Burau generator matrices as buffers."""
        generators = []
        for i in range(self.n_generators):
            generators.append(self.braid_group.generator_matrix(i, inverse=False))
            generators.append(self.braid_group.generator_matrix(i, inverse=True))

        # Shape: (2 * n_generators, rep_dim, rep_dim)
        self.register_buffer(
            "generators",
            torch.stack(generators),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_info: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """
        Encode sequence into braid group representation.

        Args:
            x: Input tensor of shape (B, T, D)
            mask: Optional attention mask of shape (B, T)
            return_info: If True, return additional information dict

        Returns:
            Braid representation of shape (B, rep_dim, rep_dim)
            If return_info=True: tuple of (representation, info_dict)
        """
        B, T, D = x.shape

        # Project to crossing logits: (B, T, 2 * n_generators)
        crossing_logits = self.crossing_proj(x)

        # Apply attention weighting over sequence positions
        if self.use_attention:
            # Compute position weights: (B, T, 1)
            position_weights = self.attention(x)
            if mask is not None:
                position_weights = position_weights.masked_fill(
                    ~mask.unsqueeze(-1), float("-inf")
                )
            position_weights = F.softmax(position_weights, dim=1)

            # Weight crossing logits: (B, 2 * n_generators)
            weighted_logits = (crossing_logits * position_weights).sum(dim=1)
        else:
            # Simple mean over positions
            if mask is not None:
                crossing_logits = crossing_logits.masked_fill(~mask.unsqueeze(-1), 0.0)
                weighted_logits = crossing_logits.sum(dim=1) / mask.sum(
                    dim=1, keepdim=True
                )
            else:
                weighted_logits = crossing_logits.mean(dim=1)

        # Softmax to get generator weights: (B, 2 * n_generators)
        generator_weights = F.softmax(weighted_logits, dim=-1)

        # Apply dropout
        generator_weights = self.dropout(generator_weights)

        # Get generator matrices: (2 * n_generators, rep_dim, rep_dim)
        gen_matrices = self.generators

        # Weighted combination of generators: (B, rep_dim, rep_dim)
        # gen_weights: (B, 2*n_gen) -> (B, 2*n_gen, 1, 1)
        # gen_matrices: (2*n_gen, rep_dim, rep_dim)
        weights_expanded = generator_weights.view(B, -1, 1, 1)
        representation = (weights_expanded * gen_matrices).sum(dim=1)

        # Normalize output for stability
        rep_flat = representation.view(B, -1)
        rep_flat = self.output_norm(rep_flat)
        representation = rep_flat.view(B, self.rep_dim, self.rep_dim)

        if return_info:
            info = {
                "generator_weights": generator_weights,
                "crossing_logits": weighted_logits,
                "position_weights": position_weights if self.use_attention else None,
                "dominant_generator": generator_weights.argmax(dim=-1),
            }
            return representation, info

        return representation

    def compute_writhe(self, generator_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute the writhe (sum of signed crossings) from generator weights.

        The writhe is a topological invariant related to the linking number.
        Positive generators contribute +1, inverse generators contribute -1.

        Args:
            generator_weights: Weights of shape (B, 2 * n_generators)

        Returns:
            Writhe of shape (B,)
        """
        # Separate positive and negative generators
        pos_weights = generator_weights[:, 0::2]  # σ_i weights
        neg_weights = generator_weights[:, 1::2]  # σ_i^{-1} weights

        writhe = pos_weights.sum(dim=-1) - neg_weights.sum(dim=-1)
        return writhe

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, "
            f"n_strands={self.n_strands}, "
            f"n_crossings={self.n_crossings}, "
            f"rep_dim={self.rep_dim}, "
            f"attention={self.use_attention}"
        )


class BraidCrossing(nn.Module):
    """
    Single braid crossing operation for use in neural network layers.

    Applies a learned braid crossing to an input sequence, modeling
    the interaction between adjacent positions as strand crossings.

    This can be used as a drop-in replacement for attention in some contexts,
    providing topological inductive bias.
    """

    def __init__(
        self,
        embed_dim: int,
        n_strands: int = 4,
        dropout: float = 0.0,
    ):
        """
        Initialize braid crossing layer.

        Args:
            embed_dim: Embedding dimension
            n_strands: Number of strands
            dropout: Dropout probability
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.n_strands = n_strands
        self.n_generators = n_strands - 1

        # Learn which crossing to apply based on position pair
        self.crossing_selector = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 2 * self.n_generators),
        )

        # Mixing matrices for applying crossing transformation
        self.mix_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Apply braid crossing to sequence.

        Args:
            x: Input of shape (B, T, D)
            positions: Optional tuple of (i, j) positions to cross.
                      If None, applies learned crossings to all adjacent pairs.

        Returns:
            Output of shape (B, T, D) with crossings applied
        """
        B, T, D = x.shape

        if positions is not None:
            # Apply crossing at specific positions
            i, j = positions
            pair_embed = torch.cat([x[:, i], x[:, j]], dim=-1)
            crossing_logits = self.crossing_selector(pair_embed)
            crossing_weights = F.softmax(crossing_logits, dim=-1)

            # Apply crossing as weighted mixing
            mixed = self.mix_proj(x[:, i]) * crossing_weights[
                :, : self.n_generators
            ].mean(dim=-1, keepdim=True)
            mixed += self.mix_proj(x[:, j]) * crossing_weights[
                :, self.n_generators :
            ].mean(dim=-1, keepdim=True)

            out = x.clone()
            out[:, i] = self.layer_norm(x[:, i] + self.dropout(mixed))
            out[:, j] = self.layer_norm(x[:, j] + self.dropout(mixed))
            return out
        else:
            # Apply to all adjacent pairs
            out = x.clone()
            for i in range(T - 1):
                pair_embed = torch.cat([x[:, i], x[:, i + 1]], dim=-1)
                crossing_logits = self.crossing_selector(pair_embed)
                crossing_weights = F.softmax(crossing_logits, dim=-1)

                # Symmetric mixing based on crossing type
                scale = crossing_weights.mean(dim=-1, keepdim=True)
                mixed_i = self.mix_proj(x[:, i + 1]) * scale
                mixed_ip1 = self.mix_proj(x[:, i]) * scale

                out[:, i] = self.layer_norm(out[:, i] + self.dropout(mixed_i))
                out[:, i + 1] = self.layer_norm(out[:, i + 1] + self.dropout(mixed_ip1))

            return out

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, n_strands={self.n_strands}"


class TemperleyLiebAlgebra:
    """
    Temperley-Lieb algebra for Jones polynomial computation.

    The Temperley-Lieb algebra TL_n(δ) provides a representation of the
    braid group that factors through the Kauffman bracket. This is used
    to compute the Jones polynomial from braid representations.

    Elements are diagrams connecting 2n points with n non-crossing arcs.
    The algebra has generators e_1, ..., e_{n-1} satisfying:
    - e_i² = δ·e_i
    - e_i e_j = e_j e_i for |i-j| > 1
    - e_i e_{i±1} e_i = e_i

    Reference: Kauffman, L. "State Models and the Jones Polynomial"
    """

    def __init__(
        self,
        n: int,
        delta: float = None,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize Temperley-Lieb algebra TL_n(δ).

        Args:
            n: Number of pairs (algebra is TL_n)
            delta: Loop value. If None, uses -A² - A^{-2} with A=e^{iπ/4}
            device: Device for tensors
            dtype: Data type for tensors
        """
        self.n = n
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        if delta is None:
            # Default: quantum group parameter for Jones polynomial
            # δ = -q - q^{-1} where q = e^{iπ/4}
            self.delta = torch.tensor(-2.0, device=self.device, dtype=self.dtype)
        else:
            self.delta = torch.tensor(delta, device=self.device, dtype=self.dtype)

        # Dimension of basis (Catalan number)
        self.dim = self._catalan(n)

    @staticmethod
    def _catalan(n: int) -> int:
        """Compute n-th Catalan number."""
        if n <= 1:
            return 1
        catalan = [0] * (n + 1)
        catalan[0] = catalan[1] = 1
        for i in range(2, n + 1):
            for j in range(i):
                catalan[i] += catalan[j] * catalan[i - 1 - j]
        return catalan[n]

    def generator_matrix(self, i: int) -> torch.Tensor:
        """
        Get matrix representation of generator e_i.

        In the standard basis of non-crossing arc diagrams:
        e_i connects points i and i+1 on top, i and i+1 on bottom.

        Args:
            i: Generator index (1-indexed in standard notation, 0-indexed here)

        Returns:
            Matrix of shape (dim, dim) in the diagram basis
        """
        # Simplified implementation: return identity scaled by delta for now
        # Full implementation would require explicit diagram basis enumeration
        matrix = torch.eye(self.dim, device=self.device, dtype=self.dtype)
        matrix = matrix * self.delta.sqrt()
        return matrix

    def __repr__(self) -> str:
        return f"TemperleyLiebAlgebra(n={self.n}, delta={self.delta.item():.3f}, dim={self.dim})"
