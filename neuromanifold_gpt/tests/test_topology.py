# neuromanifold_gpt/tests/test_topology.py
"""
Comprehensive tests for topology components.

Tests cover:
- BraidConfig: Configuration dataclass
- BraidGroup: Mathematical structure for braid groups
- BraidEncoder: Neural encoder producing braid representations
- BraidCrossing: Single braid crossing operation
- TemperleyLiebAlgebra: TL algebra for Jones polynomial computation
- JonesConfig: Configuration for Jones polynomial
- KauffmanBracketNetwork: Neural network computing Kauffman bracket
- JonesEvaluator: Evaluates Jones polynomial at t values
- JonesApproximator: Main Jones polynomial approximator
- JonesLoss: Loss function based on Jones polynomial
- TopologicalHeadConfig: Configuration for TopologicalHead
- TopologicalHead: Head module for topological loss computation
"""

import pytest
import torch
import torch.nn as nn

from neuromanifold_gpt.model.topology import (
    BraidGroup,
    BraidEncoder,
    BraidCrossing,
    BraidConfig,
    TemperleyLiebAlgebra,
    JonesApproximator,
    JonesConfig,
    JonesEvaluator,
    JonesLoss,
    KauffmanBracketNetwork,
    TopologicalHead,
    TopologicalHeadConfig,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def device():
    """Return CPU device for testing."""
    return torch.device("cpu")


@pytest.fixture
def dtype():
    """Return default dtype for testing."""
    return torch.float32


@pytest.fixture
def batch_size():
    """Standard batch size for tests."""
    return 2


@pytest.fixture
def seq_len():
    """Standard sequence length for tests."""
    return 32


@pytest.fixture
def embed_dim():
    """Standard embedding dimension for tests."""
    return 384


@pytest.fixture
def n_strands():
    """Standard number of braid strands."""
    return 4


@pytest.fixture
def n_coefficients():
    """Standard number of polynomial coefficients."""
    return 8


@pytest.fixture
def sample_input(batch_size, seq_len, embed_dim, device, dtype):
    """Create sample input tensor."""
    return torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)


@pytest.fixture
def small_input(device, dtype):
    """Create small input for quick tests."""
    return torch.randn(2, 16, 128, device=device, dtype=dtype)


# ==============================================================================
# BraidConfig Tests
# ==============================================================================

class TestBraidConfig:
    """Tests for BraidConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BraidConfig()
        assert config.n_strands == 4
        assert config.use_burau is True
        assert config.use_reduced is True
        assert config.t_param == -1.0
        assert config.dropout == 0.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = BraidConfig(
            n_strands=6,
            use_burau=False,
            use_reduced=False,
            t_param=0.5,
            dropout=0.1,
        )
        assert config.n_strands == 6
        assert config.use_burau is False
        assert config.use_reduced is False
        assert config.t_param == 0.5
        assert config.dropout == 0.1

    def test_config_is_dataclass(self):
        """Test that BraidConfig is a proper dataclass."""
        from dataclasses import is_dataclass
        assert is_dataclass(BraidConfig)


# ==============================================================================
# BraidGroup Tests
# ==============================================================================

class TestBraidGroup:
    """Tests for BraidGroup mathematical structure."""

    def test_instantiation(self, n_strands):
        """Test basic instantiation."""
        bg = BraidGroup(n_strands=n_strands)
        assert bg.n_strands == n_strands
        assert bg.n_generators == n_strands - 1
        assert bg.rep_dim == n_strands - 1  # Reduced representation

    def test_instantiation_full_burau(self, n_strands):
        """Test instantiation with full Burau representation."""
        bg = BraidGroup(n_strands=n_strands, use_reduced=False)
        assert bg.rep_dim == n_strands

    def test_instantiation_with_params(self):
        """Test instantiation with custom parameters."""
        bg = BraidGroup(
            n_strands=5,
            t_param=0.5,
            use_reduced=True,
        )
        assert bg.n_strands == 5
        assert bg.t_param == 0.5
        assert bg.use_reduced is True

    def test_generator_matrix_shape(self, n_strands):
        """Test generator matrix has correct shape."""
        bg = BraidGroup(n_strands=n_strands)
        for i in range(bg.n_generators):
            matrix = bg.generator_matrix(i)
            assert matrix.shape == (bg.rep_dim, bg.rep_dim)

    def test_generator_matrix_inverse_shape(self, n_strands):
        """Test inverse generator matrix has correct shape."""
        bg = BraidGroup(n_strands=n_strands)
        for i in range(bg.n_generators):
            matrix = bg.generator_matrix(i, inverse=True)
            assert matrix.shape == (bg.rep_dim, bg.rep_dim)

    def test_generator_index_validation(self, n_strands):
        """Test that invalid generator indices raise errors."""
        bg = BraidGroup(n_strands=n_strands)
        with pytest.raises(ValueError):
            bg.generator_matrix(-1)
        with pytest.raises(ValueError):
            bg.generator_matrix(bg.n_generators)

    def test_get_all_generators(self, n_strands):
        """Test getting all generators at once."""
        bg = BraidGroup(n_strands=n_strands)
        generators, inverses = bg.get_all_generators()
        assert generators.shape == (bg.n_generators, bg.rep_dim, bg.rep_dim)
        assert inverses.shape == (bg.n_generators, bg.rep_dim, bg.rep_dim)

    def test_verify_braid_relation(self, n_strands):
        """Test that Yang-Baxter braid relation holds."""
        bg = BraidGroup(n_strands=n_strands)
        assert bg.verify_braid_relation() is True

    def test_braid_relation_with_different_t(self):
        """Test braid relation with different t parameters."""
        for t in [-1.0, 0.5, 2.0]:
            bg = BraidGroup(n_strands=4, t_param=t)
            assert bg.verify_braid_relation() is True

    def test_word_to_matrix(self, n_strands):
        """Test braid word to matrix conversion."""
        bg = BraidGroup(n_strands=n_strands)
        # Simple word: σ_0 σ_1
        word = [(0, 1), (1, 1)]
        matrix = bg.word_to_matrix(word)
        assert matrix.shape == (bg.rep_dim, bg.rep_dim)

    def test_word_to_matrix_with_inverses(self, n_strands):
        """Test word to matrix with inverse generators."""
        bg = BraidGroup(n_strands=n_strands)
        # Word: σ_0 σ_1^{-1}
        word = [(0, 1), (1, -1)]
        matrix = bg.word_to_matrix(word)
        assert matrix.shape == (bg.rep_dim, bg.rep_dim)

    def test_empty_word(self, n_strands):
        """Test that empty word gives identity matrix."""
        bg = BraidGroup(n_strands=n_strands)
        matrix = bg.word_to_matrix([])
        identity = torch.eye(bg.rep_dim)
        assert torch.allclose(matrix, identity)

    def test_random_braid_word(self, n_strands):
        """Test random braid word generation."""
        bg = BraidGroup(n_strands=n_strands)
        word = bg.random_braid_word(length=5)
        assert len(word) == 5
        for gen_idx, power in word:
            assert 0 <= gen_idx < bg.n_generators
            assert power in [-1, 1]

    def test_repr(self, n_strands):
        """Test string representation."""
        bg = BraidGroup(n_strands=n_strands, t_param=-1.0, use_reduced=True)
        repr_str = repr(bg)
        assert f'n_strands={n_strands}' in repr_str
        assert 't=-1.0' in repr_str


# ==============================================================================
# BraidEncoder Tests
# ==============================================================================

class TestBraidEncoder:
    """Tests for BraidEncoder neural network."""

    def test_instantiation(self, embed_dim, n_strands):
        """Test basic instantiation."""
        encoder = BraidEncoder(embed_dim=embed_dim, n_strands=n_strands)
        assert encoder.embed_dim == embed_dim
        assert encoder.n_strands == n_strands
        assert encoder.rep_dim == n_strands - 1

    def test_instantiation_with_params(self, embed_dim):
        """Test instantiation with custom parameters."""
        encoder = BraidEncoder(
            embed_dim=embed_dim,
            n_strands=5,
            n_crossings=10,
            t_param=0.5,
            use_reduced=True,
            use_attention=False,
            dropout=0.1,
        )
        assert encoder.n_strands == 5
        assert encoder.n_crossings == 10
        assert encoder.use_attention is False

    def test_forward_shape(self, sample_input, embed_dim, n_strands):
        """Test forward pass output shape."""
        encoder = BraidEncoder(embed_dim=embed_dim, n_strands=n_strands)
        rep = encoder(sample_input)
        rep_dim = n_strands - 1
        assert rep.shape == (sample_input.shape[0], rep_dim, rep_dim)

    def test_forward_with_mask(self, sample_input, embed_dim, n_strands):
        """Test forward pass with attention mask."""
        encoder = BraidEncoder(embed_dim=embed_dim, n_strands=n_strands)
        mask = torch.ones(sample_input.shape[0], sample_input.shape[1], dtype=torch.bool)
        mask[:, sample_input.shape[1] // 2:] = False
        rep = encoder(sample_input, mask=mask)
        rep_dim = n_strands - 1
        assert rep.shape == (sample_input.shape[0], rep_dim, rep_dim)

    def test_forward_return_info(self, sample_input, embed_dim, n_strands):
        """Test forward pass with return_info=True."""
        encoder = BraidEncoder(embed_dim=embed_dim, n_strands=n_strands)
        rep, info = encoder(sample_input, return_info=True)

        assert 'generator_weights' in info
        assert 'crossing_logits' in info
        assert 'dominant_generator' in info
        assert info['generator_weights'].shape[0] == sample_input.shape[0]

    def test_forward_without_attention(self, sample_input, embed_dim, n_strands):
        """Test forward pass without attention weighting."""
        encoder = BraidEncoder(
            embed_dim=embed_dim,
            n_strands=n_strands,
            use_attention=False,
        )
        rep = encoder(sample_input)
        rep_dim = n_strands - 1
        assert rep.shape == (sample_input.shape[0], rep_dim, rep_dim)

    def test_compute_writhe(self, embed_dim, n_strands, device, dtype):
        """Test writhe computation."""
        encoder = BraidEncoder(embed_dim=embed_dim, n_strands=n_strands)
        # Create fake generator weights
        n_gens = n_strands - 1
        weights = torch.softmax(torch.randn(2, 2 * n_gens, device=device, dtype=dtype), dim=-1)
        writhe = encoder.compute_writhe(weights)
        assert writhe.shape == (2,)

    def test_extra_repr(self, embed_dim, n_strands):
        """Test string representation."""
        encoder = BraidEncoder(embed_dim=embed_dim, n_strands=n_strands)
        repr_str = encoder.extra_repr()
        assert f'embed_dim={embed_dim}' in repr_str
        assert f'n_strands={n_strands}' in repr_str


# ==============================================================================
# BraidCrossing Tests
# ==============================================================================

class TestBraidCrossing:
    """Tests for BraidCrossing layer."""

    def test_instantiation(self, embed_dim, n_strands):
        """Test basic instantiation."""
        crossing = BraidCrossing(embed_dim=embed_dim, n_strands=n_strands)
        assert crossing.embed_dim == embed_dim
        assert crossing.n_strands == n_strands

    def test_forward_shape(self, small_input):
        """Test forward pass output shape."""
        crossing = BraidCrossing(embed_dim=small_input.shape[-1], n_strands=4)
        out = crossing(small_input)
        assert out.shape == small_input.shape

    def test_forward_with_positions(self, small_input):
        """Test forward pass with specific positions."""
        crossing = BraidCrossing(embed_dim=small_input.shape[-1], n_strands=4)
        out = crossing(small_input, positions=(0, 1))
        assert out.shape == small_input.shape

    def test_forward_all_adjacent_pairs(self, small_input):
        """Test forward pass applying to all adjacent pairs."""
        crossing = BraidCrossing(embed_dim=small_input.shape[-1], n_strands=4)
        out = crossing(small_input, positions=None)  # All adjacent pairs
        assert out.shape == small_input.shape

    def test_extra_repr(self, embed_dim, n_strands):
        """Test string representation."""
        crossing = BraidCrossing(embed_dim=embed_dim, n_strands=n_strands)
        repr_str = crossing.extra_repr()
        assert f'embed_dim={embed_dim}' in repr_str


# ==============================================================================
# TemperleyLiebAlgebra Tests
# ==============================================================================

class TestTemperleyLiebAlgebra:
    """Tests for TemperleyLiebAlgebra."""

    def test_instantiation(self, n_strands):
        """Test basic instantiation."""
        tl = TemperleyLiebAlgebra(n=n_strands)
        assert tl.n == n_strands
        assert tl.dim > 0  # Catalan number

    def test_instantiation_with_delta(self):
        """Test instantiation with custom delta."""
        tl = TemperleyLiebAlgebra(n=4, delta=2.0)
        assert torch.isclose(tl.delta, torch.tensor(2.0))

    def test_catalan_number(self):
        """Test Catalan number computation."""
        # Catalan numbers: 1, 1, 2, 5, 14, 42, ...
        expected = [1, 1, 2, 5, 14, 42]
        for n, expected_val in enumerate(expected):
            assert TemperleyLiebAlgebra._catalan(n) == expected_val

    def test_generator_matrix_shape(self, n_strands):
        """Test generator matrix shape."""
        tl = TemperleyLiebAlgebra(n=n_strands)
        for i in range(n_strands - 1):
            matrix = tl.generator_matrix(i)
            assert matrix.shape == (tl.dim, tl.dim)

    def test_repr(self, n_strands):
        """Test string representation."""
        tl = TemperleyLiebAlgebra(n=n_strands)
        repr_str = repr(tl)
        assert f'n={n_strands}' in repr_str
        assert 'dim=' in repr_str


# ==============================================================================
# JonesConfig Tests
# ==============================================================================

class TestJonesConfig:
    """Tests for JonesConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = JonesConfig()
        assert config.embed_dim == 384
        assert config.n_strands == 4
        assert config.n_coefficients == 8
        assert config.use_kauffman_bracket is True
        assert config.use_temperley_lieb is True
        assert config.dropout == 0.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = JonesConfig(
            embed_dim=512,
            n_strands=6,
            n_coefficients=12,
            use_kauffman_bracket=False,
            use_temperley_lieb=False,
            t_values=(-0.5, 1.0),
            dropout=0.1,
        )
        assert config.embed_dim == 512
        assert config.n_strands == 6
        assert config.n_coefficients == 12
        assert config.use_kauffman_bracket is False
        assert config.t_values == (-0.5, 1.0)

    def test_config_is_dataclass(self):
        """Test that JonesConfig is a proper dataclass."""
        from dataclasses import is_dataclass
        assert is_dataclass(JonesConfig)


# ==============================================================================
# KauffmanBracketNetwork Tests
# ==============================================================================

class TestKauffmanBracketNetwork:
    """Tests for KauffmanBracketNetwork."""

    def test_instantiation(self, embed_dim, n_coefficients):
        """Test basic instantiation."""
        kb = KauffmanBracketNetwork(
            embed_dim=embed_dim,
            rep_dim=3,
            n_coefficients=n_coefficients,
        )
        assert kb.rep_dim == 3
        assert kb.n_coefficients == n_coefficients

    def test_forward_shape(self, n_coefficients, device, dtype):
        """Test forward pass output shape."""
        kb = KauffmanBracketNetwork(
            embed_dim=384,
            rep_dim=3,
            n_coefficients=n_coefficients,
        )
        braid_rep = torch.randn(2, 3, 3, device=device, dtype=dtype)
        bracket = kb(braid_rep)
        assert bracket.shape == (2, n_coefficients * 2)

    def test_forward_return_complex(self, n_coefficients, device, dtype):
        """Test forward pass with complex output."""
        kb = KauffmanBracketNetwork(
            embed_dim=384,
            rep_dim=3,
            n_coefficients=n_coefficients,
        )
        braid_rep = torch.randn(2, 3, 3, device=device, dtype=dtype)
        bracket = kb(braid_rep, return_complex=True)
        assert bracket.shape == (2, n_coefficients)
        assert bracket.is_complex()

    def test_extra_repr(self, n_coefficients):
        """Test string representation."""
        kb = KauffmanBracketNetwork(
            embed_dim=384,
            rep_dim=3,
            n_coefficients=n_coefficients,
        )
        repr_str = kb.extra_repr()
        assert 'rep_dim=3' in repr_str
        assert f'n_coefficients={n_coefficients}' in repr_str


# ==============================================================================
# JonesEvaluator Tests
# ==============================================================================

class TestJonesEvaluator:
    """Tests for JonesEvaluator."""

    def test_instantiation(self, n_coefficients):
        """Test basic instantiation."""
        evaluator = JonesEvaluator(n_coefficients=n_coefficients)
        assert evaluator.n_coefficients == n_coefficients
        assert evaluator.n_evaluations == 4  # Default t_values

    def test_instantiation_with_t_values(self):
        """Test instantiation with custom t values."""
        t_values = (-0.5, 0.0, 0.5, 1.5, 2.5)
        evaluator = JonesEvaluator(n_coefficients=8, t_values=t_values)
        assert evaluator.n_evaluations == 5

    def test_forward_shape(self, n_coefficients, device, dtype):
        """Test forward pass output shape."""
        evaluator = JonesEvaluator(n_coefficients=n_coefficients)
        coefficients = torch.randn(2, n_coefficients * 2, device=device, dtype=dtype)
        evaluations = evaluator(coefficients)
        assert evaluations.shape == (2, evaluator.n_evaluations)

    def test_forward_complex_input(self, n_coefficients, device, dtype):
        """Test forward pass with complex input."""
        evaluator = JonesEvaluator(n_coefficients=n_coefficients)
        # Create complex coefficients
        coef_real = torch.randn(2, n_coefficients, device=device, dtype=dtype)
        coef_imag = torch.randn(2, n_coefficients, device=device, dtype=dtype)
        coefficients = torch.complex(coef_real, coef_imag)
        evaluations = evaluator(coefficients)
        assert evaluations.shape == (2, evaluator.n_evaluations)

    def test_extra_repr(self, n_coefficients):
        """Test string representation."""
        evaluator = JonesEvaluator(n_coefficients=n_coefficients)
        repr_str = evaluator.extra_repr()
        assert f'n_coefficients={n_coefficients}' in repr_str


# ==============================================================================
# JonesApproximator Tests
# ==============================================================================

class TestJonesApproximator:
    """Tests for JonesApproximator."""

    def test_instantiation(self, embed_dim, n_strands):
        """Test basic instantiation."""
        jones = JonesApproximator(embed_dim=embed_dim, n_strands=n_strands)
        assert jones.embed_dim == embed_dim
        assert jones.n_strands == n_strands
        assert jones.rep_dim == n_strands - 1

    def test_instantiation_with_params(self, embed_dim):
        """Test instantiation with custom parameters."""
        jones = JonesApproximator(
            embed_dim=embed_dim,
            n_strands=5,
            n_coefficients=12,
            t_values=(-0.5, 1.0, 2.0),
            hidden_dim=256,
            dropout=0.1,
            use_kauffman_bracket=False,
            use_temperley_lieb=False,
        )
        assert jones.n_strands == 5
        assert jones.n_coefficients == 12
        assert jones.use_kauffman_bracket is False
        assert jones.use_temperley_lieb is False

    def test_forward_shape(self, sample_input, embed_dim, n_strands):
        """Test forward pass output shape."""
        jones = JonesApproximator(embed_dim=embed_dim, n_strands=n_strands)
        poly, info = jones(sample_input)
        assert poly.shape[0] == sample_input.shape[0]
        assert poly.shape[1] == jones.output_dim

    def test_forward_info_dict(self, sample_input, embed_dim, n_strands):
        """Test that forward returns proper info dictionary."""
        jones = JonesApproximator(embed_dim=embed_dim, n_strands=n_strands)
        _, info = jones(sample_input)

        assert 'braid_rep' in info
        assert 'generator_weights' in info
        assert 'dominant_generator' in info
        assert 'writhe' in info

    def test_forward_with_mask(self, sample_input, embed_dim, n_strands):
        """Test forward pass with attention mask."""
        jones = JonesApproximator(embed_dim=embed_dim, n_strands=n_strands)
        mask = torch.ones(sample_input.shape[0], sample_input.shape[1], dtype=torch.bool)
        mask[:, sample_input.shape[1] // 2:] = False
        poly, _ = jones(sample_input, mask=mask)
        assert poly.shape[0] == sample_input.shape[0]

    def test_compute_invariants(self, embed_dim, n_strands, device, dtype):
        """Test invariant computation from braid representation."""
        jones = JonesApproximator(embed_dim=embed_dim, n_strands=n_strands)
        rep_dim = n_strands - 1
        braid_rep = torch.randn(2, rep_dim, rep_dim, device=device, dtype=dtype)
        invariants, info = jones.compute_invariants(braid_rep)
        assert invariants.shape[0] == 2

    def test_compute_jones_distance(self, embed_dim, n_strands, device, dtype):
        """Test Jones polynomial distance computation."""
        jones = JonesApproximator(embed_dim=embed_dim, n_strands=n_strands)
        poly1 = torch.randn(2, jones.output_dim, device=device, dtype=dtype)
        poly2 = torch.randn(2, jones.output_dim, device=device, dtype=dtype)
        dist = jones.compute_jones_distance(poly1, poly2)
        assert dist.shape == (2,)
        assert (dist >= 0).all()

    def test_topological_similarity(self, sample_input, embed_dim, n_strands):
        """Test topological similarity computation."""
        jones = JonesApproximator(embed_dim=embed_dim, n_strands=n_strands)
        x1 = sample_input
        x2 = sample_input.clone() + 0.1 * torch.randn_like(sample_input)
        sim = jones.topological_similarity(x1, x2)
        assert sim.shape == (sample_input.shape[0],)
        assert (sim >= -1).all() and (sim <= 1).all()

    def test_without_kauffman_bracket(self, sample_input, embed_dim, n_strands):
        """Test forward pass without Kauffman bracket."""
        jones = JonesApproximator(
            embed_dim=embed_dim,
            n_strands=n_strands,
            use_kauffman_bracket=False,
        )
        poly, _ = jones(sample_input)
        assert poly.shape[0] == sample_input.shape[0]

    def test_without_temperley_lieb(self, sample_input, embed_dim, n_strands):
        """Test forward pass without Temperley-Lieb."""
        jones = JonesApproximator(
            embed_dim=embed_dim,
            n_strands=n_strands,
            use_temperley_lieb=False,
        )
        poly, _ = jones(sample_input)
        assert poly.shape[0] == sample_input.shape[0]

    def test_extra_repr(self, embed_dim, n_strands):
        """Test string representation."""
        jones = JonesApproximator(embed_dim=embed_dim, n_strands=n_strands)
        repr_str = jones.extra_repr()
        assert f'embed_dim={embed_dim}' in repr_str
        assert f'n_strands={n_strands}' in repr_str


# ==============================================================================
# JonesLoss Tests
# ==============================================================================

class TestJonesLoss:
    """Tests for JonesLoss."""

    def test_instantiation(self, embed_dim):
        """Test basic instantiation."""
        loss_fn = JonesLoss(embed_dim=embed_dim)
        assert loss_fn.margin == 0.5
        assert loss_fn.invariance_weight == 1.0
        assert loss_fn.distinctiveness_weight == 0.5

    def test_instantiation_with_params(self, embed_dim):
        """Test instantiation with custom parameters."""
        loss_fn = JonesLoss(
            embed_dim=embed_dim,
            margin=1.0,
            invariance_weight=2.0,
            distinctiveness_weight=0.3,
        )
        assert loss_fn.margin == 1.0
        assert loss_fn.invariance_weight == 2.0
        assert loss_fn.distinctiveness_weight == 0.3

    def test_forward_basic(self, sample_input, embed_dim):
        """Test forward pass with only input."""
        loss_fn = JonesLoss(embed_dim=embed_dim)
        loss, info = loss_fn(sample_input)
        assert loss.dim() == 0  # Scalar
        assert 'total_loss' in info
        assert 'reg_loss' in info

    def test_forward_with_augmented(self, sample_input, embed_dim):
        """Test forward pass with augmented sample."""
        loss_fn = JonesLoss(embed_dim=embed_dim)
        x_aug = sample_input + 0.1 * torch.randn_like(sample_input)
        loss, info = loss_fn(sample_input, x_aug=x_aug)
        assert loss.dim() == 0
        assert 'invariance_loss' in info

    def test_forward_with_negative(self, sample_input, embed_dim):
        """Test forward pass with negative sample."""
        loss_fn = JonesLoss(embed_dim=embed_dim)
        x_neg = torch.randn_like(sample_input)  # Different sample
        loss, info = loss_fn(sample_input, x_neg=x_neg)
        assert loss.dim() == 0
        assert 'distinctiveness_loss' in info

    def test_forward_with_both(self, sample_input, embed_dim):
        """Test forward pass with augmented and negative samples."""
        loss_fn = JonesLoss(embed_dim=embed_dim)
        x_aug = sample_input + 0.1 * torch.randn_like(sample_input)
        x_neg = torch.randn_like(sample_input)
        loss, info = loss_fn(sample_input, x_aug=x_aug, x_neg=x_neg)
        assert loss.dim() == 0
        assert 'invariance_loss' in info
        assert 'distinctiveness_loss' in info

    def test_extra_repr(self, embed_dim):
        """Test string representation."""
        loss_fn = JonesLoss(embed_dim=embed_dim)
        repr_str = loss_fn.extra_repr()
        assert 'margin=' in repr_str


# ==============================================================================
# TopologicalHeadConfig Tests
# ==============================================================================

class TestTopologicalHeadConfig:
    """Tests for TopologicalHeadConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TopologicalHeadConfig()
        assert config.embed_dim == 384
        assert config.n_strands == 4
        assert config.n_coefficients == 8
        assert config.smoothness_weight == 1.0
        assert config.consistency_weight == 0.5
        assert config.regularity_weight == 0.1
        assert config.coherence_weight == 0.5
        assert config.use_jones is True
        assert config.use_braid is True
        assert config.use_temporal_coherence is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TopologicalHeadConfig(
            embed_dim=512,
            n_strands=6,
            n_coefficients=12,
            smoothness_weight=2.0,
            use_jones=False,
            use_braid=True,
            use_temporal_coherence=False,
        )
        assert config.embed_dim == 512
        assert config.n_strands == 6
        assert config.smoothness_weight == 2.0
        assert config.use_jones is False
        assert config.use_temporal_coherence is False

    def test_config_is_dataclass(self):
        """Test that TopologicalHeadConfig is a proper dataclass."""
        from dataclasses import is_dataclass
        assert is_dataclass(TopologicalHeadConfig)


# ==============================================================================
# TopologicalHead Tests
# ==============================================================================

class TestTopologicalHead:
    """Tests for TopologicalHead."""

    def test_instantiation(self, embed_dim, n_strands):
        """Test basic instantiation."""
        head = TopologicalHead(embed_dim=embed_dim, n_strands=n_strands)
        assert head.embed_dim == embed_dim
        assert head.n_strands == n_strands
        assert head.use_jones is True
        assert head.use_braid is True

    def test_instantiation_with_params(self, embed_dim):
        """Test instantiation with custom parameters."""
        head = TopologicalHead(
            embed_dim=embed_dim,
            n_strands=5,
            n_coefficients=12,
            smoothness_weight=2.0,
            consistency_weight=1.0,
            regularity_weight=0.2,
            coherence_weight=0.8,
            use_jones=True,
            use_braid=False,
            use_temporal_coherence=True,
            smoothness_order=3,
            coherence_window=5,
            dropout=0.1,
        )
        assert head.n_strands == 5
        assert head.smoothness_weight == 2.0
        assert head.use_braid is False
        assert head.smoothness_order == 3
        assert head.coherence_window == 5

    def test_forward_shape(self, sample_input, embed_dim, n_strands):
        """Test forward pass output shape."""
        head = TopologicalHead(embed_dim=embed_dim, n_strands=n_strands)
        loss, info = head(sample_input)
        assert loss.dim() == 0  # Scalar

    def test_forward_info_dict(self, sample_input, embed_dim, n_strands):
        """Test that forward returns proper info dictionary."""
        head = TopologicalHead(embed_dim=embed_dim, n_strands=n_strands)
        _, info = head(sample_input)

        assert 'total_loss' in info
        assert 'temperature' in info
        assert 'smoothness_loss' in info
        assert 'regularity_loss' in info
        assert 'consistency_loss' in info
        assert 'coherence_loss' in info

    def test_forward_with_mask(self, sample_input, embed_dim, n_strands):
        """Test forward pass with attention mask."""
        head = TopologicalHead(embed_dim=embed_dim, n_strands=n_strands)
        mask = torch.ones(sample_input.shape[0], sample_input.shape[1], dtype=torch.bool)
        mask[:, sample_input.shape[1] // 2:] = False
        loss, _ = head(sample_input, mask=mask)
        assert loss.dim() == 0

    def test_compute_smoothness_loss(self, embed_dim, n_strands, device, dtype):
        """Test smoothness loss computation."""
        head = TopologicalHead(embed_dim=embed_dim, n_strands=n_strands)
        coefficients = torch.randn(2, 16, device=device, dtype=dtype)
        loss = head.compute_smoothness_loss(coefficients)
        assert loss.dim() == 0
        assert loss >= 0

    def test_compute_consistency_loss(self, embed_dim, n_strands, device, dtype):
        """Test consistency loss computation."""
        head = TopologicalHead(embed_dim=embed_dim, n_strands=n_strands)
        rep_dim = n_strands - 1
        braid_rep = torch.randn(2, rep_dim, rep_dim, device=device, dtype=dtype)
        loss = head.compute_consistency_loss(braid_rep)
        assert loss.dim() == 0
        assert loss >= 0

    def test_compute_regularity_loss(self, embed_dim, n_strands, device, dtype):
        """Test regularity loss computation."""
        head = TopologicalHead(embed_dim=embed_dim, n_strands=n_strands)
        invariants = torch.randn(2, 16, device=device, dtype=dtype)
        loss = head.compute_regularity_loss(invariants)
        assert loss.dim() == 0
        assert loss >= 0

    def test_compute_coherence_loss(self, sample_input, embed_dim, n_strands):
        """Test coherence loss computation."""
        head = TopologicalHead(embed_dim=embed_dim, n_strands=n_strands)
        loss = head.compute_coherence_loss(sample_input)
        assert loss.dim() == 0
        assert loss >= 0

    def test_compute_coherence_loss_short_sequence(self, embed_dim, n_strands, device, dtype):
        """Test coherence loss with sequence length 1."""
        head = TopologicalHead(embed_dim=embed_dim, n_strands=n_strands)
        x = torch.randn(2, 1, embed_dim, device=device, dtype=dtype)
        loss = head.compute_coherence_loss(x)
        assert loss == 0.0  # Should be 0 for T < 2

    def test_get_topological_features(self, sample_input, embed_dim, n_strands):
        """Test feature extraction without loss computation."""
        head = TopologicalHead(embed_dim=embed_dim, n_strands=n_strands)
        features = head.get_topological_features(sample_input)

        assert 'jones_poly' in features
        assert 'braid_rep' in features
        assert 'position_features' in features
        assert features['braid_rep'].shape == (sample_input.shape[0], n_strands - 1, n_strands - 1)

    def test_topological_distance(self, sample_input, embed_dim, n_strands):
        """Test topological distance computation."""
        head = TopologicalHead(embed_dim=embed_dim, n_strands=n_strands)
        x1 = sample_input
        x2 = sample_input.clone() + 0.1 * torch.randn_like(sample_input)
        dist = head.topological_distance(x1, x2)
        assert dist.shape == (sample_input.shape[0],)  # Batch-wise distance
        assert (dist >= 0).all()

    def test_jones_only(self, sample_input, embed_dim, n_strands):
        """Test with only Jones enabled."""
        head = TopologicalHead(
            embed_dim=embed_dim,
            n_strands=n_strands,
            use_jones=True,
            use_braid=False,
            use_temporal_coherence=False,
        )
        loss, info = head(sample_input)
        assert 'smoothness_loss' in info
        assert 'consistency_loss' not in info
        assert 'coherence_loss' not in info

    def test_braid_only(self, sample_input, embed_dim, n_strands):
        """Test with only braid enabled."""
        head = TopologicalHead(
            embed_dim=embed_dim,
            n_strands=n_strands,
            use_jones=False,
            use_braid=True,
            use_temporal_coherence=False,
        )
        loss, info = head(sample_input)
        assert 'smoothness_loss' not in info
        assert 'consistency_loss' in info
        assert 'coherence_loss' not in info

    def test_coherence_only(self, sample_input, embed_dim, n_strands):
        """Test with only temporal coherence enabled."""
        head = TopologicalHead(
            embed_dim=embed_dim,
            n_strands=n_strands,
            use_jones=False,
            use_braid=False,
            use_temporal_coherence=True,
        )
        loss, info = head(sample_input)
        assert 'smoothness_loss' not in info
        assert 'consistency_loss' not in info
        assert 'coherence_loss' in info

    def test_extra_repr(self, embed_dim, n_strands):
        """Test string representation."""
        head = TopologicalHead(embed_dim=embed_dim, n_strands=n_strands)
        repr_str = head.extra_repr()
        assert f'embed_dim={embed_dim}' in repr_str
        assert f'n_strands={n_strands}' in repr_str


# ==============================================================================
# Edge Cases and Numerical Stability Tests
# ==============================================================================

class TestTopologyEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_zero_input_braid_encoder(self, embed_dim, n_strands, device, dtype):
        """Test BraidEncoder with zero input."""
        encoder = BraidEncoder(embed_dim=embed_dim, n_strands=n_strands)
        x = torch.zeros(2, 16, embed_dim, device=device, dtype=dtype)
        rep = encoder(x)
        assert torch.isfinite(rep).all()

    def test_zero_input_jones_approximator(self, embed_dim, n_strands, device, dtype):
        """Test JonesApproximator with zero input."""
        jones = JonesApproximator(embed_dim=embed_dim, n_strands=n_strands)
        x = torch.zeros(2, 16, embed_dim, device=device, dtype=dtype)
        poly, _ = jones(x)
        assert torch.isfinite(poly).all()

    def test_zero_input_topological_head(self, embed_dim, n_strands, device, dtype):
        """Test TopologicalHead with zero input."""
        head = TopologicalHead(embed_dim=embed_dim, n_strands=n_strands)
        x = torch.zeros(2, 16, embed_dim, device=device, dtype=dtype)
        loss, _ = head(x)
        assert torch.isfinite(loss)

    def test_large_input_stability(self, embed_dim, n_strands, device, dtype):
        """Test numerical stability with large inputs."""
        jones = JonesApproximator(embed_dim=embed_dim, n_strands=n_strands)
        x = torch.randn(2, 16, embed_dim, device=device, dtype=dtype) * 100
        poly, _ = jones(x)
        assert torch.isfinite(poly).all()

    def test_batch_size_one(self, embed_dim, n_strands, device, dtype):
        """Test with batch size of 1."""
        jones = JonesApproximator(embed_dim=embed_dim, n_strands=n_strands)
        x = torch.randn(1, 16, embed_dim, device=device, dtype=dtype)
        poly, _ = jones(x)
        assert poly.shape[0] == 1

    def test_sequence_length_one(self, embed_dim, n_strands, device, dtype):
        """Test with sequence length of 1."""
        head = TopologicalHead(embed_dim=embed_dim, n_strands=n_strands)
        x = torch.randn(2, 1, embed_dim, device=device, dtype=dtype)
        loss, _ = head(x)
        assert torch.isfinite(loss)


# ==============================================================================
# Gradient Flow Tests
# ==============================================================================

class TestTopologyGradientFlow:
    """Tests for gradient flow through topology components."""

    def test_gradient_flow_braid_encoder(self, sample_input, embed_dim, n_strands):
        """Test that gradients flow through BraidEncoder."""
        encoder = BraidEncoder(embed_dim=embed_dim, n_strands=n_strands)
        x = sample_input.clone().requires_grad_(True)
        rep = encoder(x)
        loss = rep.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_gradient_flow_braid_crossing(self, small_input):
        """Test that gradients flow through BraidCrossing."""
        crossing = BraidCrossing(embed_dim=small_input.shape[-1], n_strands=4)
        x = small_input.clone().requires_grad_(True)
        out = crossing(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_gradient_flow_kauffman_bracket(self, device, dtype):
        """Test that gradients flow through KauffmanBracketNetwork."""
        kb = KauffmanBracketNetwork(embed_dim=384, rep_dim=3, n_coefficients=8)
        braid_rep = torch.randn(2, 3, 3, device=device, dtype=dtype, requires_grad=True)
        bracket = kb(braid_rep)
        loss = bracket.sum()
        loss.backward()
        assert braid_rep.grad is not None
        assert torch.isfinite(braid_rep.grad).all()

    def test_gradient_flow_jones_approximator(self, sample_input, embed_dim, n_strands):
        """Test that gradients flow through JonesApproximator."""
        jones = JonesApproximator(embed_dim=embed_dim, n_strands=n_strands)
        x = sample_input.clone().requires_grad_(True)
        poly, _ = jones(x)
        loss = poly.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_gradient_flow_jones_loss(self, sample_input, embed_dim):
        """Test that gradients flow through JonesLoss."""
        loss_fn = JonesLoss(embed_dim=embed_dim)
        x = sample_input.clone().requires_grad_(True)
        loss, _ = loss_fn(x)
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_gradient_flow_topological_head(self, sample_input, embed_dim, n_strands):
        """Test that gradients flow through TopologicalHead."""
        head = TopologicalHead(embed_dim=embed_dim, n_strands=n_strands)
        x = sample_input.clone().requires_grad_(True)
        loss, _ = head(x)
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


# ==============================================================================
# Parameter Learning Tests
# ==============================================================================

class TestTopologyParameterLearning:
    """Tests for learnable parameter behavior."""

    def test_braid_encoder_learnable_params(self, embed_dim, n_strands):
        """Test BraidEncoder has learnable parameters."""
        encoder = BraidEncoder(embed_dim=embed_dim, n_strands=n_strands)
        params = list(encoder.parameters())
        assert len(params) > 0
        assert any('crossing_proj' in name for name, _ in encoder.named_parameters())

    def test_kauffman_bracket_learnable_params(self):
        """Test KauffmanBracketNetwork has learnable parameters."""
        kb = KauffmanBracketNetwork(embed_dim=384, rep_dim=3, n_coefficients=8)
        params = list(kb.parameters())
        assert len(params) > 0
        assert any('output_scale' in name for name, _ in kb.named_parameters())

    def test_jones_evaluator_learnable_params(self):
        """Test JonesEvaluator has learnable parameters."""
        evaluator = JonesEvaluator(n_coefficients=8)
        params = list(evaluator.parameters())
        assert len(params) > 0
        assert any('refine' in name for name, _ in evaluator.named_parameters())

    def test_jones_approximator_learnable_params(self, embed_dim, n_strands):
        """Test JonesApproximator has learnable parameters."""
        jones = JonesApproximator(embed_dim=embed_dim, n_strands=n_strands)
        params = list(jones.parameters())
        assert len(params) > 0
        assert any('braid_encoder' in name for name, _ in jones.named_parameters())
        assert any('output_proj' in name for name, _ in jones.named_parameters())

    def test_topological_head_learnable_params(self, embed_dim, n_strands):
        """Test TopologicalHead has learnable parameters."""
        head = TopologicalHead(embed_dim=embed_dim, n_strands=n_strands)
        params = list(head.parameters())
        assert len(params) > 0
        assert any('log_temperature' in name for name, _ in head.named_parameters())
        assert any('jones' in name for name, _ in head.named_parameters())
        assert any('braid_encoder' in name for name, _ in head.named_parameters())


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestTopologyIntegration:
    """Integration tests for topology components working together."""

    def test_braid_to_jones_pipeline(self, sample_input, embed_dim, n_strands):
        """Test pipeline from BraidEncoder to JonesApproximator."""
        encoder = BraidEncoder(embed_dim=embed_dim, n_strands=n_strands)
        jones = JonesApproximator(embed_dim=embed_dim, n_strands=n_strands)

        # Encode to braid representation
        braid_rep = encoder(sample_input)
        assert braid_rep.shape == (sample_input.shape[0], n_strands - 1, n_strands - 1)

        # Compute Jones polynomial
        poly, info = jones(sample_input)
        assert torch.isfinite(poly).all()

    def test_full_topological_pipeline(self, sample_input, embed_dim, n_strands):
        """Test full topological loss computation pipeline."""
        head = TopologicalHead(embed_dim=embed_dim, n_strands=n_strands)

        # Forward pass
        loss, info = head(sample_input)

        # Check all components present
        assert loss.dim() == 0
        assert 'smoothness_loss' in info
        assert 'consistency_loss' in info
        assert 'regularity_loss' in info
        assert 'coherence_loss' in info
        assert 'total_loss' in info

        # Check loss is non-negative
        assert loss >= 0

    def test_braid_group_to_encoder_consistency(self, embed_dim, n_strands, device, dtype):
        """Test that BraidEncoder uses BraidGroup correctly."""
        encoder = BraidEncoder(embed_dim=embed_dim, n_strands=n_strands)
        bg = BraidGroup(n_strands=n_strands)

        # Check encoder's internal braid group matches
        assert encoder.braid_group.n_strands == bg.n_strands
        assert encoder.braid_group.rep_dim == bg.rep_dim

        # Check generator matrices are registered as buffers
        assert hasattr(encoder, 'generators')
        assert encoder.generators.shape[0] == 2 * (n_strands - 1)


# ==============================================================================
# Run tests if executed directly
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
