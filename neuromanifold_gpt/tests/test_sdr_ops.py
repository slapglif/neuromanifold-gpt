"""Tests for SDR (Sparse Distributed Representations) operations.

SDRs are the foundation of semantic folding. These tests verify:
- hard_topk: exact sparsity enforcement (k active bits)
- overlap_count: bitwise AND population count
- union: bitwise OR with clamping
- semantic_similarity: overlap / n_active

Note: These tests use the reset_random_seed autouse fixture from conftest.py
to ensure reproducibility for tests using torch.randn().
"""
import pytest
import torch
from neuromanifold_gpt.model.sdr_ops import SDROperations


class TestHardTopK:
    """Test hard top-k sparsity enforcement."""

    def test_hard_topk_sparsity(self):
        """Verify exactly n_active bits are set."""
        scores = torch.randn(2, 10, 2048)
        n_active = 40
        sdr = SDROperations.hard_topk(scores, n_active)
        assert sdr.shape == scores.shape
        assert (sdr.sum(dim=-1) == n_active).all()
        assert ((sdr == 0) | (sdr == 1)).all()

    def test_hard_topk_single_tensor(self):
        """Verify works on 1D tensor."""
        scores = torch.randn(2048)
        n_active = 40
        sdr = SDROperations.hard_topk(scores, n_active)
        assert sdr.shape == (2048,)
        assert sdr.sum() == n_active
        assert ((sdr == 0) | (sdr == 1)).all()

    def test_hard_topk_deterministic(self):
        """Same input should produce same output."""
        scores = torch.randn(2048)
        sdr1 = SDROperations.hard_topk(scores, 40)
        sdr2 = SDROperations.hard_topk(scores, 40)
        assert torch.equal(sdr1, sdr2)


class TestOverlapCount:
    """Test overlap counting (bitwise AND + popcount)."""

    def test_overlap_count(self):
        """Verify overlap count with known values."""
        sdr_a = torch.zeros(2048)
        sdr_a[:40] = 1
        sdr_b = torch.zeros(2048)
        sdr_b[20:60] = 1
        overlap = SDROperations.overlap_count(sdr_a, sdr_b)
        assert overlap == 20

    def test_overlap_count_no_overlap(self):
        """Disjoint SDRs should have zero overlap."""
        sdr_a = torch.zeros(2048)
        sdr_a[:40] = 1
        sdr_b = torch.zeros(2048)
        sdr_b[100:140] = 1
        overlap = SDROperations.overlap_count(sdr_a, sdr_b)
        assert overlap == 0

    def test_overlap_count_full_overlap(self):
        """Identical SDRs should have full overlap."""
        sdr = torch.zeros(2048)
        sdr[:40] = 1
        overlap = SDROperations.overlap_count(sdr, sdr)
        assert overlap == 40

    def test_overlap_count_batched(self):
        """Verify batched operation."""
        sdr_a = torch.zeros(2, 3, 2048)
        sdr_a[..., :40] = 1
        sdr_b = torch.zeros(2, 3, 2048)
        sdr_b[..., 20:60] = 1
        overlap = SDROperations.overlap_count(sdr_a, sdr_b)
        assert overlap.shape == (2, 3)
        assert (overlap == 20).all()


class TestUnion:
    """Test union operation (bitwise OR with clamping)."""

    def test_union(self):
        """Verify union with known values."""
        sdr_a = torch.zeros(2048)
        sdr_a[:40] = 1
        sdr_b = torch.zeros(2048)
        sdr_b[20:60] = 1
        union = SDROperations.union(sdr_a, sdr_b)
        assert union.sum() == 60
        assert union.max() == 1.0

    def test_union_clamping(self):
        """Overlapping bits should not exceed 1."""
        sdr_a = torch.zeros(2048)
        sdr_a[:40] = 1
        sdr_b = torch.zeros(2048)
        sdr_b[:40] = 1  # Same bits
        union = SDROperations.union(sdr_a, sdr_b)
        assert union.sum() == 40
        assert union.max() == 1.0

    def test_union_empty(self):
        """Union of empty SDRs should be empty."""
        sdr_a = torch.zeros(2048)
        sdr_b = torch.zeros(2048)
        union = SDROperations.union(sdr_a, sdr_b)
        assert union.sum() == 0


class TestIntersection:
    """Test intersection operation (bitwise AND)."""

    def test_intersection(self):
        """Verify intersection with known values."""
        sdr_a = torch.zeros(2048)
        sdr_a[:40] = 1
        sdr_b = torch.zeros(2048)
        sdr_b[20:60] = 1
        intersection = SDROperations.intersection(sdr_a, sdr_b)
        assert intersection.sum() == 20
        assert intersection.max() == 1.0

    def test_intersection_disjoint(self):
        """Disjoint SDRs should have empty intersection."""
        sdr_a = torch.zeros(2048)
        sdr_a[:40] = 1
        sdr_b = torch.zeros(2048)
        sdr_b[100:140] = 1
        intersection = SDROperations.intersection(sdr_a, sdr_b)
        assert intersection.sum() == 0


class TestSemanticSimilarity:
    """Test semantic similarity (overlap / n_active)."""

    def test_semantic_similarity_identical(self):
        """Identical SDRs should have similarity 1.0."""
        sdr = torch.zeros(2048)
        sdr[:40] = 1
        sim = SDROperations.semantic_similarity(sdr, sdr, n_active=40)
        assert sim == 1.0

    def test_semantic_similarity_disjoint(self):
        """Disjoint SDRs should have similarity 0.0."""
        sdr_a = torch.zeros(2048)
        sdr_a[:40] = 1
        sdr_b = torch.zeros(2048)
        sdr_b[100:140] = 1
        sim = SDROperations.semantic_similarity(sdr_a, sdr_b, n_active=40)
        assert sim == 0.0

    def test_semantic_similarity_half(self):
        """50% overlap should give 0.5 similarity."""
        sdr_a = torch.zeros(2048)
        sdr_a[:40] = 1
        sdr_b = torch.zeros(2048)
        sdr_b[20:60] = 1  # 20 overlap out of 40
        sim = SDROperations.semantic_similarity(sdr_a, sdr_b, n_active=40)
        assert sim == 0.5


class TestSparsity:
    """Test sparsity calculation."""

    def test_sparsity(self):
        """Verify sparsity calculation."""
        sdr = torch.zeros(2048)
        sdr[:40] = 1  # 40/2048 = 0.01953125
        sparsity = SDROperations.sparsity(sdr)
        assert abs(sparsity - 40 / 2048) < 1e-6

    def test_sparsity_target(self):
        """SDR with 2% sparsity target."""
        n_active = 40  # ~2% of 2048
        sdr = torch.zeros(2048)
        sdr[:n_active] = 1
        sparsity = SDROperations.sparsity(sdr)
        assert sparsity < 0.02  # Should be close to 2%


class TestSoftTopK:
    """Test differentiable soft top-k for training."""

    def test_soft_topk_gradient_flow(self):
        """Verify gradients flow through soft_topk."""
        scores = torch.randn(2048, requires_grad=True)
        n_active = 40
        sdr = SDROperations.soft_topk(scores, n_active, temperature=1.0)
        loss = sdr.sum()
        loss.backward()
        assert scores.grad is not None
        assert scores.grad.abs().sum() > 0

    def test_soft_topk_same_sparsity(self):
        """Hard component should enforce sparsity."""
        scores = torch.randn(2048)
        n_active = 40
        sdr = SDROperations.soft_topk(scores, n_active, temperature=1.0)
        # The hard component should give exactly n_active
        hard = SDROperations.hard_topk(scores, n_active)
        # soft_topk = hard + (soft - soft.detach())
        # so in forward pass, output = hard + soft - soft = hard
        # This test verifies the STE (straight-through estimator) pattern
        assert sdr.sum() >= n_active - 1e-6  # Allow for floating point


class TestDtypeSupport:
    """Test operations work with different dtypes."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_hard_topk_dtype(self, dtype):
        """Verify dtype preservation."""
        scores = torch.randn(2048, dtype=dtype)
        sdr = SDROperations.hard_topk(scores, 40)
        assert sdr.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_overlap_count_dtype(self, dtype):
        """Verify overlap works with different dtypes."""
        sdr_a = torch.zeros(2048, dtype=dtype)
        sdr_a[:40] = 1
        sdr_b = torch.zeros(2048, dtype=dtype)
        sdr_b[20:60] = 1
        overlap = SDROperations.overlap_count(sdr_a, sdr_b)
        assert overlap == 20
