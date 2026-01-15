"""Tests for SDR metrics evaluation.

These tests verify the correctness of SDR quality metrics:
- compute_sparsity: fraction of active bits (target ~2%)
- compute_entropy: information content and bit distribution
- compute_overlap_statistics: pairwise overlap analysis
- compute_all: aggregated metrics computation
"""
import pytest
import torch
from neuromanifold_gpt.evaluation.sdr_metrics import SDRMetrics


class TestComputeSparsity:
    """Test sparsity calculation (active bits / total bits)."""

    def test_sparsity_exact_2_percent(self):
        """Verify sparsity calculation for 2% target."""
        sdr = torch.zeros(2048)
        sdr[:40] = 1  # 40/2048 ≈ 0.01953125
        sparsity = SDRMetrics.compute_sparsity(sdr)
        expected = 40 / 2048
        assert abs(sparsity - expected) < 1e-6

    def test_sparsity_empty_sdr(self):
        """Empty SDR should have zero sparsity."""
        sdr = torch.zeros(2048)
        sparsity = SDRMetrics.compute_sparsity(sdr)
        assert sparsity == 0.0

    def test_sparsity_full_sdr(self):
        """Fully active SDR should have sparsity 1.0."""
        sdr = torch.ones(2048)
        sparsity = SDRMetrics.compute_sparsity(sdr)
        assert sparsity == 1.0

    def test_sparsity_batched(self):
        """Verify batched sparsity computation."""
        sdr = torch.zeros(2, 3, 2048)
        sdr[..., :40] = 1  # 40/2048 for all
        sparsity = SDRMetrics.compute_sparsity(sdr)
        assert sparsity.shape == (2, 3)
        expected = 40 / 2048
        assert torch.allclose(sparsity, torch.full((2, 3), expected))

    def test_sparsity_varying_activation(self):
        """Different activation levels should give different sparsities."""
        sdr = torch.zeros(3, 2048)
        sdr[0, :20] = 1   # 20/2048
        sdr[1, :40] = 1   # 40/2048
        sdr[2, :80] = 1   # 80/2048
        sparsity = SDRMetrics.compute_sparsity(sdr)
        assert sparsity[0] < sparsity[1] < sparsity[2]
        assert abs(sparsity[0] - 20/2048) < 1e-6
        assert abs(sparsity[1] - 40/2048) < 1e-6
        assert abs(sparsity[2] - 80/2048) < 1e-6


class TestComputeEntropy:
    """Test normalized entropy calculation."""

    def test_entropy_uniform_distribution(self):
        """Uniformly distributed bits should have high entropy."""
        sdr = torch.zeros(2048)
        n_active = 40
        sdr[:n_active] = 1
        entropy = SDRMetrics.compute_entropy(sdr, n_active)
        # Uniform distribution over n_active bits should have entropy close to 1
        assert entropy > 0.99

    def test_entropy_empty_sdr(self):
        """Empty SDR should have zero entropy."""
        sdr = torch.zeros(2048)
        entropy = SDRMetrics.compute_entropy(sdr, n_active=40)
        # With no active bits, entropy should be 0 or very close
        assert entropy < 0.01

    def test_entropy_single_bit(self):
        """Single active bit should have zero entropy (deterministic)."""
        sdr = torch.zeros(2048)
        sdr[0] = 1
        entropy = SDRMetrics.compute_entropy(sdr, n_active=1)
        # Single bit concentrated in one position has zero entropy
        # (log(1) = 0, so normalized entropy approaches 0)
        assert entropy < 0.01

    def test_entropy_batched(self):
        """Verify batched entropy computation."""
        sdr = torch.zeros(2, 3, 2048)
        n_active = 40
        sdr[..., :n_active] = 1
        entropy = SDRMetrics.compute_entropy(sdr, n_active)
        assert entropy.shape == (2, 3)
        # All should have uniform distribution
        assert (entropy > 0.99).all()

    def test_entropy_normalization(self):
        """Entropy should be non-negative and finite."""
        sdr = torch.zeros(5, 2048)
        sdr[0, :40] = 1    # Uniform - 40 bits
        sdr[1, :1] = 1     # Single bit
        sdr[2, :10] = 1    # Few bits
        sdr[3, :100] = 1   # Many bits (more than n_active)
        sdr[4, :] = 1      # All bits
        entropy = SDRMetrics.compute_entropy(sdr, n_active=40)
        # All entropies should be non-negative and finite
        assert (entropy >= 0).all()
        assert torch.isfinite(entropy).all()
        # SDR with exactly n_active bits uniformly distributed should have entropy ~1
        assert entropy[0] > 0.99


class TestComputeOverlapStatistics:
    """Test pairwise overlap statistics."""

    def test_overlap_identical_sdrs(self):
        """Identical SDRs should have maximum overlap."""
        sdr = torch.zeros(2, 1, 2048)
        n_active = 40
        sdr[..., :n_active] = 1
        stats = SDRMetrics.compute_overlap_statistics(sdr, n_active)
        # Only one unique pair: identical SDRs
        assert stats['overlap_mean'] == n_active
        assert stats['overlap_max'] == n_active
        assert stats['overlap_min'] == n_active
        assert abs(stats['overlap_mean_norm'] - 1.0) < 1e-6

    def test_overlap_disjoint_sdrs(self):
        """Disjoint SDRs should have zero overlap."""
        sdr = torch.zeros(2, 1, 2048)
        sdr[0, 0, :40] = 1   # First SDR
        sdr[1, 0, 100:140] = 1  # Second SDR (disjoint)
        stats = SDRMetrics.compute_overlap_statistics(sdr, n_active=40)
        assert stats['overlap_mean'] == 0.0
        assert stats['overlap_min'] == 0.0
        assert stats['overlap_max'] == 0.0
        assert stats['overlap_mean_norm'] == 0.0

    def test_overlap_partial(self):
        """Partially overlapping SDRs."""
        sdr = torch.zeros(2, 1, 2048)
        n_active = 40
        sdr[0, 0, :n_active] = 1      # [0:40]
        sdr[1, 0, 20:60] = 1           # [20:60]
        # Overlap is 20 bits (20:40)
        stats = SDRMetrics.compute_overlap_statistics(sdr, n_active)
        assert stats['overlap_mean'] == 20.0
        assert stats['overlap_min'] == 20.0
        assert stats['overlap_max'] == 20.0
        assert abs(stats['overlap_mean_norm'] - 0.5) < 1e-6

    def test_overlap_single_sdr(self):
        """Single SDR should return zeros (no pairs)."""
        sdr = torch.zeros(1, 1, 2048)
        sdr[0, 0, :40] = 1
        stats = SDRMetrics.compute_overlap_statistics(sdr, n_active=40)
        assert stats['overlap_mean'] == 0.0
        assert stats['overlap_std'] == 0.0
        assert stats['overlap_min'] == 0.0
        assert stats['overlap_max'] == 0.0

    def test_overlap_multiple_pairs(self):
        """Test with multiple SDR pairs."""
        sdr = torch.zeros(3, 1, 2048)
        n_active = 40
        # Three SDRs with varying overlaps
        sdr[0, 0, :n_active] = 1           # [0:40]
        sdr[1, 0, 10:50] = 1               # [10:50] -> overlap 30 with first
        sdr[2, 0, 30:70] = 1               # [30:70] -> overlap 10 with first, 20 with second
        stats = SDRMetrics.compute_overlap_statistics(sdr, n_active)
        # Pairs: (0,1)=30, (0,2)=10, (1,2)=20 -> mean=20
        assert abs(stats['overlap_mean'] - 20.0) < 1e-6
        assert stats['overlap_min'] == 10.0
        assert stats['overlap_max'] == 30.0
        assert stats['overlap_std'] > 0  # Should have non-zero std

    def test_overlap_batch_and_sequence(self):
        """Test with batch and sequence dimensions."""
        sdr = torch.zeros(2, 3, 2048)  # batch=2, seq=3
        n_active = 40
        # Fill all with same pattern
        sdr[..., :n_active] = 1
        stats = SDRMetrics.compute_overlap_statistics(sdr, n_active)
        # All 6 SDRs are identical, so all pairwise overlaps = 40
        assert stats['overlap_mean'] == n_active
        assert stats['overlap_std'] == 0.0


class TestComputeAll:
    """Test aggregated metrics computation."""

    def test_compute_all_structure(self):
        """Verify compute_all returns all expected metrics."""
        sdr = torch.zeros(2, 3, 2048)
        n_active = 40
        sdr[..., :n_active] = 1
        metrics = SDRMetrics.compute_all(sdr, n_active)

        # Check all expected keys are present
        expected_keys = {
            'sparsity', 'sparsity_std',
            'entropy', 'entropy_std',
            'overlap_mean', 'overlap_std',
            'overlap_min', 'overlap_max',
            'overlap_mean_norm'
        }
        assert set(metrics.keys()) == expected_keys

        # Check all values are floats
        for key, value in metrics.items():
            assert isinstance(value, float)

    def test_compute_all_uniform_sdrs(self):
        """Test compute_all with uniform SDRs."""
        sdr = torch.zeros(2, 2, 2048)
        n_active = 40
        sdr[..., :n_active] = 1
        metrics = SDRMetrics.compute_all(sdr, n_active)

        # All SDRs are identical
        assert abs(metrics['sparsity'] - n_active/2048) < 1e-6
        assert metrics['sparsity_std'] == 0.0
        assert metrics['entropy'] > 0.99  # Uniform distribution
        assert metrics['entropy_std'] == 0.0
        assert metrics['overlap_mean'] == n_active
        assert metrics['overlap_mean_norm'] == 1.0

    def test_compute_all_varying_sdrs(self):
        """Test compute_all with varying SDRs."""
        sdr = torch.zeros(3, 1, 2048)
        n_active = 40
        # Three different SDRs
        sdr[0, 0, :20] = 1      # Low sparsity
        sdr[1, 0, :40] = 1      # Target sparsity
        sdr[2, 0, :80] = 1      # High sparsity
        metrics = SDRMetrics.compute_all(sdr, n_active)

        # Should have varying sparsity
        assert metrics['sparsity_std'] > 0
        # Entropy should vary
        assert metrics['entropy_std'] >= 0
        # Overlaps should be present
        assert metrics['overlap_mean'] > 0

    def test_compute_all_empty_batch(self):
        """Test compute_all with minimal valid input."""
        sdr = torch.zeros(1, 1, 2048)
        n_active = 40
        sdr[0, 0, :n_active] = 1
        metrics = SDRMetrics.compute_all(sdr, n_active)

        # Single SDR: no pairs for overlap
        assert metrics['overlap_mean'] == 0.0
        assert metrics['overlap_std'] == 0.0
        # But should have valid sparsity and entropy
        assert abs(metrics['sparsity'] - n_active/2048) < 1e-6
        assert metrics['entropy'] > 0.99

    def test_compute_all_realistic_batch(self):
        """Test with realistic batch dimensions."""
        batch_size = 4
        seq_len = 8
        sdr_dim = 2048
        n_active = 40

        # Create realistic SDRs with some variation
        sdr = torch.zeros(batch_size, seq_len, sdr_dim)
        for i in range(batch_size):
            for j in range(seq_len):
                # Each SDR has n_active bits at different positions
                start = (i * seq_len + j) * 10 % (sdr_dim - n_active)
                sdr[i, j, start:start + n_active] = 1

        metrics = SDRMetrics.compute_all(sdr, n_active)

        # All should have target sparsity
        assert abs(metrics['sparsity'] - n_active/sdr_dim) < 1e-4
        # All have uniform distribution
        assert metrics['entropy'] > 0.95
        # Some overlap expected due to limited space
        assert 0 <= metrics['overlap_mean_norm'] <= 1.0
        # Should have reasonable statistics
        assert metrics['overlap_std'] >= 0
        assert metrics['overlap_min'] <= metrics['overlap_mean'] <= metrics['overlap_max']


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_dimension_tensor(self):
        """Verify 1D tensor handling."""
        sdr = torch.zeros(2048)
        sdr[:40] = 1
        sparsity = SDRMetrics.compute_sparsity(sdr)
        assert sparsity.shape == torch.Size([])
        assert abs(sparsity - 40/2048) < 1e-6

    def test_high_dimensional_tensor(self):
        """Verify high-dimensional tensor handling."""
        sdr = torch.zeros(2, 3, 4, 5, 2048)
        n_active = 40
        sdr[..., :n_active] = 1
        sparsity = SDRMetrics.compute_sparsity(sdr)
        assert sparsity.shape == (2, 3, 4, 5)
        assert torch.allclose(sparsity, torch.full((2, 3, 4, 5), n_active/2048))

    def test_different_sdr_dimensions(self):
        """Verify works with different SDR dimensions."""
        for sdr_dim in [512, 1024, 2048, 4096]:
            n_active = int(sdr_dim * 0.02)  # 2% sparsity
            sdr = torch.zeros(sdr_dim)
            sdr[:n_active] = 1
            sparsity = SDRMetrics.compute_sparsity(sdr)
            assert abs(sparsity - 0.02) < 1e-3

    def test_numerical_stability_entropy(self):
        """Verify entropy computation is numerically stable."""
        sdr = torch.zeros(2048)
        sdr[:40] = 1
        # Should not produce NaN or inf
        entropy = SDRMetrics.compute_entropy(sdr, n_active=40)
        assert torch.isfinite(torch.tensor(entropy))
        assert 0 <= entropy <= 1.0

    def test_overlap_with_empty_sdr(self):
        """Test overlap statistics with empty SDR."""
        sdr = torch.zeros(2, 1, 2048)
        # Leave all zeros
        stats = SDRMetrics.compute_overlap_statistics(sdr, n_active=40)
        assert stats['overlap_mean'] == 0.0
        # std of constant values is NaN in PyTorch (mathematically undefined with ddof=1)
        import math
        assert stats['overlap_std'] == 0.0 or math.isnan(stats['overlap_std'])
