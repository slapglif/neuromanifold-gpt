"""Tests for MTP and memory metrics evaluation.

These tests verify the correctness of:
- MTP (Multi-Token Prediction) accuracy metrics
- Memory utilization metrics
- Integration patterns with model info dict
"""
import pytest
import torch

from neuromanifold_gpt.evaluation.memory_metrics import MemoryMetrics
from neuromanifold_gpt.evaluation.mtp_metrics import MTPMetrics


class TestMTPComputeAccuracy:
    """Test basic MTP accuracy computation."""

    def test_accuracy_perfect_predictions(self):
        """Perfect predictions should have 100% accuracy."""
        vocab_size = 100
        logits = torch.zeros(2, 10, vocab_size)
        targets = torch.randint(0, vocab_size, (2, 10))

        # Set correct predictions
        for b in range(2):
            for t in range(10):
                logits[b, t, targets[b, t]] = 10.0

        metrics = MTPMetrics.compute_accuracy(logits, targets, top_k=1)
        assert metrics["accuracy"] == 1.0
        assert metrics["num_correct"] == 20
        assert metrics["num_total"] == 20

    def test_accuracy_zero_predictions(self):
        """All wrong predictions should have 0% accuracy."""
        vocab_size = 100
        logits = torch.zeros(2, 10, vocab_size)
        targets = torch.randint(0, vocab_size, (2, 10))

        # Set wrong predictions (target + 1)
        for b in range(2):
            for t in range(10):
                wrong_idx = (targets[b, t] + 1) % vocab_size
                logits[b, t, wrong_idx] = 10.0

        metrics = MTPMetrics.compute_accuracy(logits, targets, top_k=1)
        assert metrics["accuracy"] == 0.0
        assert metrics["num_correct"] == 0
        assert metrics["num_total"] == 20

    def test_accuracy_partial_correct(self):
        """Partial correct predictions should have correct fraction."""
        vocab_size = 100
        logits = torch.zeros(2, 10, vocab_size)
        targets = torch.randint(0, vocab_size, (2, 10))

        # First 10 correct, second 10 wrong
        for t in range(10):
            logits[0, t, targets[0, t]] = 10.0
            wrong_idx = (targets[1, t] + 1) % vocab_size
            logits[1, t, wrong_idx] = 10.0

        metrics = MTPMetrics.compute_accuracy(logits, targets, top_k=1)
        assert metrics["accuracy"] == 0.5
        assert metrics["num_correct"] == 10
        assert metrics["num_total"] == 20

    def test_accuracy_top_k(self):
        """Top-k accuracy should count if target is in top k predictions."""
        vocab_size = 100
        logits = torch.zeros(2, 5, vocab_size)
        targets = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])

        # Set top-3 predictions (target at position 2)
        for b in range(2):
            for t in range(5):
                logits[b, t, (targets[b, t] + 10) % vocab_size] = 3.0  # rank 1
                logits[b, t, (targets[b, t] + 5) % vocab_size] = 2.0  # rank 2
                logits[b, t, targets[b, t]] = 1.0  # rank 3 (target)

        metrics = MTPMetrics.compute_accuracy(logits, targets, top_k=3)
        assert metrics["accuracy"] == 1.0  # All targets in top-3

        metrics = MTPMetrics.compute_accuracy(logits, targets, top_k=1)
        assert metrics["accuracy"] == 0.0  # No targets at rank 1


class TestMTPTopKAccuracies:
    """Test multiple top-k accuracy computation."""

    def test_top_k_accuracies_keys(self):
        """Should return metrics for all k values."""
        logits = torch.randn(2, 10, 100)
        targets = torch.randint(0, 100, (2, 10))

        metrics = MTPMetrics.compute_top_k_accuracies(
            logits, targets, k_values=[1, 5, 10]
        )

        assert "top_1_accuracy" in metrics
        assert "top_5_accuracy" in metrics
        assert "top_10_accuracy" in metrics
        assert len(metrics) == 3

    def test_top_k_accuracies_increasing(self):
        """Higher k should give equal or better accuracy."""
        logits = torch.randn(2, 20, 100)
        targets = torch.randint(0, 100, (2, 20))

        metrics = MTPMetrics.compute_top_k_accuracies(
            logits, targets, k_values=[1, 5, 10, 20]
        )

        # top_k accuracy should be monotonically non-decreasing
        assert metrics["top_1_accuracy"] <= metrics["top_5_accuracy"]
        assert metrics["top_5_accuracy"] <= metrics["top_10_accuracy"]
        assert metrics["top_10_accuracy"] <= metrics["top_20_accuracy"]

    def test_top_k_accuracies_exceeds_vocab(self):
        """k larger than vocab_size should be capped."""
        vocab_size = 50
        logits = torch.randn(2, 10, vocab_size)
        targets = torch.randint(0, vocab_size, (2, 10))

        # Request k=100 but vocab is only 50
        metrics = MTPMetrics.compute_top_k_accuracies(logits, targets, k_values=[100])

        # Should not crash and should return a valid accuracy
        assert "top_100_accuracy" in metrics
        assert 0.0 <= metrics["top_100_accuracy"] <= 1.0


class TestMTPConfidenceStatistics:
    """Test prediction confidence statistics."""

    def test_confidence_statistics_keys(self):
        """Should return all confidence metrics."""
        logits = torch.randn(2, 10, 100)

        metrics = MTPMetrics.compute_confidence_statistics(logits)

        expected_keys = {
            "confidence_mean",
            "confidence_std",
            "confidence_min",
            "confidence_max",
            "entropy_mean",
        }
        assert set(metrics.keys()) == expected_keys

    def test_confidence_high_certainty(self):
        """High confidence predictions should have low entropy."""
        vocab_size = 100
        logits = torch.zeros(2, 10, vocab_size)
        logits[:, :, 0] = 10.0  # Very confident about token 0

        metrics = MTPMetrics.compute_confidence_statistics(logits)

        # High confidence (close to 1.0)
        assert metrics["confidence_mean"] > 0.9
        # Low entropy (model is certain)
        assert metrics["entropy_mean"] < 1.0

    def test_confidence_uniform_distribution(self):
        """Uniform distribution should have high entropy."""
        vocab_size = 100
        logits = torch.zeros(2, 10, vocab_size)  # All equal -> uniform probs

        metrics = MTPMetrics.compute_confidence_statistics(logits)

        # With uniform distribution, max prob is 1/vocab_size
        expected_conf = 1.0 / vocab_size
        assert abs(metrics["confidence_mean"] - expected_conf) < 0.01
        # High entropy (very uncertain)
        assert metrics["entropy_mean"] > 3.0

    def test_confidence_bounds(self):
        """Confidence should be bounded in [0, 1]."""
        logits = torch.randn(5, 20, 100)

        metrics = MTPMetrics.compute_confidence_statistics(logits)

        assert 0.0 <= metrics["confidence_min"] <= 1.0
        assert 0.0 <= metrics["confidence_max"] <= 1.0
        assert 0.0 <= metrics["confidence_mean"] <= 1.0
        assert metrics["confidence_std"] >= 0.0


class TestMTPPerDepthAccuracy:
    """Test per-depth accuracy for multi-token prediction."""

    def test_per_depth_single_depth(self):
        """Single depth should work like basic accuracy."""
        logits = torch.randn(2, 10, 100)
        targets = torch.randint(0, 100, (2, 11))

        metrics = MTPMetrics.compute_per_depth_accuracy([logits], targets)

        assert "depth_1_accuracy" in metrics
        assert "mean_depth_accuracy" in metrics
        assert metrics["mean_depth_accuracy"] == metrics["depth_1_accuracy"]

    def test_per_depth_multiple_depths(self):
        """Multiple depths should return separate accuracies."""
        vocab_size = 100
        logits_d1 = torch.randn(2, 10, vocab_size)
        logits_d2 = torch.randn(2, 10, vocab_size)
        logits_d3 = torch.randn(2, 10, vocab_size)
        targets = torch.randint(0, vocab_size, (2, 13))

        mtp_logits = [logits_d1, logits_d2, logits_d3]
        metrics = MTPMetrics.compute_per_depth_accuracy(mtp_logits, targets)

        assert "depth_1_accuracy" in metrics
        assert "depth_2_accuracy" in metrics
        assert "depth_3_accuracy" in metrics
        assert "mean_depth_accuracy" in metrics

        # Mean should be average of individual depths
        expected_mean = (
            metrics["depth_1_accuracy"]
            + metrics["depth_2_accuracy"]
            + metrics["depth_3_accuracy"]
        ) / 3
        assert abs(metrics["mean_depth_accuracy"] - expected_mean) < 1e-6

    def test_per_depth_custom_labels(self):
        """Custom depth labels should be used."""
        logits_d1 = torch.randn(2, 10, 100)
        logits_d2 = torch.randn(2, 10, 100)
        targets = torch.randint(0, 100, (2, 12))

        metrics = MTPMetrics.compute_per_depth_accuracy(
            [logits_d1, logits_d2], targets, depths=[2, 4]  # Custom labels
        )

        assert "depth_2_accuracy" in metrics
        assert "depth_4_accuracy" in metrics
        assert "depth_1_accuracy" not in metrics

    def test_per_depth_target_alignment(self):
        """Targets should be correctly aligned for each depth."""
        vocab_size = 50
        seq_len = 10

        # Create logits that predict specific targets
        logits_d1 = torch.zeros(1, seq_len, vocab_size)
        logits_d2 = torch.zeros(1, seq_len, vocab_size)

        # Create targets: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        targets = torch.arange(12).unsqueeze(0)

        # Depth 1 predicts targets[0:10] (indices 0-9)
        for t in range(seq_len):
            logits_d1[0, t, targets[0, t]] = 10.0

        # Depth 2 predicts targets[1:11] (indices 1-10)
        for t in range(seq_len):
            logits_d2[0, t, targets[0, t + 1]] = 10.0

        metrics = MTPMetrics.compute_per_depth_accuracy([logits_d1, logits_d2], targets)

        # Both should be perfect with correct alignment
        assert metrics["depth_1_accuracy"] == 1.0
        assert metrics["depth_2_accuracy"] == 1.0


class TestMTPComputeAll:
    """Test aggregated MTP metrics computation."""

    def test_compute_all_keys(self):
        """Should return all metric types."""
        logits = torch.randn(2, 10, 100)
        targets = torch.randint(0, 100, (2, 10))

        metrics = MTPMetrics.compute_all(logits, targets)

        # Should have accuracy metrics
        assert "accuracy" in metrics
        assert "top_1_accuracy" in metrics
        assert "top_5_accuracy" in metrics
        assert "top_10_accuracy" in metrics

        # Should have confidence metrics
        assert "confidence_mean" in metrics
        assert "confidence_std" in metrics
        assert "entropy_mean" in metrics

    def test_compute_all_consistency(self):
        """accuracy and top_1_accuracy should match."""
        logits = torch.randn(2, 10, 100)
        targets = torch.randint(0, 100, (2, 10))

        metrics = MTPMetrics.compute_all(logits, targets)

        # Basic accuracy and top_1 should be the same
        assert metrics["accuracy"] == metrics["top_1_accuracy"]


class TestMemoryCapacityStatistics:
    """Test memory capacity and utilization statistics."""

    def test_capacity_statistics_basic(self):
        """Should extract memory_size and compute utilization."""
        memory_stats = {"memory_size": 50, "capacity": 100}

        metrics = MemoryMetrics.compute_capacity_statistics(memory_stats)

        assert metrics["memory_size"] == 50.0
        assert metrics["memory_capacity"] == 100.0
        assert metrics["memory_utilization"] == 0.5

    def test_capacity_statistics_full(self):
        """Full memory should have utilization 1.0."""
        memory_stats = {"memory_size": 100, "capacity": 100}

        metrics = MemoryMetrics.compute_capacity_statistics(memory_stats)

        assert metrics["memory_utilization"] == 1.0

    def test_capacity_statistics_empty(self):
        """Empty memory should have utilization 0.0."""
        memory_stats = {"memory_size": 0, "capacity": 100}

        metrics = MemoryMetrics.compute_capacity_statistics(memory_stats)

        assert metrics["memory_utilization"] == 0.0

    def test_capacity_statistics_tensor_inputs(self):
        """Should handle tensor inputs."""
        memory_stats = {"memory_size": torch.tensor(75), "capacity": torch.tensor(100)}

        metrics = MemoryMetrics.compute_capacity_statistics(memory_stats)

        assert metrics["memory_size"] == 75.0
        assert metrics["memory_capacity"] == 100.0
        assert metrics["memory_utilization"] == 0.75

    def test_capacity_statistics_dimensions(self):
        """Should extract SDR and content dimensions."""
        memory_stats = {
            "memory_size": 50,
            "capacity": 100,
            "sdr_size": 2048,
            "content_dim": 384,
        }

        metrics = MemoryMetrics.compute_capacity_statistics(memory_stats)

        assert metrics["sdr_size"] == 2048.0
        assert metrics["content_dim"] == 384.0


class TestMemoryLayerStatistics:
    """Test layer-wise memory statistics."""

    def test_layer_statistics_two_layers(self):
        """Should compute L1 and L2 statistics."""
        memory_stats = {"l1_count": 50, "l2_count": 30}

        metrics = MemoryMetrics.compute_layer_statistics(memory_stats)

        assert metrics["l1_count"] == 50.0
        assert metrics["l2_count"] == 30.0
        assert metrics["total_layered_memories"] == 80.0
        assert metrics["l1_fraction"] == 50.0 / 80.0
        assert metrics["l2_fraction"] == 30.0 / 80.0

    def test_layer_statistics_single_layer(self):
        """Single layer should have fraction 1.0."""
        memory_stats = {"l1_count": 100}

        metrics = MemoryMetrics.compute_layer_statistics(memory_stats)

        assert metrics["l1_count"] == 100.0
        assert metrics["total_layered_memories"] == 100.0
        assert metrics["l1_fraction"] == 1.0

    def test_layer_statistics_multiple_layers(self):
        """Should handle multiple layers (L1-L4)."""
        memory_stats = {"l1_count": 40, "l2_count": 30, "l3_count": 20, "l4_count": 10}

        metrics = MemoryMetrics.compute_layer_statistics(memory_stats)

        assert metrics["total_layered_memories"] == 100.0
        assert metrics["l1_fraction"] == 0.4
        assert metrics["l2_fraction"] == 0.3
        assert metrics["l3_fraction"] == 0.2
        assert metrics["l4_fraction"] == 0.1

    def test_layer_statistics_tensor_inputs(self):
        """Should handle tensor inputs."""
        memory_stats = {"l1_count": torch.tensor(60), "l2_count": torch.tensor(40)}

        metrics = MemoryMetrics.compute_layer_statistics(memory_stats)

        assert metrics["l1_count"] == 60.0
        assert metrics["l2_count"] == 40.0
        assert metrics["total_layered_memories"] == 100.0


class TestMemoryRetrievalStatistics:
    """Test memory retrieval statistics."""

    def test_retrieval_statistics_basic(self):
        """Should extract retrieval metrics."""
        memory_stats = {
            "num_retrievals": 100,
            "avg_similarity": 0.75,
            "retrieval_hit_rate": 0.85,
            "avg_retrieved": 3.5,
        }

        metrics = MemoryMetrics.compute_retrieval_statistics(memory_stats)

        assert metrics["num_retrievals"] == 100.0
        assert metrics["avg_similarity"] == 0.75
        assert metrics["retrieval_hit_rate"] == 0.85
        assert metrics["avg_retrieved"] == 3.5

    def test_retrieval_statistics_partial(self):
        """Should handle partial retrieval info."""
        memory_stats = {"num_retrievals": 50, "avg_similarity": 0.6}

        metrics = MemoryMetrics.compute_retrieval_statistics(memory_stats)

        assert metrics["num_retrievals"] == 50.0
        assert metrics["avg_similarity"] == 0.6
        assert "retrieval_hit_rate" not in metrics
        assert "avg_retrieved" not in metrics

    def test_retrieval_statistics_tensor_inputs(self):
        """Should handle tensor inputs."""
        memory_stats = {
            "num_retrievals": torch.tensor(200),
            "avg_similarity": torch.tensor(0.8),
        }

        metrics = MemoryMetrics.compute_retrieval_statistics(memory_stats)

        assert metrics["num_retrievals"] == 200.0
        assert abs(metrics["avg_similarity"] - 0.8) < 1e-6


class TestMemoryComputeAll:
    """Test aggregated memory metrics computation."""

    def test_compute_all_comprehensive(self):
        """Should compute all available metrics."""
        memory_stats = {
            "memory_size": 100,
            "capacity": 200,
            "l1_count": 50,
            "l2_count": 30,
            "num_retrievals": 150,
            "avg_similarity": 0.7,
        }

        metrics = MemoryMetrics.compute_all(memory_stats)

        # Capacity metrics
        assert "memory_size" in metrics
        assert "memory_capacity" in metrics
        assert "memory_utilization" in metrics
        assert "total_size" in metrics  # Alias for memory_size

        # Layer metrics
        assert "l1_count" in metrics
        assert "l2_count" in metrics
        assert "total_layered_memories" in metrics

        # Retrieval metrics
        assert "num_retrievals" in metrics
        assert "avg_similarity" in metrics

    def test_compute_all_total_size_alias(self):
        """total_size should match memory_size."""
        memory_stats = {"memory_size": 75}

        metrics = MemoryMetrics.compute_all(memory_stats)

        assert metrics["total_size"] == metrics["memory_size"]

    def test_compute_all_empty_stats(self):
        """Should handle empty or minimal stats."""
        memory_stats = {"memory_size": 0}

        metrics = MemoryMetrics.compute_all(memory_stats)

        # Should at least have memory_size and total_size
        assert "memory_size" in metrics
        assert "total_size" in metrics
        assert metrics["memory_size"] == 0.0


class TestIntegrationPatterns:
    """Test integration with model info dict patterns."""

    def test_mtp_with_model_info_pattern(self):
        """MTP metrics should work with typical model info dict."""
        # Simulate model output
        vocab_size = 50257  # GPT-2 vocab
        batch_size = 2
        seq_len = 128

        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Compute metrics as would be done in evaluation
        metrics = MTPMetrics.compute_all(logits, targets, k_values=[1, 5, 10])

        assert "accuracy" in metrics
        assert "top_5_accuracy" in metrics
        assert "confidence_mean" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_memory_with_engram_pattern(self):
        """Memory metrics should work with SDREngramMemory stats."""
        # Simulate engram memory state
        memory_stats = {
            "memory_size": 85,
            "capacity": 100,
            "sdr_size": 2048,
            "content_dim": 384,
            "l1_count": 50,
            "l2_count": 35,
        }

        metrics = MemoryMetrics.compute_all(memory_stats)

        assert metrics["memory_utilization"] == 0.85
        assert metrics["total_layered_memories"] == 85.0
        assert metrics["l1_fraction"] > 0.5  # More in L1
