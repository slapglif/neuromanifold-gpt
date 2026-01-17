"""Tests for FHN metrics evaluation.

These tests verify the correctness of FHN wave stability metrics:
- compute_state_statistics: membrane potential statistics
- compute_pulse_width_statistics: wave propagation width analysis
- compute_wave_stability: stability indicators
- compute_all: aggregated metrics computation from info dict
"""
import pytest
import torch

from neuromanifold_gpt.evaluation.fhn_metrics import FHNMetrics


class TestComputeStateStatistics:
    """Test FHN membrane potential state statistics."""

    def test_state_statistics_keys(self):
        """Verify all expected keys are present."""
        fhn_state = torch.randn(2, 10, 8)
        stats = FHNMetrics.compute_state_statistics(fhn_state)

        expected_keys = {
            "fhn_state_mean",
            "fhn_state_std",
            "fhn_state_min",
            "fhn_state_max",
            "fhn_state_range",
        }
        assert set(stats.keys()) == expected_keys

        # All values should be floats
        for key, value in stats.items():
            assert isinstance(value, float)

    def test_state_statistics_constant_state(self):
        """Constant state should have zero std and zero range."""
        fhn_state = torch.full((2, 10, 8), 0.5)
        stats = FHNMetrics.compute_state_statistics(fhn_state)

        assert stats["fhn_state_mean"] == 0.5
        assert stats["fhn_state_std"] == 0.0
        assert stats["fhn_state_min"] == 0.5
        assert stats["fhn_state_max"] == 0.5
        assert stats["fhn_state_range"] == 0.0

    def test_state_statistics_bounded_values(self):
        """State within [-3, 3] should be correctly characterized."""
        fhn_state = torch.tensor([-3.0, -1.5, 0.0, 1.5, 3.0])
        stats = FHNMetrics.compute_state_statistics(fhn_state)

        assert stats["fhn_state_mean"] == 0.0
        assert stats["fhn_state_min"] == -3.0
        assert stats["fhn_state_max"] == 3.0
        assert stats["fhn_state_range"] == 6.0

    def test_state_statistics_positive_std(self):
        """Non-constant state should have positive std."""
        fhn_state = torch.randn(100)
        stats = FHNMetrics.compute_state_statistics(fhn_state)

        assert stats["fhn_state_std"] > 0

    def test_state_statistics_range_consistency(self):
        """Range should equal max - min."""
        fhn_state = torch.randn(2, 10, 8)
        stats = FHNMetrics.compute_state_statistics(fhn_state)

        expected_range = stats["fhn_state_max"] - stats["fhn_state_min"]
        assert abs(stats["fhn_state_range"] - expected_range) < 1e-6


class TestComputePulseWidthStatistics:
    """Test FHN pulse width statistics."""

    def test_pulse_width_statistics_keys(self):
        """Verify all expected keys are present."""
        pulse_widths = torch.randn(2, 10, 8).abs() + 1.0  # Positive
        stats = FHNMetrics.compute_pulse_width_statistics(pulse_widths)

        expected_keys = {
            "pulse_width_mean",
            "pulse_width_std",
            "pulse_width_min",
            "pulse_width_max",
        }
        assert set(stats.keys()) == expected_keys

        # All values should be floats
        for key, value in stats.items():
            assert isinstance(value, float)

    def test_pulse_width_statistics_constant_widths(self):
        """Constant pulse widths should have zero std."""
        pulse_widths = torch.full((2, 10, 8), 4.0)
        stats = FHNMetrics.compute_pulse_width_statistics(pulse_widths)

        assert stats["pulse_width_mean"] == 4.0
        assert stats["pulse_width_std"] == 0.0
        assert stats["pulse_width_min"] == 4.0
        assert stats["pulse_width_max"] == 4.0

    def test_pulse_width_statistics_positive_values(self):
        """Pulse widths should remain positive (as per FHN implementation)."""
        pulse_widths = torch.tensor([2.0, 3.5, 4.0, 5.5, 6.0])
        stats = FHNMetrics.compute_pulse_width_statistics(pulse_widths)

        assert stats["pulse_width_mean"] > 0
        assert stats["pulse_width_min"] > 0
        assert stats["pulse_width_max"] > 0

    def test_pulse_width_statistics_varying_widths(self):
        """Varying pulse widths should have positive std."""
        pulse_widths = torch.tensor([2.0, 4.0, 6.0, 8.0])
        stats = FHNMetrics.compute_pulse_width_statistics(pulse_widths)

        assert stats["pulse_width_std"] > 0
        assert stats["pulse_width_min"] == 2.0
        assert stats["pulse_width_max"] == 8.0

    def test_pulse_width_statistics_base_value(self):
        """Pulse widths around base value 4.0 should be detected."""
        # Typical base value with small modulation
        pulse_widths = torch.randn(100) * 0.5 + 4.0
        stats = FHNMetrics.compute_pulse_width_statistics(pulse_widths)

        # Mean should be close to 4.0
        assert 3.0 < stats["pulse_width_mean"] < 5.0


class TestComputeWaveStability:
    """Test FHN wave stability indicators."""

    def test_wave_stability_keys(self):
        """Verify all expected keys are present."""
        fhn_state = torch.randn(2, 10, 8)
        stats = FHNMetrics.compute_wave_stability(fhn_state)

        expected_keys = {"fhn_stability_bounded", "fhn_state_abs_mean"}
        assert set(stats.keys()) == expected_keys

        # All values should be floats
        for key, value in stats.items():
            assert isinstance(value, float)

    def test_wave_stability_bounded_positive(self):
        """States within [-3, 3] should be marked as stable."""
        fhn_state = torch.randn(100).clamp(-3.0, 3.0)
        stats = FHNMetrics.compute_wave_stability(fhn_state)

        assert stats["fhn_stability_bounded"] == 1.0

    def test_wave_stability_bounded_negative(self):
        """States outside [-3, 3] should be marked as unstable."""
        fhn_state = torch.tensor([-4.0, -2.0, 0.0, 2.0, 4.0])
        stats = FHNMetrics.compute_wave_stability(fhn_state)

        # Contains values outside [-3, 3]
        assert stats["fhn_stability_bounded"] == 0.0

    def test_wave_stability_abs_mean_positive(self):
        """Absolute mean should always be non-negative."""
        fhn_state = torch.randn(100)
        stats = FHNMetrics.compute_wave_stability(fhn_state)

        assert stats["fhn_state_abs_mean"] >= 0

    def test_wave_stability_abs_mean_zero_state(self):
        """Zero state should have zero absolute mean."""
        fhn_state = torch.zeros(100)
        stats = FHNMetrics.compute_wave_stability(fhn_state)

        assert stats["fhn_state_abs_mean"] == 0.0

    def test_wave_stability_abs_mean_symmetric(self):
        """Symmetric states should have same abs mean as positive."""
        fhn_state_pos = torch.tensor([1.0, 2.0, 3.0])
        fhn_state_neg = torch.tensor([-1.0, -2.0, -3.0])

        stats_pos = FHNMetrics.compute_wave_stability(fhn_state_pos)
        stats_neg = FHNMetrics.compute_wave_stability(fhn_state_neg)

        assert stats_pos["fhn_state_abs_mean"] == stats_neg["fhn_state_abs_mean"]

    def test_wave_stability_boundary_exact(self):
        """Exact boundary values [-3, 3] should be stable."""
        fhn_state = torch.tensor([-3.0, 0.0, 3.0])
        stats = FHNMetrics.compute_wave_stability(fhn_state)

        assert stats["fhn_stability_bounded"] == 1.0


class TestComputeAll:
    """Test aggregated metrics computation from info dict."""

    def test_compute_all_complete_info(self):
        """Verify compute_all with complete info dict."""
        info = {
            "fhn_state": torch.randn(2, 10, 8),
            "pulse_widths": torch.randn(2, 10, 8).abs() + 2.0,
        }
        metrics = FHNMetrics.compute_all(info)

        # Should contain all metrics from both state and pulse width
        expected_keys = {
            "fhn_state_mean",
            "fhn_state_std",
            "fhn_state_min",
            "fhn_state_max",
            "fhn_state_range",
            "pulse_width_mean",
            "pulse_width_std",
            "pulse_width_min",
            "pulse_width_max",
            "fhn_stability_bounded",
            "fhn_state_abs_mean",
        }
        assert set(metrics.keys()) == expected_keys

    def test_compute_all_state_only(self):
        """Verify compute_all with only fhn_state."""
        info = {"fhn_state": torch.randn(2, 10, 8)}
        metrics = FHNMetrics.compute_all(info)

        # Should have state metrics but not pulse width metrics
        assert "fhn_state_mean" in metrics
        assert "fhn_stability_bounded" in metrics
        assert "pulse_width_mean" not in metrics

    def test_compute_all_pulse_widths_only(self):
        """Verify compute_all with only pulse_widths."""
        info = {"pulse_widths": torch.randn(2, 10, 8).abs() + 2.0}
        metrics = FHNMetrics.compute_all(info)

        # Should have pulse width metrics but not state metrics
        assert "pulse_width_mean" in metrics
        assert "fhn_state_mean" not in metrics

    def test_compute_all_empty_info(self):
        """Verify compute_all with empty info dict."""
        info = {}
        metrics = FHNMetrics.compute_all(info)

        # Should return empty dict
        assert metrics == {}

    def test_compute_all_scalar_tensors(self):
        """Verify compute_all handles scalar tensors (0D)."""
        info = {"fhn_state": torch.tensor(0.5), "pulse_widths": torch.tensor(4.5)}
        metrics = FHNMetrics.compute_all(info)

        # Should successfully compute metrics from scalars
        assert "fhn_state_mean" in metrics
        assert "pulse_width_mean" in metrics
        assert metrics["fhn_state_mean"] == 0.5
        assert metrics["pulse_width_mean"] == 4.5

    def test_compute_all_realistic_values(self):
        """Test with realistic FHN dynamics values."""
        # Simulate realistic FHN state and pulse widths
        batch_size = 2
        seq_len = 10
        n_heads = 8

        # State typically bounded in [-3, 3]
        fhn_state = torch.randn(batch_size, seq_len, n_heads).clamp(-2.5, 2.5)
        # Pulse widths around base value 4.0
        pulse_widths = torch.randn(batch_size, seq_len, n_heads) * 0.5 + 4.0

        info = {"fhn_state": fhn_state, "pulse_widths": pulse_widths}
        metrics = FHNMetrics.compute_all(info)

        # State should be bounded
        assert metrics["fhn_stability_bounded"] == 1.0
        # State mean should be near zero (random)
        assert -1.0 < metrics["fhn_state_mean"] < 1.0
        # Pulse width mean should be near 4.0
        assert 3.0 < metrics["pulse_width_mean"] < 5.0
        # All metrics should be finite
        for key, value in metrics.items():
            assert torch.isfinite(torch.tensor(value))


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_value_tensors(self):
        """Verify handling of single-value tensors."""
        import math

        fhn_state = torch.tensor([1.0])
        stats = FHNMetrics.compute_state_statistics(fhn_state)

        assert stats["fhn_state_mean"] == 1.0
        # std of single value is NaN in PyTorch (mathematically undefined with ddof=1)
        assert stats["fhn_state_std"] == 0.0 or math.isnan(stats["fhn_state_std"])
        assert stats["fhn_state_range"] == 0.0

    def test_large_tensors(self):
        """Verify handling of large tensors."""
        fhn_state = torch.randn(100, 200, 16)
        pulse_widths = torch.randn(100, 200, 16).abs() + 2.0

        info = {"fhn_state": fhn_state, "pulse_widths": pulse_widths}
        metrics = FHNMetrics.compute_all(info)

        # Should compute without errors
        assert "fhn_state_mean" in metrics
        assert "pulse_width_mean" in metrics

    def test_extreme_values(self):
        """Verify handling of extreme but valid values."""
        # Extreme state values at boundaries
        fhn_state = torch.tensor([-3.0, 3.0, -3.0, 3.0])
        stats = FHNMetrics.compute_state_statistics(fhn_state)

        assert stats["fhn_state_mean"] == 0.0
        assert stats["fhn_state_min"] == -3.0
        assert stats["fhn_state_max"] == 3.0
        assert stats["fhn_state_range"] == 6.0

    def test_numerical_stability(self):
        """Verify metrics are numerically stable."""
        # Very small variations
        fhn_state = torch.full((100,), 0.5) + torch.randn(100) * 1e-6
        stats = FHNMetrics.compute_state_statistics(fhn_state)

        # Should not produce NaN or inf
        for key, value in stats.items():
            assert torch.isfinite(torch.tensor(value))

    def test_none_values_in_info(self):
        """Verify handling of None values in info dict."""
        info = {"fhn_state": None, "pulse_widths": torch.randn(10).abs() + 2.0}
        metrics = FHNMetrics.compute_all(info)

        # Should only compute pulse width metrics
        assert "pulse_width_mean" in metrics
        assert "fhn_state_mean" not in metrics

    def test_mixed_dimensions(self):
        """Verify handling of different tensor shapes."""
        # 2D state, 3D pulse widths
        info = {
            "fhn_state": torch.randn(10, 8),
            "pulse_widths": torch.randn(2, 10, 8).abs() + 2.0,
        }
        metrics = FHNMetrics.compute_all(info)

        # Should compute metrics for both despite different shapes
        assert "fhn_state_mean" in metrics
        assert "pulse_width_mean" in metrics


class TestIntegrationWithFHNAttention:
    """Test integration patterns with FHN attention info dict."""

    def test_typical_fhn_attention_output(self):
        """Test with typical FHN attention output format."""
        # Simulate typical FHN attention info dict
        batch_size = 2
        seq_len = 20
        n_heads = 8

        info = {
            "pulse_widths": torch.randn(batch_size, seq_len, n_heads).abs() + 4.0,
            "fhn_state": torch.randn(batch_size, seq_len, n_heads).clamp(-3.0, 3.0),
        }

        metrics = FHNMetrics.compute_all(info)

        # Verify all expected metrics are present
        assert "fhn_state_mean" in metrics
        assert "pulse_width_mean" in metrics
        assert "fhn_stability_bounded" in metrics

        # Verify stability (clamped to [-3, 3])
        assert metrics["fhn_stability_bounded"] == 1.0

        # Verify pulse widths are positive
        assert metrics["pulse_width_mean"] > 0
        assert metrics["pulse_width_min"] > 0

    def test_multi_layer_aggregation(self):
        """Test metrics from multiple layers (typical use case)."""
        # Simulate info from multiple transformer blocks
        layer_infos = []
        for layer in range(4):
            info = {
                "pulse_widths": torch.randn(2, 10, 8).abs() + 4.0,
                "fhn_state": torch.randn(2, 10, 8).clamp(-3.0, 3.0),
            }
            layer_infos.append(FHNMetrics.compute_all(info))

        # Aggregate metrics across layers
        avg_pulse_width = sum(info["pulse_width_mean"] for info in layer_infos) / len(
            layer_infos
        )
        avg_stability = sum(
            info["fhn_stability_bounded"] for info in layer_infos
        ) / len(layer_infos)

        # All layers should have positive pulse widths
        assert avg_pulse_width > 0
        # All layers should be stable
        assert avg_stability == 1.0
