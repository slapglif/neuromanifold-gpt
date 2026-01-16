
import torch
from neuromanifold_gpt.evaluation.component_metrics import ComponentMetricsAggregator
from neuromanifold_gpt.config.base import NeuroManifoldConfigNano

def test_full_component_metrics_integration():
    """Smoke test for ComponentMetricsAggregator with all components."""
    print("Initializing ComponentMetricsAggregator...")

    # Configure for full component metrics
    config = NeuroManifoldConfigNano()
    config.use_sdr = True
    config.sdr_size = 2048
    config.sdr_n_active = 40
    config.use_mtp = True

    # Initialize aggregator
    aggregator = ComponentMetricsAggregator()

    # Create mock model info dict with all components
    B, T = 2, 64
    vocab_size = config.vocab_size

    print(f"Creating mock info dict with batch={B}, seq_len={T}...")

    # SDR component: (batch, seq_len, sdr_size)
    sdr = torch.zeros(B, T, config.sdr_size)
    for i in range(B):
        for j in range(T):
            # Set n_active bits randomly
            indices = torch.randperm(config.sdr_size)[:config.sdr_n_active]
            sdr[i, j, indices] = 1.0

    # FHN component: block_infos with FHN state and pulse widths
    n_blocks = config.n_layer
    block_infos = []
    for block_idx in range(n_blocks):
        block_info = {
            'fhn_state': torch.randn(B, T, config.n_embd) * 0.5,  # Bounded FHN state
            'pulse_widths': torch.rand(B, T, config.n_heads) * config.pulse_width_base,
        }
        block_infos.append(block_info)

    # Memory component: memory_stats dict
    memory_stats = {
        'memory_size': 100,
        'capacity': 1000,
        'sdr_size': config.sdr_size,
        'content_dim': config.n_embd,
    }

    # MTP component: logits and targets
    logits = torch.randn(B, T, vocab_size)
    targets = torch.randint(0, vocab_size, (B, T))

    # Assemble full info dict
    info = {
        'sdr': sdr,
        'block_infos': block_infos,
        'memory_stats': memory_stats,
    }

    print("Computing component metrics...")
    metrics = aggregator.compute(info, config, logits=logits, targets=targets)

    print(f"Component metrics computed: {list(metrics.keys())}")

    # Verify all expected components are present
    expected_components = ['sdr', 'fhn', 'mtp', 'memory']
    for component in expected_components:
        assert component in metrics, f"Missing component: {component}"
        print(f"  {component}: {list(metrics[component].keys())}")

    # Check SDR metrics
    print("\nVerifying SDR metrics...")
    sdr_metrics = metrics['sdr']
    assert 'sparsity' in sdr_metrics
    assert 'entropy' in sdr_metrics
    assert 'overlap_mean' in sdr_metrics
    assert not any(v != v for v in sdr_metrics.values()), "NaNs in SDR metrics"
    print(f"  Sparsity: {sdr_metrics['sparsity']:.4f}")
    print(f"  Entropy: {sdr_metrics['entropy']:.4f}")
    print(f"  Overlap mean: {sdr_metrics['overlap_mean']:.4f}")

    # Check FHN metrics
    print("\nVerifying FHN metrics...")
    fhn_metrics = metrics['fhn']
    assert 'fhn_state_mean' in fhn_metrics
    assert 'pulse_width_mean' in fhn_metrics
    assert 'fhn_stability_bounded' in fhn_metrics
    assert not any(v != v for v in fhn_metrics.values()), "NaNs in FHN metrics"
    print(f"  FHN state mean: {fhn_metrics['fhn_state_mean']:.4f}")
    print(f"  Pulse width mean: {fhn_metrics['pulse_width_mean']:.4f}")
    print(f"  FHN stability bounded: {fhn_metrics['fhn_stability_bounded']:.4f}")

    # Check MTP metrics
    print("\nVerifying MTP metrics...")
    mtp_metrics = metrics['mtp']
    assert 'accuracy' in mtp_metrics
    assert 'top_1_accuracy' in mtp_metrics
    assert 'confidence_mean' in mtp_metrics
    assert not any(v != v for v in mtp_metrics.values()), "NaNs in MTP metrics"
    print(f"  Accuracy: {mtp_metrics['accuracy']:.4f}")
    print(f"  Top-1 accuracy: {mtp_metrics['top_1_accuracy']:.4f}")
    print(f"  Confidence mean: {mtp_metrics['confidence_mean']:.4f}")

    # Check Memory metrics
    print("\nVerifying Memory metrics...")
    memory_metrics = metrics['memory']
    assert 'memory_size' in memory_metrics
    assert 'memory_utilization' in memory_metrics
    assert not any(v != v for v in memory_metrics.values()), "NaNs in Memory metrics"
    print(f"  Memory size: {memory_metrics['memory_size']:.0f}")
    print(f"  Memory utilization: {memory_metrics['memory_utilization']:.4f}")

    print("\nIntegration test PASSED!")


def test_component_metrics_partial_info():
    """Test ComponentMetricsAggregator with partial info dict (some components missing)."""
    print("Testing partial info dict (SDR and FHN only)...")

    config = NeuroManifoldConfigNano()
    config.use_sdr = True
    config.sdr_size = 2048
    config.sdr_n_active = 40
    config.use_mtp = False  # MTP disabled

    aggregator = ComponentMetricsAggregator()

    B, T = 2, 32

    # Only SDR and FHN components
    sdr = torch.zeros(B, T, config.sdr_size)
    for i in range(B):
        for j in range(T):
            indices = torch.randperm(config.sdr_size)[:config.sdr_n_active]
            sdr[i, j, indices] = 1.0

    block_infos = [
        {
            'fhn_state': torch.randn(B, T, config.n_embd) * 0.5,
        }
        for _ in range(config.n_layer)
    ]

    info = {
        'sdr': sdr,
        'block_infos': block_infos,
    }

    print("Computing metrics with partial info...")
    metrics = aggregator.compute(info, config)

    # Should have SDR and FHN, but not MTP or Memory
    assert 'sdr' in metrics
    assert 'fhn' in metrics
    assert 'mtp' not in metrics  # MTP disabled in config
    assert 'memory' not in metrics  # No memory_stats provided

    print(f"Components present: {list(metrics.keys())}")
    print("Partial info test PASSED!")


def test_component_metrics_no_sdr():
    """Test ComponentMetricsAggregator with SDR disabled."""
    print("Testing with SDR disabled...")

    config = NeuroManifoldConfigNano()
    config.use_sdr = False  # Dense embeddings

    aggregator = ComponentMetricsAggregator()

    B, T = 2, 32

    # Only FHN component
    block_infos = [
        {
            'fhn_state': torch.randn(B, T, config.n_embd) * 0.5,
            'pulse_widths': torch.rand(B, T, config.n_heads) * config.pulse_width_base,
        }
        for _ in range(config.n_layer)
    ]

    info = {
        'block_infos': block_infos,
    }

    print("Computing metrics without SDR...")
    metrics = aggregator.compute(info, config)

    # Should not have SDR metrics
    assert 'sdr' not in metrics
    assert 'fhn' in metrics

    print(f"Components present: {list(metrics.keys())}")
    print("No SDR test PASSED!")


def test_component_metrics_empty_info():
    """Test ComponentMetricsAggregator with empty info dict."""
    print("Testing with empty info dict...")

    config = NeuroManifoldConfigNano()
    aggregator = ComponentMetricsAggregator()

    info = {}

    print("Computing metrics from empty info...")
    metrics = aggregator.compute(info, config)

    # Should return empty metrics dict
    assert len(metrics) == 0

    print("Empty info test PASSED!")


def test_component_metrics_aggregation_across_blocks():
    """Test that FHN metrics are properly aggregated across multiple blocks."""
    print("Testing FHN metrics aggregation across blocks...")

    config = NeuroManifoldConfigNano()
    aggregator = ComponentMetricsAggregator()

    B, T = 2, 32
    n_blocks = 4

    # Create distinct FHN states for each block
    block_infos = []
    for block_idx in range(n_blocks):
        # Each block has different FHN characteristics
        block_info = {
            'fhn_state': torch.randn(B, T, config.n_embd) * (0.1 + 0.1 * block_idx),
            'pulse_widths': torch.ones(B, T, config.n_heads) * (2.0 + block_idx),
        }
        block_infos.append(block_info)

    info = {
        'block_infos': block_infos,
    }

    print(f"Computing aggregated metrics from {n_blocks} blocks...")
    metrics = aggregator.compute(info, config)

    assert 'fhn' in metrics
    fhn_metrics = metrics['fhn']

    # Verify aggregated statistics
    assert 'fhn_state_mean' in fhn_metrics
    assert 'fhn_state_std' in fhn_metrics
    assert 'pulse_width_mean' in fhn_metrics
    assert 'pulse_width_std' in fhn_metrics

    print(f"  Aggregated FHN state mean: {fhn_metrics['fhn_state_mean']:.4f}")
    print(f"  Aggregated FHN state std: {fhn_metrics['fhn_state_std']:.4f}")
    print(f"  Aggregated pulse width mean: {fhn_metrics['pulse_width_mean']:.4f}")
    print(f"  Aggregated pulse width std: {fhn_metrics['pulse_width_std']:.4f}")

    # Pulse width mean should be around the middle of the range (2, 3, 4, 5 -> mean ~3.5)
    assert 2.5 < fhn_metrics['pulse_width_mean'] < 4.5

    print("Aggregation test PASSED!")


if __name__ == "__main__":
    test_full_component_metrics_integration()
    print("\n" + "="*60 + "\n")
    test_component_metrics_partial_info()
    print("\n" + "="*60 + "\n")
    test_component_metrics_no_sdr()
    print("\n" + "="*60 + "\n")
    test_component_metrics_empty_info()
    print("\n" + "="*60 + "\n")
    test_component_metrics_aggregation_across_blocks()
