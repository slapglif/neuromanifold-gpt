
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

    # KAN component: activations, basis output, and spline weights
    kan_activations = torch.randn(B, T, config.n_embd) * 0.5  # Bounded activations
    kan_basis_output = torch.randn(B, T, config.n_embd, 8)  # 8 basis functions
    kan_spline_weights = torch.randn(config.n_embd, 8) * 0.1  # Spline weights

    # Spectral component: spectral basis, frequencies, and orthogonality loss
    n_eig = 32  # Number of spectral modes
    spectral_basis = torch.randn(B, T, n_eig) * 0.3  # Spectral coefficients
    spectral_freqs = torch.rand(B, n_eig) * 2.0  # Eigenvalue estimates
    ortho_loss = torch.tensor(0.05)  # Orthogonality loss

    # Assemble full info dict
    info = {
        'sdr': sdr,
        'block_infos': block_infos,
        'memory_stats': memory_stats,
        'kan_activations': kan_activations,
        'kan_basis_output': kan_basis_output,
        'kan_spline_weights': kan_spline_weights,
        'spectral_basis': spectral_basis,
        'spectral_freqs': spectral_freqs,
        'ortho_loss': ortho_loss,
    }

    print("Computing component metrics...")
    metrics = aggregator.compute(info, config, logits=logits, targets=targets)

    print(f"Component metrics computed: {list(metrics.keys())}")

    # Verify all expected components are present
    expected_components = ['sdr', 'fhn', 'mtp', 'memory', 'kan', 'spectral']
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

    # Check KAN metrics
    print("\nVerifying KAN metrics...")
    kan_metrics = metrics['kan']
    assert 'activation_mean' in kan_metrics
    assert 'activation_std' in kan_metrics
    assert 'grid_utilization_mean' in kan_metrics
    assert 'spline_weight_mean' in kan_metrics
    assert not any(v != v for v in kan_metrics.values()), "NaNs in KAN metrics"
    print(f"  Activation mean: {kan_metrics['activation_mean']:.4f}")
    print(f"  Grid utilization mean: {kan_metrics['grid_utilization_mean']:.4f}")
    print(f"  Spline weight mean: {kan_metrics['spline_weight_mean']:.4f}")

    # Check Spectral metrics
    print("\nVerifying Spectral metrics...")
    spectral_metrics = metrics['spectral']
    assert 'eigenvalue_mean' in spectral_metrics
    assert 'eigenvalue_std' in spectral_metrics
    assert 'basis_mean' in spectral_metrics
    assert 'ortho_loss' in spectral_metrics
    assert not any(v != v for v in spectral_metrics.values()), "NaNs in Spectral metrics"
    print(f"  Eigenvalue mean: {spectral_metrics['eigenvalue_mean']:.4f}")
    print(f"  Basis mean: {spectral_metrics['basis_mean']:.4f}")
    print(f"  Ortho loss: {spectral_metrics['ortho_loss']:.4f}")

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


def test_kan_metrics_integration():
    """Test ComponentMetricsAggregator with KAN metrics."""
    print("Testing KAN metrics integration...")

    config = NeuroManifoldConfigNano()
    aggregator = ComponentMetricsAggregator()

    B, T = 2, 32

    # Create KAN component data
    kan_activations = torch.randn(B, T, config.n_embd) * 0.5
    kan_basis_output = torch.randn(B, T, config.n_embd, 8)
    kan_spline_weights = torch.randn(config.n_embd, 8) * 0.1

    info = {
        'kan_activations': kan_activations,
        'kan_basis_output': kan_basis_output,
        'kan_spline_weights': kan_spline_weights,
    }

    print("Computing KAN metrics...")
    metrics = aggregator.compute(info, config)

    # Should have KAN metrics only
    assert 'kan' in metrics
    kan_metrics = metrics['kan']

    # Verify all expected KAN metrics are present
    expected_kan_metrics = [
        'activation_mean', 'activation_std', 'activation_min', 'activation_max',
        'grid_utilization_mean', 'grid_utilization_std',
        'spline_weight_mean', 'spline_weight_std',
    ]
    for metric in expected_kan_metrics:
        assert metric in kan_metrics, f"Missing KAN metric: {metric}"

    # Verify no NaNs
    assert not any(v != v for v in kan_metrics.values()), "NaNs in KAN metrics"

    print(f"  KAN metrics computed: {list(kan_metrics.keys())}")
    print(f"  Activation mean: {kan_metrics['activation_mean']:.4f}")
    print(f"  Grid utilization mean: {kan_metrics['grid_utilization_mean']:.4f}")
    print(f"  Spline weight mean: {kan_metrics['spline_weight_mean']:.4f}")

    print("KAN metrics integration test PASSED!")


def test_spectral_metrics_integration():
    """Test ComponentMetricsAggregator with Spectral metrics."""
    print("Testing Spectral metrics integration...")

    config = NeuroManifoldConfigNano()
    aggregator = ComponentMetricsAggregator()

    B, T = 2, 32
    n_eig = 32

    # Create Spectral component data
    spectral_basis = torch.randn(B, T, n_eig) * 0.3
    spectral_freqs = torch.rand(B, n_eig) * 2.0
    ortho_loss = torch.tensor(0.05)

    info = {
        'spectral_basis': spectral_basis,
        'spectral_freqs': spectral_freqs,
        'ortho_loss': ortho_loss,
    }

    print("Computing Spectral metrics...")
    metrics = aggregator.compute(info, config)

    # Should have Spectral metrics only
    assert 'spectral' in metrics
    spectral_metrics = metrics['spectral']

    # Verify all expected Spectral metrics are present
    expected_spectral_metrics = [
        'eigenvalue_mean', 'eigenvalue_std', 'eigenvalue_min', 'eigenvalue_max',
        'basis_mean', 'basis_std', 'basis_abs_mean',
        'ortho_loss', 'basis_norm_mean', 'basis_norm_std',
    ]
    for metric in expected_spectral_metrics:
        assert metric in spectral_metrics, f"Missing Spectral metric: {metric}"

    # Verify no NaNs
    assert not any(v != v for v in spectral_metrics.values()), "NaNs in Spectral metrics"

    print(f"  Spectral metrics computed: {list(spectral_metrics.keys())}")
    print(f"  Eigenvalue mean: {spectral_metrics['eigenvalue_mean']:.4f}")
    print(f"  Basis mean: {spectral_metrics['basis_mean']:.4f}")
    print(f"  Ortho loss: {spectral_metrics['ortho_loss']:.4f}")
    print(f"  Basis norm mean: {spectral_metrics['basis_norm_mean']:.4f}")

    print("Spectral metrics integration test PASSED!")


def test_kan_and_spectral_together():
    """Test ComponentMetricsAggregator with both KAN and Spectral metrics together."""
    print("Testing KAN and Spectral metrics together...")

    config = NeuroManifoldConfigNano()
    aggregator = ComponentMetricsAggregator()

    B, T = 2, 32
    n_eig = 32

    # Create both KAN and Spectral component data
    kan_activations = torch.randn(B, T, config.n_embd) * 0.5
    kan_basis_output = torch.randn(B, T, config.n_embd, 8)

    spectral_basis = torch.randn(B, T, n_eig) * 0.3
    spectral_freqs = torch.rand(B, n_eig) * 2.0
    ortho_loss = torch.tensor(0.05)

    info = {
        'kan_activations': kan_activations,
        'kan_basis_output': kan_basis_output,
        'spectral_basis': spectral_basis,
        'spectral_freqs': spectral_freqs,
        'ortho_loss': ortho_loss,
    }

    print("Computing both KAN and Spectral metrics...")
    metrics = aggregator.compute(info, config)

    # Should have both KAN and Spectral metrics
    assert 'kan' in metrics
    assert 'spectral' in metrics

    print(f"  Components present: {list(metrics.keys())}")
    print(f"  KAN activation mean: {metrics['kan']['activation_mean']:.4f}")
    print(f"  Spectral eigenvalue mean: {metrics['spectral']['eigenvalue_mean']:.4f}")

    print("KAN and Spectral together test PASSED!")


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
    print("\n" + "="*60 + "\n")
    test_kan_metrics_integration()
    print("\n" + "="*60 + "\n")
    test_spectral_metrics_integration()
    print("\n" + "="*60 + "\n")
    test_kan_and_spectral_together()
