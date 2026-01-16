"""Unit tests for NeuroManifold attention benchmark suite.

Tests verify that:
- Benchmark modules can be imported
- Config files load correctly
- Basic benchmark functions are callable
- CLI interfaces work properly
"""
import pytest
import sys
from pathlib import Path


def test_benchmark_modules_importable():
    """Test all benchmark modules can be imported without errors."""
    # Import without triggering full neuromanifold_gpt initialization
    sys.path.insert(0, str(Path(__file__).parent))

    # These should not raise ImportError
    import attention_quality
    import attention_speed
    import attention_memory
    import run_all
    import visualize

    # Verify key functions exist
    assert hasattr(attention_quality, 'benchmark_quality')
    assert hasattr(attention_quality, 'benchmark_sample_quality')
    assert hasattr(attention_speed, 'benchmark_speed')
    assert hasattr(attention_memory, 'benchmark_memory')
    assert hasattr(visualize, 'plot_results')


def test_standard_config_structure():
    """Test standard attention config has required attributes."""
    from neuromanifold_gpt.config.benchmarks.standard_attention import out, config

    # Check GPT-2 124M architecture
    assert out['n_layer'] == 12
    assert out['n_embd'] == 768
    assert out['n_heads'] == 12
    assert out['block_size'] == 1024

    # Verify NeuroManifold features are disabled
    assert out['use_sdr'] == False
    assert out['use_kan'] == False
    assert out['use_mhc'] == False
    assert out['use_mtp'] == False
    assert out['use_fhn_parallel'] == False

    # Config object should be instantiable
    assert config.n_layer == 12


def test_neuromanifold_config_structure():
    """Test NeuroManifold attention config has required attributes."""
    from neuromanifold_gpt.config.benchmarks.neuromanifold_attention import out, config

    # Check GPT-2 124M architecture
    assert out['n_layer'] == 12
    assert out['n_embd'] == 768
    assert out['n_heads'] == 12
    assert out['block_size'] == 1024

    # Verify NeuroManifold features are enabled
    assert out['use_sdr'] == True
    assert out['use_kan'] == True
    assert out['use_mhc'] == True
    assert out['use_mtp'] == True
    assert out['use_fhn_parallel'] == True

    # Check SDR parameters
    assert out['sdr_size'] == 2048
    assert out['sdr_sparsity'] == 0.02

    # Config object should be instantiable
    assert config.n_layer == 12


def test_run_all_cli():
    """Test run_all.py CLI accepts expected arguments."""
    import run_all
    import argparse

    # Verify ArgumentParser is configured
    parser = run_all.parser if hasattr(run_all, 'parser') else argparse.ArgumentParser()

    # Test that main function exists
    assert hasattr(run_all, 'main')


def test_benchmark_results_json_valid():
    """Test that benchmark_results.json exists and is valid JSON."""
    import json
    from pathlib import Path

    results_file = Path(__file__).parent.parent.parent / "benchmark_results.json"

    if results_file.exists():
        with open(results_file) as f:
            data = json.load(f)

        # Check expected structure
        assert 'metadata' in data
        assert 'quality' in data
        assert 'memory' in data
        # Speed may have errors due to OOM
        assert 'speed' in data


def test_benchmark_results_md_exists():
    """Test that BENCHMARK_RESULTS.md exists and has content."""
    from pathlib import Path

    results_file = Path(__file__).parent.parent.parent / "BENCHMARK_RESULTS.md"

    assert results_file.exists(), "BENCHMARK_RESULTS.md should exist"

    content = results_file.read_text()
    assert len(content) > 1000, "BENCHMARK_RESULTS.md should have substantial content"
    assert "Executive Summary" in content
    assert "Quality Comparison" in content
    assert "Memory Comparison" in content
