"""NeuroManifold Attention Benchmarks.

Comprehensive benchmarks comparing Soliton-Spectral Attention and SDR Memory
against standard attention on quality, speed, and memory metrics.

This module provides:
- Quality benchmarks (perplexity, convergence)
- Speed benchmarks (tokens/sec, forward/backward timing)
- Memory benchmarks (peak memory, scaling analysis)
- Visualization and result analysis tools

Example:
    from neuromanifold_gpt.benchmarks import run_all

    results = run_all(output_file='benchmark_results.json')
"""

__all__ = []
