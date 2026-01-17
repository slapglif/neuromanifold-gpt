"""Benchmark configurations for NeuroManifoldGPT.

This package contains baseline configurations for comparing NeuroManifold
attention against standard transformer architectures.
"""

from neuromanifold_gpt.config.benchmarks.standard_attention import (
    config as standard_attention_config,
)

__all__ = ["standard_attention_config"]
