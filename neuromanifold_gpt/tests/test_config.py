"""Tests for NeuroManifoldConfig dataclasses."""

import pytest


class TestNeuroManifoldConfig:
    """Test suite for the main configuration dataclass."""

    def test_config_defaults(self):
        """Test default SDR configuration values."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfig

        config = NeuroManifoldConfig()
        assert config.sdr_size == 2048
        assert config.sdr_sparsity == 0.02
        assert config.sdr_n_active == 40  # 2048 * 0.02 = 40.96, int = 40

    def test_config_n_embd_divisible_by_heads(self):
        """Test that n_embd must be divisible by n_heads."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfig
        from neuromanifold_gpt.errors import ConfigurationError

        with pytest.raises(ConfigurationError):
            NeuroManifoldConfig(n_embd=100, n_heads=8)  # 100 % 8 != 0

    def test_config_valid_n_embd_and_heads(self):
        """Test valid n_embd / n_heads combinations."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfig

        config = NeuroManifoldConfig(n_embd=128, n_heads=8)  # 128 % 8 == 0
        assert config.n_embd == 128
        assert config.n_heads == 8

    def test_config_sdr_n_active_computed(self):
        """Test that sdr_n_active is computed from sdr_size * sdr_sparsity."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfig

        config = NeuroManifoldConfig(sdr_size=1024, sdr_sparsity=0.05)
        assert config.sdr_n_active == 51  # int(1024 * 0.05) = 51

    def test_config_model_architecture_defaults(self):
        """Test default model architecture values."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfig

        config = NeuroManifoldConfig()
        assert config.vocab_size == 50304
        assert config.block_size == 1024
        assert config.n_layer == 6
        assert config.n_embd == 384
        assert config.n_heads == 8
        assert config.dropout == 0.0
        assert config.bias is False

    def test_config_manifold_defaults(self):
        """Test default manifold projection values."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfig

        config = NeuroManifoldConfig()
        assert config.manifold_dim == 64
        assert config.n_neighbors == 15
        assert config.n_eigenvectors == 32
        assert config.spectral_sigma == 1.0

    def test_config_fhn_defaults(self):
        """Test default FHN attention values."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfig

        config = NeuroManifoldConfig()
        assert config.fhn_threshold == 0.5
        assert config.fhn_tau == 12.5  # Fixed: proper slow-fast separation
        assert config.fhn_velocity == 1.0
        assert config.pulse_width_base == 4
        assert config.n_fhn_steps == 2  # IMEX allows 2 steps
        assert config.use_fhn_imex is True  # Semi-implicit scheme

    def test_config_engram_defaults(self):
        """Test default engram memory values."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfig

        config = NeuroManifoldConfig()
        assert config.engram_capacity == 1000
        assert config.engram_threshold == 0.3
        assert config.l1_capacity == 100
        assert config.l2_capacity == 500
        assert config.l3_capacity == 1000

    def test_config_dag_and_imagination_defaults(self):
        """Test default DAG planner and imagination module values."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfig

        config = NeuroManifoldConfig()
        assert config.max_dag_depth == 8
        assert config.imagination_steps == 4
        assert config.imagination_dim == 256


class TestNeuroManifoldConfigNano:
    """Test suite for the nano preset configuration."""

    def test_nano_preset_exists(self):
        """Test that nano preset can be imported."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfigNano

        config = NeuroManifoldConfigNano()
        assert config is not None

    def test_nano_preset_values(self):
        """Test nano preset specific values."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfigNano

        config = NeuroManifoldConfigNano()
        assert config.n_layer == 4
        assert config.n_embd == 128
        assert config.n_heads == 4
        assert config.block_size == 256

    def test_nano_preset_reduced_manifold(self):
        """Test nano preset has reduced manifold dimensions."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfigNano

        config = NeuroManifoldConfigNano()
        assert config.manifold_dim == 32
        assert config.n_eigenvectors == 16
        assert config.sdr_size == 1024

    def test_nano_preset_sdr_n_active_computed(self):
        """Test nano preset computes sdr_n_active correctly."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfigNano

        config = NeuroManifoldConfigNano()
        # sdr_size=1024, sdr_sparsity=0.02 (inherited) -> 20 active bits
        assert config.sdr_n_active == int(config.sdr_size * config.sdr_sparsity)

    def test_nano_preset_inherits_from_base(self):
        """Test nano preset inherits non-overridden values from base."""
        from neuromanifold_gpt.config.base import (
            NeuroManifoldConfig,
            NeuroManifoldConfigNano,
        )

        base = NeuroManifoldConfig()
        nano = NeuroManifoldConfigNano()

        # These should be inherited unchanged
        assert nano.vocab_size == base.vocab_size
        assert nano.sdr_sparsity == base.sdr_sparsity
        assert nano.dropout == base.dropout
        assert nano.bias == base.bias


class TestConfigValidation:
    """Test configuration validation logic."""

    def test_sdr_sparsity_affects_n_active(self):
        """Test various sparsity values compute n_active correctly."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfig

        # 2% of 2048 = 40
        config1 = NeuroManifoldConfig(sdr_sparsity=0.02)
        assert config1.sdr_n_active == 40

        # 5% of 2048 = 102
        config2 = NeuroManifoldConfig(sdr_sparsity=0.05)
        assert config2.sdr_n_active == 102

    def test_head_dim_computed_correctly(self):
        """Test that head_dim can be computed from n_embd / n_heads."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfig

        config = NeuroManifoldConfig(n_embd=384, n_heads=8)
        # head_dim should be 384 / 8 = 48
        assert config.n_embd // config.n_heads == 48
