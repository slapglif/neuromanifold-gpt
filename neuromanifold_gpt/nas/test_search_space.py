"""Unit tests for search space module.

Tests cover:
- ArchitectureConfig validation
- SearchSpace sampling
- Config serialization (to_dict/from_dict)
- Config conversion (to_config)
- Parameter space definitions
"""

import pytest

from neuromanifold_gpt.nas.search_space import ArchitectureConfig, SearchSpace


class TestArchitectureConfigValidation:
    """Test ArchitectureConfig validation logic."""

    def test_valid_configuration(self):
        """Test that valid configurations pass validation."""
        config = ArchitectureConfig(n_embd=384, n_heads=8)
        is_valid, error = config.validate()
        assert is_valid
        assert error is None

    def test_invalid_divisibility(self):
        """Test that n_embd not divisible by n_heads fails validation."""
        config = ArchitectureConfig(n_embd=384, n_heads=7)
        is_valid, error = config.validate()
        assert not is_valid
        assert "divisible" in error.lower()

    def test_invalid_attention_type(self):
        """Test that invalid attention type fails validation."""
        config = ArchitectureConfig(attention_type="invalid")
        is_valid, error = config.validate()
        assert not is_valid
        assert "attention_type" in error.lower()

    def test_invalid_kan_type(self):
        """Test that invalid KAN type fails validation."""
        config = ArchitectureConfig(kan_type="invalid")
        is_valid, error = config.validate()
        assert not is_valid
        assert "kan_type" in error.lower()

    def test_invalid_n_layer(self):
        """Test that invalid n_layer fails validation."""
        config = ArchitectureConfig(n_layer=0)
        is_valid, error = config.validate()
        assert not is_valid
        assert "n_layer" in error.lower()

    def test_invalid_dropout_range(self):
        """Test that dropout out of range fails validation."""
        config = ArchitectureConfig(dropout=1.5)
        is_valid, error = config.validate()
        assert not is_valid
        assert "dropout" in error.lower()

    def test_invalid_fhn_threshold(self):
        """Test that negative fhn_threshold fails validation."""
        config = ArchitectureConfig(fhn_threshold=-0.1)
        is_valid, error = config.validate()
        assert not is_valid
        assert "fhn_threshold" in error.lower()

    def test_invalid_sdr_sparsity(self):
        """Test that invalid sdr_sparsity fails validation."""
        config = ArchitectureConfig(sdr_sparsity=1.5)
        is_valid, error = config.validate()
        assert not is_valid
        assert "sdr_sparsity" in error.lower()

    def test_invalid_engram_threshold(self):
        """Test that invalid engram_threshold fails validation."""
        config = ArchitectureConfig(engram_threshold=2.0)
        is_valid, error = config.validate()
        assert not is_valid
        assert "engram_threshold" in error.lower()


class TestSearchSpaceSampling:
    """Test SearchSpace sampling functionality."""

    def test_sample_returns_valid_architecture(self):
        """Test that sampling produces valid architectures."""
        space = SearchSpace()
        for _ in range(20):
            arch = space.sample()
            is_valid, error = arch.validate()
            assert is_valid, f"Sampled invalid architecture: {error}"

    def test_sample_random_returns_valid_architecture(self):
        """Test that sample_random produces valid architectures."""
        space = SearchSpace()
        for _ in range(20):
            arch = space.sample_random()
            is_valid, error = arch.validate()
            assert is_valid, f"Sampled invalid architecture: {error}"

    def test_sample_respects_divisibility_constraint(self):
        """Test that sampled architectures respect n_embd divisible by n_heads."""
        space = SearchSpace()
        for _ in range(20):
            arch = space.sample()
            assert (
                arch.n_embd % arch.n_heads == 0
            ), f"n_embd={arch.n_embd} not divisible by n_heads={arch.n_heads}"

    def test_sample_produces_variety(self):
        """Test that sampling produces different architectures."""
        space = SearchSpace()
        architectures = [space.sample() for _ in range(10)]
        # Convert to dicts for comparison
        dicts = [arch.to_dict() for arch in architectures]
        # Check that not all are identical
        assert (
            len(set(tuple(sorted(d.items())) for d in dicts)) > 1
        ), "All sampled architectures are identical"

    def test_sdr_parameters_in_sample(self):
        """Test that sampled architectures include SDR parameters."""
        space = SearchSpace()
        arch = space.sample()
        assert hasattr(arch, "use_sdr")
        assert hasattr(arch, "sdr_size")
        assert hasattr(arch, "sdr_sparsity")
        assert hasattr(arch, "engram_capacity")
        assert hasattr(arch, "engram_threshold")


class TestConfigSerialization:
    """Test ArchitectureConfig serialization."""

    def test_to_dict_round_trip(self):
        """Test to_dict/from_dict preserves configuration."""
        original = ArchitectureConfig(
            n_layer=8,
            n_embd=512,
            n_heads=8,
            attention_type="fhn",
            use_kan=True,
            kan_type="wave",
            use_sdr=True,
            sdr_size=2048,
        )
        dict_repr = original.to_dict()
        restored = ArchitectureConfig.from_dict(dict_repr)

        # Check all fields match
        assert original.n_layer == restored.n_layer
        assert original.n_embd == restored.n_embd
        assert original.n_heads == restored.n_heads
        assert original.attention_type == restored.attention_type
        assert original.use_kan == restored.use_kan
        assert original.kan_type == restored.kan_type
        assert original.use_sdr == restored.use_sdr
        assert original.sdr_size == restored.sdr_size

    def test_to_dict_includes_all_fields(self):
        """Test that to_dict includes all configuration fields."""
        config = ArchitectureConfig()
        dict_repr = config.to_dict()

        # Check essential fields
        required_fields = [
            "n_layer",
            "n_embd",
            "n_heads",
            "attention_type",
            "use_kan",
            "kan_type",
            "use_sdr",
            "sdr_size",
            "sdr_sparsity",
            "engram_capacity",
            "engram_threshold",
            "dropout",
        ]
        for field in required_fields:
            assert field in dict_repr, f"Missing field: {field}"

    def test_from_dict_with_partial_data(self):
        """Test from_dict with partial data uses defaults."""
        partial = {"n_layer": 10, "n_embd": 768, "n_heads": 12}
        config = ArchitectureConfig.from_dict(partial)
        assert config.n_layer == 10
        assert config.n_embd == 768
        assert config.n_heads == 12
        # Check defaults are used for other fields
        assert config.attention_type == "fhn"  # Default value


class TestConfigConversion:
    """Test ArchitectureConfig to NeuroManifoldConfig conversion."""

    def test_to_config_creates_valid_config(self):
        """Test conversion to NeuroManifoldConfig."""
        # Note: This test requires NeuroManifoldConfig to be importable
        # Skip if not available (e.g., torch not installed)
        try:
            from neuromanifold_gpt.config import NeuroManifoldConfig
        except ImportError:
            pytest.skip("NeuroManifoldConfig not available (torch not installed)")

        arch = ArchitectureConfig(n_layer=6, n_embd=384, n_heads=8)
        config = arch.to_config(vocab_size=65, block_size=512)

        assert isinstance(config, NeuroManifoldConfig)
        assert config.n_layer == 6
        assert config.n_embd == 384
        assert config.n_heads == 8
        assert config.vocab_size == 65
        assert config.block_size == 512

    def test_to_config_includes_sdr_parameters(self):
        """Test that to_config passes SDR parameters."""
        try:
            from neuromanifold_gpt.config import NeuroManifoldConfig
        except ImportError:
            pytest.skip("NeuroManifoldConfig not available (torch not installed)")

        arch = ArchitectureConfig(
            use_sdr=True,
            sdr_size=4096,
            sdr_sparsity=0.03,
            engram_capacity=2000,
            engram_threshold=0.4,
        )
        config = arch.to_config(vocab_size=65)

        assert config.use_sdr
        assert config.sdr_size == 4096
        assert config.sdr_sparsity == 0.03
        assert config.engram_capacity == 2000
        assert config.engram_threshold == 0.4


class TestSearchSpaceDefinition:
    """Test SearchSpace parameter space definition."""

    def test_get_default(self):
        """Test get_default returns default configuration."""
        space = SearchSpace()
        default = space.get_default()
        assert isinstance(default, ArchitectureConfig)
        is_valid, error = default.validate()
        assert is_valid, f"Default config is invalid: {error}"

    def test_get_parameter_space(self):
        """Test get_parameter_space returns valid parameter definitions."""
        space = SearchSpace()
        param_space = space.get_parameter_space()

        # Check key parameters are present
        assert "n_layer" in param_space
        assert "n_embd" in param_space
        assert "attention_type" in param_space
        assert "use_sdr" in param_space
        assert "sdr_size" in param_space

        # Check types
        assert isinstance(param_space["n_layer"], list)
        assert isinstance(param_space["attention_type"], list)
        assert isinstance(param_space["dropout"], tuple)  # Range

    def test_get_search_space_size(self):
        """Test search space size calculation."""
        space = SearchSpace()
        size = space.get_search_space_size()

        # Should be a large number
        assert size > 1000, "Search space seems too small"
        assert isinstance(size, int)

    def test_search_space_includes_sdr_choices(self):
        """Test that search space includes SDR parameter choices."""
        space = SearchSpace()

        assert hasattr(space, "use_sdr_choices")
        assert hasattr(space, "sdr_size_choices")
        assert hasattr(space, "sdr_sparsity_choices")
        assert hasattr(space, "engram_capacity_choices")
        assert hasattr(space, "engram_threshold_choices")

        # Check they're not empty
        assert len(space.use_sdr_choices) > 0
        assert len(space.sdr_size_choices) > 0
