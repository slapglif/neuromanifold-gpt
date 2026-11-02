"""Tests for configuration dependency validation."""

import pytest


class TestConfigDependencyValidation:
    """Test suite for configuration dependency validation."""

    def test_n_embd_divisible_by_n_heads_valid(self):
        """Test valid n_embd / n_heads combinations."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfig

        # Valid combinations
        config1 = NeuroManifoldConfig(n_embd=384, n_heads=8)
        assert config1.n_embd == 384
        assert config1.n_heads == 8

        config2 = NeuroManifoldConfig(n_embd=256, n_heads=4)
        assert config2.n_embd == 256
        assert config2.n_heads == 4

        config3 = NeuroManifoldConfig(n_embd=512, n_heads=16)
        assert config3.n_embd == 512
        assert config3.n_heads == 16

    def test_n_embd_not_divisible_by_n_heads_raises(self):
        """Test that n_embd must be divisible by n_heads."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfig

        with pytest.raises(AssertionError):
            NeuroManifoldConfig(n_embd=100, n_heads=8)

        with pytest.raises(AssertionError):
            NeuroManifoldConfig(n_embd=385, n_heads=8)

        with pytest.raises(AssertionError):
            NeuroManifoldConfig(n_embd=127, n_heads=4)

    def test_memory_active_retrieval_requires_use_sdr(self):
        """Test that memory_active_retrieval requires use_sdr=True."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfig

        # Should raise error when memory_active_retrieval=True and use_sdr=False
        with pytest.raises(
            ValueError, match="memory_active_retrieval=True requires use_sdr=True"
        ):
            NeuroManifoldConfig(memory_active_retrieval=True, use_sdr=False)

    def test_memory_active_retrieval_with_use_sdr_valid(self):
        """Test that memory_active_retrieval works when use_sdr=True."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfig

        config = NeuroManifoldConfig(memory_active_retrieval=True, use_sdr=True)
        assert config.memory_active_retrieval is True
        assert config.use_sdr is True

    def test_memory_active_retrieval_false_with_use_sdr_false_valid(self):
        """Test that memory_active_retrieval=False works with use_sdr=False."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfig

        config = NeuroManifoldConfig(memory_active_retrieval=False, use_sdr=False)
        assert config.memory_active_retrieval is False
        assert config.use_sdr is False

    def test_fast_mode_enables_fast_path_optimizations(self):
        """Test that fast_mode enables all fast-path optimizations."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfig

        config = NeuroManifoldConfig(fast_mode=True)
        assert config.fast_mode is True
        assert config.skip_context_encoder is True
        assert config.skip_semantic_retina is True
        assert config.skip_metric_tensor is True
        assert config.n_fhn_steps == 1

    def test_fast_mode_caps_sdr_size(self):
        """Test that fast_mode caps SDR size at 512."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfig

        config = NeuroManifoldConfig(fast_mode=True, sdr_size=2048)
        assert config.sdr_size == 512

        config2 = NeuroManifoldConfig(fast_mode=True, sdr_size=256)
        assert config2.sdr_size == 256  # Should not increase

    def test_sdr_n_active_computed_from_size_and_sparsity(self):
        """Test that sdr_n_active is computed from sdr_size * sdr_sparsity."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfig

        config1 = NeuroManifoldConfig(sdr_size=2048, sdr_sparsity=0.02)
        assert config1.sdr_n_active == int(2048 * 0.02)  # 40

        config2 = NeuroManifoldConfig(sdr_size=1024, sdr_sparsity=0.05)
        assert config2.sdr_n_active == int(1024 * 0.05)  # 51

        config3 = NeuroManifoldConfig(sdr_size=512, sdr_sparsity=0.1)
        assert config3.sdr_n_active == int(512 * 0.1)  # 51

    def test_head_dim_property_computed_correctly(self):
        """Test that head_dim property is computed correctly."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfig

        config1 = NeuroManifoldConfig(n_embd=384, n_heads=8)
        assert config1.head_dim == 48

        config2 = NeuroManifoldConfig(n_embd=256, n_heads=4)
        assert config2.head_dim == 64

        config3 = NeuroManifoldConfig(n_embd=512, n_heads=16)
        assert config3.head_dim == 32


class TestConfigNanoValidation:
    """Test suite for nano configuration validation."""

    def test_nano_config_inherits_validation(self):
        """Test that nano config inherits validation from base."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfigNano

        # Should fail with invalid n_embd / n_heads
        with pytest.raises(AssertionError):
            NeuroManifoldConfigNano(n_embd=100, n_heads=8)

    def test_nano_config_memory_retrieval_dependency(self):
        """Test that nano config enforces memory_active_retrieval dependency."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfigNano

        with pytest.raises(
            ValueError, match="memory_active_retrieval=True requires use_sdr=True"
        ):
            NeuroManifoldConfigNano(memory_active_retrieval=True, use_sdr=False)

    def test_nano_config_sdr_n_active_computed(self):
        """Test that nano config computes sdr_n_active correctly."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfigNano

        config = NeuroManifoldConfigNano()
        assert config.sdr_n_active == int(config.sdr_size * config.sdr_sparsity)

    def test_nano_config_head_dim_valid(self):
        """Test that nano config has valid head_dim."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfigNano

        config = NeuroManifoldConfigNano()
        # n_embd=128, n_heads=4 -> head_dim=32
        assert config.head_dim == 32
        assert config.n_embd % config.n_heads == 0

    def test_nano_config_fast_mode_enables_optimizations(self):
        """Test that fast_mode works correctly in nano config."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfigNano

        config = NeuroManifoldConfigNano(fast_mode=True)
        assert config.skip_context_encoder is True
        assert config.skip_semantic_retina is True
        assert config.skip_metric_tensor is True
        assert config.n_fhn_steps == 1


class TestMultiscaleManifoldValidation:
    """Test suite for multiscale manifold configuration validation."""

    def test_multiscale_dimensions_hierarchical(self):
        """Test that multiscale dimensions follow expected hierarchy."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfig

        config = NeuroManifoldConfig()
        assert config.multiscale_coarse_dim < config.multiscale_medium_dim
        assert config.multiscale_medium_dim < config.multiscale_fine_dim

    def test_multiscale_fine_dim_matches_manifold_dim(self):
        """Test that multiscale_fine_dim matches manifold_dim."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfig

        config = NeuroManifoldConfig()
        assert config.multiscale_fine_dim == config.manifold_dim

    def test_multiscale_custom_dimensions(self):
        """Test multiscale with custom dimensions."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfig

        config = NeuroManifoldConfig(
            manifold_dim=128,
            multiscale_coarse_dim=32,
            multiscale_medium_dim=64,
            multiscale_fine_dim=128,
        )
        assert config.multiscale_coarse_dim == 32
        assert config.multiscale_medium_dim == 64
        assert config.multiscale_fine_dim == 128
        assert config.manifold_dim == 128


class TestAdvancedFeatureValidation:
    """Test suite for advanced feature configuration validation."""

    def test_moe_configuration_valid(self):
        """Test that MoE configuration has valid parameters."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfig

        config = NeuroManifoldConfig(use_moe=True, moe_n_experts=8, moe_n_active=2)
        assert config.use_moe is True
        assert config.moe_n_experts == 8
        assert config.moe_n_active == 2
        assert config.moe_n_active <= config.moe_n_experts

    def test_mla_configuration_valid(self):
        """Test that MLA configuration has valid parameters."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfig

        config = NeuroManifoldConfig(use_mla=True, mla_latent_dim=64, mla_rope_dim=32)
        assert config.use_mla is True
        assert config.mla_latent_dim == 64
        assert config.mla_rope_dim == 32

    def test_mtp_configuration_valid(self):
        """Test that MTP configuration has valid parameters."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfig

        config = NeuroManifoldConfig(use_mtp=True, mtp_n_predict=4, mtp_loss_weight=0.1)
        assert config.use_mtp is True
        assert config.mtp_n_predict == 4
        assert config.mtp_loss_weight == 0.1

    def test_hybrid_reasoning_configuration_valid(self):
        """Test that hybrid reasoning configuration has valid parameters."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfig

        config = NeuroManifoldConfig(
            use_hybrid_reasoning=True, n_thinking_layers=2, thinking_threshold=0.5
        )
        assert config.use_hybrid_reasoning is True
        assert config.n_thinking_layers == 2
        assert config.thinking_threshold == 0.5

    def test_dag_planner_configuration_valid(self):
        """Test that DAG planner configuration has valid parameters."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfig

        config = NeuroManifoldConfig(
            use_dag_planner=True, dag_max_nodes=32, dag_min_nodes=3
        )
        assert config.use_dag_planner is True
        assert config.dag_max_nodes == 32
        assert config.dag_min_nodes == 3
        assert config.dag_min_nodes <= config.dag_max_nodes

    def test_hierarchical_memory_configuration_valid(self):
        """Test that hierarchical memory configuration has valid parameters."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfig

        config = NeuroManifoldConfig(
            use_hierarchical_memory=True,
            hierarchical_l1_capacity=64,
            hierarchical_l2_capacity=512,
            hierarchical_l3_capacity=4096,
        )
        assert config.use_hierarchical_memory is True
        assert config.hierarchical_l1_capacity == 64
        assert config.hierarchical_l2_capacity == 512
        assert config.hierarchical_l3_capacity == 4096

    def test_imagination_configuration_valid(self):
        """Test that imagination configuration has valid parameters."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfig

        config = NeuroManifoldConfig(
            use_imagination=True, imagination_steps=4, imagination_n_alternatives=4
        )
        assert config.use_imagination is True
        assert config.imagination_steps == 4
        assert config.imagination_n_alternatives == 4


class TestTrainingConfigValidation:
    """Test suite for training configuration validation."""

    def test_lr_schedule_valid_values(self):
        """Test that lr_schedule accepts valid values."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfig

        config1 = NeuroManifoldConfig(lr_schedule="wsd")
        assert config1.lr_schedule == "wsd"

        config2 = NeuroManifoldConfig(lr_schedule="cosine")
        assert config2.lr_schedule == "cosine"

    def test_lr_schedule_ratios_sum_to_one(self):
        """Test that warmup, stable, and decay ratios sum to 1.0."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfig

        config = NeuroManifoldConfig()
        ratio_sum = config.warmup_ratio + config.stable_ratio + config.decay_ratio
        assert abs(ratio_sum - 1.0) < 1e-6  # Allow for float precision

    def test_optimizer_parameters_valid(self):
        """Test that optimizer parameters have valid values."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfig

        config = NeuroManifoldConfig()
        assert 0 < config.learning_rate < 1.0
        assert 0 <= config.weight_decay < 1.0
        assert 0 < config.beta1 < 1.0
        assert 0 < config.beta2 < 1.0
        assert config.optimizer_eps > 0
        assert config.grad_clip > 0

    def test_label_smoothing_valid_range(self):
        """Test that label_smoothing is in valid range [0, 1)."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfig

        config1 = NeuroManifoldConfig(label_smoothing=0.0)
        assert config1.label_smoothing == 0.0

        config2 = NeuroManifoldConfig(label_smoothing=0.1)
        assert config2.label_smoothing == 0.1

        config3 = NeuroManifoldConfig(label_smoothing=0.5)
        assert config3.label_smoothing == 0.5


class TestKANConfigValidation:
    """Test suite for KAN configuration validation."""

    def test_kan_type_valid_values(self):
        """Test that kan_type accepts valid values."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfig

        config1 = NeuroManifoldConfig(use_kan=True, kan_type="faster")
        assert config1.kan_type == "faster"

        config2 = NeuroManifoldConfig(use_kan=True, kan_type="wave")
        assert config2.kan_type == "wave"

        config3 = NeuroManifoldConfig(use_kan=True, kan_type="cheby")
        assert config3.kan_type == "cheby"

    def test_kan_wavelet_valid_values(self):
        """Test that kan_wavelet has valid value."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfig

        config = NeuroManifoldConfig(use_kan=True, kan_wavelet="dog")
        assert config.kan_wavelet == "dog"

    def test_kan_num_centers_positive(self):
        """Test that kan_num_centers is positive."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfig

        config = NeuroManifoldConfig(use_kan=True, kan_num_centers=3)
        assert config.kan_num_centers > 0


class TestMHCConfigValidation:
    """Test suite for mHC (Manifold-Constrained Hyper-Connections) validation."""

    def test_mhc_configuration_valid(self):
        """Test that mHC configuration has valid parameters."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfig

        config = NeuroManifoldConfig(
            use_mhc=True,
            use_full_mhc=True,
            mhc_n_streams=2,
            mhc_residual_weight=0.9,
            mhc_sinkhorn_iters=5,
        )
        assert config.use_mhc is True
        assert config.use_full_mhc is True
        assert config.mhc_n_streams == 2
        assert config.mhc_residual_weight == 0.9
        assert config.mhc_sinkhorn_iters == 5

    def test_mhc_residual_weight_valid_range(self):
        """Test that mhc_residual_weight is in valid range."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfig

        config = NeuroManifoldConfig(mhc_residual_weight=0.9)
        assert 0 <= config.mhc_residual_weight <= 1.0

    def test_mhc_sinkhorn_iters_positive(self):
        """Test that mhc_sinkhorn_iters is positive."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfig

        config = NeuroManifoldConfig(mhc_sinkhorn_iters=5)
        assert config.mhc_sinkhorn_iters > 0
