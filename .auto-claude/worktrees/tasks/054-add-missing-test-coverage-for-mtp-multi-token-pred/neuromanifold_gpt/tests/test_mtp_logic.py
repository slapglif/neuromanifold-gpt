# neuromanifold_gpt/tests/test_mtp_logic.py
"""Tests for Multi-Token Prediction (MTP) logic."""
import pytest
import torch
from neuromanifold_gpt.config import NeuroManifoldConfigNano
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT


class TestMTPInitialization:
    """Tests for MTP initialization and configuration."""

    def test_mtp_disabled_by_default(self):
        """MTP should be disabled by default in base config."""
        config = NeuroManifoldConfigNano()
        # Override the base config's default (which is True) for this test
        config.use_mtp = False
        model = NeuroManifoldGPT(config)

        assert model.use_mtp is False
        assert not hasattr(model, 'mtp_projs')

    def test_mtp_enabled_config_params(self):
        """MTP config parameters should be loaded correctly."""
        config = NeuroManifoldConfigNano()
        config.use_mtp = True
        config.mtp_n_predict = 4
        config.mtp_loss_weight = 0.1

        model = NeuroManifoldGPT(config)

        assert model.use_mtp is True
        assert model.mtp_n_predict == 4
        assert model.mtp_loss_weight == 0.1

    def test_mtp_creates_correct_number_of_heads(self):
        """MTP should create n_predict - 1 projection heads."""
        config = NeuroManifoldConfigNano()
        config.use_mtp = True
        config.mtp_n_predict = 4

        model = NeuroManifoldGPT(config)

        # Should create 3 heads (n_predict - 1, since main lm_head handles t+1)
        assert hasattr(model, 'mtp_projs')
        assert len(model.mtp_projs) == 3

    def test_mtp_projection_architecture(self):
        """MTP projection heads should have Linear -> SiLU architecture."""
        config = NeuroManifoldConfigNano()
        config.use_mtp = True
        config.mtp_n_predict = 4

        model = NeuroManifoldGPT(config)

        # Check each projection head
        for proj in model.mtp_projs:
            # Should be a Sequential module
            assert isinstance(proj, torch.nn.Sequential)
            # Should have 2 layers: Linear and SiLU
            assert len(proj) == 2
            assert isinstance(proj[0], torch.nn.Linear)
            assert isinstance(proj[1], torch.nn.SiLU)
            # Linear layer should map n_embd -> n_embd
            assert proj[0].in_features == config.n_embd
            assert proj[0].out_features == config.n_embd

    def test_mtp_with_different_n_predict_values(self):
        """MTP should work with different n_predict values."""
        config = NeuroManifoldConfigNano()
        config.use_mtp = True

        for n_predict in [2, 3, 4, 8]:
            config.mtp_n_predict = n_predict
            model = NeuroManifoldGPT(config)

            assert model.mtp_n_predict == n_predict
            assert len(model.mtp_projs) == n_predict - 1

    def test_mtp_single_prediction_no_heads(self):
        """MTP with n_predict=1 should not create auxiliary heads."""
        config = NeuroManifoldConfigNano()
        config.use_mtp = True
        config.mtp_n_predict = 1

        model = NeuroManifoldGPT(config)

        # n_predict=1 means only main lm_head, no auxiliary heads
        assert not hasattr(model, 'mtp_projs')

    def test_mtp_disabled_no_heads(self):
        """Disabled MTP should not create projection heads."""
        config = NeuroManifoldConfigNano()
        config.use_mtp = False
        config.mtp_n_predict = 4  # Even with n_predict > 1

        model = NeuroManifoldGPT(config)

        assert not hasattr(model, 'mtp_projs')

    def test_mtp_with_fp32_lm_head(self):
        """MTP should work with FP32 lm_head enabled."""
        config = NeuroManifoldConfigNano()
        config.use_mtp = True
        config.mtp_n_predict = 4
        config.lm_head_fp32 = True

        model = NeuroManifoldGPT(config)

        assert model.lm_head_fp32 is True
        assert model.use_mtp is True
        assert len(model.mtp_projs) == 3
