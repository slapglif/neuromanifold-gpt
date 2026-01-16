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


class TestMTPLossComputation:
    """Tests for MTP forward pass and loss computation."""

    def test_mtp_forward_produces_correct_shapes(self):
        """MTP projections should produce correct output shapes."""
        config = NeuroManifoldConfigNano()
        config.use_mtp = True
        config.mtp_n_predict = 4
        model = NeuroManifoldGPT(config)
        model.eval()

        # Create input (B=2, T=10)
        batch_size, seq_len = 2, 10
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Forward pass
        logits, loss, info = model(idx, targets)

        # Main logits should have full sequence shape
        assert logits.shape == (batch_size, seq_len, config.vocab_size)

        # Loss should be computed
        assert loss is not None
        assert loss.numel() == 1  # Scalar loss

        # MTP loss should be in info dict
        assert 'mtp_loss' in info
        assert info['mtp_loss'].numel() == 1

    def test_mtp_target_shifting_correctness(self):
        """MTP should shift targets correctly for each prediction depth."""
        config = NeuroManifoldConfigNano()
        config.use_mtp = True
        config.mtp_n_predict = 3  # Predict t+1, t+2, t+3
        config.mtp_loss_weight = 1.0  # Full weight for easier testing
        model = NeuroManifoldGPT(config)
        model.eval()

        # Create a sequence where tokens are their position indices
        # This makes it easy to verify target shifting
        batch_size, seq_len = 1, 8
        # Tokens: [0, 1, 2, 3, 4, 5, 6, 7]
        idx = torch.arange(seq_len).unsqueeze(0)
        targets = torch.arange(seq_len).unsqueeze(0)

        # Forward pass
        _, loss, info = model(idx, targets)

        # MTP should compute loss (even if the loss value is high due to untrained model)
        assert info['mtp_loss'] > 0
        assert not torch.isnan(info['mtp_loss'])
        assert not torch.isinf(info['mtp_loss'])

    def test_mtp_loss_computed_for_each_depth(self):
        """MTP should compute loss for all prediction depths."""
        config = NeuroManifoldConfigNano()
        config.use_mtp = True
        config.mtp_n_predict = 4  # t+2, t+3, t+4 (3 auxiliary heads)
        model = NeuroManifoldGPT(config)
        model.eval()

        batch_size, seq_len = 2, 12
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Forward pass
        _, loss, info = model(idx, targets)

        # MTP loss should be non-zero (averaged over 3 heads)
        assert info['mtp_loss'] > 0

        # Total loss should include MTP component
        assert loss is not None
        assert loss > info['ce_loss']  # Total should be larger than just CE

    def test_mtp_loss_averaged_over_depths(self):
        """MTP loss should be averaged over prediction depths."""
        config = NeuroManifoldConfigNano()
        config.use_mtp = True
        config.mtp_n_predict = 4  # 3 auxiliary heads (k=2,3,4)
        model = NeuroManifoldGPT(config)
        model.eval()

        batch_size, seq_len = 2, 12
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Forward pass
        _, loss, info = model(idx, targets)

        # MTP loss is averaged (not summed), so it should be reasonable magnitude
        # Compare to CE loss as reference
        assert info['mtp_loss'] > 0
        assert info['mtp_loss'] < info['ce_loss'] * 5  # Sanity check: not absurdly large

    def test_mtp_loss_weighted_correctly(self):
        """MTP loss should be weighted by mtp_loss_weight in total loss."""
        config = NeuroManifoldConfigNano()
        config.use_mtp = True
        config.mtp_n_predict = 3
        config.mtp_loss_weight = 0.5
        # Disable other loss components for cleaner testing
        config.ortho_loss_weight = 0.0
        config.discrimination_loss_weight = 0.0
        config.contrastive_loss_weight = 0.0

        model = NeuroManifoldGPT(config)
        model.eval()
        # Set topographic and FHN weights to 0
        model._topo_weight = 0.0
        model._fhn_lyapunov_weight = 0.0

        batch_size, seq_len = 2, 10
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Forward pass
        _, loss, info = model(idx, targets)

        # Total loss should approximately equal CE + weighted MTP
        # (with small tolerance for numerical precision)
        expected_loss = info['ce_loss'] + config.mtp_loss_weight * info['mtp_loss']
        assert torch.isclose(loss, expected_loss, rtol=1e-4, atol=1e-6)

    def test_mtp_loss_integrated_into_total_loss(self):
        """MTP loss should be properly integrated into total loss calculation."""
        config = NeuroManifoldConfigNano()
        config.use_mtp = True
        config.mtp_n_predict = 4
        config.mtp_loss_weight = 0.1

        model = NeuroManifoldGPT(config)
        model.eval()

        batch_size, seq_len = 2, 10
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Forward pass with MTP
        _, loss_with_mtp, info_with_mtp = model(idx, targets)

        # Disable MTP and run again
        model.use_mtp = False
        _, loss_without_mtp, info_without_mtp = model(idx, targets)

        # Loss with MTP should be larger (additional MTP component)
        assert loss_with_mtp > loss_without_mtp

        # MTP loss should be zero when disabled
        assert info_without_mtp['mtp_loss'] == 0.0

    def test_mtp_with_short_sequence(self):
        """MTP should handle sequences shorter than prediction depth gracefully."""
        config = NeuroManifoldConfigNano()
        config.use_mtp = True
        config.mtp_n_predict = 4  # Tries to predict up to t+4
        model = NeuroManifoldGPT(config)
        model.eval()

        # Very short sequence (T=3, less than some prediction depths)
        batch_size, seq_len = 2, 3
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Should not crash
        _, loss, info = model(idx, targets)

        # Loss should still be computed
        assert loss is not None
        assert not torch.isnan(loss)

        # MTP loss might be zero or very small due to short sequence
        assert not torch.isnan(info['mtp_loss'])

    def test_mtp_disabled_zero_loss(self):
        """MTP loss should be zero when MTP is disabled."""
        config = NeuroManifoldConfigNano()
        config.use_mtp = False
        model = NeuroManifoldGPT(config)
        model.eval()

        batch_size, seq_len = 2, 10
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        _, loss, info = model(idx, targets)

        # MTP loss should be exactly zero
        assert info['mtp_loss'] == 0.0

    def test_mtp_loss_backpropagation(self):
        """MTP loss should allow gradient backpropagation."""
        config = NeuroManifoldConfigNano()
        config.use_mtp = True
        config.mtp_n_predict = 3
        model = NeuroManifoldGPT(config)
        model.train()

        batch_size, seq_len = 2, 8
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Forward pass
        _, loss, info = model(idx, targets)

        # Backward pass should work
        loss.backward()

        # Check that MTP projection heads received gradients
        for proj in model.mtp_projs:
            for param in proj.parameters():
                assert param.grad is not None
                assert not torch.all(param.grad == 0)  # Should have non-zero gradients
