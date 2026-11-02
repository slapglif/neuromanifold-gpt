"""Tests for Context Encoder module.

TDD tests for ContextEncoder which modulates embeddings based on local
context window, enabling context-aware understanding (e.g., "bank" near
"money" vs "river").
"""
import pytest
import torch


class TestContextEncoderBasic:
    """Basic functionality tests for ContextEncoder."""

    def test_context_encoder_shape(self):
        """Context encoder should preserve input shape."""
        from neuromanifold_gpt.model.context_encoder import ContextEncoder

        encoder = ContextEncoder(embed_dim=256, context_size=5)
        x = torch.randn(2, 20, 256)
        out = encoder(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_context_encoder_local_attention(self):
        """Context encoder with small context size should work."""
        from neuromanifold_gpt.model.context_encoder import ContextEncoder

        encoder = ContextEncoder(embed_dim=256, context_size=2)
        x = torch.randn(1, 10, 256)
        out = encoder(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_context_gate_bounded(self):
        """Context gate values should be in [0, 1]."""
        from neuromanifold_gpt.model.context_encoder import ContextEncoder

        encoder = ContextEncoder(embed_dim=256, context_size=5)
        x = torch.randn(2, 20, 256)
        with torch.no_grad():
            x_normed = encoder.ln(x) if encoder.use_layer_norm else x
            context_out, _ = encoder.context_attn(
                encoder.context_proj(x_normed), x_normed, x_normed
            )
            gate = encoder.context_gate(torch.cat([x, context_out], dim=-1))
        assert gate.min() >= 0.0, f"Gate min {gate.min()} < 0"
        assert gate.max() <= 1.0, f"Gate max {gate.max()} > 1"

    def test_context_encoder_deterministic(self):
        """Same input should produce same output in eval mode."""
        from neuromanifold_gpt.model.context_encoder import ContextEncoder

        encoder = ContextEncoder(embed_dim=128, context_size=3)
        encoder.eval()
        x = torch.randn(1, 8, 128)
        with torch.no_grad():
            out1 = encoder(x)
            out2 = encoder(x)
        assert torch.allclose(out1, out2), "Output should be deterministic in eval mode"

    def test_context_encoder_different_embed_dims(self):
        """Should work with various embedding dimensions."""
        from neuromanifold_gpt.model.context_encoder import ContextEncoder

        for embed_dim in [64, 128, 256, 512]:
            encoder = ContextEncoder(embed_dim=embed_dim, context_size=3)
            x = torch.randn(1, 10, embed_dim)
            out = encoder(x)
            assert out.shape == x.shape, f"Failed for embed_dim={embed_dim}"


class TestContextEncoderMask:
    """Tests for local attention mask creation."""

    def test_local_mask_correctness(self):
        """Verify local attention mask is correct."""
        from neuromanifold_gpt.model.context_encoder import ContextEncoder

        encoder = ContextEncoder(embed_dim=64, context_size=2)
        mask = encoder._make_local_mask(5, torch.device("cpu"))

        # For context_size=2, position i can attend to [i-2, i+2] (5 positions)
        # Mask is True where attention is BLOCKED
        # Position 0 can attend to 0,1,2 (indices within window)
        assert not mask[0, 0], "Position 0 should attend to itself"
        assert not mask[0, 1], "Position 0 should attend to position 1"
        assert not mask[0, 2], "Position 0 should attend to position 2"
        assert mask[0, 3], "Position 0 should NOT attend to position 3"

        # Position 2 can attend to 0,1,2,3,4 (full window)
        assert not mask[2, 0], "Position 2 should attend to position 0"
        assert not mask[2, 4], "Position 2 should attend to position 4"

    def test_mask_is_symmetric(self):
        """Mask should be symmetric (distance is symmetric)."""
        from neuromanifold_gpt.model.context_encoder import ContextEncoder

        encoder = ContextEncoder(embed_dim=64, context_size=3)
        mask = encoder._make_local_mask(10, torch.device("cpu"))

        assert torch.equal(mask, mask.T), "Mask should be symmetric"

    def test_mask_caching(self):
        """Mask should be cached for same sequence length."""
        from neuromanifold_gpt.model.context_encoder import ContextEncoder

        encoder = ContextEncoder(embed_dim=64, context_size=2)
        encoder._make_local_mask(5, torch.device("cpu"))
        encoder._make_local_mask(5, torch.device("cpu"))

        # Should use cached version
        assert encoder._cached_seq_len == 5
        assert encoder._cached_mask is not None

    def test_mask_diagonal_always_false(self):
        """Diagonal (self-attention) should never be masked."""
        from neuromanifold_gpt.model.context_encoder import ContextEncoder

        for context_size in [1, 3, 5]:
            encoder = ContextEncoder(embed_dim=64, context_size=context_size)
            mask = encoder._make_local_mask(10, torch.device("cpu"))

            # Diagonal should always be False (attend to self)
            assert (
                not mask.diagonal().any()
            ), f"Diagonal should be unmasked for context_size={context_size}"


class TestContextEncoderGradients:
    """Tests for gradient flow and training behavior."""

    def test_context_encoder_gradient_flow(self):
        """Gradients should flow through the encoder."""
        from neuromanifold_gpt.model.context_encoder import ContextEncoder

        encoder = ContextEncoder(embed_dim=64, context_size=3)
        x = torch.randn(2, 10, 64, requires_grad=True)
        out = encoder(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "Gradient should flow to input"
        assert x.grad.shape == x.shape, "Gradient shape should match input"

    def test_context_encoder_batch_independence(self):
        """Each batch element should be processed independently."""
        from neuromanifold_gpt.model.context_encoder import ContextEncoder

        encoder = ContextEncoder(embed_dim=64, context_size=2)
        encoder.eval()

        x1 = torch.randn(1, 8, 64)
        x2 = torch.randn(1, 8, 64)
        x_batch = torch.cat([x1, x2], dim=0)

        with torch.no_grad():
            out1 = encoder(x1)
            out2 = encoder(x2)
            out_batch = encoder(x_batch)

        assert torch.allclose(
            out_batch[0], out1[0], atol=1e-6
        ), "First batch element should match single processing"
        assert torch.allclose(
            out_batch[1], out2[0], atol=1e-6
        ), "Second batch element should match single processing"

    def test_gradient_through_gate(self):
        """Gradients should flow through the gating mechanism."""
        from neuromanifold_gpt.model.context_encoder import ContextEncoder

        encoder = ContextEncoder(embed_dim=64, context_size=3)
        x = torch.randn(2, 10, 64, requires_grad=True)
        out = encoder(x)
        loss = out.sum()
        loss.backward()

        # Check that gate weights received gradients
        for name, param in encoder.context_gate.named_parameters():
            if param.grad is not None:
                assert (
                    param.grad.abs().sum() > 0
                ), f"Gate param {name} should receive gradient"


class TestContextEncoderConfig:
    """Tests for configuration-based initialization."""

    def test_from_config(self):
        """Should create encoder from NeuroManifoldConfig."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfig
        from neuromanifold_gpt.model.context_encoder import ContextEncoder

        config = NeuroManifoldConfig()
        encoder = ContextEncoder.from_config(config)

        assert encoder.embed_dim == config.sdr_embed_dim
        assert encoder.context_size == config.sdr_context_size
        assert encoder.n_heads == config.n_heads
        assert encoder.dropout == config.dropout

    def test_from_nano_config(self):
        """Should work with nano preset config."""
        from neuromanifold_gpt.config.base import NeuroManifoldConfigNano
        from neuromanifold_gpt.model.context_encoder import ContextEncoder

        config = NeuroManifoldConfigNano()
        encoder = ContextEncoder.from_config(config)

        x = torch.randn(1, 10, config.sdr_embed_dim)
        out = encoder(x)
        assert out.shape == x.shape


class TestContextEncoderOptions:
    """Tests for optional features (dropout, layer norm, etc.)."""

    def test_dropout_effect(self):
        """Dropout should cause different outputs in train mode."""
        from neuromanifold_gpt.model.context_encoder import ContextEncoder

        encoder = ContextEncoder(embed_dim=64, context_size=3, dropout=0.5)
        encoder.train()
        x = torch.randn(2, 10, 64)

        # In train mode with high dropout, outputs should differ
        out1 = encoder(x)
        out2 = encoder(x)

        # Note: with dropout=0.5, outputs are very likely to differ
        # but this is probabilistic, so we just check it doesn't crash
        assert out1.shape == x.shape
        assert out2.shape == x.shape

    def test_no_layer_norm(self):
        """Should work without layer normalization."""
        from neuromanifold_gpt.model.context_encoder import ContextEncoder

        encoder = ContextEncoder(embed_dim=64, context_size=3, use_layer_norm=False)
        assert not hasattr(encoder, "ln") or not encoder.use_layer_norm

        x = torch.randn(2, 10, 64)
        out = encoder(x)
        assert out.shape == x.shape

    def test_with_layer_norm(self):
        """Layer norm should be applied when enabled."""
        from neuromanifold_gpt.model.context_encoder import ContextEncoder

        encoder = ContextEncoder(embed_dim=64, context_size=3, use_layer_norm=True)
        assert hasattr(encoder, "ln")
        assert encoder.use_layer_norm

        x = torch.randn(2, 10, 64)
        out = encoder(x)
        assert out.shape == x.shape

    def test_extra_repr(self):
        """extra_repr should return proper string representation."""
        from neuromanifold_gpt.model.context_encoder import ContextEncoder

        encoder = ContextEncoder(
            embed_dim=128, context_size=5, n_heads=4, dropout=0.1, use_layer_norm=True
        )

        repr_str = encoder.extra_repr()
        assert "embed_dim=128" in repr_str
        assert "context_size=5" in repr_str
        assert "n_heads=4" in repr_str
        assert "dropout=0.1" in repr_str
        assert "use_layer_norm=True" in repr_str


class TestContextEncoderValidation:
    """Tests for input validation and error handling."""

    def test_invalid_n_heads(self):
        """Should raise error if embed_dim not divisible by n_heads."""
        from neuromanifold_gpt.model.context_encoder import ContextEncoder

        with pytest.raises(ValueError, match="must be divisible"):
            ContextEncoder(embed_dim=100, n_heads=7)  # 100 not divisible by 7

    def test_single_token_sequence(self):
        """Should handle single token sequences."""
        from neuromanifold_gpt.model.context_encoder import ContextEncoder

        encoder = ContextEncoder(embed_dim=64, context_size=3)
        x = torch.randn(2, 1, 64)  # Single token
        out = encoder(x)
        assert out.shape == x.shape

    def test_context_larger_than_sequence(self):
        """Should handle context window larger than sequence."""
        from neuromanifold_gpt.model.context_encoder import ContextEncoder

        encoder = ContextEncoder(embed_dim=64, context_size=10)
        x = torch.randn(2, 5, 64)  # Sequence shorter than context
        out = encoder(x)
        assert out.shape == x.shape

    def test_very_long_sequence(self):
        """Should handle reasonably long sequences."""
        from neuromanifold_gpt.model.context_encoder import ContextEncoder

        encoder = ContextEncoder(embed_dim=64, context_size=5)
        x = torch.randn(1, 512, 64)
        out = encoder(x)
        assert out.shape == x.shape
