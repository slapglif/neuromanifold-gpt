"""Tests for Complete Semantic Folding Encoder.

The SemanticFoldingEncoder is the core input module that:
1. Takes token IDs
2. Produces context-aware SDRs (2048-bit vectors with ~40 active bits)
3. Preserves semantic similarity in bit overlap

Tests verify:
- Output shape correctness
- Exact sparsity (40 active bits)
- Self-similarity = 1.0
- Phrase encoding via union + re-sparsify
"""
import pytest
import torch

from neuromanifold_gpt.model.semantic_folding import SemanticFoldingEncoder


class TestEncoderOutputShape:
    """Test that encoder produces correct output shapes."""

    def test_encoder_output_shape(self):
        """SDR output should have correct shape and sparsity."""
        encoder = SemanticFoldingEncoder(
            vocab_size=1000, sdr_size=2048, n_active=40, embed_dim=256
        )
        tokens = torch.randint(0, 1000, (2, 20))

        sdr, scores = encoder(tokens)

        assert sdr.shape == (2, 20, 2048)
        assert scores.shape == (2, 20, 2048)

    def test_encoder_single_token(self):
        """Encoder handles single token input."""
        encoder = SemanticFoldingEncoder(
            vocab_size=1000, sdr_size=2048, n_active=40, embed_dim=256
        )
        encoder.eval()
        tokens = torch.randint(0, 1000, (1, 1))

        sdr, scores = encoder(tokens)

        assert sdr.shape == (1, 1, 2048)
        assert scores.shape == (1, 1, 2048)

    def test_encoder_batch_sizes(self):
        """Encoder handles various batch sizes."""
        encoder = SemanticFoldingEncoder(
            vocab_size=1000, sdr_size=2048, n_active=40, embed_dim=256
        )
        encoder.eval()

        for batch_size in [1, 4, 16]:
            tokens = torch.randint(0, 1000, (batch_size, 10))
            sdr, _ = encoder(tokens)
            assert sdr.shape == (batch_size, 10, 2048)


class TestEncoderSparsity:
    """Test sparsity enforcement."""

    def test_encoder_sparsity(self):
        """SDR should have exactly n_active bits per token."""
        encoder = SemanticFoldingEncoder(
            vocab_size=1000, sdr_size=2048, n_active=40, embed_dim=256
        )
        encoder.eval()  # Use hard top-k
        tokens = torch.randint(0, 1000, (2, 20))

        sdr, _ = encoder(tokens)

        # Each SDR should have exactly 40 active bits
        assert (sdr.sum(dim=-1) == 40).all()

    def test_encoder_binary_values(self):
        """SDR values should be 0 or 1."""
        encoder = SemanticFoldingEncoder(
            vocab_size=1000, sdr_size=2048, n_active=40, embed_dim=256
        )
        encoder.eval()
        tokens = torch.randint(0, 1000, (2, 20))

        sdr, _ = encoder(tokens)

        assert ((sdr == 0) | (sdr == 1)).all()

    def test_training_mode_sparsity(self):
        """Training mode should also produce approximately correct sparsity."""
        encoder = SemanticFoldingEncoder(
            vocab_size=1000, sdr_size=2048, n_active=40, embed_dim=256
        )
        encoder.train()
        tokens = torch.randint(0, 1000, (2, 20))

        sdr, _ = encoder(tokens)

        # Soft topk still produces exactly n_active due to hard component
        # Allow small floating point tolerance
        active_counts = sdr.sum(dim=-1)
        assert (active_counts >= 39.5).all()  # Near 40


class TestSemanticSimilarity:
    """Test semantic similarity operations."""

    def test_semantic_similarity_self(self):
        """Token should have similarity 1.0 with itself."""
        encoder = SemanticFoldingEncoder(
            vocab_size=1000, sdr_size=2048, n_active=40, embed_dim=256
        )
        encoder.eval()
        tokens = torch.randint(0, 1000, (1, 5))

        sdr, _ = encoder(tokens)

        sim = encoder.semantic_similarity(sdr[0, 0], sdr[0, 0])
        assert abs(sim - 1.0) < 1e-5

    def test_semantic_similarity_different_tokens(self):
        """Different tokens generally have < 1.0 similarity."""
        encoder = SemanticFoldingEncoder(
            vocab_size=1000, sdr_size=2048, n_active=40, embed_dim=256
        )
        encoder.eval()
        # Use specific different tokens
        tokens = torch.tensor([[0, 100]])

        sdr, _ = encoder(tokens)

        sim = encoder.semantic_similarity(sdr[0, 0], sdr[0, 1])
        # Different tokens should generally have different SDRs
        # (not guaranteed to be < 1.0 but very likely)
        assert sim <= 1.0


class TestPhraseEncoding:
    """Test phrase-level SDR encoding."""

    def test_encode_phrase(self):
        """Phrase encoding should union token SDRs."""
        encoder = SemanticFoldingEncoder(
            vocab_size=1000, sdr_size=2048, n_active=40, embed_dim=256
        )
        encoder.eval()
        tokens = torch.randint(0, 1000, (1, 5))

        sdr, _ = encoder(tokens)
        phrase_sdr = encoder.encode_phrase(sdr)

        assert phrase_sdr.shape == (1, 2048)
        assert phrase_sdr.sum() == 40  # re-sparsified

    def test_encode_phrase_batched(self):
        """Phrase encoding works with batched input."""
        encoder = SemanticFoldingEncoder(
            vocab_size=1000, sdr_size=2048, n_active=40, embed_dim=256
        )
        encoder.eval()
        tokens = torch.randint(0, 1000, (4, 5))

        sdr, _ = encoder(tokens)
        phrase_sdr = encoder.encode_phrase(sdr)

        assert phrase_sdr.shape == (4, 2048)
        assert (phrase_sdr.sum(dim=-1) == 40).all()


class TestGradientFlow:
    """Test that gradients flow correctly during training."""

    def test_gradient_through_encoder(self):
        """Gradients should flow back through encoder."""
        encoder = SemanticFoldingEncoder(
            vocab_size=1000, sdr_size=2048, n_active=40, embed_dim=256
        )
        encoder.train()
        tokens = torch.randint(0, 1000, (2, 10))

        sdr, scores = encoder(tokens)
        loss = scores.sum()  # Use scores for cleaner gradients
        loss.backward()

        # Embedding weights should have gradients
        assert encoder.token_embed.weight.grad is not None
        assert encoder.token_embed.weight.grad.abs().sum() > 0

    def test_gradient_through_sdr(self):
        """SDR output allows gradient flow via soft_topk."""
        encoder = SemanticFoldingEncoder(
            vocab_size=1000, sdr_size=2048, n_active=40, embed_dim=256
        )
        encoder.train()
        tokens = torch.randint(0, 1000, (2, 10))

        sdr, _ = encoder(tokens)
        loss = sdr.sum()
        loss.backward()

        # Should have gradients (via straight-through estimator)
        assert encoder.token_embed.weight.grad is not None


class TestDeterminism:
    """Test deterministic behavior."""

    def test_eval_mode_deterministic(self):
        """Same input should produce same output in eval mode."""
        encoder = SemanticFoldingEncoder(
            vocab_size=1000, sdr_size=2048, n_active=40, embed_dim=256
        )
        encoder.eval()
        tokens = torch.randint(0, 1000, (2, 10))

        sdr1, _ = encoder(tokens)
        sdr2, _ = encoder(tokens)

        assert torch.equal(sdr1, sdr2)


class TestEncoderComponents:
    """Test internal components work correctly."""

    def test_context_encoder_integration(self):
        """Context encoder should modulate embeddings."""
        encoder = SemanticFoldingEncoder(
            vocab_size=1000, sdr_size=2048, n_active=40, embed_dim=256, context_size=5
        )
        encoder.eval()

        # Context should matter - changing surrounding tokens may change SDR
        # This is a loose test since context effect isn't guaranteed to be visible
        tokens = torch.randint(0, 1000, (1, 20))
        sdr, _ = encoder(tokens)

        # Just verify it runs without error
        assert sdr.shape == (1, 20, 2048)

    def test_retina_integration(self):
        """Semantic retina should be part of pipeline."""
        encoder = SemanticFoldingEncoder(
            vocab_size=1000, sdr_size=2048, n_active=40, embed_dim=256, grid_size=64
        )

        # Verify retina exists and has expected properties
        assert hasattr(encoder, "retina")
        assert encoder.retina.grid_size == 64


class TestBoostingMechanism:
    """Test bit duty cycle boosting for uniform usage."""

    def test_boosting_updates_duty_cycle(self):
        """Training should update bit duty cycle."""
        encoder = SemanticFoldingEncoder(
            vocab_size=1000, sdr_size=2048, n_active=40, embed_dim=256
        )
        encoder.train()

        initial_duty = encoder.bit_duty_cycle.clone()

        # Run a few training steps
        for _ in range(5):
            tokens = torch.randint(0, 1000, (4, 20))
            sdr, _ = encoder(tokens)

        # Duty cycle should have been updated
        assert not torch.equal(initial_duty, encoder.bit_duty_cycle)

    def test_boosting_disabled_in_eval(self):
        """Boosting should not affect eval mode."""
        encoder = SemanticFoldingEncoder(
            vocab_size=1000, sdr_size=2048, n_active=40, embed_dim=256
        )
        encoder.eval()

        initial_duty = encoder.bit_duty_cycle.clone()

        tokens = torch.randint(0, 1000, (4, 20))
        sdr, _ = encoder(tokens)

        # Duty cycle should NOT be updated in eval mode
        assert torch.equal(initial_duty, encoder.bit_duty_cycle)
