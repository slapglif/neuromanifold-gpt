#!/usr/bin/env python3
"""Tests for text generation with NeuroManifoldGPT."""

import pytest
import torch
import torch.nn.functional as F
from neuromanifold_gpt.config import NeuroManifoldConfigNano
from neuromanifold_gpt.config.base import NeuroManifoldConfig
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT


class TestBasicGeneration:
    """Test basic generation functionality."""

    def test_generate_default_params(self, gpt_model, tokens_1x10):
        """Test generation with default parameters."""
        gpt_model.eval()

        with torch.no_grad():
            generated = gpt_model.generate(tokens_1x10, max_new_tokens=5)

        assert generated.shape == (1, 15)
        # Verify prompt is unchanged
        assert torch.equal(generated[:, :10], tokens_1x10)

    def test_generate_with_temperature(self, gpt_model, tokens_1x10):
        """Test generation with different temperatures."""
        gpt_model.eval()

        for temp in [0.5, 0.8, 1.0, 1.5]:
            with torch.no_grad():
                generated = gpt_model.generate(
                    tokens_1x10,
                    max_new_tokens=10,
                    temperature=temp
                )

            assert generated.shape == (1, 20)
            assert torch.equal(generated[:, :10], tokens_1x10)

    def test_generate_with_top_k(self, gpt_model, tokens_1x10):
        """Test generation with top-k sampling."""
        gpt_model.eval()

        for top_k in [10, 20, 40]:
            with torch.no_grad():
                generated = gpt_model.generate(
                    tokens_1x10,
                    max_new_tokens=10,
                    top_k=top_k
                )

            assert generated.shape == (1, 20)
            assert torch.equal(generated[:, :10], tokens_1x10)

    def test_generate_batch(self, gpt_model, tokens_2x20):
        """Test generation with batched input."""
        gpt_model.eval()

        with torch.no_grad():
            generated = gpt_model.generate(tokens_2x20, max_new_tokens=10)

        assert generated.shape == (2, 30)
        # Verify prompts are unchanged
        assert torch.equal(generated[:, :20], tokens_2x20)


class TestGenerationWithSDR:
    """Test generation with SDR mode enabled."""

    def test_generate_sdr_mode(self, nano_config, tokens_1x10):
        """Test generation with SDR enabled."""
        nano_config.use_sdr = True
        model = NeuroManifoldGPT(nano_config)
        model.eval()

        with torch.no_grad():
            generated = model.generate(tokens_1x10, max_new_tokens=10)

        assert generated.shape == (1, 20)
        assert torch.equal(generated[:, :10], tokens_1x10)

    def test_generate_dense_mode(self, nano_config, tokens_1x10):
        """Test generation with SDR disabled."""
        nano_config.use_sdr = False
        model = NeuroManifoldGPT(nano_config)
        model.eval()

        with torch.no_grad():
            generated = model.generate(tokens_1x10, max_new_tokens=10)

        assert generated.shape == (1, 20)
        assert torch.equal(generated[:, :10], tokens_1x10)


class TestGenerationOutputDistribution:
    """Test generation output distribution properties."""

    def test_logits_valid_distribution(self, gpt_model, tokens_1x20):
        """Test that logits form a valid probability distribution."""
        gpt_model.eval()

        with torch.no_grad():
            logits, _, _ = gpt_model(tokens_1x20)
            last_logits = logits[0, -1, :]
            probs = F.softmax(last_logits, dim=-1)

        # Check probabilities are valid
        assert torch.all(probs >= 0)
        assert torch.all(probs <= 1)
        assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)

    def test_top_k_tokens_reasonable(self, gpt_model, tokens_1x20):
        """Test that top-k predictions are reasonable."""
        gpt_model.eval()

        with torch.no_grad():
            logits, _, _ = gpt_model(tokens_1x20)
            last_logits = logits[0, -1, :]
            probs = F.softmax(last_logits, dim=-1)
            top_probs, top_ids = torch.topk(probs, 10)

        # Top probabilities should be in descending order
        assert torch.all(top_probs[:-1] >= top_probs[1:])

        # All top IDs should be valid tokens
        assert torch.all(top_ids >= 0)
        assert torch.all(top_ids < gpt_model.config.vocab_size)

    def test_generation_deterministic_with_temp_zero(self, gpt_model, tokens_1x10):
        """Test that temperature=0 gives deterministic generation."""
        gpt_model.eval()

        # Small epsilon for temperature (true 0 would cause division by zero)
        temp = 1e-8

        with torch.no_grad():
            gen1 = gpt_model.generate(tokens_1x10.clone(), max_new_tokens=5, temperature=temp)
            gen2 = gpt_model.generate(tokens_1x10.clone(), max_new_tokens=5, temperature=temp)

        # Should produce identical outputs
        assert torch.equal(gen1, gen2)


class TestGenerationEdgeCases:
    """Test edge cases in generation."""

    def test_generate_zero_tokens(self, gpt_model, tokens_1x10):
        """Test generation with max_new_tokens=0."""
        gpt_model.eval()

        with torch.no_grad():
            generated = gpt_model.generate(tokens_1x10, max_new_tokens=0)

        # Should return unchanged prompt
        assert torch.equal(generated, tokens_1x10)

    def test_generate_single_token(self, gpt_model, tokens_1x10):
        """Test generation of a single token."""
        gpt_model.eval()

        with torch.no_grad():
            generated = gpt_model.generate(tokens_1x10, max_new_tokens=1)

        assert generated.shape == (1, 11)
        assert torch.equal(generated[:, :10], tokens_1x10)

    def test_generate_long_sequence(self, gpt_model, tokens_1x10):
        """Test generation of longer sequences."""
        gpt_model.eval()

        with torch.no_grad():
            generated = gpt_model.generate(tokens_1x10, max_new_tokens=50)

        assert generated.shape == (1, 60)
        assert torch.equal(generated[:, :10], tokens_1x10)

    def test_generate_respects_vocab_size(self, gpt_model, tokens_1x10):
        """Test that generated tokens are within vocab bounds."""
        gpt_model.eval()

        with torch.no_grad():
            generated = gpt_model.generate(tokens_1x10, max_new_tokens=20)

        # All tokens should be valid
        assert torch.all(generated >= 0)
        assert torch.all(generated < gpt_model.config.vocab_size)


class TestGenerationTrainingMode:
    """Test generation behavior in different model modes."""

    def test_eval_mode_required_for_generation(self, gpt_model, tokens_1x10):
        """Test that generation works correctly in eval mode."""
        gpt_model.eval()

        # Should work without errors
        with torch.no_grad():
            generated = gpt_model.generate(tokens_1x10, max_new_tokens=5)

        assert generated.shape == (1, 15)

    def test_generation_in_train_mode(self, gpt_model, tokens_1x10):
        """Test that generation still works in train mode (though not recommended)."""
        gpt_model.train()

        # Should still work, though results may differ
        with torch.no_grad():
            generated = gpt_model.generate(tokens_1x10, max_new_tokens=5)

        assert generated.shape == (1, 15)


if __name__ == "__main__":
    # Allow running as standalone script for debugging
    pytest.main([__file__, "-v"])
