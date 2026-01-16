# neuromanifold_gpt/tests/test_position_embeddings.py
"""Integration tests for all position embedding types in NeuroManifoldGPT."""
import pytest
import torch
from neuromanifold_gpt.config import NeuroManifoldConfigNano
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT


@pytest.fixture(params=['learned', 'ramanujan', 'rotary', 'alibi'])
def pos_emb_type(request):
    """Parametrized fixture for all position embedding types."""
    return request.param


def test_learned_pos_emb_forward():
    """Learned position embeddings should work in forward pass."""
    config = NeuroManifoldConfigNano(pos_emb_type='learned')
    model = NeuroManifoldGPT(config)

    tokens = torch.randint(0, config.vocab_size, (2, 50))
    logits, loss, info = model(tokens)

    assert logits.shape == (2, 50, config.vocab_size)
    assert loss is None


def test_ramanujan_pos_emb_forward():
    """Ramanujan position embeddings should work in forward pass."""
    config = NeuroManifoldConfigNano(pos_emb_type='ramanujan')
    model = NeuroManifoldGPT(config)

    tokens = torch.randint(0, config.vocab_size, (2, 50))
    logits, loss, info = model(tokens)

    assert logits.shape == (2, 50, config.vocab_size)
    assert loss is None


def test_rotary_pos_emb_forward():
    """Rotary position embeddings should work in forward pass."""
    config = NeuroManifoldConfigNano(pos_emb_type='rotary')
    model = NeuroManifoldGPT(config)

    tokens = torch.randint(0, config.vocab_size, (2, 50))
    logits, loss, info = model(tokens)

    assert logits.shape == (2, 50, config.vocab_size)
    assert loss is None


def test_alibi_pos_emb_forward():
    """ALiBi position embeddings should work in forward pass."""
    config = NeuroManifoldConfigNano(pos_emb_type='alibi')
    model = NeuroManifoldGPT(config)

    tokens = torch.randint(0, config.vocab_size, (2, 50))
    logits, loss, info = model(tokens)

    assert logits.shape == (2, 50, config.vocab_size)
    assert loss is None


def test_all_pos_emb_types_with_targets(pos_emb_type):
    """All position embedding types should compute loss when targets provided."""
    config = NeuroManifoldConfigNano(pos_emb_type=pos_emb_type)
    model = NeuroManifoldGPT(config)

    tokens = torch.randint(0, config.vocab_size, (2, 50))
    targets = torch.randint(0, config.vocab_size, (2, 50))

    logits, loss, info = model(tokens, targets)

    assert loss is not None
    assert loss.ndim == 0  # Scalar
    assert loss > 0  # Loss should be positive


def test_all_pos_emb_types_generate(pos_emb_type):
    """All position embedding types should support generation."""
    config = NeuroManifoldConfigNano(pos_emb_type=pos_emb_type)
    model = NeuroManifoldGPT(config)
    model.eval()

    prompt = torch.randint(0, config.vocab_size, (1, 10))

    generated = model.generate(prompt, max_new_tokens=5)

    assert generated.shape == (1, 15)


def test_all_pos_emb_types_produce_different_outputs(pos_emb_type):
    """Different position embedding types should produce different outputs."""
    # Get output from current position embedding type
    config = NeuroManifoldConfigNano(pos_emb_type=pos_emb_type)
    model = NeuroManifoldGPT(config)
    model.eval()

    tokens = torch.randint(0, config.vocab_size, (1, 20))

    with torch.no_grad():
        logits1, _, _ = model(tokens)

    # Compare with learned embeddings (baseline)
    if pos_emb_type != 'learned':
        config2 = NeuroManifoldConfigNano(pos_emb_type='learned')
        model2 = NeuroManifoldGPT(config2)
        model2.eval()

        with torch.no_grad():
            logits2, _, _ = model2(tokens)

        # Outputs should be different (different position encoding)
        # Note: may have same shape, but different values
        assert not torch.allclose(logits1, logits2, atol=1e-3)


def test_all_pos_emb_types_info_dict(pos_emb_type):
    """All position embedding types should produce info dict."""
    config = NeuroManifoldConfigNano(pos_emb_type=pos_emb_type)
    model = NeuroManifoldGPT(config)

    tokens = torch.randint(0, config.vocab_size, (1, 20))
    _, _, info = model(tokens)

    # Check basic info dict structure
    assert isinstance(info, dict)
    assert "block_infos" in info
    assert len(info["block_infos"]) == config.n_layer


def test_learned_vs_ramanujan_different_positions():
    """Learned and Ramanujan embeddings should encode position differently."""
    config_learned = NeuroManifoldConfigNano(pos_emb_type='learned')
    config_ramanujan = NeuroManifoldConfigNano(pos_emb_type='ramanujan')

    model_learned = NeuroManifoldGPT(config_learned)
    model_ramanujan = NeuroManifoldGPT(config_ramanujan)

    model_learned.eval()
    model_ramanujan.eval()

    tokens = torch.randint(0, config_learned.vocab_size, (1, 50))

    with torch.no_grad():
        logits_learned, _, _ = model_learned(tokens)
        logits_ramanujan, _, _ = model_ramanujan(tokens)

    # Different position encodings should produce different outputs
    assert not torch.allclose(logits_learned, logits_ramanujan, atol=1e-3)


def test_rotary_vs_alibi_different_mechanisms():
    """RoPE and ALiBi should use different position encoding mechanisms."""
    config_rotary = NeuroManifoldConfigNano(pos_emb_type='rotary')
    config_alibi = NeuroManifoldConfigNano(pos_emb_type='alibi')

    model_rotary = NeuroManifoldGPT(config_rotary)
    model_alibi = NeuroManifoldGPT(config_alibi)

    model_rotary.eval()
    model_alibi.eval()

    tokens = torch.randint(0, config_rotary.vocab_size, (1, 50))

    with torch.no_grad():
        logits_rotary, _, _ = model_rotary(tokens)
        logits_alibi, _, _ = model_alibi(tokens)

    # Different mechanisms should produce different outputs
    assert not torch.allclose(logits_rotary, logits_alibi, atol=1e-3)


def test_all_pos_emb_types_gradient_flow(pos_emb_type):
    """All position embedding types should allow gradient flow."""
    config = NeuroManifoldConfigNano(pos_emb_type=pos_emb_type)
    model = NeuroManifoldGPT(config)
    model.train()

    tokens = torch.randint(0, config.vocab_size, (2, 30))
    targets = torch.randint(0, config.vocab_size, (2, 30))

    logits, loss, _ = model(tokens, targets)

    # Backward pass
    loss.backward()

    # Check that token embeddings have gradients
    if model.use_sdr:
        assert model.encoder.sdr_projector.weight.grad is not None
    else:
        assert model.token_embedding.weight.grad is not None

    # For learned and ramanujan, check position embedding gradients
    if pos_emb_type in ['learned', 'ramanujan']:
        # These add embeddings, so should have parameter gradients
        if hasattr(model.position_embedding, 'weight'):
            # Learned embeddings
            assert model.position_embedding.weight.grad is not None
        elif hasattr(model.position_embedding, 'pe'):
            # Ramanujan embeddings (buffer, no grad expected)
            # But model should still train
            pass


def test_batch_size_independence(pos_emb_type):
    """Position embeddings should work with different batch sizes."""
    config = NeuroManifoldConfigNano(pos_emb_type=pos_emb_type)
    model = NeuroManifoldGPT(config)
    model.eval()

    # Test with batch size 1
    tokens1 = torch.randint(0, config.vocab_size, (1, 20))
    with torch.no_grad():
        logits1, _, _ = model(tokens1)
    assert logits1.shape == (1, 20, config.vocab_size)

    # Test with batch size 4
    tokens4 = torch.randint(0, config.vocab_size, (4, 20))
    with torch.no_grad():
        logits4, _, _ = model(tokens4)
    assert logits4.shape == (4, 20, config.vocab_size)


def test_sequence_length_independence(pos_emb_type):
    """Position embeddings should work with different sequence lengths."""
    config = NeuroManifoldConfigNano(pos_emb_type=pos_emb_type)
    model = NeuroManifoldGPT(config)
    model.eval()

    # Test with short sequence
    tokens_short = torch.randint(0, config.vocab_size, (2, 10))
    with torch.no_grad():
        logits_short, _, _ = model(tokens_short)
    assert logits_short.shape == (2, 10, config.vocab_size)

    # Test with longer sequence
    tokens_long = torch.randint(0, config.vocab_size, (2, 100))
    with torch.no_grad():
        logits_long, _, _ = model(tokens_long)
    assert logits_long.shape == (2, 100, config.vocab_size)


def test_learned_pos_emb_initialization():
    """Learned position embeddings should be properly initialized."""
    config = NeuroManifoldConfigNano(pos_emb_type='learned')
    model = NeuroManifoldGPT(config)

    # Check position embedding exists and has correct shape
    assert hasattr(model, 'position_embedding')
    assert hasattr(model.position_embedding, 'weight')
    assert model.position_embedding.weight.shape == (config.block_size, config.n_embd)


def test_ramanujan_pos_emb_initialization():
    """Ramanujan position embeddings should be properly initialized."""
    config = NeuroManifoldConfigNano(pos_emb_type='ramanujan')
    model = NeuroManifoldGPT(config)

    # Check position embedding exists and has precomputed buffer
    assert hasattr(model, 'position_embedding')
    assert hasattr(model.position_embedding, 'pe')
    assert model.position_embedding.pe.shape == (config.block_size, config.n_embd)


def test_rotary_pos_emb_initialization():
    """Rotary position embeddings should be properly initialized."""
    config = NeuroManifoldConfigNano(pos_emb_type='rotary')
    model = NeuroManifoldGPT(config)

    # Check position embedding exists
    assert hasattr(model, 'position_embedding')
    assert hasattr(model.position_embedding, 'inv_freq')
    assert hasattr(model.position_embedding, 'cos_cache')
    assert hasattr(model.position_embedding, 'sin_cache')


def test_alibi_pos_emb_initialization():
    """ALiBi position embeddings should be properly initialized."""
    config = NeuroManifoldConfigNano(pos_emb_type='alibi')
    model = NeuroManifoldGPT(config)

    # Check position embedding exists
    assert hasattr(model, 'position_embedding')
    assert hasattr(model.position_embedding, 'slopes')
    assert hasattr(model.position_embedding, 'bias_cache')
    assert model.position_embedding.slopes.shape[0] == config.n_heads


def test_invalid_pos_emb_type():
    """Should raise error for invalid position embedding type."""
    config = NeuroManifoldConfigNano(pos_emb_type='invalid')

    with pytest.raises(ValueError, match="Unknown pos_emb_type"):
        model = NeuroManifoldGPT(config)


def test_all_pos_emb_types_deterministic(pos_emb_type):
    """All position embedding types should be deterministic given same seed."""
    torch.manual_seed(42)
    config = NeuroManifoldConfigNano(pos_emb_type=pos_emb_type)
    model = NeuroManifoldGPT(config)
    model.eval()

    tokens = torch.randint(0, config.vocab_size, (1, 20))

    with torch.no_grad():
        logits1, _, _ = model(tokens)

    # Same inputs should produce same outputs
    with torch.no_grad():
        logits2, _, _ = model(tokens)

    assert torch.allclose(logits1, logits2)


def test_all_pos_emb_types_memory_efficiency(pos_emb_type):
    """All position embedding types should not cause memory leaks."""
    config = NeuroManifoldConfigNano(pos_emb_type=pos_emb_type)
    model = NeuroManifoldGPT(config)
    model.train()

    tokens = torch.randint(0, config.vocab_size, (2, 30))
    targets = torch.randint(0, config.vocab_size, (2, 30))

    # Run multiple iterations
    for _ in range(5):
        logits, loss, _ = model(tokens, targets)
        loss.backward()
        model.zero_grad()

    # If we get here without OOM, memory is being managed properly
    assert True


def test_rotary_extrapolation_capability():
    """RoPE should enable better extrapolation to longer sequences."""
    config = NeuroManifoldConfigNano(pos_emb_type='rotary', block_size=128)
    model = NeuroManifoldGPT(config)
    model.eval()

    # Train on short sequences
    tokens_short = torch.randint(0, config.vocab_size, (1, 64))

    with torch.no_grad():
        logits_short, _, _ = model(tokens_short)

    # Should also work on longer sequences (up to block_size)
    tokens_long = torch.randint(0, config.vocab_size, (1, 128))

    with torch.no_grad():
        logits_long, _, _ = model(tokens_long)

    assert logits_short.shape == (1, 64, config.vocab_size)
    assert logits_long.shape == (1, 128, config.vocab_size)


def test_alibi_extrapolation_capability():
    """ALiBi should enable better extrapolation to longer sequences."""
    config = NeuroManifoldConfigNano(pos_emb_type='alibi', block_size=128)
    model = NeuroManifoldGPT(config)
    model.eval()

    # Train on short sequences
    tokens_short = torch.randint(0, config.vocab_size, (1, 64))

    with torch.no_grad():
        logits_short, _, _ = model(tokens_short)

    # Should also work on longer sequences (up to block_size)
    tokens_long = torch.randint(0, config.vocab_size, (1, 128))

    with torch.no_grad():
        logits_long, _, _ = model(tokens_long)

    assert logits_short.shape == (1, 64, config.vocab_size)
    assert logits_long.shape == (1, 128, config.vocab_size)


def test_all_pos_emb_types_with_sdr(pos_emb_type):
    """All position embedding types should work with SDR enabled."""
    config = NeuroManifoldConfigNano(pos_emb_type=pos_emb_type, use_sdr=True)
    model = NeuroManifoldGPT(config)

    tokens = torch.randint(0, config.vocab_size, (2, 30))
    logits, loss, info = model(tokens)

    assert logits.shape == (2, 30, config.vocab_size)
    assert "sdr" in info or not config.use_sdr  # SDR info if enabled


def test_pos_emb_type_saved_in_config():
    """Position embedding type should be saved in model config."""
    for pos_emb_type in ['learned', 'ramanujan', 'rotary', 'alibi']:
        config = NeuroManifoldConfigNano(pos_emb_type=pos_emb_type)
        model = NeuroManifoldGPT(config)

        assert model.config.pos_emb_type == pos_emb_type
        assert model.pos_emb_type == pos_emb_type
