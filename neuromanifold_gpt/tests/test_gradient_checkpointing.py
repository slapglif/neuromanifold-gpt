# neuromanifold_gpt/tests/test_gradient_checkpointing.py
"""Tests for gradient checkpointing in GPT models."""
import pytest
import torch
from neuromanifold_gpt.config import NeuroManifoldConfigNano
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT
from model import GPT, GPTConfig


def test_neuromanifold_gpt_with_gradient_checkpointing(nano_config, tokens_2x20):
    """NeuroManifoldGPT should work with gradient_checkpointing=True."""
    nano_config.gradient_checkpointing = True
    model = NeuroManifoldGPT(nano_config)
    model.train()

    targets = torch.randint(0, nano_config.vocab_size, (2, 20))

    logits, loss, info = model(tokens_2x20, targets)

    assert logits.shape == (2, 20, nano_config.vocab_size)
    assert loss is not None
    assert loss.ndim == 0  # Scalar


def test_standard_gpt_with_gradient_checkpointing():
    """Standard GPT should work with gradient_checkpointing=True."""
    config = GPTConfig(
        n_layer=2,
        n_head=2,
        n_embd=64,
        block_size=32,
        vocab_size=100,
        gradient_checkpointing=True
    )
    model = GPT(config)
    model.train()

    tokens = torch.randint(0, 100, (2, 16))
    targets = torch.randint(0, 100, (2, 16))

    logits, loss = model(tokens, targets)

    assert logits.shape == (2, 16, 100)
    assert loss is not None
    assert loss.ndim == 0  # Scalar


def test_gradient_checkpointing_output_consistency(nano_config, tokens_2x20):
    """Output should be consistent with and without gradient checkpointing."""
    nano_config.n_layer = 2
    nano_config.use_sdr = False  # Disable SDR for deterministic comparison

    # Model without checkpointing
    model_no_ckpt = NeuroManifoldGPT(nano_config)
    model_no_ckpt.train()

    # Model with checkpointing
    config_ckpt = NeuroManifoldConfigNano()
    config_ckpt.n_layer = 2
    config_ckpt.use_sdr = False
    config_ckpt.gradient_checkpointing = True
    model_with_ckpt = NeuroManifoldGPT(config_ckpt)
    model_with_ckpt.train()

    # Copy weights to ensure same starting point
    model_with_ckpt.load_state_dict(model_no_ckpt.state_dict())

    # Test forward pass
    with torch.no_grad():
        logits_no_ckpt, _, _ = model_no_ckpt(tokens_2x20)
        logits_with_ckpt, _, _ = model_with_ckpt(tokens_2x20)

    # Outputs should have same shape
    assert logits_no_ckpt.shape == logits_with_ckpt.shape
    # Outputs should be very close (same model, same weights, same input)
    assert torch.allclose(logits_no_ckpt, logits_with_ckpt, atol=1e-5)


def test_gradient_checkpointing_backward_pass(nano_config, tokens_2x20):
    """Gradients should be computed correctly with gradient checkpointing."""
    nano_config.gradient_checkpointing = True
    nano_config.n_layer = 2
    model = NeuroManifoldGPT(nano_config)
    model.train()

    targets = torch.randint(0, nano_config.vocab_size, (2, 20))

    logits, loss, _ = model(tokens_2x20, targets)

    # Backward pass should work without errors
    loss.backward()

    # Check that gradients exist for model parameters that are used in computation
    grad_count = 0
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            assert not torch.isnan(param.grad).any()
            grad_count += 1

    # Should have gradients for multiple parameters
    assert grad_count > 0


def test_gradient_checkpointing_only_in_training_mode(nano_config, tokens_1x10):
    """Gradient checkpointing should only be used in training mode."""
    nano_config.gradient_checkpointing = True
    model = NeuroManifoldGPT(nano_config)

    # In eval mode, should work without checkpointing
    model.eval()
    with torch.no_grad():
        logits_eval, _, _ = model(tokens_1x10)
        assert logits_eval.shape == (1, 10, nano_config.vocab_size)

    # In train mode, should use checkpointing
    model.train()
    logits_train, _, _ = model(tokens_1x10)
    assert logits_train.shape == (1, 10, nano_config.vocab_size)


def test_gradient_checkpointing_with_sdr_mode(nano_config):
    """Gradient checkpointing should work with SDR mode enabled."""
    nano_config.gradient_checkpointing = True
    nano_config.use_sdr = True
    nano_config.n_layer = 2
    model = NeuroManifoldGPT(nano_config)
    model.train()

    tokens = torch.randint(0, nano_config.vocab_size, (2, 15))
    targets = torch.randint(0, nano_config.vocab_size, (2, 15))

    logits, loss, info = model(tokens, targets)

    assert logits.shape == (2, 15, nano_config.vocab_size)
    assert loss is not None
    # Verify SDR info is present
    assert "sdr" in info
    assert "sdr_scores" in info


def test_gradient_checkpointing_with_dense_mode(nano_config):
    """Gradient checkpointing should work with dense (non-SDR) mode."""
    nano_config.gradient_checkpointing = True
    nano_config.use_sdr = False
    nano_config.n_layer = 2
    model = NeuroManifoldGPT(nano_config)
    model.train()

    tokens = torch.randint(0, nano_config.vocab_size, (2, 15))
    targets = torch.randint(0, nano_config.vocab_size, (2, 15))

    logits, loss, info = model(tokens, targets)

    assert logits.shape == (2, 15, nano_config.vocab_size)
    assert loss is not None


def test_gradient_flow_with_checkpointing(nano_config, tokens_1x10):
    """Gradients should flow properly through checkpointed blocks."""
    nano_config.gradient_checkpointing = True
    nano_config.n_layer = 3
    nano_config.use_sdr = False
    model = NeuroManifoldGPT(nano_config)
    model.train()

    # tokens_1x10 is (1, 10), but we need (2, 10)
    tokens = torch.randint(0, nano_config.vocab_size, (2, 10))
    targets = torch.randint(0, nano_config.vocab_size, (2, 10))

    # Forward and backward pass
    logits, loss, _ = model(tokens, targets)
    loss.backward()

    # Check gradients exist and are non-zero for block parameters
    blocks_with_gradients = 0
    for block in model.blocks:
        block_has_grad = False
        for param in block.parameters():
            if param.requires_grad and param.grad is not None:
                # Check gradient is not all zeros
                if param.grad.abs().sum() > 0:
                    block_has_grad = True
                    break
        if block_has_grad:
            blocks_with_gradients += 1

    # All blocks should have non-zero gradients
    assert blocks_with_gradients == nano_config.n_layer


def test_standard_gpt_gradient_flow():
    """Standard GPT gradients should flow properly with checkpointing."""
    config = GPTConfig(
        n_layer=3,
        n_head=2,
        n_embd=64,
        block_size=32,
        vocab_size=100,
        gradient_checkpointing=True
    )
    model = GPT(config)
    model.train()

    tokens = torch.randint(0, 100, (2, 16))
    targets = torch.randint(0, 100, (2, 16))

    # Forward and backward pass
    logits, loss = model(tokens, targets)
    loss.backward()

    # Check that transformer blocks have gradients
    blocks_with_gradients = 0
    for block in model.transformer.h:
        block_has_grad = False
        for param in block.parameters():
            if param.requires_grad and param.grad is not None:
                if param.grad.abs().sum() > 0:
                    block_has_grad = True
                    break
        if block_has_grad:
            blocks_with_gradients += 1

    # All blocks should have gradients
    assert blocks_with_gradients == config.n_layer


def test_gradient_checkpointing_disabled_by_default():
    """Gradient checkpointing should be disabled by default."""
    # NeuroManifoldGPT
    config_nm = NeuroManifoldConfigNano()
    assert config_nm.gradient_checkpointing is False

    # Standard GPT
    config_gpt = GPTConfig()
    assert config_gpt.gradient_checkpointing is False


def test_neuromanifold_generate_with_checkpointing(nano_config, tokens_1x10):
    """Generate should work correctly with gradient checkpointing enabled."""
    nano_config.gradient_checkpointing = True
    model = NeuroManifoldGPT(nano_config)
    model.eval()

    with torch.no_grad():
        generated = model.generate(tokens_1x10, max_new_tokens=5)

    assert generated.shape == (1, 15)
    # Verify prompt is unchanged
    assert torch.equal(generated[:, :10], tokens_1x10)


def test_gradient_checkpointing_batch_sizes(nano_config, tokens_1x10):
    """Gradient checkpointing should work with different batch sizes."""
    nano_config.gradient_checkpointing = True
    nano_config.n_layer = 2
    model = NeuroManifoldGPT(nano_config)
    model.train()

    # Test batch size 1
    logits_1, _, _ = model(tokens_1x10)
    assert logits_1.shape == (1, 10, nano_config.vocab_size)

    # Test batch size 4
    tokens_4 = torch.randint(0, nano_config.vocab_size, (4, 10))
    logits_4, _, _ = model(tokens_4)
    assert logits_4.shape == (4, 10, nano_config.vocab_size)

    # Test batch size 8
    tokens_8 = torch.randint(0, nano_config.vocab_size, (8, 10))
    logits_8, _, _ = model(tokens_8)
    assert logits_8.shape == (8, 10, nano_config.vocab_size)


def test_gradient_checkpointing_sequence_lengths(nano_config, tokens_2x50):
    """Gradient checkpointing should work with different sequence lengths."""
    nano_config.gradient_checkpointing = True
    nano_config.n_layer = 2
    model = NeuroManifoldGPT(nano_config)
    model.train()

    # Test short sequence
    tokens_short = torch.randint(0, nano_config.vocab_size, (2, 5))
    logits_short, _, _ = model(tokens_short)
    assert logits_short.shape == (2, 5, nano_config.vocab_size)

    # Test medium sequence
    logits_med, _, _ = model(tokens_2x50)
    assert logits_med.shape == (2, 50, nano_config.vocab_size)

    # Test long sequence (up to block_size)
    seq_len = min(100, nano_config.block_size)
    tokens_long = torch.randint(0, nano_config.vocab_size, (2, seq_len))
    logits_long, _, _ = model(tokens_long)
    assert logits_long.shape == (2, seq_len, nano_config.vocab_size)
