# neuromanifold_gpt/tests/test_hybrid_reasoning.py
"""Tests for HybridReasoningModule and ThinkingLayer."""
import pytest
import torch
from neuromanifold_gpt.model.hybrid_reasoning import (
    HybridReasoningModule,
    ThinkingLayer,
)


# ThinkingLayer Tests
def test_thinking_layer_forward_shape():
    """ThinkingLayer should preserve input shape."""
    layer = ThinkingLayer(embed_dim=384, n_heads=8)
    x = torch.randn(2, 20, 384)

    out = layer(x)

    assert out.shape == (2, 20, 384)


def test_thinking_layer_gradient_flow():
    """Gradients should flow through ThinkingLayer."""
    layer = ThinkingLayer(embed_dim=384, n_heads=8)
    x = torch.randn(2, 20, 384, requires_grad=True)

    out = layer(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_thinking_layer_with_dropout():
    """ThinkingLayer should work with dropout enabled."""
    layer = ThinkingLayer(embed_dim=384, n_heads=8, dropout=0.1)
    layer.train()
    x = torch.randn(2, 20, 384)

    out = layer(x)

    assert out.shape == (2, 20, 384)


# HybridReasoningModule Tests
def test_hybrid_reasoning_forward_shape():
    """HybridReasoningModule should preserve input shape."""
    module = HybridReasoningModule(
        embed_dim=384,
        n_thinking_layers=2,
        n_heads=8,
    )
    x = torch.randn(2, 20, 384)

    out, info = module(x)

    assert out.shape == (2, 20, 384)
    assert 'thinking_probs' in info
    assert 'mode_selections' in info


def test_hybrid_reasoning_gradient_flow():
    """Gradients should flow through HybridReasoningModule."""
    module = HybridReasoningModule(
        embed_dim=384,
        n_thinking_layers=2,
        n_heads=8,
    )
    x = torch.randn(2, 20, 384, requires_grad=True)

    out, _ = module(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_hybrid_reasoning_thinking_probs_range():
    """Thinking probabilities should be in [0, 1] range."""
    module = HybridReasoningModule(embed_dim=384)
    x = torch.randn(4, 20, 384)

    _, info = module(x)
    thinking_probs = info['thinking_probs']

    assert thinking_probs.shape == (4,)
    assert (thinking_probs >= 0).all()
    assert (thinking_probs <= 1).all()


def test_hybrid_reasoning_mode_selections_binary():
    """Mode selections should be binary (0 or 1)."""
    module = HybridReasoningModule(embed_dim=384)
    module.eval()
    x = torch.randn(4, 20, 384)

    _, info = module(x)
    mode_selections = info['mode_selections']

    assert mode_selections.shape == (4,)
    assert ((mode_selections == 0) | (mode_selections == 1)).all()


def test_hybrid_reasoning_with_e7_prior():
    """Should accept and use E7 curriculum tier."""
    module = HybridReasoningModule(
        embed_dim=384,
        use_e7_prior=True,
    )
    x = torch.randn(2, 20, 384)
    e7_tier = torch.tensor([0, 6])  # Min and max tiers

    out, info = module(x, e7_tier=e7_tier)

    assert out.shape == (2, 20, 384)
    assert 'thinking_probs' in info


def test_hybrid_reasoning_without_e7_prior():
    """Should work without E7 prior."""
    module = HybridReasoningModule(
        embed_dim=384,
        use_e7_prior=False,
    )
    x = torch.randn(2, 20, 384)

    out, info = module(x)

    assert out.shape == (2, 20, 384)


def test_hybrid_reasoning_e7_default_when_none():
    """Should use default E7 tier when enabled but tier not provided."""
    module = HybridReasoningModule(
        embed_dim=384,
        use_e7_prior=True,
    )
    x = torch.randn(2, 20, 384)

    out, info = module(x, e7_tier=None)

    assert out.shape == (2, 20, 384)
    assert 'thinking_probs' in info


def test_hybrid_reasoning_training_mode():
    """Should use probabilistic selection in training mode."""
    module = HybridReasoningModule(embed_dim=384)
    module.train()
    x = torch.randn(4, 20, 384)

    out, info = module(x)
    mode_selections = info['mode_selections']

    # In training mode, selections should be from Gumbel-Softmax
    assert mode_selections.shape == (4,)
    assert out.shape == (4, 20, 384)


def test_hybrid_reasoning_eval_mode_threshold():
    """Should use threshold-based selection in eval mode."""
    module = HybridReasoningModule(
        embed_dim=384,
        thinking_threshold=0.5,
    )
    module.eval()
    x = torch.randn(4, 20, 384)

    _, info = module(x)
    thinking_probs = info['thinking_probs']
    mode_selections = info['mode_selections']

    # Mode selections should match threshold comparison
    expected_selections = (thinking_probs > 0.5).float()
    assert torch.allclose(mode_selections, expected_selections)


def test_hybrid_reasoning_custom_threshold():
    """Should respect custom thinking threshold."""
    module = HybridReasoningModule(
        embed_dim=384,
        thinking_threshold=0.8,
    )
    module.eval()
    x = torch.randn(4, 20, 384)

    _, info = module(x)
    thinking_probs = info['thinking_probs']
    mode_selections = info['mode_selections']

    # Mode selections should match custom threshold
    expected_selections = (thinking_probs > 0.8).float()
    assert torch.allclose(mode_selections, expected_selections)


def test_hybrid_reasoning_multiple_thinking_layers():
    """Should work with different numbers of thinking layers."""
    for n_layers in [1, 2, 4]:
        module = HybridReasoningModule(
            embed_dim=384,
            n_thinking_layers=n_layers,
        )
        x = torch.randn(2, 20, 384)

        out, info = module(x)

        assert out.shape == (2, 20, 384)
        assert len(module.thinking_layers) == n_layers


def test_hybrid_reasoning_batch_consistency():
    """Different batch items should get independent routing."""
    module = HybridReasoningModule(embed_dim=384)
    module.eval()

    # Create batch with very different inputs
    x1 = torch.randn(1, 20, 384) * 10  # High variance
    x2 = torch.randn(1, 20, 384) * 0.1  # Low variance
    x = torch.cat([x1, x2], dim=0)

    _, info = module(x)
    thinking_probs = info['thinking_probs']

    # Both items should get predictions
    assert thinking_probs.shape == (2,)


def test_hybrid_reasoning_complexity_features():
    """Internal complexity features should be computed correctly."""
    module = HybridReasoningModule(embed_dim=384)
    x = torch.randn(2, 20, 384)

    features = module._compute_complexity_features(x)

    # Should concatenate mean, max, std
    assert features.shape == (2, 384 * 3)


def test_hybrid_reasoning_router_output_stability():
    """Router should produce stable outputs for same input."""
    module = HybridReasoningModule(embed_dim=384)
    module.eval()
    x = torch.randn(2, 20, 384)

    _, info1 = module(x)
    _, info2 = module(x)

    # Same input should give same predictions in eval mode
    assert torch.allclose(info1['thinking_probs'], info2['thinking_probs'])
    assert torch.allclose(info1['mode_selections'], info2['mode_selections'])


def test_hybrid_reasoning_fast_path_when_disabled():
    """When all samples select fast path, output should match input pattern."""
    module = HybridReasoningModule(
        embed_dim=384,
        thinking_threshold=0.99,  # Very high threshold
    )
    module.eval()
    x = torch.randn(2, 20, 384)

    out, info = module(x)

    # If router learns to output low probabilities, fast path is used
    # Output should still be valid
    assert out.shape == (2, 20, 384)
    assert not torch.isnan(out).any()


def test_hybrid_reasoning_with_dropout():
    """Should work with dropout enabled."""
    module = HybridReasoningModule(
        embed_dim=384,
        dropout=0.1,
    )
    module.train()
    x = torch.randn(2, 20, 384)

    out, info = module(x)

    assert out.shape == (2, 20, 384)
    assert not torch.isnan(out).any()
