"""Test topographic loss implementation."""

import torch

from neuromanifold_gpt.model.semantic_folding import SemanticFoldingEncoder


def test_topographic_loss_basic():
    vocab_size = 100
    batch_size = 2
    seq_len = 8

    encoder = SemanticFoldingEncoder(
        vocab_size=vocab_size,
        sdr_size=2048,
        n_active=40,
        embed_dim=256,
    )
    encoder.train()

    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))

    sdr, scores, topo_loss = encoder(tokens)

    print(f"SDR shape: {sdr.shape}")
    print(f"Scores shape: {scores.shape}")
    print(f"Topographic loss: {topo_loss.item():.4f}")

    assert sdr.shape == (batch_size, seq_len, 2048)
    assert scores.shape == (batch_size, seq_len, 2048)
    assert topo_loss.ndim == 0
    assert topo_loss.item() >= 0.0

    print("✓ Basic topographic loss test passed")


def test_topographic_loss_gradient():
    vocab_size = 50
    batch_size = 2
    seq_len = 4

    encoder = SemanticFoldingEncoder(
        vocab_size=vocab_size,
        sdr_size=2048,
        n_active=40,
        embed_dim=128,
    )
    encoder.train()

    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))

    sdr, scores, topo_loss = encoder(tokens)

    topo_loss.backward()

    has_gradients = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters()
    )

    print(f"Gradients present: {has_gradients}")
    print(f"Loss value: {topo_loss.item():.4f}")

    assert has_gradients, "No gradients computed!"
    print("✓ Gradient flow test passed")


def test_topographic_loss_inference():
    vocab_size = 50
    batch_size = 1
    seq_len = 4

    encoder = SemanticFoldingEncoder(
        vocab_size=vocab_size,
        sdr_size=2048,
        n_active=40,
    )
    encoder.eval()

    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        sdr, scores, topo_loss = encoder(tokens)

    print(f"Inference loss: {topo_loss.item():.4f}")
    assert topo_loss.item() == 0.0, "Loss should be zero during inference"
    print("✓ Inference mode test passed")


def test_full_model_integration():
    from neuromanifold_gpt.config import NeuroManifoldConfig
    from neuromanifold_gpt.model.gpt import NeuroManifoldGPT

    config = NeuroManifoldConfig(
        vocab_size=100,
        block_size=16,
        n_layer=2,
        n_heads=4,
        n_embd=128,
        use_sdr=True,
        sdr_size=2048,
        sdr_sparsity=0.02,
    )

    model = NeuroManifoldGPT(config)
    model.train()

    batch_size = 2
    seq_len = 8
    tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, loss, info = model(tokens, targets)

    print(f"Total loss: {loss.item():.4f}")
    print(f"Topographic loss: {info['topographic_loss'].item():.4f}")
    print(f"Ortho loss: {info['ortho_loss'].item():.4f}")

    assert "topographic_loss" in info
    assert info["topographic_loss"].item() >= 0.0

    loss.backward()

    print("✓ Full model integration test passed")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Topographic Loss Implementation")
    print("=" * 60)

    test_topographic_loss_basic()
    print()

    test_topographic_loss_gradient()
    print()

    test_topographic_loss_inference()
    print()

    test_full_model_integration()
    print()

    print("=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
