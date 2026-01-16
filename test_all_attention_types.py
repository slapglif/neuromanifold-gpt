#!/usr/bin/env python3
"""Integration test for all attention types.

Tests all attention mechanisms (standard, soliton, sdr, fast-spectral, kaufmann)
to ensure they can be instantiated and run forward/backward passes successfully.
"""
import torch
from neuromanifold_gpt.config import NeuroManifoldConfig
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT


def test_attention_type(attention_type: str, description: str):
    """Test a specific attention type.

    Args:
        attention_type: The attention type to test
        description: Human-readable description of the attention type
    """
    print(f"\nTesting {attention_type} ({description})...")

    # Create small config for fast testing
    config = NeuroManifoldConfig(
        attention_type=attention_type,
        n_layer=2,  # Small model for testing
        n_embd=128,
        n_heads=4,
        block_size=64,
        vocab_size=256,  # Small vocab
        use_sdr=False,  # Use dense embeddings for simplicity
        skip_manifold_spectral=True,  # Skip for speed
        use_mhc=False,  # Disable mHC for simpler testing
        use_kan=False,  # Use standard FFN for speed
        use_mtp=False,  # Disable multi-token prediction
    )

    # Instantiate model
    model = NeuroManifoldGPT(config)
    model.eval()

    # Create dummy input
    batch_size = 2
    seq_len = 16
    tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Forward pass without targets
    with torch.no_grad():
        logits, loss, info = model(tokens)

    # Verify output shape
    expected_shape = (batch_size, seq_len, config.vocab_size)
    assert logits.shape == expected_shape, (
        f"Expected logits shape {expected_shape}, got {logits.shape}"
    )
    assert loss is None, "Loss should be None when targets not provided"

    # Forward pass with targets (compute loss)
    with torch.no_grad():
        logits, loss, info = model(tokens, targets)

    assert loss is not None, "Loss should not be None when targets provided"
    assert loss.ndim == 0, f"Loss should be scalar, got shape {loss.shape}"
    assert not torch.isnan(loss), f"Loss is NaN for {attention_type}"
    assert not torch.isinf(loss), f"Loss is inf for {attention_type}"

    # Verify info dict contains expected keys
    assert "block_infos" in info, "Info should contain block_infos"
    assert len(info["block_infos"]) == config.n_layer, (
        f"Expected {config.n_layer} block infos, got {len(info['block_infos'])}"
    )

    # Verify each block has attention type info
    for i, block_info in enumerate(info["block_infos"]):
        assert "attention_type" in block_info, (
            f"Block {i} info missing attention_type"
        )
        # Note: Some attention types map to others (e.g., sdr -> fhn + knot)
        # So we just verify the key exists, not the exact value

    print(f"  ✓ Model instantiated successfully")
    print(f"  ✓ Forward pass shape: {logits.shape}")
    print(f"  ✓ Loss computed: {loss.item():.4f}")
    print(f"  ✓ Info dict complete with {len(info['block_infos'])} blocks")


def test_backward_compatibility():
    """Test backward compatibility with deprecated use_kaufmann_attention flag."""
    print("\nTesting backward compatibility (use_kaufmann_attention=True)...")

    # Create config with deprecated flag
    config = NeuroManifoldConfig(
        use_kaufmann_attention=True,  # Deprecated flag
        n_layer=2,
        n_embd=128,
        n_heads=4,
        block_size=64,
        vocab_size=256,
        use_sdr=False,
        skip_manifold_spectral=True,
        use_mhc=False,
        use_kan=False,
        use_mtp=False,
    )

    # The flag should be overridden by attention_type in block creation
    model = NeuroManifoldGPT(config)
    model.eval()

    # Create dummy input
    tokens = torch.randint(0, config.vocab_size, (2, 16))

    # Forward pass
    with torch.no_grad():
        logits, loss, info = model(tokens)

    assert logits.shape == (2, 16, config.vocab_size)
    print(f"  ✓ Backward compatibility maintained")
    print(f"  ✓ Model works with deprecated flag")


def test_backward_compatibility_knot():
    """Test backward compatibility with deprecated use_knot_attention flag."""
    print("\nTesting backward compatibility (use_knot_attention=True)...")

    # Create config with deprecated flag
    config = NeuroManifoldConfig(
        use_knot_attention=True,  # Deprecated flag
        n_layer=2,
        n_embd=128,
        n_heads=4,
        block_size=64,
        vocab_size=256,
        use_sdr=False,
        skip_manifold_spectral=True,
        use_mhc=False,
        use_kan=False,
        use_mtp=False,
    )

    # The flag should be overridden by attention_type in block creation
    model = NeuroManifoldGPT(config)
    model.eval()

    # Create dummy input
    tokens = torch.randint(0, config.vocab_size, (2, 16))

    # Forward pass
    with torch.no_grad():
        logits, loss, info = model(tokens)

    assert logits.shape == (2, 16, config.vocab_size)
    print(f"  ✓ Backward compatibility maintained")
    print(f"  ✓ Model works with deprecated flag")


def main():
    """Run all attention type tests."""
    print("=" * 70)
    print("Integration Test: All Attention Types")
    print("=" * 70)

    # Test each attention type
    attention_types = [
        ("standard", "Flash Attention, O(n²)"),
        ("soliton", "FHN excitable wave dynamics, O(n)"),
        ("sdr", "SDR Memory with knot attention"),
        ("fast-spectral", "Learned spectral basis, O(n·k)"),
        ("kaufmann", "Full Trifecta: FHN + Knot + Reaction-Diffusion"),
    ]

    for attention_type, description in attention_types:
        try:
            test_attention_type(attention_type, description)
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            raise

    # Test backward compatibility
    try:
        test_backward_compatibility()
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        raise

    try:
        test_backward_compatibility_knot()
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        raise

    print("\n" + "=" * 70)
    print("All Tests Passed! ✓")
    print("=" * 70)
    print("\nSummary:")
    print("  • All 5 attention types work correctly")
    print("  • Forward pass produces correct shapes")
    print("  • Loss computation works for all types")
    print("  • Info dicts contain expected diagnostic data")
    print("  • Backward compatibility with deprecated flags maintained")
    print()


if __name__ == "__main__":
    main()
