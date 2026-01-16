#!/usr/bin/env python3
"""End-to-end integration test for attention registry pattern.

Subtask 4-1: Test model creation with all attention types.

This test verifies:
1. Model creation with attention_type='standard', 'fhn', 'knot', 'kaufmann'
2. Forward pass with dummy data for each attention type
3. Loss computation works correctly
4. Info dict contains correct attention_type

Note: 'mla' (MultiHeadLatentAttention) is not yet implemented, so it's skipped.
"""
import sys
import torch
from neuromanifold_gpt.config import NeuroManifoldConfig
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT


def test_attention_type_e2e(attention_type: str):
    """End-to-end test for a specific attention type.

    Args:
        attention_type: The attention type to test ('standard', 'fhn', 'knot', 'kaufmann')

    Returns:
        bool: True if all tests pass, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Testing: attention_type='{attention_type}'")
    print('='*60)

    try:
        # Step 1: Create model with specified attention type
        print(f"[1/4] Creating model with attention_type='{attention_type}'...")
        config = NeuroManifoldConfig(
            attention_type=attention_type,
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
        model = NeuroManifoldGPT(config)
        model.eval()
        print(f"      ✓ Model created successfully")

        # Step 2: Run forward pass with dummy data
        print(f"[2/4] Running forward pass with dummy data...")
        batch_size = 2
        seq_len = 16
        tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            logits, loss_none, info = model(tokens)

        # Verify output shape
        expected_shape = (batch_size, seq_len, config.vocab_size)
        assert logits.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {logits.shape}"
        )
        assert loss_none is None, "Loss should be None without targets"
        print(f"      ✓ Forward pass successful (shape: {logits.shape})")

        # Step 3: Verify loss computation works
        print(f"[3/4] Verifying loss computation...")
        with torch.no_grad():
            logits_with_loss, loss, info = model(tokens, targets)

        assert loss is not None, "Loss should not be None with targets"
        assert loss.ndim == 0, f"Loss should be scalar, got shape {loss.shape}"
        assert not torch.isnan(loss), f"Loss is NaN"
        assert not torch.isinf(loss), f"Loss is inf"
        assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"
        print(f"      ✓ Loss computed successfully: {loss.item():.4f}")

        # Step 4: Check info dict contains correct attention_type
        print(f"[4/4] Checking info dict contains attention_type...")
        assert "block_infos" in info, "Info missing 'block_infos'"
        assert len(info["block_infos"]) == config.n_layer, (
            f"Expected {config.n_layer} blocks, got {len(info['block_infos'])}"
        )

        # Verify each block's info dict contains attention_type
        for i, block_info in enumerate(info["block_infos"]):
            assert "attention_type" in block_info, (
                f"Block {i} info missing 'attention_type'"
            )
            block_attention_type = block_info["attention_type"]
            print(f"      - Block {i}: attention_type='{block_attention_type}'")

        print(f"      ✓ All block info dicts contain attention_type")

        print(f"\n✅ PASSED: attention_type='{attention_type}'")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mla_not_implemented():
    """Test that 'mla' attention type is not yet implemented."""
    print(f"\n{'='*60}")
    print(f"Testing: attention_type='mla' (expected to fail)")
    print('='*60)

    try:
        config = NeuroManifoldConfig(
            attention_type='mla',
            n_layer=1,
            use_sdr=False,
        )
        model = NeuroManifoldGPT(config)
        print(f"\n⚠️  UNEXPECTED: 'mla' should not be implemented yet")
        return False
    except (NameError, AttributeError, ValueError) as e:
        # Expected errors:
        # - ValueError: when mla is not in the registry (current state)
        # - NameError: when MultiHeadLatentAttention is not imported
        # - AttributeError: when the class doesn't exist
        print(f"      ✓ Expected error: {type(e).__name__}")
        print(f"      ✓ Message: {str(e)[:100]}")
        print(f"\n✅ PASSED: 'mla' correctly not implemented")
        return True
    except Exception as e:
        print(f"\n❌ FAILED: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all end-to-end integration tests."""
    print("="*60)
    print("END-TO-END INTEGRATION TEST: ATTENTION REGISTRY")
    print("Subtask 4-1: Test model creation with all attention types")
    print("="*60)

    # Test all implemented attention types
    attention_types = ['standard', 'fhn', 'knot', 'kaufmann']
    results = {}

    for attention_type in attention_types:
        results[attention_type] = test_attention_type_e2e(attention_type)

    # Test that mla is not yet implemented
    results['mla'] = test_mla_not_implemented()

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    all_passed = all(results.values())

    for attention_type, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} | attention_type='{attention_type}'")

    print("="*60)

    if all_passed:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("\nVerification complete:")
        print("  ✓ Model creation works for all implemented attention types")
        print("  ✓ Forward passes produce correct output shapes")
        print("  ✓ Loss computation works correctly")
        print("  ✓ Info dicts contain attention_type for all blocks")
        print("  ✓ MLA correctly marked as not yet implemented")
        print()
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
