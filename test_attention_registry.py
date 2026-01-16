#!/usr/bin/env python3
"""Test attention_type registry pattern."""

import torch
from neuromanifold_gpt.config.block_config import NeuroManifoldBlockConfig
from neuromanifold_gpt.model.block import NeuroManifoldBlock

def test_attention_types():
    """Test all three attention types."""

    # Test FHN attention
    print("Testing FHN attention...")
    config_fhn = NeuroManifoldBlockConfig(
        sdr_size=128,
        embed_dim=128,
        attention_type="fhn"
    )
    block_fhn = NeuroManifoldBlock(config_fhn)
    print("✓ FHN attention initialized successfully")

    # Test Knot attention
    print("Testing Knot attention...")
    config_knot = NeuroManifoldBlockConfig(
        sdr_size=128,
        embed_dim=128,
        attention_type="knot"
    )
    block_knot = NeuroManifoldBlock(config_knot)
    print("✓ Knot attention initialized successfully")

    # Test Kaufmann attention
    print("Testing Kaufmann attention...")
    config_kaufmann = NeuroManifoldBlockConfig(
        sdr_size=128,
        embed_dim=128,
        attention_type="kaufmann"
    )
    block_kaufmann = NeuroManifoldBlock(config_kaufmann)
    print("✓ Kaufmann attention initialized successfully")

    # Test invalid attention type
    print("Testing invalid attention type...")
    try:
        config_invalid = NeuroManifoldBlockConfig(
            sdr_size=128,
            embed_dim=128,
            attention_type="invalid"
        )
        block_invalid = NeuroManifoldBlock(config_invalid)
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")

    # Test forward pass with FHN
    print("\nTesting forward pass...")
    sdr = torch.randn(2, 10, 128)  # (B=2, T=10, sdr_size=128)
    output, info = block_fhn(sdr)
    assert output.shape == (2, 10, 128), f"Expected shape (2, 10, 128), got {output.shape}"
    print(f"✓ Forward pass successful, output shape: {output.shape}")

    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)

if __name__ == "__main__":
    test_attention_types()
