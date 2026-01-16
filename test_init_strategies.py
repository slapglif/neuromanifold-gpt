#!/usr/bin/env python3
"""Test script for initialization strategies.

This verifies that all initialization strategies work correctly.
Run this after installing dependencies: pip install -e .
"""

import torch
from neuromanifold_gpt.config.base import NeuroManifoldConfig
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT


def test_strategy(strategy_name, **kwargs):
    """Test a specific initialization strategy."""
    print(f"\nTesting {strategy_name} initialization...")

    cfg = NeuroManifoldConfig(
        vocab_size=100,
        block_size=64,
        n_layer=2,
        n_heads=2,
        n_embd=64,
        init_strategy=strategy_name,
        **kwargs
    )

    model = NeuroManifoldGPT(cfg)

    # Check for NaN or Inf in weights
    has_nan = False
    has_inf = False
    weight_stds = []

    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            has_nan = True
            print(f"  WARNING: NaN found in {name}")
        if torch.isinf(param).any():
            has_inf = True
            print(f"  WARNING: Inf found in {name}")

        std = param.std().item()
        weight_stds.append((name, std))

    if not has_nan and not has_inf:
        print(f"  ✓ No NaN or Inf values")

    # Print some weight statistics
    print(f"  Weight std statistics:")
    for name, std in weight_stds[:5]:  # Show first 5
        print(f"    {name}: {std:.6f}")

    return not (has_nan or has_inf)


def main():
    """Test all initialization strategies."""
    print("=" * 60)
    print("Testing Weight Initialization Strategies")
    print("=" * 60)

    results = {}

    # Test DeepSeek (default)
    results['deepseek'] = test_strategy('deepseek')

    # Test GPT-2
    results['gpt2'] = test_strategy('gpt2')

    # Test GPT-2 scaled
    results['gpt2_scaled'] = test_strategy('gpt2_scaled')

    # Test muP with different base widths
    results['mup_32'] = test_strategy('mup', mup_base_width=32)
    results['mup_128'] = test_strategy('mup', mup_base_width=128)

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\n✓ All initialization strategies working correctly!")
        print("OK")
    else:
        print("\n✗ Some strategies failed")
        exit(1)


if __name__ == '__main__':
    main()
