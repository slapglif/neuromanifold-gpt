#!/usr/bin/env python3
"""
Verification script for gradient checkpointing implementation (no GPU required).

This script verifies the code implementation is correct without running actual training.
For full memory reduction testing, use verify_gradient_checkpointing.py with GPU.
"""

import sys
from pathlib import Path


def verify_implementation():
    """Verify gradient checkpointing implementation in code."""

    print("\n" + "="*80)
    print("GRADIENT CHECKPOINTING IMPLEMENTATION VERIFICATION (No GPU)")
    print("="*80)
    print("\nThis test verifies:")
    print("1. Configuration fields are added to all model configs")
    print("2. Checkpoint imports are present")
    print("3. Forward passes use checkpointing when enabled")
    print("4. Training integration is complete")
    print()

    all_passed = True

    # Test 1: model.py (GPT)
    print("[1/4] Verifying model.py (GPT)...")
    try:
        with open('model.py', 'r') as f:
            content = f.read()

        checks = {
            'checkpoint import': 'from torch.utils.checkpoint import checkpoint' in content,
            'GPTConfig.gradient_checkpointing field': 'gradient_checkpointing: bool' in content and 'class GPTConfig' in content,
            'forward pass conditional': 'self.config.gradient_checkpointing and self.training' in content,
            'checkpoint call with use_reentrant=False': 'checkpoint(block, x, use_reentrant=False)' in content,
        }

        for check, passed in checks.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check}")
            if not passed:
                all_passed = False

    except Exception as e:
        print(f"  ✗ Error reading model.py: {e}")
        all_passed = False

    # Test 2: neuromanifold_gpt/config/base.py
    print("\n[2/4] Verifying neuromanifold_gpt/config/base.py...")
    try:
        with open('neuromanifold_gpt/config/base.py', 'r') as f:
            content = f.read()

        checks = {
            'NeuroManifoldConfig.gradient_checkpointing field': 'gradient_checkpointing: bool' in content,
        }

        for check, passed in checks.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check}")
            if not passed:
                all_passed = False

    except Exception as e:
        print(f"  ✗ Error reading neuromanifold_gpt/config/base.py: {e}")
        all_passed = False

    # Test 3: neuromanifold_gpt/model/gpt.py
    print("\n[3/4] Verifying neuromanifold_gpt/model/gpt.py...")
    try:
        with open('neuromanifold_gpt/model/gpt.py', 'r') as f:
            content = f.read()

        checks = {
            'checkpoint import': 'from torch.utils.checkpoint import checkpoint' in content,
            'forward pass conditional': 'self.config.gradient_checkpointing and self.training' in content,
            'checkpoint usage in forward': 'checkpoint(' in content and 'use_reentrant=False' in content,
        }

        for check, passed in checks.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check}")
            if not passed:
                all_passed = False

    except Exception as e:
        print(f"  ✗ Error reading neuromanifold_gpt/model/gpt.py: {e}")
        all_passed = False

    # Test 4: train.py
    print("\n[4/4] Verifying train.py...")
    try:
        with open('train.py', 'r') as f:
            content = f.read()

        checks = {
            'TrainConfig.gradient_checkpointing field': 'gradient_checkpointing: bool' in content and 'class TrainConfig' in content,
            'passed to model configs': 'gradient_checkpointing=' in content and 'config.gradient_checkpointing' in content,
        }

        for check, passed in checks.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check}")
            if not passed:
                all_passed = False

    except Exception as e:
        print(f"  ✗ Error reading train.py: {e}")
        all_passed = False

    # Summary
    print("\n" + "="*80)
    if all_passed:
        print("VERIFICATION PASSED ✓")
        print("="*80)
        print("\n✓ All gradient checkpointing code is correctly implemented!")
        print("\nNOTE: This verification only checks code structure.")
        print("For full memory reduction testing (20-40%), run verify_gradient_checkpointing.py")
        print("on a system with PyTorch and GPU hardware.")
    else:
        print("VERIFICATION FAILED ✗")
        print("="*80)
        print("\n✗ Some implementation checks failed. See details above.")

    return all_passed


if __name__ == "__main__":
    # Check if we're in the right directory
    if not Path("train.py").exists():
        print("ERROR: train.py not found. Please run this script from the repository root.")
        sys.exit(1)

    success = verify_implementation()
    sys.exit(0 if success else 1)
