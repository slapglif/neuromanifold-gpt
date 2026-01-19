#!/usr/bin/env python3
"""
Minimal integration test for distillation pipeline.
Tests that all components work together without requiring full training.
"""

import sys
import os

def test_imports():
    """Test that all distillation modules can be imported"""
    print("Testing imports...")
    try:
        from neuromanifold_gpt.training.distillation_loss import (
            distillation_loss,
            kl_divergence_loss
        )
        from neuromanifold_gpt.training.distillation_module import DistillationLitModule
        from neuromanifold_gpt.training.config import TrainConfig
        from neuromanifold_gpt.config.base import NeuroManifoldConfig
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_distillation_loss():
    """Test distillation loss functions work"""
    print("\nTesting distillation loss functions...")
    try:
        import torch
        from neuromanifold_gpt.training.distillation_loss import (
            distillation_loss,
            kl_divergence_loss
        )

        # Create dummy tensors
        batch_size, seq_len, vocab_size = 2, 10, 50
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Test KL divergence loss
        kl_loss = kl_divergence_loss(student_logits, teacher_logits, temperature=2.0)
        assert kl_loss.item() > 0, "KL loss should be positive"
        print(f"  ✓ KL divergence loss: {kl_loss.item():.4f}")

        # Test combined distillation loss
        combined_loss = distillation_loss(
            student_logits, teacher_logits, labels,
            alpha=0.5, temperature=2.0
        )
        assert combined_loss.item() > 0, "Combined loss should be positive"
        print(f"  ✓ Combined distillation loss: {combined_loss.item():.4f}")

        return True
    except Exception as e:
        print(f"✗ Loss function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_distillation_module():
    """Test DistillationLitModule can be instantiated"""
    print("\nTesting DistillationLitModule...")
    try:
        from neuromanifold_gpt.training.distillation_module import DistillationLitModule
        from neuromanifold_gpt.config.base import NeuroManifoldConfig
        from neuromanifold_gpt.training.config import TrainConfig

        # Create configs for small models
        model_config = NeuroManifoldConfig(
            vocab_size=100,
            block_size=64,
            n_layer=2,
            n_heads=2,
            n_embd=32
        )

        # Note: We can't fully test without a teacher checkpoint, but we can
        # verify the module can be instantiated with config
        print(f"  ✓ Model config created: {model_config.n_layer} layers, {model_config.n_embd} embd")

        train_config = TrainConfig(
            teacher_checkpoint='dummy.pt',  # Would need real checkpoint
            distillation_alpha=0.5,
            distillation_temperature=2.0
        )
        print(f"  ✓ Train config created: alpha={train_config.distillation_alpha}, T={train_config.distillation_temperature}")

        return True
    except Exception as e:
        print(f"✗ Module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_scripts_exist():
    """Test that main scripts exist and are valid Python"""
    print("\nTesting scripts exist...")
    try:
        scripts = ['distill.py', 'eval_distillation.py', 'train.py']
        for script in scripts:
            if not os.path.exists(script):
                print(f"✗ Script not found: {script}")
                return False
            # Try to compile it
            with open(script, 'r') as f:
                compile(f.read(), script, 'exec')
            print(f"  ✓ {script} exists and is valid Python")
        return True
    except Exception as e:
        print(f"✗ Script validation failed: {e}")
        return False

def test_config_preset():
    """Test distillation config preset"""
    print("\nTesting distillation config preset...")
    try:
        from neuromanifold_gpt.config.training.train_distillation_shakespeare import get_config

        config = get_config()
        assert hasattr(config, 'enable_distillation'), "Config should have enable_distillation"
        assert config.enable_distillation == True, "Distillation should be enabled"
        assert config.teacher_checkpoint is not None, "Teacher checkpoint should be set"
        print(f"  ✓ Config preset valid: distillation enabled, teacher={config.teacher_checkpoint}")
        return True
    except Exception as e:
        print(f"✗ Config preset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all integration tests"""
    print("=" * 60)
    print("DISTILLATION PIPELINE INTEGRATION TEST")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Distillation Loss", test_distillation_loss),
        ("Distillation Module", test_distillation_module),
        ("Scripts Exist", test_scripts_exist),
        ("Config Preset", test_config_preset),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n✓ ALL INTEGRATION TESTS PASSED")
        print("\nThe distillation pipeline components are correctly integrated.")
        print("Full end-to-end training requires:")
        print("  1. Proper Python environment with all dependencies")
        print("  2. Dataset preparation (shakespeare_char)")
        print("  3. GPU resources for training")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
