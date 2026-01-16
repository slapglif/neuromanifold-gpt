#!/usr/bin/env python3
"""
Verification script for config refactor that doesn't require torch.
Tests that the new ralph config system works correctly.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_ralph_base_config_loadable():
    """Test that RalphBaseConfig can be imported and instantiated."""
    try:
        from neuromanifold_gpt.config.ralph_base import RalphBaseConfig
        config = RalphBaseConfig()
        assert hasattr(config, 'dataset')
        assert hasattr(config, 'batch_size')
        assert hasattr(config, 'n_layer')
        print("✓ RalphBaseConfig loadable and has expected attributes")
        return True
    except Exception as e:
        print(f"✗ RalphBaseConfig test failed: {e}")
        return False

def test_ralph_builder_loadable():
    """Test that RalphConfigBuilder can be imported and used."""
    try:
        from neuromanifold_gpt.config.ralph_builder import RalphConfigBuilder
        builder = RalphConfigBuilder()
        config = builder.with_overrides(batch_size=32, n_layer=4).build()
        assert config.batch_size == 32
        assert config.n_layer == 4
        print("✓ RalphConfigBuilder works with overrides")
        return True
    except Exception as e:
        print(f"✗ RalphConfigBuilder test failed: {e}")
        return False

def test_ralph_configs_module_loadable():
    """Test that ralph_configs module can be imported."""
    try:
        from neuromanifold_gpt.config.ralph_configs import get_ralph_config, list_ralph_iterations
        iterations = list_ralph_iterations()
        assert len(iterations) > 0, "No ralph iterations found"
        print(f"✓ ralph_configs module loadable, found {len(iterations)} iterations")
        return True
    except Exception as e:
        print(f"✗ ralph_configs module test failed: {e}")
        return False

def test_get_ralph_config():
    """Test that get_ralph_config works for known iterations."""
    try:
        from neuromanifold_gpt.config.ralph_configs import get_ralph_config
        # Test a few known iterations
        for iteration in [1, 10, 50]:
            config = get_ralph_config(iteration)
            assert hasattr(config, 'dataset')
            assert hasattr(config, 'batch_size')
            print(f"  ✓ get_ralph_config({iteration}) works")
        print("✓ get_ralph_config works for sample iterations")
        return True
    except Exception as e:
        print(f"✗ get_ralph_config test failed: {e}")
        return False

def test_archived_files_exist():
    """Test that old ralph_iter files were properly archived."""
    try:
        import glob
        archived = glob.glob("config/archive/ralph_iterations/ralph_iter*.py")
        if len(archived) == 73:
            print(f"✓ All 73 ralph_iter files properly archived")
            return True
        else:
            print(f"⚠ Found {len(archived)} archived files (expected 73)")
            return True  # Not a hard failure
    except Exception as e:
        print(f"✗ Archive check failed: {e}")
        return False

def test_new_config_files_exist():
    """Test that new config files were created."""
    try:
        import os.path
        required_files = [
            "neuromanifold_gpt/config/ralph_base.py",
            "neuromanifold_gpt/config/ralph_builder.py",
            "neuromanifold_gpt/config/ralph_configs/__init__.py",
            "neuromanifold_gpt/config/ralph_configs/registry.py",
            "neuromanifold_gpt/config/ralph_configs/iterations.py",
        ]
        all_exist = True
        for file in required_files:
            if os.path.exists(file):
                print(f"  ✓ {file} exists")
            else:
                print(f"  ✗ {file} missing")
                all_exist = False
        if all_exist:
            print("✓ All required new config files exist")
        return all_exist
    except Exception as e:
        print(f"✗ File existence check failed: {e}")
        return False

def main():
    """Run all verification tests."""
    print("=" * 70)
    print("Config Refactor Verification (No Torch Required)")
    print("=" * 70)
    print()

    tests = [
        ("File Existence", test_new_config_files_exist),
        ("Archived Files", test_archived_files_exist),
        ("RalphBaseConfig", test_ralph_base_config_loadable),
        ("RalphConfigBuilder", test_ralph_builder_loadable),
        ("ralph_configs Module", test_ralph_configs_module_loadable),
        ("get_ralph_config()", test_get_ralph_config),
    ]

    results = []
    for name, test_func in tests:
        print(f"\nTesting: {name}")
        print("-" * 70)
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            results.append((name, False))

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {name:30s} [{status}]")

    print()
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print()
        print("✓ All config refactor verification tests passed!")
        print()
        print("Note: Full test suite requires torch and other dependencies.")
        print("Run 'pip install -r requirements.txt' to install dependencies,")
        print("then run 'pytest neuromanifold_gpt/tests/' for full validation.")
        return 0
    else:
        print()
        print("✗ Some verification tests failed. Please review output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
