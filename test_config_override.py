#!/usr/bin/env python3
"""Test script to verify config file override backward compatibility."""

import sys
import subprocess
from pathlib import Path

def test_config_with_help(script, config_file):
    """Test that --help works with config file."""
    cmd = [sys.executable, script, config_file, "--help"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ FAIL: {script} {config_file} --help returned {result.returncode}")
        print(f"STDERR: {result.stderr}")
        return False

    if "usage:" not in result.stdout:
        print(f"❌ FAIL: {script} {config_file} --help did not show usage")
        return False

    print(f"✓ {script} with {config_file} --help works")
    return True

def test_sample_with_config():
    """Test sample.py with config file."""
    # Create a simple test config
    test_config = Path("test_config_temp.py")
    test_config.write_text("""
# Test config
num_samples = 3
temperature = 0.9
start = "Test prompt"
""")

    try:
        # Test that help works with config
        result = test_config_with_help("sample.py", str(test_config))
        return result
    finally:
        # Cleanup
        if test_config.exists():
            test_config.unlink()

def main():
    """Run all backward compatibility tests."""
    print("Testing backward compatibility with config file overrides...\n")

    all_passed = True

    # Test 1: sample.py with finetune_shakespeare.py config
    if Path("config/finetune_shakespeare.py").exists():
        passed = test_config_with_help("sample.py", "config/finetune_shakespeare.py")
        all_passed = all_passed and passed
    else:
        print("⚠ Skipping: config/finetune_shakespeare.py not found")

    # Test 2: bench.py with train_gpt2.py config
    if Path("config/train_gpt2.py").exists():
        passed = test_config_with_help("bench.py", "config/train_gpt2.py")
        all_passed = all_passed and passed
    else:
        print("⚠ Skipping: config/train_gpt2.py not found")

    # Test 3: sample.py with custom config
    passed = test_sample_with_config()
    all_passed = all_passed and passed

    # Test 4: neuromanifold_gpt scripts (with PYTHONPATH)
    if Path("config/train_shakespeare_char.py").exists():
        cmd = ["bash", "-c",
               f"PYTHONPATH=. {sys.executable} neuromanifold_gpt/train_nanogpt.py config/train_shakespeare_char.py --help"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and "usage:" in result.stdout:
            print("✓ neuromanifold_gpt/train_nanogpt.py with config/train_shakespeare_char.py --help works")
        else:
            print("❌ FAIL: neuromanifold_gpt/train_nanogpt.py with config --help failed")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All backward compatibility tests PASSED")
        return 0
    else:
        print("❌ Some tests FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
