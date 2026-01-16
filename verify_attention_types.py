#!/usr/bin/env python3
"""Syntax verification for all attention types integration.

Verifies that:
1. All Python files can be parsed without syntax errors
2. Config has attention_type field with correct default
3. Block factory method maps all attention types
4. GPT model passes attention_type to blocks
5. Backward compatibility flags work
"""
import ast
import sys
from pathlib import Path


def check_syntax(filepath: Path) -> bool:
    """Check if a Python file has valid syntax."""
    try:
        with open(filepath, 'r') as f:
            ast.parse(f.read())
        return True
    except SyntaxError as e:
        print(f"  ✗ Syntax error in {filepath}: {e}")
        return False


def verify_config():
    """Verify NeuroManifoldConfig has attention_type field."""
    print("\nVerifying NeuroManifoldConfig...")

    config_file = Path("neuromanifold_gpt/config/base.py")
    with open(config_file, 'r') as f:
        content = f.read()

    # Check attention_type field exists
    assert "attention_type: str = 'standard'" in content, (
        "Config should have attention_type field with default 'standard'"
    )

    # Check backward compatibility flags exist
    assert "use_kaufmann_attention: bool = False" in content, (
        "Config should have deprecated use_kaufmann_attention flag"
    )
    assert "use_knot_attention: bool = False" in content, (
        "Config should have deprecated use_knot_attention flag"
    )

    print("  ✓ attention_type field exists with default 'standard'")
    print("  ✓ Backward compatibility flags preserved")


def verify_block_factory():
    """Verify block has factory method for all attention types."""
    print("\nVerifying NeuroManifoldBlock factory...")

    block_file = Path("neuromanifold_gpt/model/block.py")
    with open(block_file, 'r') as f:
        content = f.read()

    # Check factory method exists
    assert "_create_attention" in content, (
        "Block should have _create_attention factory method"
    )

    # Check all attention types are mapped
    attention_types = ["standard", "soliton", "sdr", "fast-spectral", "kaufmann"]
    for atype in attention_types:
        assert f'"{atype}"' in content, (
            f"Factory should map attention type '{atype}'"
        )

    # Check backward compatibility mapping
    assert "if use_kaufmann_attention:" in content, (
        "Should map use_kaufmann_attention to attention_type"
    )
    assert "if use_knot_attention:" in content, (
        "Should map use_knot_attention to attention_type"
    )

    print("  ✓ _create_attention factory method exists")
    print("  ✓ All attention types mapped: " + ", ".join(attention_types))
    print("  ✓ Backward compatibility mapping present")


def verify_gpt_integration():
    """Verify GPT model passes attention_type to blocks."""
    print("\nVerifying NeuroManifoldGPT integration...")

    gpt_file = Path("neuromanifold_gpt/model/gpt.py")
    with open(gpt_file, 'r') as f:
        content = f.read()

    # Check that attention_type is passed to blocks
    assert "attention_type=getattr(config, 'attention_type', 'standard')" in content, (
        "GPT should pass attention_type from config to blocks"
    )

    print("  ✓ GPT passes attention_type to blocks")
    print("  ✓ Uses getattr with default for backward compatibility")


def verify_standard_attention():
    """Verify StandardAttention class exists."""
    print("\nVerifying StandardAttention...")

    standard_file = Path("neuromanifold_gpt/model/attention/standard.py")
    assert standard_file.exists(), (
        "StandardAttention file should exist"
    )

    with open(standard_file, 'r') as f:
        content = f.read()

    # Check class exists
    assert "class StandardAttention" in content, (
        "StandardAttention class should exist"
    )

    # Check it has forward method with correct signature
    assert "def forward(self, x" in content, (
        "StandardAttention should have forward method"
    )

    print("  ✓ StandardAttention class exists")
    print("  ✓ Forward method implemented")


def verify_attention_exports():
    """Verify all attention classes are exported."""
    print("\nVerifying attention module exports...")

    init_file = Path("neuromanifold_gpt/model/attention/__init__.py")
    with open(init_file, 'r') as f:
        content = f.read()

    # Check StandardAttention is exported
    assert "StandardAttention" in content, (
        "StandardAttention should be exported"
    )

    print("  ✓ StandardAttention exported in __init__.py")


def verify_all_syntax():
    """Verify all Python files have valid syntax."""
    print("\nVerifying Python syntax...")

    files_to_check = [
        "neuromanifold_gpt/config/base.py",
        "neuromanifold_gpt/model/block.py",
        "neuromanifold_gpt/model/gpt.py",
        "neuromanifold_gpt/model/attention/standard.py",
        "neuromanifold_gpt/model/attention/__init__.py",
    ]

    all_valid = True
    for filepath in files_to_check:
        path = Path(filepath)
        if path.exists():
            if check_syntax(path):
                print(f"  ✓ {filepath}")
            else:
                all_valid = False
        else:
            print(f"  ✗ File not found: {filepath}")
            all_valid = False

    assert all_valid, "Some files have syntax errors or are missing"


def main():
    """Run all verification checks."""
    print("=" * 70)
    print("Syntax Verification: All Attention Types Integration")
    print("=" * 70)

    try:
        verify_all_syntax()
        verify_config()
        verify_standard_attention()
        verify_attention_exports()
        verify_block_factory()
        verify_gpt_integration()

        print("\n" + "=" * 70)
        print("All Syntax Checks Passed! ✓")
        print("=" * 70)
        print("\nSummary:")
        print("  • All Python files have valid syntax")
        print("  • Config has attention_type field with correct default")
        print("  • StandardAttention class implemented and exported")
        print("  • Block factory maps all 5 attention types")
        print("  • GPT model passes attention_type to blocks")
        print("  • Backward compatibility flags preserved")
        print("\nNote: Full runtime verification requires PyTorch environment")
        print("      Run 'python test_all_attention_types.py' with PyTorch installed")
        print()

        return 0
    except AssertionError as e:
        print(f"\n✗ Verification failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
