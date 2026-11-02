#!/usr/bin/env python3
"""Minimal structural verification for NAS implementation.

This script verifies the NAS implementation structure without requiring
full dependencies (torch, etc.). It checks:
- Module structure
- Class definitions
- Method signatures
- Import paths
"""

import sys
from pathlib import Path
import ast
import inspect

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def check_file_exists(filepath):
    """Check if a file exists."""
    path = Path(filepath)
    if path.exists():
        print(f"✓ {filepath}")
        return True
    else:
        print(f"✗ {filepath} - NOT FOUND")
        return False

def check_class_in_module(module_path, class_name):
    """Check if a class exists in a module."""
    try:
        with open(module_path, 'r') as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return True
        return False
    except Exception as e:
        print(f"  Error checking {module_path}: {e}")
        return False

def check_function_in_module(module_path, func_name):
    """Check if a function exists in a module."""
    try:
        with open(module_path, 'r') as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                return True
        return False
    except Exception as e:
        print(f"  Error checking {module_path}: {e}")
        return False

def main():
    print("=" * 70)
    print("NAS IMPLEMENTATION - STRUCTURAL VERIFICATION")
    print("=" * 70)
    print()

    all_passed = True

    # Check file structure
    print("1. Checking file structure...")
    print("-" * 70)

    required_files = [
        "neuromanifold_gpt/nas/__init__.py",
        "neuromanifold_gpt/nas/search_space.py",
        "neuromanifold_gpt/nas/evaluator.py",
        "neuromanifold_gpt/nas/searcher.py",
        "neuromanifold_gpt/nas/export.py",
        "neuromanifold_gpt/nas/strategies/__init__.py",
        "neuromanifold_gpt/nas/strategies/random_search.py",
        "neuromanifold_gpt/nas/strategies/evolutionary.py",
        "examples/nas_search.py",
        "examples/nas_export_best.py",
    ]

    for filepath in required_files:
        if not check_file_exists(filepath):
            all_passed = False

    print()

    # Check class definitions
    print("2. Checking class definitions...")
    print("-" * 70)

    classes_to_check = [
        ("neuromanifold_gpt/nas/search_space.py", "SearchSpace"),
        ("neuromanifold_gpt/nas/search_space.py", "ArchitectureConfig"),
        ("neuromanifold_gpt/nas/evaluator.py", "ArchitectureEvaluator"),
        ("neuromanifold_gpt/nas/evaluator.py", "ComputeBudget"),
        ("neuromanifold_gpt/nas/evaluator.py", "EvaluationResult"),
        ("neuromanifold_gpt/nas/searcher.py", "Searcher"),
        ("neuromanifold_gpt/nas/strategies/random_search.py", "RandomSearch"),
        ("neuromanifold_gpt/nas/strategies/random_search.py", "SearchResults"),
        ("neuromanifold_gpt/nas/strategies/evolutionary.py", "EvolutionarySearch"),
    ]

    for module_path, class_name in classes_to_check:
        if check_class_in_module(module_path, class_name):
            print(f"✓ {class_name} in {module_path}")
        else:
            print(f"✗ {class_name} NOT FOUND in {module_path}")
            all_passed = False

    print()

    # Check key functions
    print("3. Checking key functions...")
    print("-" * 70)

    functions_to_check = [
        ("neuromanifold_gpt/nas/export.py", "export_config"),
        ("neuromanifold_gpt/nas/export.py", "export_to_json"),
        ("neuromanifold_gpt/nas/export.py", "generate_summary_report"),
    ]

    for module_path, func_name in functions_to_check:
        if check_function_in_module(module_path, func_name):
            print(f"✓ {func_name} in {module_path}")
        else:
            print(f"✗ {func_name} NOT FOUND in {module_path}")
            all_passed = False

    print()

    # Check imports (without requiring dependencies)
    print("4. Checking import structure...")
    print("-" * 70)

    # Check __init__.py exports
    init_file = Path("neuromanifold_gpt/nas/__init__.py")
    if init_file.exists():
        with open(init_file, 'r') as f:
            init_content = f.read()

        expected_exports = [
            'SearchSpace',
            'ArchitectureConfig',
            'ArchitectureEvaluator',
            'RandomSearch',
            'EvolutionarySearch',
            'export_config',
        ]

        for export in expected_exports:
            if export in init_content:
                print(f"✓ {export} exported from neuromanifold_gpt.nas")
            else:
                print(f"✗ {export} NOT exported from neuromanifold_gpt.nas")
                all_passed = False
    else:
        print("✗ neuromanifold_gpt/nas/__init__.py not found")
        all_passed = False

    print()

    # Check example scripts
    print("5. Checking example scripts...")
    print("-" * 70)

    example_files = [
        "examples/nas_search.py",
        "examples/nas_export_best.py",
    ]

    for example in example_files:
        if Path(example).exists():
            # Check if script is executable
            import os
            is_executable = os.access(example, os.X_OK)
            status = "✓" if is_executable else "⚠"
            print(f"{status} {example} {'(executable)' if is_executable else '(not executable)'}")

            # Check for shebang
            with open(example, 'r') as f:
                first_line = f.readline()
                if first_line.startswith('#!/'):
                    print(f"  ✓ Has shebang: {first_line.strip()}")
                else:
                    print(f"  ⚠ No shebang line")
        else:
            print(f"✗ {example} - NOT FOUND")
            all_passed = False

    print()

    # Check data directory
    print("6. Checking data availability...")
    print("-" * 70)

    data_dir = Path("data/shakespeare_char")
    if data_dir.exists():
        print(f"✓ {data_dir} exists")

        required_data_files = ["train.bin", "val.bin", "meta.pkl"]
        for data_file in required_data_files:
            filepath = data_dir / data_file
            if filepath.exists():
                size_mb = filepath.stat().st_size / (1024 * 1024)
                print(f"  ✓ {data_file} ({size_mb:.2f} MB)")
            else:
                print(f"  ⚠ {data_file} - NOT FOUND")
    else:
        print(f"⚠ {data_dir} not found")
        print("  Run: python data/shakespeare_char/prepare.py")

    print()
    print("=" * 70)

    if all_passed:
        print("✅ STRUCTURAL VERIFICATION PASSED")
        print()
        print("All required files, classes, and functions are present.")
        print("The NAS implementation structure is complete.")
        print()
        print("Next steps:")
        print("  1. Ensure dependencies are installed: pip install -r requirements.txt")
        print("  2. Run full verification: ./verify_nas_e2e.sh")
        print("  3. Or try a quick search: python examples/nas_search.py --help")
        return 0
    else:
        print("❌ STRUCTURAL VERIFICATION FAILED")
        print()
        print("Some required components are missing.")
        print("Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
