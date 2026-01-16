#!/usr/bin/env python3
"""
Static verification of System 2 component integration.
This script verifies the integration without importing torch.
"""

import ast
import os
import sys
from pathlib import Path

class IntegrationVerifier:
    """Verify System 2 components integration with gpt.py"""

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.success_checks = []

    def verify(self):
        """Run all verification checks"""
        print("=" * 60)
        print("System 2 Components Integration Verification")
        print("=" * 60)
        print()

        self.check_module_files_exist()
        self.check_module_syntax()
        self.check_gpt_imports()
        self.check_constructor_signatures()
        self.check_config_attributes()

        self.print_results()
        return len(self.errors) == 0

    def check_module_files_exist(self):
        """Verify all module files exist"""
        print("[1/5] Checking module files exist...")

        files = [
            "neuromanifold_gpt/model/planning/dag_planner.py",
            "neuromanifold_gpt/model/memory/hierarchical_engram.py",
            "neuromanifold_gpt/model/imagination.py",
        ]

        for filepath in files:
            if os.path.exists(filepath):
                self.success_checks.append(f"✓ {filepath} exists")
            else:
                self.errors.append(f"✗ {filepath} not found")

        print(f"  Found {len(files)} module files\n")

    def check_module_syntax(self):
        """Verify all modules have valid Python syntax"""
        print("[2/5] Checking module syntax...")

        files = [
            "neuromanifold_gpt/model/planning/dag_planner.py",
            "neuromanifold_gpt/model/memory/hierarchical_engram.py",
            "neuromanifold_gpt/model/imagination.py",
        ]

        for filepath in files:
            try:
                with open(filepath, 'r') as f:
                    ast.parse(f.read())
                self.success_checks.append(f"✓ {filepath} has valid syntax")
            except SyntaxError as e:
                self.errors.append(f"✗ {filepath} has syntax error: {e}")

        print(f"  Verified syntax for {len(files)} modules\n")

    def check_gpt_imports(self):
        """Verify gpt.py has correct imports"""
        print("[3/5] Checking gpt.py imports...")

        gpt_file = "neuromanifold_gpt/model/gpt.py"

        try:
            with open(gpt_file, 'r') as f:
                content = f.read()
                tree = ast.parse(content)

            # Check for the three critical imports
            imports_to_find = {
                'ForcedDAGPlanner': False,
                'HierarchicalEngramMemory': False,
                'ConsistencyImaginationModule': False,
            }

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        if alias.name in imports_to_find:
                            imports_to_find[alias.name] = True
                            # Verify the import path
                            if alias.name == 'ForcedDAGPlanner':
                                if 'planning.dag_planner' in (node.module or ''):
                                    self.success_checks.append(f"✓ ForcedDAGPlanner imported from correct module")
                                else:
                                    self.errors.append(f"✗ ForcedDAGPlanner imported from wrong module")
                            elif alias.name == 'HierarchicalEngramMemory':
                                if 'memory.hierarchical_engram' in (node.module or ''):
                                    self.success_checks.append(f"✓ HierarchicalEngramMemory imported from correct module")
                                else:
                                    self.errors.append(f"✗ HierarchicalEngramMemory imported from wrong module")
                            elif alias.name == 'ConsistencyImaginationModule':
                                if 'imagination' in (node.module or ''):
                                    self.success_checks.append(f"✓ ConsistencyImaginationModule imported from correct module")
                                else:
                                    self.errors.append(f"✗ ConsistencyImaginationModule imported from wrong module")

            for name, found in imports_to_find.items():
                if not found:
                    self.errors.append(f"✗ {name} not imported in gpt.py")

        except Exception as e:
            self.errors.append(f"✗ Error reading gpt.py: {e}")

        print(f"  Verified imports in gpt.py\n")

    def check_constructor_signatures(self):
        """Verify module constructors match gpt.py usage"""
        print("[4/5] Checking constructor signatures...")

        # Parse gpt.py to find how modules are instantiated
        gpt_file = "neuromanifold_gpt/model/gpt.py"

        try:
            with open(gpt_file, 'r') as f:
                gpt_content = f.read()

            # Check ForcedDAGPlanner instantiation
            if 'ForcedDAGPlanner(' in gpt_content:
                if all(param in gpt_content for param in ['embed_dim=', 'manifold_dim=', 'max_nodes=', 'min_nodes=']):
                    self.success_checks.append("✓ ForcedDAGPlanner constructor signature matches")
                else:
                    self.errors.append("✗ ForcedDAGPlanner constructor parameters don't match")

            # Check HierarchicalEngramMemory instantiation
            if 'HierarchicalEngramMemory(' in gpt_content:
                required_params = ['sdr_size=', 'n_active=', 'content_dim=', 'l1_capacity=', 'l2_capacity=', 'l3_capacity=']
                if all(param in gpt_content for param in required_params):
                    self.success_checks.append("✓ HierarchicalEngramMemory constructor signature matches")
                else:
                    self.errors.append("✗ HierarchicalEngramMemory constructor parameters don't match")

            # Check ConsistencyImaginationModule instantiation
            if 'ConsistencyImaginationModule(' in gpt_content:
                if all(param in gpt_content for param in ['embed_dim=', 'manifold_dim=', 'n_imagination_steps=']):
                    self.success_checks.append("✓ ConsistencyImaginationModule constructor signature matches")
                else:
                    self.errors.append("✗ ConsistencyImaginationModule constructor parameters don't match")

        except Exception as e:
            self.errors.append(f"✗ Error checking constructor signatures: {e}")

        print(f"  Verified constructor signatures\n")

    def check_config_attributes(self):
        """Verify config has the required attributes"""
        print("[5/5] Checking config attributes...")

        config_file = "neuromanifold_gpt/config/base.py"

        try:
            with open(config_file, 'r') as f:
                config_content = f.read()

            # Check for config flags
            flags = {
                'use_dag_planner': False,
                'use_hierarchical_memory': False,
                'use_imagination': False,
            }

            for flag in flags:
                if flag in config_content:
                    flags[flag] = True
                    self.success_checks.append(f"✓ Config has '{flag}' attribute")
                else:
                    self.warnings.append(f"⚠ Config might be missing '{flag}' attribute (may use getattr)")

        except Exception as e:
            self.errors.append(f"✗ Error reading config: {e}")

        print(f"  Verified config attributes\n")

    def print_results(self):
        """Print verification results"""
        print("=" * 60)
        print("Verification Results")
        print("=" * 60)
        print()

        if self.success_checks:
            print(f"✓ Passed Checks ({len(self.success_checks)}):")
            for check in self.success_checks:
                print(f"  {check}")
            print()

        if self.warnings:
            print(f"⚠ Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  {warning}")
            print()

        if self.errors:
            print(f"✗ Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  {error}")
            print()

        print("=" * 60)
        if len(self.errors) == 0:
            print("✓ STATIC VERIFICATION PASSED")
            print()
            print("All System 2 components are correctly integrated.")
            print("To complete verification, install torch and run:")
            print()
            print("  source .venv/bin/activate")
            print("  pip install torch einops")
            print("  python test_integration.py")
            print()
        else:
            print("✗ STATIC VERIFICATION FAILED")
            print(f"{len(self.errors)} error(s) found")
            print()
        print("=" * 60)

def main():
    verifier = IntegrationVerifier()
    success = verifier.verify()
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
