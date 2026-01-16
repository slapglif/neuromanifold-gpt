"""Pytest configuration for config loader tests.

This conftest.py loads config modules directly before pytest imports the package,
bypassing the torch dependency in neuromanifold_gpt/__init__.py.
"""

import sys
import importlib.util
from pathlib import Path

# Get the project root directory
_conftest_dir = Path(__file__).parent
_project_root = _conftest_dir.parent.parent


def _load_module_direct(module_name, file_path):
    """Load a module directly from file path without triggering package imports."""
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Pre-load modules to bypass torch dependencies
_load_module_direct(
    'neuromanifold_gpt.errors',
    _project_root / 'neuromanifold_gpt' / 'errors.py'
)
_load_module_direct(
    'neuromanifold_gpt.config.training',
    _project_root / 'neuromanifold_gpt' / 'config' / 'training.py'
)
_load_module_direct(
    'neuromanifold_gpt.config.loader',
    _project_root / 'neuromanifold_gpt' / 'config' / 'loader.py'
)
