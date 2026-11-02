#!/usr/bin/env python3
"""Test architecture sampling - loading module directly."""

import sys
import importlib.util

# Load the search_space module directly without triggering package imports
spec = importlib.util.spec_from_file_location(
    "search_space",
    "./neuromanifold_gpt/nas/search_space.py"
)
search_space_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(search_space_module)

# Get the classes
SearchSpace = search_space_module.SearchSpace
ArchitectureConfig = search_space_module.ArchitectureConfig

# Create search space
search_space = SearchSpace()

# Sample architecture
arch = search_space.sample()

# Validate
is_valid, error_msg = arch.validate()

if not is_valid:
    print(f"Invalid architecture: {error_msg}")
    sys.exit(1)

print("OK")
