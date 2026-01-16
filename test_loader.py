#!/usr/bin/env python3
"""Test script for config loader without requiring torch."""

import sys
import importlib.util

# Load errors module directly (bypasses torch dependency in __init__.py)
errors_spec = importlib.util.spec_from_file_location(
    "neuromanifold_gpt.errors",
    "./neuromanifold_gpt/errors.py"
)
errors = importlib.util.module_from_spec(errors_spec)
sys.modules['neuromanifold_gpt.errors'] = errors
errors_spec.loader.exec_module(errors)

# Load config modules directly (bypasses torch dependency)
training_spec = importlib.util.spec_from_file_location(
    "neuromanifold_gpt.config.training",
    "./neuromanifold_gpt/config/training.py"
)
training = importlib.util.module_from_spec(training_spec)
sys.modules['neuromanifold_gpt.config.training'] = training
training_spec.loader.exec_module(training)

loader_spec = importlib.util.spec_from_file_location(
    "neuromanifold_gpt.config.loader",
    "./neuromanifold_gpt/config/loader.py"
)
loader_module = importlib.util.module_from_spec(loader_spec)
sys.modules['neuromanifold_gpt.config.loader'] = loader_module
loader_spec.loader.exec_module(loader_module)

TrainingConfig = training.TrainingConfig
SamplingConfig = training.SamplingConfig
EvalConfig = training.EvalConfig
BenchConfig = training.BenchConfig
load_config = loader_module.load_config

# Test 1: Load default config
print("Test 1: Loading default TrainingConfig...")
config = load_config(TrainingConfig, [], show_help=False)
assert config.batch_size == 64
assert config.learning_rate == 1e-3
print("  ✓ Default config loaded successfully")

# Test 2: Load with CLI overrides
print("Test 2: Loading with CLI overrides...")
config = load_config(TrainingConfig, ['--batch_size=32', '--learning_rate=0.001'], show_help=False)
assert config.batch_size == 32
assert config.learning_rate == 0.001
print("  ✓ CLI overrides applied successfully")

# Test 3: Load SamplingConfig
print("Test 3: Loading SamplingConfig...")
config = load_config(SamplingConfig, ['--temperature=0.9', '--top_k=100'], show_help=False)
assert config.temperature == 0.9
assert config.top_k == 100
print("  ✓ SamplingConfig loaded successfully")

# Test 4: Test error handling for invalid key
print("Test 4: Testing error handling...")
try:
    config = load_config(TrainingConfig, ['--invalid_key=123'], show_help=False)
    print("  ✗ Should have raised ValidationError")
    sys.exit(1)
except errors.ValidationError:
    print("  ✓ ValidationError raised correctly for invalid key")

# Test 5: Test type validation
print("Test 5: Testing type validation...")
try:
    config = load_config(TrainingConfig, ['--batch_size=not_a_number'], show_help=False)
    print("  ✗ Should have raised ValidationError")
    sys.exit(1)
except errors.ValidationError:
    print("  ✓ ValidationError raised correctly for type mismatch")

print("\nAll tests passed! ✓")
print("OK")
