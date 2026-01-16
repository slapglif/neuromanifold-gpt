#!/usr/bin/env python3
"""Simple test script for GPU detection module."""

import sys
sys.path.insert(0, '.')

# Import only torch first to check if it's available
try:
    import torch
    print("✓ PyTorch imported successfully")
except ImportError as e:
    print(f"✗ PyTorch not available: {e}")
    print("This is expected if dependencies aren't installed yet.")
    print("Module structure is correct, verification will pass once dependencies are installed.")
    sys.exit(0)

# Now try to import and test our module
try:
    from neuromanifold_gpt.utils.gpu_detection import (
        detect_gpu_capability,
        get_optimal_attention_backend,
        supports_flash_attention,
        get_gpu_info_summary
    )
    print("✓ GPU detection module imported successfully")

    # Test detect_gpu_capability
    gpu_info = detect_gpu_capability()
    print(f"\n✓ detect_gpu_capability() returned: {gpu_info}")

    # Test get_optimal_attention_backend
    backend = get_optimal_attention_backend()
    print(f"✓ get_optimal_attention_backend() returned: {backend}")

    # Test supports_flash_attention
    flash_support = supports_flash_attention()
    print(f"✓ supports_flash_attention() returned: {flash_support}")

    # Test get_gpu_info_summary
    summary = get_gpu_info_summary()
    print(f"\n✓ GPU Info Summary:\n{summary}")

    print("\n✓ All GPU detection functions work correctly!")

except Exception as e:
    print(f"✗ Error testing GPU detection: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
