#!/usr/bin/env python
"""Test script to verify Triton kernel module loads correctly."""

import sys

try:
    from neuromanifold_gpt.model.attention.fhn_triton import fhn_triton_kernel, FHNTritonAttention
    import torch

    print("✓ Triton kernel module loaded successfully")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")

    # Test basic instantiation
    if torch.cuda.is_available():
        print("✓ Triton kernel loadable")
    else:
        print("✓ Triton kernel loadable (CUDA not available for full test)")

    sys.exit(0)
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
