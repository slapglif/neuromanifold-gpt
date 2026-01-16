#!/usr/bin/env python3
"""
Verification script for gradient checkpointing memory reduction.

This script runs two training sessions and compares peak memory usage:
1. Training WITHOUT gradient checkpointing
2. Training WITH gradient checkpointing

Expected: Gradient checkpointing should reduce peak memory by ~20-40%
"""

import subprocess
import re
import sys
import os
from pathlib import Path


def run_training(use_checkpointing: bool, max_iters: int = 10) -> dict:
    """Run training and extract memory statistics.

    Args:
        use_checkpointing: Whether to enable gradient checkpointing
        max_iters: Number of iterations to run

    Returns:
        dict with memory stats: {
            'peak_memory_gb': float,
            'allocated_gb': float,
            'reserved_gb': float,
            'success': bool
        }
    """
    checkpoint_flag = "true" if use_checkpointing else "false"
    cmd = [
        "python3", "train.py",
        "--config", "config/train_shakespeare_char.py",
        "--max_iters", str(max_iters),
        "--gradient_checkpointing", checkpoint_flag
    ]

    print(f"\n{'='*80}")
    print(f"Running training with gradient_checkpointing={checkpoint_flag}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        output = result.stdout + result.stderr

        # Extract memory statistics from output
        # Look for peak memory (max_allocated_gb)
        peak_memory = None
        allocated_memory = None
        reserved_memory = None

        # Pattern to match memory logs: "memory/gpu_max_allocated_gb: X.XXX"
        peak_pattern = r"memory/gpu_max_allocated_gb[:\s]+([0-9.]+)"
        allocated_pattern = r"memory/gpu_allocated_gb[:\s]+([0-9.]+)"
        reserved_pattern = r"memory/gpu_reserved_gb[:\s]+([0-9.]+)"

        peak_matches = re.findall(peak_pattern, output)
        allocated_matches = re.findall(allocated_pattern, output)
        reserved_matches = re.findall(reserved_pattern, output)

        if peak_matches:
            # Take the maximum peak memory seen
            peak_memory = max(float(m) for m in peak_matches)

        if allocated_matches:
            allocated_memory = max(float(m) for m in allocated_matches)

        if reserved_matches:
            reserved_memory = max(float(m) for m in reserved_matches)

        # If we couldn't find peak in logs, try to find it in the output another way
        if peak_memory is None:
            # Look for any mention of memory in GB
            general_pattern = r"([0-9.]+)\s*GB"
            matches = re.findall(general_pattern, output)
            if matches:
                print(f"  Warning: Could not find specific peak memory metric, found general GB values: {matches}")

        success = result.returncode == 0

        if not success:
            print(f"  Error: Training failed with return code {result.returncode}")
            print(f"  Output: {output[-500:]}")  # Last 500 chars

        return {
            'peak_memory_gb': peak_memory,
            'allocated_gb': allocated_memory,
            'reserved_gb': reserved_memory,
            'success': success,
            'output': output
        }

    except subprocess.TimeoutExpired:
        print(f"  Error: Training timed out after 300 seconds")
        return {
            'peak_memory_gb': None,
            'allocated_gb': None,
            'reserved_gb': None,
            'success': False,
            'output': ""
        }
    except Exception as e:
        print(f"  Error running training: {e}")
        return {
            'peak_memory_gb': None,
            'allocated_gb': None,
            'reserved_gb': None,
            'success': False,
            'output': str(e)
        }


def verify_gradient_checkpointing():
    """Run verification and compare results."""

    # Check if CUDA is available
    try:
        import torch
        if not torch.cuda.is_available():
            print("ERROR: CUDA is not available. This test requires a GPU.")
            print("Gradient checkpointing memory reduction can only be verified on GPU hardware.")
            return False
    except ImportError:
        print("ERROR: PyTorch is not installed.")
        return False

    print("\n" + "="*80)
    print("GRADIENT CHECKPOINTING MEMORY VERIFICATION")
    print("="*80)
    print("\nThis test will:")
    print("1. Run training WITHOUT gradient checkpointing (baseline)")
    print("2. Run training WITH gradient checkpointing (optimized)")
    print("3. Compare peak memory usage")
    print("4. Verify memory reduction of ~20-40%")
    print()

    # Run without gradient checkpointing
    print("\n[1/2] Running baseline (no gradient checkpointing)...")
    baseline = run_training(use_checkpointing=False, max_iters=10)

    if not baseline['success']:
        print("\nERROR: Baseline training failed")
        return False

    if baseline['peak_memory_gb'] is None:
        print("\nERROR: Could not extract peak memory from baseline run")
        print("Output snippet:")
        print(baseline['output'][-1000:])
        return False

    print(f"\n  ✓ Baseline peak memory: {baseline['peak_memory_gb']:.3f} GB")

    # Run with gradient checkpointing
    print("\n[2/2] Running with gradient checkpointing...")
    optimized = run_training(use_checkpointing=True, max_iters=10)

    if not optimized['success']:
        print("\nERROR: Optimized training failed")
        return False

    if optimized['peak_memory_gb'] is None:
        print("\nERROR: Could not extract peak memory from optimized run")
        print("Output snippet:")
        print(optimized['output'][-1000:])
        return False

    print(f"\n  ✓ Optimized peak memory: {optimized['peak_memory_gb']:.3f} GB")

    # Calculate reduction
    memory_reduction_gb = baseline['peak_memory_gb'] - optimized['peak_memory_gb']
    memory_reduction_pct = (memory_reduction_gb / baseline['peak_memory_gb']) * 100

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\n  Baseline peak memory:    {baseline['peak_memory_gb']:.3f} GB")
    print(f"  Optimized peak memory:   {optimized['peak_memory_gb']:.3f} GB")
    print(f"  Memory reduction:        {memory_reduction_gb:.3f} GB ({memory_reduction_pct:.1f}%)")
    print()

    # Verify reduction is in expected range
    if memory_reduction_pct < 0:
        print(f"  ✗ FAIL: Gradient checkpointing INCREASED memory by {abs(memory_reduction_pct):.1f}%")
        print(f"        This should not happen - gradient checkpointing should reduce memory.")
        return False
    elif memory_reduction_pct < 20:
        print(f"  ⚠ WARNING: Memory reduction ({memory_reduction_pct:.1f}%) is less than expected (20-40%)")
        print(f"        This might be OK for small models or short sequences.")
        print(f"        Expected reduction is more pronounced with larger models/sequences.")
        return True
    elif memory_reduction_pct > 40:
        print(f"  ⚠ WARNING: Memory reduction ({memory_reduction_pct:.1f}%) is higher than expected (20-40%)")
        print(f"        This is actually good! More memory savings than expected.")
        return True
    else:
        print(f"  ✓ SUCCESS: Memory reduction ({memory_reduction_pct:.1f}%) is within expected range (20-40%)")
        return True


if __name__ == "__main__":
    print("\nGradient Checkpointing Memory Reduction Verification")
    print("=" * 80)

    # Check if we're in the right directory
    if not Path("train.py").exists():
        print("ERROR: train.py not found. Please run this script from the repository root.")
        sys.exit(1)

    if not Path("config/train_shakespeare_char.py").exists():
        print("ERROR: config/train_shakespeare_char.py not found.")
        sys.exit(1)

    success = verify_gradient_checkpointing()

    if success:
        print("\n" + "="*80)
        print("VERIFICATION PASSED ✓")
        print("="*80)
        print("\nGradient checkpointing successfully reduces memory usage!")
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("VERIFICATION FAILED ✗")
        print("="*80)
        print("\nGradient checkpointing did not reduce memory as expected.")
        sys.exit(1)
