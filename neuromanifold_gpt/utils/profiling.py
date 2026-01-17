"""
Shared profiling utilities for performance measurement.

Provides standardized profiling functions for measuring component and model
performance across all profiling scripts. Includes timing, memory tracking,
and CUDA synchronization handling.

Usage:
    from neuromanifold_gpt.utils.profiling import profile_component, profile_forward_backward

    def make_input():
        return (torch.randn(64, 256, 384, device="cuda"),)

    module = MyModule()
    result = profile_component("MyModule", module, make_input)
    print(f"Mean: {result['mean_ms']:.2f}ms")
"""

import gc
import time
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn


def cleanup() -> None:
    """
    Clean up memory by running garbage collection and clearing CUDA cache.

    This function helps prevent OOM errors during profiling by freeing
    unused memory between profiling runs.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def profile_component(
    name: str,
    module: nn.Module,
    input_fn: Callable[[], tuple],
    n_warmup: int = 5,
    n_iters: int = 50,
    device: Optional[str] = None,
    track_memory: bool = False,
) -> Dict[str, Any]:
    """
    Profile a single component's forward pass.

    Measures inference performance by running multiple iterations with warmup
    and computing statistics (mean, min, max, std). Optionally tracks peak
    memory usage on CUDA devices.

    Args:
        name: Component name for identification in results
        module: PyTorch module to profile
        input_fn: Callable that returns a tuple of inputs for the module
        n_warmup: Number of warmup iterations before timing (default: 5)
        n_iters: Number of timed iterations for statistics (default: 50)
        device: Device to run on ("cuda" or "cpu"). Auto-detected if None.
        track_memory: If True, track peak memory usage on CUDA (default: False)

    Returns:
        Dictionary with profiling results:
            - name: Component name
            - mean_ms: Mean execution time in milliseconds
            - min_ms: Minimum execution time in milliseconds
            - max_ms: Maximum execution time in milliseconds
            - std_ms: Standard deviation of execution times
            - mem_mb: Peak memory usage in MB (only if track_memory=True on CUDA)

    Example:
        >>> def make_input():
        ...     return (torch.randn(64, 256, 384, device="cuda"),)
        >>> module = nn.Linear(384, 384)
        >>> result = profile_component("Linear", module, make_input, n_iters=10)
        >>> print(f"Mean: {result['mean_ms']:.2f}ms")
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    module = module.to(device)
    module.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            inputs = input_fn()
            _ = module(*inputs)
            if device == "cuda":
                torch.cuda.synchronize()

    # Memory tracking setup
    mem_used = 0
    if track_memory and device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(n_iters):
            inputs = input_fn()
            if device == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = module(*inputs)

            if device == "cuda":
                torch.cuda.synchronize()

            times.append((time.perf_counter() - start) * 1000)

    # Memory tracking finalization
    if track_memory and device == "cuda":
        mem_peak = torch.cuda.max_memory_allocated()
        mem_used = (mem_peak - mem_before) / 1e6

    # Compute statistics
    mean_time = sum(times) / len(times)
    result = {
        "name": name,
        "mean_ms": mean_time,
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5,
    }

    if track_memory:
        result["mem_mb"] = mem_used

    return result


def profile_forward_backward(
    name: str,
    module: nn.Module,
    input_fn: Callable[[], tuple],
    loss_fn: Callable[[Any], torch.Tensor],
    n_warmup: int = 5,
    n_iters: int = 50,
    device: Optional[str] = None,
    track_memory: bool = False,
) -> Dict[str, Any]:
    """
    Profile a training iteration (forward + backward pass).

    Measures training performance by running forward pass, loss computation,
    backward pass, and gradient zeroing. Computes timing statistics across
    multiple iterations with warmup.

    Args:
        name: Component name for identification in results
        module: PyTorch module to profile
        input_fn: Callable that returns a tuple of inputs for the module
        loss_fn: Callable that takes module output and returns a scalar loss
        n_warmup: Number of warmup iterations before timing (default: 5)
        n_iters: Number of timed iterations for statistics (default: 50)
        device: Device to run on ("cuda" or "cpu"). Auto-detected if None.
        track_memory: If True, track peak memory usage on CUDA (default: False)

    Returns:
        Dictionary with profiling results:
            - name: Component name
            - mean_ms: Mean execution time in milliseconds
            - min_ms: Minimum execution time in milliseconds
            - max_ms: Maximum execution time in milliseconds
            - std_ms: Standard deviation of execution times
            - mem_mb: Peak memory usage in MB (only if track_memory=True on CUDA)

    Example:
        >>> def make_input():
        ...     return (torch.randn(64, 256, 384, device="cuda"),)
        >>> def loss_fn(output):
        ...     return output.mean()
        >>> module = nn.Linear(384, 384)
        >>> result = profile_forward_backward(
        ...     "Linear Training", module, make_input, loss_fn, n_iters=10
        ... )
        >>> print(f"Mean: {result['mean_ms']:.2f}ms")
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    module = module.to(device)
    module.train()

    # Warmup
    for _ in range(n_warmup):
        inputs = input_fn()
        output = module(*inputs)
        loss = loss_fn(output)
        loss.backward()
        module.zero_grad()
        if device == "cuda":
            torch.cuda.synchronize()

    # Memory tracking setup
    mem_used = 0
    if track_memory and device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()

    # Timed runs
    times = []
    for _ in range(n_iters):
        inputs = input_fn()
        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        output = module(*inputs)
        loss = loss_fn(output)
        loss.backward()
        module.zero_grad()

        if device == "cuda":
            torch.cuda.synchronize()

        times.append((time.perf_counter() - start) * 1000)

    # Memory tracking finalization
    if track_memory and device == "cuda":
        mem_peak = torch.cuda.max_memory_allocated()
        mem_used = (mem_peak - mem_before) / 1e6

    # Compute statistics
    mean_time = sum(times) / len(times)
    result = {
        "name": name,
        "mean_ms": mean_time,
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5,
    }

    if track_memory:
        result["mem_mb"] = mem_used

    return result


# Alias for backward compatibility with some scripts
profile_module = profile_component
profile_fwd_bwd = profile_forward_backward
