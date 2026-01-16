"""
Memory Monitor Utility

Provides real-time GPU memory tracking during training for consumer GPUs.
Tracks current, peak, and reserved memory with formatting utilities.
"""

import torch
from collections import deque
from typing import Optional


class GPUMemoryMonitor:
    """
    Real-time GPU memory monitor for tracking memory usage during training.

    Tracks allocated, reserved, and peak memory usage with utilities for
    resetting peak stats and formatting display output.

    Example:
        >>> monitor = GPUMemoryMonitor()
        >>> stats = monitor.get_memory_stats()
        >>> print(f"Allocated: {stats['allocated_gb']:.2f}GB")
        >>> print(monitor.format_memory_display())
    """

    def __init__(self, device: int = 0, history_size: int = 1000):
        """
        Initialize GPU memory monitor.

        Args:
            device: CUDA device index to monitor (default: 0)
            history_size: Maximum number of memory samples to keep in history (default: 1000)
        """
        self.device = device
        self.cuda_available = torch.cuda.is_available()

        # Memory history tracking
        self.memory_history = deque(maxlen=history_size)

        if self.cuda_available:
            # Get total memory capacity
            self.total_memory = torch.cuda.get_device_properties(device).total_memory
        else:
            self.total_memory = 0

    def get_memory_stats(self) -> dict:
        """
        Get current GPU memory statistics.

        Returns:
            dict: Memory statistics with keys:
                - allocated_gb: Currently allocated memory in GB
                - reserved_gb: Memory reserved by PyTorch in GB
                - peak_gb: Peak allocated memory since last reset in GB
                - total_gb: Total GPU memory in GB
                - free_gb: Available memory in GB
                - utilization: Memory utilization as fraction (0.0-1.0)
        """
        if not self.cuda_available:
            return {
                "allocated_gb": 0.0,
                "reserved_gb": 0.0,
                "peak_gb": 0.0,
                "total_gb": 0.0,
                "free_gb": 0.0,
                "utilization": 0.0,
            }

        # Get memory stats in bytes
        allocated_bytes = torch.cuda.memory_allocated(self.device)
        reserved_bytes = torch.cuda.memory_reserved(self.device)
        peak_bytes = torch.cuda.max_memory_allocated(self.device)

        # Convert to GB
        allocated_gb = allocated_bytes / 1e9
        reserved_gb = reserved_bytes / 1e9
        peak_gb = peak_bytes / 1e9
        total_gb = self.total_memory / 1e9
        free_gb = total_gb - allocated_gb

        # Calculate utilization
        utilization = allocated_gb / total_gb if total_gb > 0 else 0.0

        return {
            "allocated_gb": round(allocated_gb, 2),
            "reserved_gb": round(reserved_gb, 2),
            "peak_gb": round(peak_gb, 2),
            "total_gb": round(total_gb, 2),
            "free_gb": round(free_gb, 2),
            "utilization": round(utilization, 3),
        }

    def reset_peak(self):
        """
        Reset peak memory statistics.

        Useful for tracking peak memory usage for specific training phases
        (e.g., forward pass, backward pass, optimizer step).
        """
        if self.cuda_available:
            torch.cuda.reset_peak_memory_stats(self.device)

    def format_memory_display(self) -> str:
        """
        Format memory statistics as a human-readable string.

        Returns:
            str: Formatted memory display string

        Example:
            "GPU Memory: 8.45GB / 23.69GB (35.7%) | Peak: 12.34GB | Reserved: 10.50GB"
        """
        stats = self.get_memory_stats()

        if not self.cuda_available:
            return "GPU Memory: No CUDA device available"

        return (
            f"GPU Memory: {stats['allocated_gb']:.2f}GB / {stats['total_gb']:.2f}GB "
            f"({stats['utilization']*100:.1f}%) | "
            f"Peak: {stats['peak_gb']:.2f}GB | "
            f"Reserved: {stats['reserved_gb']:.2f}GB"
        )

    def get_memory_summary(self) -> str:
        """
        Get a compact memory summary for logging.

        Returns:
            str: Compact memory summary

        Example:
            "8.45GB (35.7%)"
        """
        stats = self.get_memory_stats()

        if not self.cuda_available:
            return "No GPU"

        return f"{stats['allocated_gb']:.2f}GB ({stats['utilization']*100:.1f}%)"

    def check_memory_available(self, required_gb: float) -> bool:
        """
        Check if sufficient memory is available.

        Args:
            required_gb: Required memory in GB

        Returns:
            bool: True if sufficient memory available, False otherwise
        """
        if not self.cuda_available:
            return False

        stats = self.get_memory_stats()
        return stats['free_gb'] >= required_gb

    def get_device_name(self) -> str:
        """
        Get the name of the monitored GPU device.

        Returns:
            str: GPU device name or "CPU" if no CUDA device
        """
        if not self.cuda_available:
            return "CPU"

        return torch.cuda.get_device_name(self.device)
