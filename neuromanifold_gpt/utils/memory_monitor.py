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

    def record_sample(self):
        """
        Record a memory snapshot to the history.

        Captures current memory statistics and adds them to the memory history
        for profiling and analysis. The history is limited by the history_size
        parameter set during initialization.
        """
        stats = self.get_memory_stats()
        self.memory_history.append(stats)

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

    def get_history_stats(self) -> dict:
        """
        Calculate min/max/avg statistics from memory history.

        Computes rolling statistics across all recorded memory samples.
        Useful for profiling memory usage patterns over time.

        Returns:
            dict: Statistics with keys:
                - min_allocated_gb: Minimum allocated memory in GB
                - max_allocated_gb: Maximum allocated memory in GB
                - avg_allocated_gb: Average allocated memory in GB
                - min_reserved_gb: Minimum reserved memory in GB
                - max_reserved_gb: Maximum reserved memory in GB
                - avg_reserved_gb: Average reserved memory in GB
                - min_utilization: Minimum memory utilization (0.0-1.0)
                - max_utilization: Maximum memory utilization (0.0-1.0)
                - avg_utilization: Average memory utilization (0.0-1.0)
                - sample_count: Number of samples in history

        Example:
            >>> monitor = GPUMemoryMonitor()
            >>> monitor.record_sample()
            >>> monitor.record_sample()
            >>> stats = monitor.get_history_stats()
            >>> print(f"Avg Memory: {stats['avg_allocated_gb']:.2f}GB")
        """
        if not self.memory_history:
            # Return zeros if no history available
            return {
                "min_allocated_gb": 0.0,
                "max_allocated_gb": 0.0,
                "avg_allocated_gb": 0.0,
                "min_reserved_gb": 0.0,
                "max_reserved_gb": 0.0,
                "avg_reserved_gb": 0.0,
                "min_utilization": 0.0,
                "max_utilization": 0.0,
                "avg_utilization": 0.0,
                "sample_count": 0,
            }

        # Convert deque to list for processing
        history_list = list(self.memory_history)
        sample_count = len(history_list)

        # Extract individual metric lists
        allocated_values = [sample['allocated_gb'] for sample in history_list]
        reserved_values = [sample['reserved_gb'] for sample in history_list]
        utilization_values = [sample['utilization'] for sample in history_list]

        # Calculate statistics for each metric
        return {
            "min_allocated_gb": round(min(allocated_values), 2),
            "max_allocated_gb": round(max(allocated_values), 2),
            "avg_allocated_gb": round(sum(allocated_values) / sample_count, 2),
            "min_reserved_gb": round(min(reserved_values), 2),
            "max_reserved_gb": round(max(reserved_values), 2),
            "avg_reserved_gb": round(sum(reserved_values) / sample_count, 2),
            "min_utilization": round(min(utilization_values), 3),
            "max_utilization": round(max(utilization_values), 3),
            "avg_utilization": round(sum(utilization_values) / sample_count, 3),
            "sample_count": sample_count,
        }

    def clear_history(self):
        """
        Clear all recorded memory samples from history.

        Useful for resetting memory profiling between training phases or
        experiments without recreating the monitor instance.

        Example:
            >>> monitor = GPUMemoryMonitor()
            >>> monitor.record_sample()
            >>> monitor.clear_history()
            >>> len(monitor.memory_history)
            0
        """
        self.memory_history.clear()

    def export_timeline(self) -> list:
        """
        Export memory history as a list for profiling reports.

        Returns the complete memory history as a list of dictionaries,
        where each dictionary contains a snapshot of memory statistics.
        Useful for generating memory profiling reports and visualizations.

        Returns:
            list: List of memory sample dictionaries, each containing:
                - allocated_gb: Memory allocated at sample time
                - reserved_gb: Memory reserved at sample time
                - peak_gb: Peak memory at sample time
                - total_gb: Total GPU memory
                - free_gb: Free memory at sample time
                - utilization: Memory utilization (0.0-1.0)

        Example:
            >>> monitor = GPUMemoryMonitor()
            >>> monitor.record_sample()
            >>> monitor.record_sample()
            >>> timeline = monitor.export_timeline()
            >>> print(f"Recorded {len(timeline)} samples")
        """
        return list(self.memory_history)
