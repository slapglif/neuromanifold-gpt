"""Memory utilization evaluation metrics.

This module provides metrics for evaluating SDR Engram Memory utilization:
- Memory capacity and usage statistics
- Layer-wise memory statistics (L1, L2 caches)
- Memory efficiency metrics
- Retrieval statistics

These metrics help monitor whether the engram memory system is being utilized
effectively and whether capacity limits are being approached.

Reference: neuromanifold_gpt/model/memory/engram.py (SDREngramMemory)
"""

from typing import Any, Dict


class MemoryMetrics:
    """Compute evaluation metrics for memory utilization.

    Analyzes memory usage statistics from the model's info dict or from
    direct memory state queries.
    """

    @staticmethod
    def compute_capacity_statistics(memory_stats: Dict[str, Any]) -> Dict[str, float]:
        """Compute memory capacity and utilization statistics.

        Args:
            memory_stats: Dictionary containing memory state info:
                - 'memory_size': Current number of stored memories (int or tensor)
                - 'capacity': Maximum capacity (optional, int or tensor)
                - 'sdr_size': Size of SDR vectors (optional)
                - 'content_dim': Dimension of content vectors (optional)

        Returns:
            Dictionary with capacity statistics:
                - memory_size: Current number of stored memories
                - memory_utilization: Fraction of capacity used (if capacity provided)
                - memory_capacity: Maximum capacity (if provided)
        """
        metrics = {}

        # Extract memory size
        memory_size = memory_stats.get("memory_size", 0)
        # Handle tensor inputs
        if hasattr(memory_size, "item"):
            memory_size = memory_size.item()
        metrics["memory_size"] = float(memory_size)

        # Extract capacity if available
        capacity = memory_stats.get("capacity")
        if capacity is not None:
            if hasattr(capacity, "item"):
                capacity = capacity.item()
            metrics["memory_capacity"] = float(capacity)

            # Compute utilization
            if capacity > 0:
                metrics["memory_utilization"] = float(memory_size) / float(capacity)
            else:
                metrics["memory_utilization"] = 0.0

        # Extract dimensions if available
        sdr_size = memory_stats.get("sdr_size")
        if sdr_size is not None:
            if hasattr(sdr_size, "item"):
                sdr_size = sdr_size.item()
            metrics["sdr_size"] = float(sdr_size)

        content_dim = memory_stats.get("content_dim")
        if content_dim is not None:
            if hasattr(content_dim, "item"):
                content_dim = content_dim.item()
            metrics["content_dim"] = float(content_dim)

        return metrics

    @staticmethod
    def compute_layer_statistics(memory_stats: Dict[str, Any]) -> Dict[str, float]:
        """Compute layer-wise memory statistics.

        In hierarchical memory architectures, different layers may maintain
        separate memory buffers (L1 for recent context, L2 for long-term, etc.).

        Args:
            memory_stats: Dictionary containing layer-specific counts:
                - 'l1_count': Number of L1 cache memories
                - 'l2_count': Number of L2 cache memories
                - 'l3_count': Number of L3 cache memories (optional)
                - etc.

        Returns:
            Dictionary with layer statistics:
                - l1_count: L1 cache size
                - l2_count: L2 cache size
                - total_layered_memories: Sum of all layer counts
                - l1_fraction: Fraction in L1 (if total > 0)
                - l2_fraction: Fraction in L2 (if total > 0)
        """
        metrics = {}
        layer_counts = []

        # Extract layer counts
        for layer_key in ["l1_count", "l2_count", "l3_count", "l4_count"]:
            count = memory_stats.get(layer_key)
            if count is not None:
                if hasattr(count, "item"):
                    count = count.item()
                metrics[layer_key] = float(count)
                layer_counts.append(float(count))

        # Compute total and fractions
        if layer_counts:
            total = sum(layer_counts)
            metrics["total_layered_memories"] = total

            if total > 0:
                # Compute fractions for each layer present
                if "l1_count" in metrics:
                    metrics["l1_fraction"] = metrics["l1_count"] / total
                if "l2_count" in metrics:
                    metrics["l2_fraction"] = metrics["l2_count"] / total
                if "l3_count" in metrics:
                    metrics["l3_fraction"] = metrics["l3_count"] / total
                if "l4_count" in metrics:
                    metrics["l4_fraction"] = metrics["l4_count"] / total

        return metrics

    @staticmethod
    def compute_retrieval_statistics(memory_stats: Dict[str, Any]) -> Dict[str, float]:
        """Compute memory retrieval statistics.

        Args:
            memory_stats: Dictionary containing retrieval info:
                - 'num_retrievals': Total number of retrieval operations
                - 'avg_similarity': Average retrieval similarity scores
                - 'retrieval_hit_rate': Fraction of successful retrievals
                - 'avg_retrieved': Average number of memories retrieved per query

        Returns:
            Dictionary with retrieval statistics:
                - num_retrievals: Total retrievals performed
                - avg_similarity: Average similarity scores
                - retrieval_hit_rate: Success rate
                - avg_retrieved: Average retrieved per query
        """
        metrics = {}

        for key in [
            "num_retrievals",
            "avg_similarity",
            "retrieval_hit_rate",
            "avg_retrieved",
        ]:
            value = memory_stats.get(key)
            if value is not None:
                if hasattr(value, "item"):
                    value = value.item()
                metrics[key] = float(value)

        return metrics

    @staticmethod
    def compute_all(memory_stats: Dict[str, Any]) -> Dict[str, float]:
        """Compute all memory utilization metrics.

        Args:
            memory_stats: Dictionary containing memory state information.
                Can include any combination of:
                - Basic: memory_size, capacity, sdr_size, content_dim
                - Layers: l1_count, l2_count, l3_count, ...
                - Retrieval: num_retrievals, avg_similarity, retrieval_hit_rate

        Returns:
            Dictionary with all available metrics:
                - memory_size: Current number of stored memories
                - memory_utilization: Fraction of capacity used
                - total_size: Total memory footprint (alias for memory_size)
                - l1_count, l2_count: Layer-wise counts
                - total_layered_memories: Sum of layer counts
                - l1_fraction, l2_fraction: Layer distribution
                - num_retrievals: Retrieval statistics
                - avg_similarity: Average retrieval similarity
                - (additional metrics as available)
        """
        metrics = {}

        # Compute capacity statistics
        capacity_metrics = MemoryMetrics.compute_capacity_statistics(memory_stats)
        metrics.update(capacity_metrics)

        # Add total_size as an alias for memory_size (required by verification)
        if "memory_size" in metrics:
            metrics["total_size"] = metrics["memory_size"]

        # Compute layer statistics
        layer_metrics = MemoryMetrics.compute_layer_statistics(memory_stats)
        metrics.update(layer_metrics)

        # Compute retrieval statistics
        retrieval_metrics = MemoryMetrics.compute_retrieval_statistics(memory_stats)
        metrics.update(retrieval_metrics)

        return metrics
