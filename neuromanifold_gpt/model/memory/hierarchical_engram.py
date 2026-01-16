"""
Hierarchical SDR Engram Memory - 3-tier cache architecture.

Inspired by CPU cache hierarchy (L1/L2/L3), this memory system
provides fast access to recent memories (L1) while maintaining
a larger pool of older memories in slower tiers (L2/L3).

Memory promotion:
- New memories go to L1 (hot cache)
- When L1 fills, oldest items promote to L2 (warm cache)
- When L2 fills, oldest items promote to L3 (cold storage)
- L3 evicts oldest items when full (circular buffer)
"""
import torch
import torch.nn as nn
from neuromanifold_gpt.model.memory.engram import SDREngramMemory


class HierarchicalEngramMemory(nn.Module):
    """
    3-tier hierarchical memory using SDR engrams.

    Architecture:
    - L1: Hot cache for most recent memories (fast access)
    - L2: Warm cache for recent memories (medium access)
    - L3: Cold storage for older memories (slow access)

    Retrieval priority: L1 -> L2 -> L3 (checks hot cache first)

    This design optimizes for temporal locality - recent memories
    are accessed more frequently and should be in fast tiers.

    Args:
        sdr_size: Size of SDR vectors (typically 2048)
        n_active: Number of active bits in SDRs (for similarity)
        content_dim: Dimension of content vectors (default 384)
        l1_capacity: Max items in L1 hot cache (default 128)
        l2_capacity: Max items in L2 warm cache (default 512)
        l3_capacity: Max items in L3 cold storage (default 2048)
        threshold: Minimum similarity for retrieval (default 0.3)
    """

    def __init__(
        self,
        sdr_size: int,
        n_active: int,
        content_dim: int = 384,
        l1_capacity: int = 128,
        l2_capacity: int = 512,
        l3_capacity: int = 2048,
        threshold: float = 0.3,
    ):
        super().__init__()
        self.sdr_size = sdr_size
        self.n_active = n_active
        self.content_dim = content_dim
        self.threshold = threshold

        # Create 3-tier memory hierarchy
        # L1: Hot cache (most recent, fastest access)
        self.l1 = SDREngramMemory(
            sdr_size=sdr_size,
            capacity=l1_capacity,
            n_active=n_active,
            content_dim=content_dim,
            threshold=threshold,
        )

        # L2: Warm cache (recent, medium access)
        self.l2 = SDREngramMemory(
            sdr_size=sdr_size,
            capacity=l2_capacity,
            n_active=n_active,
            content_dim=content_dim,
            threshold=threshold,
        )

        # L3: Cold storage (older memories, slower access)
        self.l3 = SDREngramMemory(
            sdr_size=sdr_size,
            capacity=l3_capacity,
            n_active=n_active,
            content_dim=content_dim,
            threshold=threshold,
        )

    def __len__(self) -> int:
        """Return total number of stored memories across all tiers."""
        return len(self.l1) + len(self.l2) + len(self.l3)

    def store(self, sdr: torch.Tensor, content: torch.Tensor) -> None:
        """
        Store SDR-content pair with tier promotion logic.

        New memories go to L1 (hot cache). When L1 fills, oldest
        items are promoted to L2, and from L2 to L3 as needed.

        Args:
            sdr: (sdr_size,) SDR vector
            content: (content_dim,) content vector
        """
        # If L1 is at capacity, promote oldest entry before overwriting
        if len(self.l1) >= self.l1.capacity:
            # Get the item about to be overwritten (at write_ptr)
            old_ptr = self.l1.write_ptr.item()
            old_sdr = self.l1.sdr_bank[old_ptr]
            old_content = self.l1.content_bank[old_ptr]

            # Promote to L2 (which may trigger L2->L3 promotion)
            if len(self.l2) >= self.l2.capacity:
                # Get item about to be overwritten in L2
                l2_ptr = self.l2.write_ptr.item()
                l2_sdr = self.l2.sdr_bank[l2_ptr]
                l2_content = self.l2.content_bank[l2_ptr]

                # Promote to L3
                self.l3.store(l2_sdr, l2_content)

            # Store old L1 item in L2
            self.l2.store(old_sdr, old_content)

        # Store new item in L1
        self.l1.store(sdr, content)

    def store_batch(self, sdrs: torch.Tensor, contents: torch.Tensor) -> None:
        """
        Store multiple SDR-content pairs with tier promotion logic (vectorized).

        New memories go to L1 (hot cache). When L1 fills, oldest
        items are promoted to L2, and from L2 to L3 as needed.

        This is the key optimization - instead of calling store() N times
        (which triggers per-item promotion overhead), we handle the entire
        batch in one pass.

        Args:
            sdrs: (N, sdr_size) batch of SDR vectors
            contents: (N, content_dim) batch of content vectors
        """
        # Detach inputs
        sdrs = sdrs.detach()
        contents = contents.detach()

        N = sdrs.shape[0]
        if N == 0:
            return

        # Calculate how many L1 items will be overwritten
        l1_count = len(self.l1)
        l1_capacity = self.l1.capacity
        n_overflow = max(0, l1_count + N - l1_capacity)

        if n_overflow > 0:
            # Get items that will be overwritten in L1
            # These are at indices [write_ptr, write_ptr + n_overflow) mod capacity
            l1_ptr = self.l1.write_ptr.item()
            overflow_indices = torch.arange(n_overflow, device=self.l1.sdr_bank.device)
            overflow_indices = (l1_ptr + overflow_indices) % l1_capacity

            l1_sdrs_to_promote = self.l1.sdr_bank[overflow_indices]
            l1_contents_to_promote = self.l1.content_bank[overflow_indices]

            # Now we need to promote these to L2, which may trigger L2->L3 promotion
            l2_count = len(self.l2)
            l2_capacity = self.l2.capacity
            n_l2_overflow = max(0, l2_count + n_overflow - l2_capacity)

            if n_l2_overflow > 0:
                # Get items that will be overwritten in L2
                l2_ptr = self.l2.write_ptr.item()
                l2_overflow_indices = torch.arange(n_l2_overflow, device=self.l2.sdr_bank.device)
                l2_overflow_indices = (l2_ptr + l2_overflow_indices) % l2_capacity

                l2_sdrs_to_promote = self.l2.sdr_bank[l2_overflow_indices]
                l2_contents_to_promote = self.l2.content_bank[l2_overflow_indices]

                # Promote to L3
                self.l3.store_batch(l2_sdrs_to_promote, l2_contents_to_promote)

            # Promote L1 items to L2
            self.l2.store_batch(l1_sdrs_to_promote, l1_contents_to_promote)

        # Store new items in L1
        self.l1.store_batch(sdrs, contents)

    def retrieve(
        self,
        query_sdr: torch.Tensor,
        top_k: int = 5,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        """
        Retrieve by SDR similarity with tier priority: L1 -> L2 -> L3.

        Searches hot cache (L1) first, then warm cache (L2), then cold
        storage (L3). Returns results from the first tier that has
        matches above threshold.

        This implements cache hierarchy - we check fastest tiers first
        and only fall back to slower tiers if needed.

        Args:
            query_sdr: (sdr_size,) query SDR
            top_k: Maximum number of results to return

        Returns:
            contents: (n_results, content_dim) retrieved content vectors
            similarities: (n_results,) overlap-based similarity scores
            tier: Which tier the results came from ('L1', 'L2', or 'L3')
        """
        # If completely empty, return empty results
        if len(self) == 0:
            device = self.l1.content_bank.device
            return (
                torch.zeros(0, self.content_dim, device=device),
                torch.zeros(0, device=device),
                "L1",
            )

        # Try L1 first (hot cache, fastest)
        contents, similarities = self.l1.retrieve(query_sdr, top_k=top_k)
        if len(similarities) > 0:
            return contents, similarities, "L1"

        # Fall back to L2 (warm cache)
        contents, similarities = self.l2.retrieve(query_sdr, top_k=top_k)
        if len(similarities) > 0:
            return contents, similarities, "L2"

        # Fall back to L3 (cold storage)
        contents, similarities = self.l3.retrieve(query_sdr, top_k=top_k)
        if len(similarities) > 0:
            return contents, similarities, "L3"

        # No results found in any tier, return empty from L1
        device = self.l1.content_bank.device
        return (
            torch.zeros(0, self.content_dim, device=device),
            torch.zeros(0, device=device),
            "L1",
        )

    def get_stats(self) -> dict:
        """
        Get statistics about memory usage across all tiers.

        Returns:
            Dictionary with tier-specific and aggregate statistics:
            - l1_size: Number of memories in L1 (hot cache)
            - l2_size: Number of memories in L2 (warm cache)
            - l3_size: Number of memories in L3 (cold storage)
            - memory_size: Total memories across all tiers
            - l1_capacity: Maximum capacity of L1
            - l2_capacity: Maximum capacity of L2
            - l3_capacity: Maximum capacity of L3
            - total_capacity: Total capacity across all tiers
        """
        return {
            "l1_size": len(self.l1),
            "l2_size": len(self.l2),
            "l3_size": len(self.l3),
            "memory_size": len(self),
            "l1_capacity": self.l1.capacity,
            "l2_capacity": self.l2.capacity,
            "l3_capacity": self.l3.capacity,
            "total_capacity": self.l1.capacity + self.l2.capacity + self.l3.capacity,
        }

    def clear(self) -> None:
        """Clear all memories across all tiers."""
        self.l1.clear()
        self.l2.clear()
        self.l3.clear()
