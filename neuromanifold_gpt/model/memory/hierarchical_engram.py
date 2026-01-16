"""
Hierarchical SDR Engram Memory - Multi-tiered memory consolidation.

Implements a 3-tier memory hierarchy inspired by biological memory systems:
- L1 (Working Memory): Fast access, small capacity, recent memories
- L2 (Short-term Memory): Medium capacity, frequently accessed memories
- L3 (Long-term Memory): Large capacity, consolidated memories

This mirrors the hippocampus -> cortex consolidation pathway, where
memories move from fast-access short-term storage to stable long-term
storage based on access patterns and time.
"""
import torch
import torch.nn as nn


class HierarchicalEngramMemory(nn.Module):
    """
    Three-tiered hierarchical memory using SDR overlap for retrieval.

    Key architectural principles:
    1. **Working Memory (L1)**: Stores most recent memories for immediate recall
       - Small capacity (default 64), circular buffer replacement
       - First checked during retrieval (fast path)

    2. **Short-term Memory (L2)**: Caches frequently accessed memories
       - Medium capacity (default 512)
       - Receives evicted L1 memories
       - Acts as a buffer before long-term consolidation

    3. **Long-term Memory (L3)**: Stable storage for consolidated knowledge
       - Large capacity (default 4096)
       - Receives evicted L2 memories
       - Represents learned knowledge base

    Memory Flow:
    - New memories enter L1 (working memory)
    - When L1 fills, oldest memories are pushed to L2
    - When L2 fills, oldest memories are pushed to L3
    - Retrieval checks L1 -> L2 -> L3 in order (fast to slow)

    This tiered approach provides:
    - Fast access to recent memories (L1 hit rate)
    - Memory consolidation without catastrophic forgetting
    - Efficient capacity scaling (only L3 needs to be large)
    - Biologically plausible memory dynamics

    Args:
        sdr_size: Size of SDR vectors (typically 2048)
        n_active: Number of active bits in SDRs (for similarity normalization)
        content_dim: Dimension of content vectors (default 384)
        l1_capacity: Working memory capacity (default 64)
        l2_capacity: Short-term memory capacity (default 512)
        l3_capacity: Long-term memory capacity (default 4096)
        threshold: Minimum similarity for retrieval (default 0.3)
    """

    def __init__(
        self,
        sdr_size: int,
        n_active: int,
        content_dim: int = 384,
        l1_capacity: int = 64,
        l2_capacity: int = 512,
        l3_capacity: int = 4096,
        threshold: float = 0.3,
    ):
        super().__init__()
        self.sdr_size = sdr_size
        self.n_active = n_active
        self.content_dim = content_dim
        self.l1_capacity = l1_capacity
        self.l2_capacity = l2_capacity
        self.l3_capacity = l3_capacity
        self.threshold = threshold

        # L1: Working Memory (fast, small)
        self.register_buffer("l1_sdr_bank", torch.zeros(l1_capacity, sdr_size))
        self.register_buffer("l1_content_bank", torch.zeros(l1_capacity, content_dim))
        self.register_buffer("l1_valid_mask", torch.zeros(l1_capacity, dtype=torch.bool))
        self.register_buffer("l1_write_ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("l1_count", torch.tensor(0, dtype=torch.long))

        # L2: Short-term Memory (medium capacity)
        self.register_buffer("l2_sdr_bank", torch.zeros(l2_capacity, sdr_size))
        self.register_buffer("l2_content_bank", torch.zeros(l2_capacity, content_dim))
        self.register_buffer("l2_valid_mask", torch.zeros(l2_capacity, dtype=torch.bool))
        self.register_buffer("l2_write_ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("l2_count", torch.tensor(0, dtype=torch.long))

        # L3: Long-term Memory (large capacity)
        self.register_buffer("l3_sdr_bank", torch.zeros(l3_capacity, sdr_size))
        self.register_buffer("l3_content_bank", torch.zeros(l3_capacity, content_dim))
        self.register_buffer("l3_valid_mask", torch.zeros(l3_capacity, dtype=torch.bool))
        self.register_buffer("l3_write_ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("l3_count", torch.tensor(0, dtype=torch.long))

    def get_size(self) -> dict[str, torch.Tensor]:
        """Return number of memories in each tier (as Tensors to avoid graph break)."""
        return {
            "l1": torch.min(self.l1_count, torch.tensor(self.l1_capacity, device=self.l1_count.device)),
            "l2": torch.min(self.l2_count, torch.tensor(self.l2_capacity, device=self.l2_count.device)),
            "l3": torch.min(self.l3_count, torch.tensor(self.l3_capacity, device=self.l3_count.device)),
        }

    def __len__(self) -> int:
        """Return total number of stored memories across all tiers (Python int, causes graph break)."""
        return int(
            min(self.l1_count.item(), self.l1_capacity)
            + min(self.l2_count.item(), self.l2_capacity)
            + min(self.l3_count.item(), self.l3_capacity)
        )

    def _evict_from_l1_to_l2(self) -> None:
        """
        Evict oldest memory from L1 to L2.

        When L1 reaches capacity, we move the oldest entry to L2.
        This implements the working memory -> short-term memory transition.
        """
        if self.l1_count == 0:
            return

        # Get the oldest entry (at current write pointer, about to be overwritten)
        evict_idx = self.l1_write_ptr

        # Copy to L2
        sdr = self.l1_sdr_bank[evict_idx:evict_idx+1]  # (1, sdr_size)
        content = self.l1_content_bank[evict_idx:evict_idx+1]  # (1, content_dim)

        ptr = self.l2_write_ptr.view(1)
        self.l2_sdr_bank.index_copy_(0, ptr, sdr)
        self.l2_content_bank.index_copy_(0, ptr, content)
        self.l2_valid_mask.index_fill_(0, ptr, torch.tensor(True, device=self.l2_valid_mask.device))

        # Advance L2 pointer
        next_ptr = (self.l2_write_ptr + 1) % self.l2_capacity
        self.l2_write_ptr.copy_(next_ptr)
        next_count = torch.clamp(self.l2_count + 1, max=self.l2_capacity)
        self.l2_count.copy_(next_count)

        # If L2 is now full, evict to L3
        if self.l2_count >= self.l2_capacity:
            self._evict_from_l2_to_l3()

    def _evict_from_l2_to_l3(self) -> None:
        """
        Evict oldest memory from L2 to L3.

        When L2 reaches capacity, we move the oldest entry to L3.
        This implements the short-term -> long-term memory consolidation.
        """
        if self.l2_count == 0:
            return

        # Get the oldest entry from L2
        evict_idx = self.l2_write_ptr

        # Copy to L3
        sdr = self.l2_sdr_bank[evict_idx:evict_idx+1]  # (1, sdr_size)
        content = self.l2_content_bank[evict_idx:evict_idx+1]  # (1, content_dim)

        ptr = self.l3_write_ptr.view(1)
        self.l3_sdr_bank.index_copy_(0, ptr, sdr)
        self.l3_content_bank.index_copy_(0, ptr, content)
        self.l3_valid_mask.index_fill_(0, ptr, torch.tensor(True, device=self.l3_valid_mask.device))

        # Advance L3 pointer
        next_ptr = (self.l3_write_ptr + 1) % self.l3_capacity
        self.l3_write_ptr.copy_(next_ptr)
        next_count = torch.clamp(self.l3_count + 1, max=self.l3_capacity)
        self.l3_count.copy_(next_count)

    def store(self, sdr: torch.Tensor, content: torch.Tensor) -> None:
        """
        Store SDR-content pair in L1 (working memory).

        If L1 is full, the oldest entry is evicted to L2 before storing.
        This maintains the memory hierarchy: new memories always enter
        at the top (L1) and flow down through the tiers over time.

        Args:
            sdr: (sdr_size,) SDR marker
            content: (content_dim,) content vector
        """
        # Detach inputs to prevent autograd issues
        sdr = sdr.detach()
        content = content.detach()

        # If L1 is full, evict oldest to L2 first
        if self.l1_count >= self.l1_capacity:
            self._evict_from_l1_to_l2()

        # Store in L1
        ptr = self.l1_write_ptr.view(1)

        # Ensure inputs are on correct device
        sdr = sdr.to(self.l1_sdr_bank.device).unsqueeze(0)  # (1, sdr_size)
        content = content.to(self.l1_content_bank.device).unsqueeze(0)  # (1, content_dim)

        # In-place update using index_copy_
        self.l1_sdr_bank.index_copy_(0, ptr, sdr)
        self.l1_content_bank.index_copy_(0, ptr, content)
        self.l1_valid_mask.index_fill_(0, ptr, torch.tensor(True, device=self.l1_valid_mask.device))

        # Advance write pointer (circular)
        next_ptr = (self.l1_write_ptr + 1) % self.l1_capacity
        self.l1_write_ptr.copy_(next_ptr)

        next_count = torch.clamp(self.l1_count + 1, max=self.l1_capacity)
        self.l1_count.copy_(next_count)

    def store_batch(self, sdrs: torch.Tensor, contents: torch.Tensor) -> None:
        """
        Store multiple SDR-content pairs efficiently (vectorized).

        All memories are stored in L1, triggering cascading evictions
        to L2 and L3 as needed. This maintains the hierarchy while
        allowing efficient batch operations.

        Args:
            sdrs: (N, sdr_size) batch of SDRs
            contents: (N, content_dim) batch of content vectors
        """
        # Detach inputs
        sdrs = sdrs.detach()
        contents = contents.detach()

        N = sdrs.shape[0]
        if N == 0:
            return

        # For simplicity, store one at a time to maintain eviction logic
        # This ensures proper cascading through the hierarchy
        for i in range(N):
            self.store(sdrs[i], contents[i])

    def _retrieve_from_tier(
        self,
        query_sdr: torch.Tensor,
        sdr_bank: torch.Tensor,
        content_bank: torch.Tensor,
        valid_mask: torch.Tensor,
        count: torch.Tensor,
        capacity: int,
        top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve from a single tier using SDR overlap.

        This implements the core HTM-style content-addressable retrieval
        using bit overlap as the similarity metric.

        Args:
            query_sdr: (sdr_size,) query SDR
            sdr_bank: (capacity, sdr_size) stored SDRs
            content_bank: (capacity, content_dim) stored contents
            valid_mask: (capacity,) valid entry mask
            count: scalar tensor with number of stored items
            capacity: tier capacity
            top_k: maximum results to return

        Returns:
            contents: (n_results, content_dim) retrieved content vectors
            similarities: (n_results,) overlap-based similarity scores
        """
        if count == 0:
            return (
                torch.zeros(0, self.content_dim, device=content_bank.device),
                torch.zeros(0, device=sdr_bank.device),
            )

        query_sdr = query_sdr.to(sdr_bank.device)

        # Compute overlap with all stored SDRs
        overlap = (query_sdr.unsqueeze(0) * sdr_bank).sum(dim=-1)

        # Normalize by n_active to get similarity in [0, 1]
        similarity = overlap / self.n_active

        # Mask invalid entries with -inf so they sort to bottom
        similarity = torch.where(
            valid_mask,
            similarity,
            torch.tensor(-float("inf"), device=similarity.device),
        )

        # Get top-k entries
        k = min(top_k, min(int(count.item()), capacity))
        if k == 0:
            return (
                torch.zeros(0, self.content_dim, device=content_bank.device),
                torch.zeros(0, device=sdr_bank.device),
            )

        top_sim, top_idx = torch.topk(similarity, k)

        # Filter by threshold
        mask = top_sim >= self.threshold
        top_sim = top_sim[mask]
        top_idx = top_idx[mask]

        return content_bank[top_idx], top_sim

    def retrieve(
        self,
        query_sdr: torch.Tensor,
        top_k: int = 5,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
        """
        Retrieve by SDR similarity across all tiers.

        Searches L1 -> L2 -> L3 in order, combining results.
        This gives priority to recent memories (L1) while still
        accessing consolidated knowledge (L2, L3) when needed.

        The hierarchical search reflects biological memory retrieval:
        - Check working memory first (fast, recent)
        - Fall back to short-term memory (medium speed)
        - Access long-term memory last (comprehensive)

        Args:
            query_sdr: (sdr_size,) query SDR
            top_k: Maximum number of results to return (split across tiers)

        Returns:
            contents: (n_results, content_dim) retrieved content vectors
            similarities: (n_results,) overlap-based similarity scores
            tier_stats: dict with number of results from each tier
        """
        # Retrieve from each tier
        l1_contents, l1_sims = self._retrieve_from_tier(
            query_sdr, self.l1_sdr_bank, self.l1_content_bank,
            self.l1_valid_mask, self.l1_count, self.l1_capacity, top_k
        )

        l2_contents, l2_sims = self._retrieve_from_tier(
            query_sdr, self.l2_sdr_bank, self.l2_content_bank,
            self.l2_valid_mask, self.l2_count, self.l2_capacity, top_k
        )

        l3_contents, l3_sims = self._retrieve_from_tier(
            query_sdr, self.l3_sdr_bank, self.l3_content_bank,
            self.l3_valid_mask, self.l3_count, self.l3_capacity, top_k
        )

        # Combine results from all tiers
        all_contents = torch.cat([l1_contents, l2_contents, l3_contents], dim=0)
        all_sims = torch.cat([l1_sims, l2_sims, l3_sims], dim=0)

        # Track where results came from
        tier_stats = {
            "l1": len(l1_sims),
            "l2": len(l2_sims),
            "l3": len(l3_sims),
        }

        # If we have results, sort by similarity and take top-k
        if len(all_sims) > 0:
            k = min(top_k, len(all_sims))
            top_sim, top_idx = torch.topk(all_sims, k)
            return all_contents[top_idx], top_sim, tier_stats
        else:
            return all_contents, all_sims, tier_stats

    def clear(self) -> None:
        """Clear all memories from all tiers."""
        # L1
        self.l1_sdr_bank.zero_()
        self.l1_content_bank.zero_()
        self.l1_valid_mask.zero_()
        self.l1_write_ptr.zero_()
        self.l1_count.zero_()

        # L2
        self.l2_sdr_bank.zero_()
        self.l2_content_bank.zero_()
        self.l2_valid_mask.zero_()
        self.l2_write_ptr.zero_()
        self.l2_count.zero_()

        # L3
        self.l3_sdr_bank.zero_()
        self.l3_content_bank.zero_()
        self.l3_valid_mask.zero_()
        self.l3_write_ptr.zero_()
        self.l3_count.zero_()
