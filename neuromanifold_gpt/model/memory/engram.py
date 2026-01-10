"""
SDR Engram Memory - Breadcrumb trail for infinite context.

Stores SDR markers as context window slides. Reconstruction via
SDR overlap lets the model "remember" information outside current window.

Inspired by hippocampus -> cortex memory consolidation.
"""
import torch
import torch.nn as nn


class SDREngramMemory(nn.Module):
    """
    HTM-style memory using SDR overlap for retrieval.

    Key insight: SDR similarity (bit overlap) naturally implements
    content-addressable memory - query with partial pattern to
    retrieve full memory.

    This acts as a "breadcrumb trail" that survives context window
    compaction, enabling effectively infinite context via semantic
    reconstruction.

    Args:
        sdr_size: Size of SDR vectors (typically 2048)
        capacity: Maximum number of memories to store
        n_active: Number of active bits in SDRs (for similarity normalization)
        content_dim: Dimension of content vectors (default 384)
        threshold: Minimum similarity for retrieval (default 0.3)
    """

    def __init__(
        self,
        sdr_size: int,
        capacity: int,
        n_active: int,
        content_dim: int = 384,
        threshold: float = 0.3,
    ):
        super().__init__()
        self.sdr_size = sdr_size
        self.capacity = capacity
        self.n_active = n_active
        self.content_dim = content_dim
        self.threshold = threshold

        # Storage as buffers (move with model to correct device)
        self.register_buffer("sdr_bank", torch.zeros(capacity, sdr_size))
        self.register_buffer("content_bank", torch.zeros(capacity, content_dim))
        self.register_buffer("valid_mask", torch.zeros(capacity, dtype=torch.bool))
        self.register_buffer("write_ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    def __len__(self) -> int:
        """Return number of stored memories."""
        return min(self.count.item(), self.capacity)

    def store(self, sdr: torch.Tensor, content: torch.Tensor) -> None:
        """
        Store SDR-content pair.

        Uses circular buffer with FIFO eviction when at capacity.

        Args:
            sdr: (sdr_size,) binary SDR marker
            content: (content_dim,) associated content vector
        """
        ptr = self.write_ptr.item()

        # Store at current write position
        self.sdr_bank[ptr] = sdr.to(self.sdr_bank.device)
        self.content_bank[ptr] = content.to(self.content_bank.device)
        self.valid_mask[ptr] = True

        # Advance write pointer (circular)
        self.write_ptr = (self.write_ptr + 1) % self.capacity
        self.count = torch.clamp(self.count + 1, max=self.capacity)

    def retrieve(
        self,
        query_sdr: torch.Tensor,
        top_k: int = 5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve by SDR similarity (bit overlap).

        The key insight from HTM: semantic similarity between SDRs
        is simply the number of overlapping active bits. This gives
        us efficient content-addressable memory without learned
        similarity functions.

        Args:
            query_sdr: (sdr_size,) query SDR
            top_k: Maximum number of results to return

        Returns:
            contents: (n_results, content_dim) retrieved content vectors
            similarities: (n_results,) overlap-based similarity scores
        """
        if self.count == 0:
            return (
                torch.zeros(0, self.content_dim, device=self.content_bank.device),
                torch.zeros(0, device=self.content_bank.device),
            )

        query_sdr = query_sdr.to(self.sdr_bank.device)

        # Compute overlap with all stored SDRs
        # overlap = sum of (query AND stored) bits
        overlap = (query_sdr.unsqueeze(0) * self.sdr_bank).sum(dim=-1)

        # Normalize by n_active to get similarity in [0, 1]
        similarity = overlap / self.n_active

        # Mask invalid entries with -inf so they sort to bottom
        similarity = torch.where(
            self.valid_mask,
            similarity,
            torch.tensor(-float("inf"), device=similarity.device),
        )

        # Get top-k entries
        k = min(top_k, len(self))
        top_sim, top_idx = torch.topk(similarity, k)

        # Filter by threshold
        mask = top_sim >= self.threshold
        top_sim = top_sim[mask]
        top_idx = top_idx[mask]

        return self.content_bank[top_idx], top_sim

    def clear(self) -> None:
        """Clear all memories."""
        self.sdr_bank.zero_()
        self.content_bank.zero_()
        self.valid_mask.zero_()
        self.write_ptr.zero_()
        self.count.zero_()
