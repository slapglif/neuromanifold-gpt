"""
Hierarchical SDR Engram Memory - Three-tier memory system.

Extends SDREngramMemory with a hierarchical structure:
- L1: Hot/fast cache (small capacity, frequent access)
- L2: Warm storage (medium capacity, moderate access)
- L3: Cold archive (large capacity, rare access)

Automatic tier promotion/demotion based on access frequency.
"""
import torch
import torch.nn as nn


class HierarchicalEngramMemory(nn.Module):
    """
    Three-tier hierarchical memory system with automatic tier management.

    Memory tiers mimic CPU cache hierarchy:
    - L1: Hot memories (recently/frequently accessed)
    - L2: Warm memories (moderately accessed)
    - L3: Cold memories (rarely accessed, potentially compressed)

    Items are automatically promoted to higher tiers when accessed frequently,
    and demoted to lower tiers when access frequency drops.

    Args:
        sdr_size: Size of SDR vectors (typically 2048)
        n_active: Number of active bits in SDRs (for similarity normalization)
        content_dim: Dimension of content vectors (default 384)
        l1_capacity: Maximum memories in L1 tier (hot cache)
        l2_capacity: Maximum memories in L2 tier (warm storage)
        l3_capacity: Maximum memories in L3 tier (cold archive)
        threshold: Minimum similarity for retrieval (default 0.3)
        promotion_threshold: Access count for promotion (default 3)
        demotion_threshold: Access count for demotion (default 1)
    """

    def __init__(
        self,
        sdr_size: int,
        n_active: int,
        content_dim: int,
        l1_capacity: int,
        l2_capacity: int,
        l3_capacity: int,
        threshold: float = 0.3,
        promotion_threshold: int = 3,
        demotion_threshold: int = 1,
    ):
        super().__init__()
        self.sdr_size = sdr_size
        self.n_active = n_active
        self.content_dim = content_dim
        self.l1_capacity = l1_capacity
        self.l2_capacity = l2_capacity
        self.l3_capacity = l3_capacity
        self.threshold = threshold
        self.promotion_threshold = promotion_threshold
        self.demotion_threshold = demotion_threshold

        # L1 tier (hot/fast)
        self.register_buffer("l1_sdr_bank", torch.zeros(l1_capacity, sdr_size))
        self.register_buffer("l1_content_bank", torch.zeros(l1_capacity, content_dim))
        self.register_buffer("l1_access_count", torch.zeros(l1_capacity, dtype=torch.long))
        self.register_buffer("l1_valid_mask", torch.zeros(l1_capacity, dtype=torch.bool))
        self.register_buffer("l1_write_ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("l1_count", torch.tensor(0, dtype=torch.long))

        # L2 tier (warm/medium)
        self.register_buffer("l2_sdr_bank", torch.zeros(l2_capacity, sdr_size))
        self.register_buffer("l2_content_bank", torch.zeros(l2_capacity, content_dim))
        self.register_buffer("l2_access_count", torch.zeros(l2_capacity, dtype=torch.long))
        self.register_buffer("l2_valid_mask", torch.zeros(l2_capacity, dtype=torch.bool))
        self.register_buffer("l2_write_ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("l2_count", torch.tensor(0, dtype=torch.long))

        # L3 tier (cold/compressed)
        self.register_buffer("l3_sdr_bank", torch.zeros(l3_capacity, sdr_size))
        self.register_buffer("l3_content_bank", torch.zeros(l3_capacity, content_dim))
        self.register_buffer("l3_access_count", torch.zeros(l3_capacity, dtype=torch.long))
        self.register_buffer("l3_valid_mask", torch.zeros(l3_capacity, dtype=torch.bool))
        self.register_buffer("l3_write_ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("l3_count", torch.tensor(0, dtype=torch.long))

    def get_stats(self) -> dict:
        """
        Return statistics about memory tier usage.

        Returns:
            dict with keys:
                - l1_count: Number of memories in L1
                - l2_count: Number of memories in L2
                - l3_count: Number of memories in L3
                - total_count: Total memories across all tiers
                - l1_avg_access: Average access count in L1
                - l2_avg_access: Average access count in L2
                - l3_avg_access: Average access count in L3
        """
        l1_count = int(min(self.l1_count.item(), self.l1_capacity))
        l2_count = int(min(self.l2_count.item(), self.l2_capacity))
        l3_count = int(min(self.l3_count.item(), self.l3_capacity))

        l1_avg = 0.0
        l2_avg = 0.0
        l3_avg = 0.0

        if l1_count > 0:
            l1_avg = float(self.l1_access_count[self.l1_valid_mask].float().mean().item())
        if l2_count > 0:
            l2_avg = float(self.l2_access_count[self.l2_valid_mask].float().mean().item())
        if l3_count > 0:
            l3_avg = float(self.l3_access_count[self.l3_valid_mask].float().mean().item())

        return {
            "l1_count": l1_count,
            "l2_count": l2_count,
            "l3_count": l3_count,
            "total_count": l1_count + l2_count + l3_count,
            "l1_avg_access": l1_avg,
            "l2_avg_access": l2_avg,
            "l3_avg_access": l3_avg,
        }

    def store(self, sdr: torch.Tensor, content: torch.Tensor) -> None:
        """
        Store SDR-content pair, starting in L1 tier.

        New memories always enter at L1 (hot tier). If L1 is full,
        the least recently accessed item in L1 is demoted to L2.

        Args:
            sdr: (sdr_size,) SDR vector
            content: (content_dim,) content vector
        """
        # Detach inputs
        sdr = sdr.detach().to(self.l1_sdr_bank.device)
        content = content.detach().to(self.l1_content_bank.device)

        # If L1 is full, demote least accessed item to L2
        if self.l1_count >= self.l1_capacity:
            self._demote_l1_to_l2()

        # Store in L1
        ptr = self.l1_write_ptr.view(1)
        sdr_exp = sdr.unsqueeze(0)
        content_exp = content.unsqueeze(0)

        self.l1_sdr_bank.index_copy_(0, ptr, sdr_exp)
        self.l1_content_bank.index_copy_(0, ptr, content_exp)
        self.l1_access_count.index_fill_(0, ptr, 1)  # Initial access count
        self.l1_valid_mask.index_fill_(0, ptr, True)

        # Advance write pointer
        next_ptr = (self.l1_write_ptr + 1) % self.l1_capacity
        self.l1_write_ptr.copy_(next_ptr)

        next_count = torch.clamp(self.l1_count + 1, max=self.l1_capacity)
        self.l1_count.copy_(next_count)

    def _demote_l1_to_l2(self) -> None:
        """Demote least accessed item from L1 to L2."""
        if self.l1_count == 0:
            return

        # Find least accessed valid item in L1
        access_counts = self.l1_access_count.clone()
        access_counts[~self.l1_valid_mask] = torch.iinfo(torch.long).max
        victim_idx = torch.argmin(access_counts)

        # If L2 is full, demote its least accessed item to L3
        if self.l2_count >= self.l2_capacity:
            self._demote_l2_to_l3()

        # Move victim from L1 to L2
        ptr = self.l2_write_ptr.view(1)
        victim_idx_view = victim_idx.view(1)

        self.l2_sdr_bank.index_copy_(0, ptr, self.l1_sdr_bank[victim_idx_view])
        self.l2_content_bank.index_copy_(0, ptr, self.l1_content_bank[victim_idx_view])
        self.l2_access_count.index_copy_(0, ptr, self.l1_access_count[victim_idx_view])
        self.l2_valid_mask.index_fill_(0, ptr, True)

        # Invalidate in L1
        self.l1_valid_mask.index_fill_(0, victim_idx_view, False)
        self.l1_count.copy_(torch.clamp(self.l1_count - 1, min=0))

        # Advance L2 write pointer
        next_ptr = (self.l2_write_ptr + 1) % self.l2_capacity
        self.l2_write_ptr.copy_(next_ptr)

        next_count = torch.clamp(self.l2_count + 1, max=self.l2_capacity)
        self.l2_count.copy_(next_count)

    def _demote_l2_to_l3(self) -> None:
        """Demote least accessed item from L2 to L3."""
        if self.l2_count == 0:
            return

        # Find least accessed valid item in L2
        access_counts = self.l2_access_count.clone()
        access_counts[~self.l2_valid_mask] = torch.iinfo(torch.long).max
        victim_idx = torch.argmin(access_counts)

        # Move victim from L2 to L3 (L3 is circular buffer, overwrites oldest)
        ptr = self.l3_write_ptr.view(1)
        victim_idx_view = victim_idx.view(1)

        self.l3_sdr_bank.index_copy_(0, ptr, self.l2_sdr_bank[victim_idx_view])
        self.l3_content_bank.index_copy_(0, ptr, self.l2_content_bank[victim_idx_view])
        self.l3_access_count.index_copy_(0, ptr, self.l2_access_count[victim_idx_view])
        self.l3_valid_mask.index_fill_(0, ptr, True)

        # Invalidate in L2
        self.l2_valid_mask.index_fill_(0, victim_idx_view, False)
        self.l2_count.copy_(torch.clamp(self.l2_count - 1, min=0))

        # Advance L3 write pointer
        next_ptr = (self.l3_write_ptr + 1) % self.l3_capacity
        self.l3_write_ptr.copy_(next_ptr)

        next_count = torch.clamp(self.l3_count + 1, max=self.l3_capacity)
        self.l3_count.copy_(next_count)

    def _promote_to_l1(self, tier: str, idx: torch.Tensor) -> None:
        """
        Promote item from L2 or L3 to L1.

        Args:
            tier: "l2" or "l3"
            idx: Index in source tier
        """
        idx_view = idx.view(1)

        # If L1 is full, demote least accessed
        if self.l1_count >= self.l1_capacity:
            self._demote_l1_to_l2()

        # Copy from source tier to L1
        ptr = self.l1_write_ptr.view(1)

        if tier == "l2":
            self.l1_sdr_bank.index_copy_(0, ptr, self.l2_sdr_bank[idx_view])
            self.l1_content_bank.index_copy_(0, ptr, self.l2_content_bank[idx_view])
            self.l1_access_count.index_copy_(0, ptr, self.l2_access_count[idx_view])

            # Invalidate in L2
            self.l2_valid_mask.index_fill_(0, idx_view, False)
            self.l2_count.copy_(torch.clamp(self.l2_count - 1, min=0))

        elif tier == "l3":
            self.l1_sdr_bank.index_copy_(0, ptr, self.l3_sdr_bank[idx_view])
            self.l1_content_bank.index_copy_(0, ptr, self.l3_content_bank[idx_view])
            self.l1_access_count.index_copy_(0, ptr, self.l3_access_count[idx_view])

            # Invalidate in L3
            self.l3_valid_mask.index_fill_(0, idx_view, False)
            self.l3_count.copy_(torch.clamp(self.l3_count - 1, min=0))

        self.l1_valid_mask.index_fill_(0, ptr, True)

        # Advance L1 write pointer
        next_ptr = (self.l1_write_ptr + 1) % self.l1_capacity
        self.l1_write_ptr.copy_(next_ptr)

        next_count = torch.clamp(self.l1_count + 1, max=self.l1_capacity)
        self.l1_count.copy_(next_count)

    def retrieve(
        self,
        query_sdr: torch.Tensor,
        top_k: int = 5,
        tier_threshold: float = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        """
        Retrieve by SDR similarity across all tiers.

        Searches L1, L2, L3 in order, returning top matches.
        Increments access counts and promotes frequently accessed items.

        Args:
            query_sdr: (sdr_size,) query SDR
            top_k: Maximum number of results to return
            tier_threshold: Override default similarity threshold (optional)

        Returns:
            contents: (n_results, content_dim) retrieved content vectors
            similarities: (n_results,) overlap-based similarity scores
            tier_info: (n_results,) list of tier names ["l1", "l2", "l3"]
        """
        threshold = tier_threshold if tier_threshold is not None else self.threshold
        query_sdr = query_sdr.to(self.l1_sdr_bank.device)

        all_contents = []
        all_similarities = []
        all_tiers = []

        # Search L1
        if self.l1_count > 0:
            overlap = (query_sdr.unsqueeze(0) * self.l1_sdr_bank).sum(dim=-1)
            similarity = overlap / self.n_active
            similarity = torch.where(
                self.l1_valid_mask,
                similarity,
                torch.tensor(-float("inf"), device=similarity.device),
            )

            valid_mask = similarity >= threshold
            if valid_mask.any():
                valid_sim = similarity[valid_mask]
                valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)

                all_contents.append(self.l1_content_bank[valid_idx])
                all_similarities.append(valid_sim)
                all_tiers.extend(["l1"] * len(valid_idx))

                # Increment access counts
                self.l1_access_count[valid_idx] = self.l1_access_count[valid_idx] + 1

        # Search L2
        if self.l2_count > 0:
            overlap = (query_sdr.unsqueeze(0) * self.l2_sdr_bank).sum(dim=-1)
            similarity = overlap / self.n_active
            similarity = torch.where(
                self.l2_valid_mask,
                similarity,
                torch.tensor(-float("inf"), device=similarity.device),
            )

            valid_mask = similarity >= threshold
            if valid_mask.any():
                valid_sim = similarity[valid_mask]
                valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)

                all_contents.append(self.l2_content_bank[valid_idx])
                all_similarities.append(valid_sim)
                all_tiers.extend(["l2"] * len(valid_idx))

                # Increment access counts and check for promotion
                for idx in valid_idx:
                    self.l2_access_count[idx] = self.l2_access_count[idx] + 1
                    if self.l2_access_count[idx] >= self.promotion_threshold:
                        self._promote_to_l1("l2", idx)

        # Search L3
        if self.l3_count > 0:
            overlap = (query_sdr.unsqueeze(0) * self.l3_sdr_bank).sum(dim=-1)
            similarity = overlap / self.n_active
            similarity = torch.where(
                self.l3_valid_mask,
                similarity,
                torch.tensor(-float("inf"), device=similarity.device),
            )

            valid_mask = similarity >= threshold
            if valid_mask.any():
                valid_sim = similarity[valid_mask]
                valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)

                all_contents.append(self.l3_content_bank[valid_idx])
                all_similarities.append(valid_sim)
                all_tiers.extend(["l3"] * len(valid_idx))

                # Increment access counts and check for promotion
                for idx in valid_idx:
                    self.l3_access_count[idx] = self.l3_access_count[idx] + 1
                    if self.l3_access_count[idx] >= self.promotion_threshold:
                        self._promote_to_l1("l3", idx)

        # Combine results from all tiers
        if not all_contents:
            return (
                torch.zeros(0, self.content_dim, device=self.l1_content_bank.device),
                torch.zeros(0, device=self.l1_sdr_bank.device),
                [],
            )

        all_contents = torch.cat(all_contents, dim=0)
        all_similarities = torch.cat(all_similarities, dim=0)

        # Sort by similarity and take top-k
        k = min(top_k, len(all_similarities))
        top_sim, top_indices = torch.topk(all_similarities, k)

        top_contents = all_contents[top_indices]
        top_tiers = [all_tiers[i] for i in top_indices.cpu().tolist()]

        return top_contents, top_sim, top_tiers

    def clear(self) -> None:
        """Clear all memories across all tiers."""
        # L1
        self.l1_sdr_bank.zero_()
        self.l1_content_bank.zero_()
        self.l1_access_count.zero_()
        self.l1_valid_mask.zero_()
        self.l1_write_ptr.zero_()
        self.l1_count.zero_()

        # L2
        self.l2_sdr_bank.zero_()
        self.l2_content_bank.zero_()
        self.l2_access_count.zero_()
        self.l2_valid_mask.zero_()
        self.l2_write_ptr.zero_()
        self.l2_count.zero_()

        # L3
        self.l3_sdr_bank.zero_()
        self.l3_content_bank.zero_()
        self.l3_access_count.zero_()
        self.l3_valid_mask.zero_()
        self.l3_write_ptr.zero_()
        self.l3_count.zero_()

    def __len__(self) -> int:
        """Return total number of memories across all tiers."""
        return int(
            min(self.l1_count.item(), self.l1_capacity)
            + min(self.l2_count.item(), self.l2_capacity)
            + min(self.l3_count.item(), self.l3_capacity)
        )
