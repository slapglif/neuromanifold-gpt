# neuromanifold_gpt/tests/test_hierarchical_engram.py
"""Tests for HierarchicalEngramMemory - 3-tier cache architecture."""
import pytest
import torch

from neuromanifold_gpt.model.memory.hierarchical_engram import HierarchicalEngramMemory


def test_hierarchical_store_retrieve():
    """Should store and retrieve by SDR similarity from L1."""
    memory = HierarchicalEngramMemory(sdr_size=2048, n_active=40, content_dim=384)

    # Create an SDR
    sdr = torch.zeros(2048)
    sdr[:40] = 1
    content = torch.randn(384)

    # Store
    memory.store(sdr, content)

    # Retrieve with same SDR
    retrieved, similarity, tier = memory.retrieve(sdr, top_k=1)

    assert retrieved.shape == (1, 384)
    assert similarity[0] > 0.9
    assert tier == "L1"  # Should come from hot cache


def test_hierarchical_total_capacity():
    """Should track total memories across all tiers."""
    memory = HierarchicalEngramMemory(
        sdr_size=2048,
        n_active=40,
        content_dim=384,
        l1_capacity=5,
        l2_capacity=10,
        l3_capacity=15,
    )

    # Total capacity should be sum of all tiers
    assert len(memory) == 0

    # Store one item
    sdr = torch.zeros(2048)
    sdr[:40] = 1
    memory.store(sdr, torch.randn(384))

    assert len(memory) == 1


def test_hierarchical_empty_retrieval():
    """Should handle retrieval from empty memory."""
    memory = HierarchicalEngramMemory(sdr_size=2048, n_active=40, content_dim=384)

    query_sdr = torch.zeros(2048)
    query_sdr[:40] = 1

    retrieved, similarity, tier = memory.retrieve(query_sdr, top_k=5)

    assert len(retrieved) == 0
    assert len(similarity) == 0
    assert tier == "L1"


def test_hierarchical_clear():
    """Should clear all memories across all tiers."""
    memory = HierarchicalEngramMemory(
        sdr_size=2048,
        n_active=40,
        content_dim=384,
        l1_capacity=5,
        l2_capacity=10,
        l3_capacity=15,
    )

    # Store some items
    for i in range(8):
        sdr = torch.zeros(2048)
        sdr[i : i + 40] = 1
        memory.store(sdr, torch.randn(384))

    assert len(memory) > 0

    memory.clear()

    assert len(memory) == 0
    assert len(memory.l1) == 0
    assert len(memory.l2) == 0
    assert len(memory.l3) == 0


def test_hierarchical_get_stats():
    """Should return statistics about memory usage."""
    memory = HierarchicalEngramMemory(
        sdr_size=2048,
        n_active=40,
        content_dim=384,
        l1_capacity=5,
        l2_capacity=10,
        l3_capacity=15,
    )

    # Store 3 items (all should be in L1)
    for i in range(3):
        sdr = torch.zeros(2048)
        sdr[i : i + 40] = 1
        memory.store(sdr, torch.randn(384))

    stats = memory.get_stats()

    assert "l1_size" in stats
    assert "l2_size" in stats
    assert "l3_size" in stats
    assert "memory_size" in stats
    assert stats["l1_size"] == 3
    assert stats["l2_size"] == 0
    assert stats["l3_size"] == 0
    assert stats["memory_size"] == 3
    assert stats["total_capacity"] == 30


def test_hierarchical_similarity_threshold():
    """Low similarity should return nothing."""
    memory = HierarchicalEngramMemory(
        sdr_size=2048, n_active=40, content_dim=384, threshold=0.5
    )

    sdr1 = torch.zeros(2048)
    sdr1[:40] = 1
    memory.store(sdr1, torch.randn(384))

    # Query with non-overlapping SDR
    sdr2 = torch.zeros(2048)
    sdr2[100:140] = 1

    retrieved, similarity, tier = memory.retrieve(sdr2, top_k=1)

    assert len(retrieved) == 0 or similarity[0] < 0.5


def test_hierarchical_partial_overlap_retrieval():
    """Should retrieve with partial SDR overlap."""
    memory = HierarchicalEngramMemory(
        sdr_size=2048, n_active=40, content_dim=384, threshold=0.3
    )

    # Store SDR with bits 0-39 active
    sdr1 = torch.zeros(2048)
    sdr1[:40] = 1
    content1 = torch.randn(384)
    memory.store(sdr1, content1)

    # Query with 50% overlap (bits 20-59 active)
    query_sdr = torch.zeros(2048)
    query_sdr[20:60] = 1

    retrieved, similarity, tier = memory.retrieve(query_sdr, top_k=1)

    # 20 bits overlap out of 40 = 0.5 similarity
    assert len(retrieved) == 1
    assert 0.4 <= similarity[0] <= 0.6  # Allow some tolerance
    assert tier == "L1"


def test_hierarchical_multiple_retrieval():
    """Should retrieve multiple results sorted by similarity."""
    memory = HierarchicalEngramMemory(
        sdr_size=2048, n_active=40, content_dim=384, threshold=0.2
    )

    # Store 3 SDRs with different overlaps to query
    contents = []
    for i in range(3):
        sdr = torch.zeros(2048)
        # i=0: bits 0-39, i=1: bits 10-49, i=2: bits 20-59
        sdr[i * 10 : i * 10 + 40] = 1
        content = torch.randn(384)
        contents.append(content)
        memory.store(sdr, content)

    # Query with bits 0-39 (exact match with first, partial with others)
    query_sdr = torch.zeros(2048)
    query_sdr[:40] = 1

    retrieved, similarity, tier = memory.retrieve(query_sdr, top_k=3)

    # Should be sorted by similarity (descending)
    assert len(retrieved) >= 2
    assert all(similarity[i] >= similarity[i + 1] for i in range(len(similarity) - 1))
    assert similarity[0] > 0.9  # First should be near-exact match
    assert tier == "L1"  # All should be in L1 hot cache


def test_store_batch_correctness():
    """Should store all items in batch and retrieve them correctly."""
    memory = HierarchicalEngramMemory(
        sdr_size=2048, n_active=40, content_dim=384, l1_capacity=10
    )

    # Create batch of 5 SDR-content pairs
    batch_size = 5
    sdrs = torch.zeros(batch_size, 2048)
    contents = torch.randn(batch_size, 384)

    for i in range(batch_size):
        sdrs[i, i * 50 : (i * 50) + 40] = 1  # Non-overlapping SDRs

    # Store batch
    memory.store_batch(sdrs, contents)

    # Verify all items are stored
    assert len(memory) == batch_size
    assert len(memory.l1) == batch_size

    # Verify each item is retrievable
    for i in range(batch_size):
        query_sdr = sdrs[i]
        retrieved, similarity, tier = memory.retrieve(query_sdr, top_k=1)

        assert len(retrieved) == 1
        assert similarity[0] > 0.9  # High similarity for exact match
        assert tier == "L1"


def test_store_batch_performance():
    """store_batch() should handle large batches correctly."""
    memory = HierarchicalEngramMemory(
        sdr_size=2048, n_active=40, content_dim=384, l1_capacity=500
    )

    # Create batch of 100 items
    batch_size = 100
    sdrs = torch.zeros(batch_size, 2048)
    contents = torch.randn(batch_size, 384)

    for i in range(batch_size):
        sdrs[i, (i * 20) % 2000 : (i * 20) % 2000 + 40] = 1

    # Store batch
    memory.store_batch(sdrs, contents)

    # All items should be stored
    assert len(memory) == batch_size

    # Random sample should be retrievable
    for i in [0, batch_size // 2, batch_size - 1]:
        retrieved, similarity, tier = memory.retrieve(sdrs[i], top_k=1)
        assert len(retrieved) == 1
        # May have overlap with other SDRs, so similarity might not be perfect
        assert similarity[0] > 0.5


def test_store_batch_with_l1_overflow():
    """Batch store should handle L1 overflow with proper tier promotion."""
    memory = HierarchicalEngramMemory(
        sdr_size=2048,
        n_active=40,
        content_dim=384,
        l1_capacity=5,
        l2_capacity=10,
        l3_capacity=20,
    )

    # Store batch larger than L1 capacity
    batch_size = 8
    sdrs = torch.zeros(batch_size, 2048)
    contents = torch.randn(batch_size, 384)

    for i in range(batch_size):
        sdrs[i, i * 50 : (i * 50) + 40] = 1

    memory.store_batch(sdrs, contents)

    # Should have all items stored
    assert len(memory) == batch_size

    # L1 should be at capacity
    assert len(memory.l1) == 5

    # Overflow should go to L2
    assert len(memory.l2) == 3

    # L3 should be empty (no L2 overflow yet)
    assert len(memory.l3) == 0


def test_store_batch_with_l2_overflow():
    """Batch store should cascade promotions through L2 to L3."""
    memory = HierarchicalEngramMemory(
        sdr_size=2048,
        n_active=40,
        content_dim=384,
        l1_capacity=3,
        l2_capacity=5,
        l3_capacity=20,
    )

    # First, fill L1 and L2
    for i in range(8):
        sdr = torch.zeros(2048)
        sdr[i * 50 : (i * 50) + 40] = 1
        memory.store(sdr, torch.randn(384))

    # Now L1 has 3, L2 has 5
    assert len(memory.l1) == 3
    assert len(memory.l2) == 5

    # Store batch that will overflow both L1 and L2
    batch_size = 5
    sdrs = torch.zeros(batch_size, 2048)
    contents = torch.randn(batch_size, 384)

    for i in range(batch_size):
        sdrs[i, (i + 10) * 50 : ((i + 10) * 50) + 40] = 1

    memory.store_batch(sdrs, contents)

    # All tiers should have items now
    assert len(memory.l1) == 3  # At capacity
    assert len(memory.l2) == 5  # At capacity
    assert len(memory.l3) > 0  # Has overflow from L2


def test_store_batch_empty():
    """Should handle empty batch gracefully."""
    memory = HierarchicalEngramMemory(sdr_size=2048, n_active=40, content_dim=384)

    # Store empty batch
    sdrs = torch.zeros(0, 2048)
    contents = torch.zeros(0, 384)

    memory.store_batch(sdrs, contents)

    # Should remain empty
    assert len(memory) == 0


def test_store_batch_single_item():
    """Should handle single-item batch correctly."""
    memory = HierarchicalEngramMemory(sdr_size=2048, n_active=40, content_dim=384)

    # Store single-item batch
    sdrs = torch.zeros(1, 2048)
    sdrs[0, :40] = 1
    contents = torch.randn(1, 384)

    memory.store_batch(sdrs, contents)

    # Should have one item
    assert len(memory) == 1
    assert len(memory.l1) == 1

    # Should be retrievable
    retrieved, similarity, tier = memory.retrieve(sdrs[0], top_k=1)
    assert len(retrieved) == 1
    assert similarity[0] > 0.9


def test_store_batch_equivalence():
    """store_batch() should store all items with proper tier distribution."""
    memory = HierarchicalEngramMemory(
        sdr_size=2048,
        n_active=40,
        content_dim=384,
        l1_capacity=10,
        l2_capacity=20,
        l3_capacity=30,
    )

    # Create test data - use non-overlapping SDRs
    batch_size = 15
    sdrs = torch.zeros(batch_size, 2048)
    contents = torch.randn(batch_size, 384)

    # Ensure SDRs don't overlap by using distinct regions
    for i in range(batch_size):
        start_pos = i * 100  # 100 bits apart to avoid any overlap
        sdrs[i, start_pos : start_pos + 40] = 1

    # Store as batch
    memory.store_batch(sdrs, contents)

    # Should have all items stored
    stats = memory.get_stats()
    assert stats["memory_size"] == batch_size

    # L1 should have 10, L2 should have 5
    assert stats["l1_size"] == 10
    assert stats["l2_size"] == 5
    assert stats["l3_size"] == 0

    # Most recent items (last 10) should be in L1 and fully retrievable
    for i in range(batch_size - 10, batch_size):
        query_sdr = sdrs[i]
        retrieved, sim, tier = memory.retrieve(query_sdr, top_k=1)

        assert len(retrieved) == 1, f"Item {i} not found"
        assert sim[0] > 0.9, f"Low similarity for item {i}: {sim[0]}"
        assert tier == "L1", f"Item {i} should be in L1 but is in {tier}"


def test_tier_promotion_l1_to_l2():
    """Items should be promoted from L1 to L2 when L1 is full."""
    memory = HierarchicalEngramMemory(
        sdr_size=2048,
        n_active=40,
        content_dim=384,
        l1_capacity=3,
        l2_capacity=10,
        l3_capacity=20,
    )

    # Store 3 items to fill L1
    for i in range(3):
        sdr = torch.zeros(2048)
        sdr[i * 50 : (i * 50) + 40] = 1
        memory.store(sdr, torch.randn(384))

    assert len(memory.l1) == 3
    assert len(memory.l2) == 0

    # Store one more item - should trigger promotion
    sdr = torch.zeros(2048)
    sdr[200:240] = 1
    memory.store(sdr, torch.randn(384))

    # L1 should still be at capacity, L2 should have one item
    assert len(memory.l1) == 3
    assert len(memory.l2) == 1
    assert len(memory.l3) == 0


def test_tier_promotion_l2_to_l3():
    """Items should be promoted from L2 to L3 when both L1 and L2 are full."""
    memory = HierarchicalEngramMemory(
        sdr_size=2048,
        n_active=40,
        content_dim=384,
        l1_capacity=2,
        l2_capacity=3,
        l3_capacity=20,
    )

    # Fill L1 and L2 completely (2 + 3 = 5 items)
    for i in range(5):
        sdr = torch.zeros(2048)
        sdr[i * 50 : (i * 50) + 40] = 1
        memory.store(sdr, torch.randn(384))

    assert len(memory.l1) == 2
    assert len(memory.l2) == 3
    assert len(memory.l3) == 0

    # Store one more - should cascade to L3
    sdr = torch.zeros(2048)
    sdr[300:340] = 1
    memory.store(sdr, torch.randn(384))

    assert len(memory.l1) == 2
    assert len(memory.l2) == 3
    assert len(memory.l3) == 1


def test_tier_promotion_cascade():
    """Should cascade promotions through all tiers correctly."""
    memory = HierarchicalEngramMemory(
        sdr_size=2048,
        n_active=40,
        content_dim=384,
        l1_capacity=2,
        l2_capacity=2,
        l3_capacity=10,
    )

    # Store items and verify cascade behavior
    for i in range(7):
        sdr = torch.zeros(2048)
        sdr[i * 100 : (i * 100) + 40] = 1
        memory.store(sdr, torch.randn(384))

    # All tiers should be populated
    assert len(memory.l1) == 2  # At capacity
    assert len(memory.l2) == 2  # At capacity
    assert len(memory.l3) == 3  # Has overflow from L2

    stats = memory.get_stats()
    assert stats["memory_size"] == 7


def test_capacity_l1_limit():
    """L1 should enforce capacity limit."""
    memory = HierarchicalEngramMemory(
        sdr_size=2048,
        n_active=40,
        content_dim=384,
        l1_capacity=5,
        l2_capacity=10,
        l3_capacity=15,
    )

    # Store more than L1 capacity
    for i in range(10):
        sdr = torch.zeros(2048)
        sdr[i * 50 : (i * 50) + 40] = 1
        memory.store(sdr, torch.randn(384))

    # L1 should be at its capacity limit
    assert len(memory.l1) == 5

    # Remaining items should be in L2
    assert len(memory.l2) == 5


def test_capacity_l2_limit():
    """L2 should enforce capacity limit."""
    memory = HierarchicalEngramMemory(
        sdr_size=2048,
        n_active=40,
        content_dim=384,
        l1_capacity=3,
        l2_capacity=5,
        l3_capacity=20,
    )

    # Store more than L1+L2 capacity
    for i in range(12):
        sdr = torch.zeros(2048)
        sdr[i * 50 : (i * 50) + 40] = 1
        memory.store(sdr, torch.randn(384))

    # L1 and L2 should be at capacity
    assert len(memory.l1) == 3
    assert len(memory.l2) == 5

    # Overflow should go to L3
    assert len(memory.l3) == 4


def test_capacity_l3_limit():
    """L3 should enforce capacity limit and evict oldest items."""
    memory = HierarchicalEngramMemory(
        sdr_size=2048,
        n_active=40,
        content_dim=384,
        l1_capacity=2,
        l2_capacity=3,
        l3_capacity=5,
    )

    # Store more than total capacity
    for i in range(15):
        sdr = torch.zeros(2048)
        sdr[i * 50 : (i * 50) + 40] = 1
        memory.store(sdr, torch.randn(384))

    # Each tier should be at capacity
    assert len(memory.l1) == 2
    assert len(memory.l2) == 3
    assert len(memory.l3) == 5

    # Total should equal sum of capacities (oldest items evicted)
    assert len(memory) == 10


def test_capacity_total_limit():
    """Total memory should not exceed sum of all tier capacities."""
    memory = HierarchicalEngramMemory(
        sdr_size=2048,
        n_active=40,
        content_dim=384,
        l1_capacity=5,
        l2_capacity=10,
        l3_capacity=15,
    )

    # Store many more items than total capacity
    for i in range(50):
        sdr = torch.zeros(2048)
        sdr[i * 40 : (i * 40) + 40] = 1
        memory.store(sdr, torch.randn(384))

    # Total should be capped at sum of capacities
    total_capacity = 5 + 10 + 15
    assert len(memory) == total_capacity

    stats = memory.get_stats()
    assert stats["memory_size"] == total_capacity
    assert stats["total_capacity"] == total_capacity
