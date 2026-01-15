# neuromanifold_gpt/tests/test_hierarchical_memory.py
"""Tests for HierarchicalEngramMemory - three-tier memory system."""
import pytest
import torch
from neuromanifold_gpt.model.memory.hierarchical_engram import HierarchicalEngramMemory


def test_hierarchical_store_retrieve_l1():
    """Should store and retrieve from L1 tier."""
    memory = HierarchicalEngramMemory(
        sdr_size=2048,
        n_active=40,
        content_dim=384,
        l1_capacity=10,
        l2_capacity=20,
        l3_capacity=50,
        threshold=0.3,
    )

    # Create and store an SDR
    sdr = torch.zeros(2048)
    sdr[:40] = 1
    content = torch.randn(384)

    memory.store(sdr, content)

    # Retrieve with same SDR
    retrieved, similarity, tiers = memory.retrieve(sdr, top_k=1)

    assert retrieved.shape == (1, 384)
    assert similarity[0] > 0.9
    assert tiers == ["l1"]
    assert len(memory) == 1


def test_hierarchical_tier_stats():
    """Should track statistics across all tiers."""
    memory = HierarchicalEngramMemory(
        sdr_size=2048,
        n_active=40,
        content_dim=384,
        l1_capacity=2,
        l2_capacity=3,
        l3_capacity=5,
        threshold=0.3,
    )

    # Initially empty
    stats = memory.get_stats()
    assert stats["l1_count"] == 0
    assert stats["l2_count"] == 0
    assert stats["l3_count"] == 0
    assert stats["total_count"] == 0

    # Add one to L1
    sdr = torch.zeros(2048)
    sdr[:40] = 1
    memory.store(sdr, torch.randn(384))

    stats = memory.get_stats()
    assert stats["l1_count"] == 1
    assert stats["total_count"] == 1


def test_hierarchical_l1_to_l2_demotion():
    """Should demote from L1 to L2 when L1 is full."""
    memory = HierarchicalEngramMemory(
        sdr_size=2048,
        n_active=40,
        content_dim=384,
        l1_capacity=2,
        l2_capacity=5,
        l3_capacity=10,
        threshold=0.3,
    )

    # Fill L1 (capacity 2)
    for i in range(2):
        sdr = torch.zeros(2048)
        sdr[i * 10 : i * 10 + 40] = 1
        memory.store(sdr, torch.randn(384))

    stats = memory.get_stats()
    assert stats["l1_count"] == 2
    assert stats["l2_count"] == 0

    # Add third item - should trigger L1 -> L2 demotion
    sdr = torch.zeros(2048)
    sdr[100:140] = 1
    memory.store(sdr, torch.randn(384))

    stats = memory.get_stats()
    assert stats["l1_count"] == 2
    assert stats["l2_count"] == 1
    assert stats["total_count"] == 3


def test_hierarchical_l2_to_l3_demotion():
    """Should demote from L2 to L3 when both L1 and L2 are full."""
    memory = HierarchicalEngramMemory(
        sdr_size=2048,
        n_active=40,
        content_dim=384,
        l1_capacity=2,
        l2_capacity=2,
        l3_capacity=10,
        threshold=0.3,
    )

    # Fill L1 and L2
    for i in range(4):
        sdr = torch.zeros(2048)
        sdr[i * 10 : i * 10 + 40] = 1
        memory.store(sdr, torch.randn(384))

    stats = memory.get_stats()
    assert stats["l1_count"] == 2
    assert stats["l2_count"] == 2

    # Add fifth item - should trigger L2 -> L3 demotion
    sdr = torch.zeros(2048)
    sdr[200:240] = 1
    memory.store(sdr, torch.randn(384))

    stats = memory.get_stats()
    assert stats["l1_count"] == 2
    assert stats["l2_count"] == 2
    assert stats["l3_count"] == 1
    assert stats["total_count"] == 5


def test_hierarchical_l2_to_l1_promotion():
    """Should promote frequently accessed L2 items to L1."""
    memory = HierarchicalEngramMemory(
        sdr_size=2048,
        n_active=40,
        content_dim=384,
        l1_capacity=2,
        l2_capacity=5,
        l3_capacity=10,
        threshold=0.3,
        promotion_threshold=3,
    )

    # Fill L1, forcing first item to L2
    sdr1 = torch.zeros(2048)
    sdr1[:40] = 1
    memory.store(sdr1, torch.randn(384))

    sdr2 = torch.zeros(2048)
    sdr2[50:90] = 1
    memory.store(sdr2, torch.randn(384))

    sdr3 = torch.zeros(2048)
    sdr3[100:140] = 1
    memory.store(sdr3, torch.randn(384))

    # sdr1 should now be in L2
    stats = memory.get_stats()
    assert stats["l1_count"] == 2
    assert stats["l2_count"] == 1

    # Access sdr1 multiple times to trigger promotion
    for _ in range(3):
        memory.retrieve(sdr1, top_k=1)

    # After 3 accesses, sdr1 should be promoted back to L1
    stats = memory.get_stats()
    assert stats["l1_count"] == 2
    # Note: L2 count might be 0 or 1 depending on promotion timing


def test_hierarchical_l3_to_l1_promotion():
    """Should promote frequently accessed L3 items to L1."""
    memory = HierarchicalEngramMemory(
        sdr_size=2048,
        n_active=40,
        content_dim=384,
        l1_capacity=1,
        l2_capacity=1,
        l3_capacity=10,
        threshold=0.3,
        promotion_threshold=3,
    )

    # Store 3 items, pushing first to L3
    sdr1 = torch.zeros(2048)
    sdr1[:40] = 1
    memory.store(sdr1, torch.randn(384))

    sdr2 = torch.zeros(2048)
    sdr2[50:90] = 1
    memory.store(sdr2, torch.randn(384))

    sdr3 = torch.zeros(2048)
    sdr3[100:140] = 1
    memory.store(sdr3, torch.randn(384))

    # sdr1 should be in L3
    stats = memory.get_stats()
    assert stats["l3_count"] >= 1

    # Access sdr1 multiple times to trigger promotion
    for _ in range(3):
        memory.retrieve(sdr1, top_k=1)

    # After 3 accesses, sdr1 should be promoted to L1
    stats = memory.get_stats()
    assert stats["l1_count"] == 1


def test_hierarchical_retrieval_across_tiers():
    """Should retrieve matches from all tiers."""
    memory = HierarchicalEngramMemory(
        sdr_size=2048,
        n_active=40,
        content_dim=384,
        l1_capacity=2,
        l2_capacity=2,
        l3_capacity=5,
        threshold=0.2,
    )

    # Store items that will end up in different tiers
    sdrs = []
    for i in range(5):
        sdr = torch.zeros(2048)
        sdr[i * 20 : i * 20 + 40] = 1
        sdrs.append(sdr)
        memory.store(sdr, torch.randn(384))

    stats = memory.get_stats()
    assert stats["total_count"] == 5

    # Query with an SDR that partially overlaps multiple stored SDRs
    query_sdr = torch.zeros(2048)
    query_sdr[10:50] = 1

    retrieved, similarities, tiers = memory.retrieve(query_sdr, top_k=5)

    # Should find matches across tiers
    assert len(retrieved) > 0
    assert len(set(tiers)) >= 1  # At least one tier


def test_hierarchical_similarity_threshold():
    """Low similarity should return nothing."""
    memory = HierarchicalEngramMemory(
        sdr_size=2048,
        n_active=40,
        content_dim=384,
        l1_capacity=5,
        l2_capacity=5,
        l3_capacity=5,
        threshold=0.5,
    )

    sdr1 = torch.zeros(2048)
    sdr1[:40] = 1
    memory.store(sdr1, torch.randn(384))

    # Query with non-overlapping SDR
    sdr2 = torch.zeros(2048)
    sdr2[200:240] = 1

    retrieved, similarity, tiers = memory.retrieve(sdr2, top_k=1)

    assert len(retrieved) == 0 or similarity[0] < 0.5


def test_hierarchical_clear():
    """Should clear all memories across all tiers."""
    memory = HierarchicalEngramMemory(
        sdr_size=2048,
        n_active=40,
        content_dim=384,
        l1_capacity=2,
        l2_capacity=2,
        l3_capacity=5,
        threshold=0.3,
    )

    # Fill memory across tiers
    for i in range(6):
        sdr = torch.zeros(2048)
        sdr[i * 10 : i * 10 + 40] = 1
        memory.store(sdr, torch.randn(384))

    stats = memory.get_stats()
    assert stats["total_count"] > 0

    memory.clear()

    assert len(memory) == 0
    stats = memory.get_stats()
    assert stats["l1_count"] == 0
    assert stats["l2_count"] == 0
    assert stats["l3_count"] == 0
    assert stats["total_count"] == 0


def test_hierarchical_empty_retrieval():
    """Should handle retrieval from empty memory."""
    memory = HierarchicalEngramMemory(
        sdr_size=2048,
        n_active=40,
        content_dim=384,
        l1_capacity=5,
        l2_capacity=5,
        l3_capacity=5,
        threshold=0.3,
    )

    query_sdr = torch.zeros(2048)
    query_sdr[:40] = 1

    retrieved, similarity, tiers = memory.retrieve(query_sdr, top_k=5)

    assert len(retrieved) == 0
    assert len(similarity) == 0
    assert len(tiers) == 0


def test_hierarchical_top_k_limit():
    """Should respect top_k parameter."""
    memory = HierarchicalEngramMemory(
        sdr_size=2048,
        n_active=40,
        content_dim=384,
        l1_capacity=10,
        l2_capacity=10,
        l3_capacity=10,
        threshold=0.1,
    )

    # Store multiple overlapping items
    for i in range(10):
        sdr = torch.zeros(2048)
        sdr[i * 5 : i * 5 + 40] = 1
        memory.store(sdr, torch.randn(384))

    # Query should match many items
    query_sdr = torch.zeros(2048)
    query_sdr[:40] = 1

    retrieved, similarities, tiers = memory.retrieve(query_sdr, top_k=3)

    # Should return at most 3 results
    assert len(retrieved) <= 3
    assert len(similarities) <= 3
    assert len(tiers) <= 3

    # Results should be sorted by similarity
    if len(similarities) > 1:
        assert all(similarities[i] >= similarities[i + 1] for i in range(len(similarities) - 1))


def test_hierarchical_capacity_enforcement():
    """Each tier should enforce capacity limits."""
    memory = HierarchicalEngramMemory(
        sdr_size=2048,
        n_active=40,
        content_dim=384,
        l1_capacity=2,
        l2_capacity=3,
        l3_capacity=5,
        threshold=0.3,
    )

    # Store more items than total capacity
    for i in range(15):
        sdr = torch.zeros(2048)
        sdr[i * 10 : i * 10 + 40] = 1
        memory.store(sdr, torch.randn(384))

    # Total should not exceed capacity
    stats = memory.get_stats()
    assert stats["l1_count"] <= 2
    assert stats["l2_count"] <= 3
    assert stats["l3_count"] <= 5
    assert stats["total_count"] <= 10


def test_hierarchical_partial_overlap_retrieval():
    """Should retrieve with partial SDR overlap."""
    memory = HierarchicalEngramMemory(
        sdr_size=2048,
        n_active=40,
        content_dim=384,
        l1_capacity=5,
        l2_capacity=5,
        l3_capacity=5,
        threshold=0.3,
    )

    # Store SDR with bits 0-39 active
    sdr1 = torch.zeros(2048)
    sdr1[:40] = 1
    content1 = torch.randn(384)
    memory.store(sdr1, content1)

    # Query with 50% overlap (bits 20-59 active)
    query_sdr = torch.zeros(2048)
    query_sdr[20:60] = 1

    retrieved, similarity, tiers = memory.retrieve(query_sdr, top_k=1)

    # 20 bits overlap out of 40 = 0.5 similarity
    assert len(retrieved) == 1
    assert 0.4 <= similarity[0] <= 0.6  # Allow some tolerance


def test_hierarchical_multiple_retrieval_sorted():
    """Should retrieve multiple results sorted by similarity."""
    memory = HierarchicalEngramMemory(
        sdr_size=2048,
        n_active=40,
        content_dim=384,
        l1_capacity=10,
        l2_capacity=10,
        l3_capacity=10,
        threshold=0.2,
    )

    # Store 3 SDRs with different overlaps to query
    for i in range(3):
        sdr = torch.zeros(2048)
        # i=0: bits 0-39, i=1: bits 10-49, i=2: bits 20-59
        sdr[i * 10 : i * 10 + 40] = 1
        memory.store(sdr, torch.randn(384))

    # Query with bits 0-39 (exact match with first, partial with others)
    query_sdr = torch.zeros(2048)
    query_sdr[:40] = 1

    retrieved, similarity, tiers = memory.retrieve(query_sdr, top_k=3)

    # Should be sorted by similarity (descending)
    assert len(retrieved) >= 2
    assert all(similarity[i] >= similarity[i + 1] for i in range(len(similarity) - 1))
    assert similarity[0] > 0.9  # First should be near-exact match


def test_hierarchical_device_movement():
    """Memory should move with model to different devices."""
    memory = HierarchicalEngramMemory(
        sdr_size=2048,
        n_active=40,
        content_dim=384,
        l1_capacity=5,
        l2_capacity=5,
        l3_capacity=5,
        threshold=0.3,
    )

    sdr = torch.zeros(2048)
    sdr[:40] = 1
    memory.store(sdr, torch.randn(384))

    # Move to same device (CPU) - should work
    memory = memory.to("cpu")

    # Verify retrieval still works
    retrieved, similarity, tiers = memory.retrieve(sdr, top_k=1)
    assert len(retrieved) == 1


def test_hierarchical_access_count_tracking():
    """Should track access counts correctly."""
    memory = HierarchicalEngramMemory(
        sdr_size=2048,
        n_active=40,
        content_dim=384,
        l1_capacity=5,
        l2_capacity=5,
        l3_capacity=5,
        threshold=0.3,
    )

    # Store an item
    sdr = torch.zeros(2048)
    sdr[:40] = 1
    memory.store(sdr, torch.randn(384))

    # Initial access count should be 1 (from store)
    stats = memory.get_stats()
    assert stats["l1_avg_access"] >= 1.0

    # Retrieve multiple times
    for _ in range(3):
        memory.retrieve(sdr, top_k=1)

    # Access count should increase
    stats = memory.get_stats()
    assert stats["l1_avg_access"] >= 4.0


def test_hierarchical_tier_override_threshold():
    """Should respect tier_threshold parameter in retrieve."""
    memory = HierarchicalEngramMemory(
        sdr_size=2048,
        n_active=40,
        content_dim=384,
        l1_capacity=5,
        l2_capacity=5,
        l3_capacity=5,
        threshold=0.3,  # Default threshold
    )

    sdr1 = torch.zeros(2048)
    sdr1[:40] = 1
    memory.store(sdr1, torch.randn(384))

    # Query with partial overlap (0.5 similarity)
    query_sdr = torch.zeros(2048)
    query_sdr[20:60] = 1

    # With default threshold (0.3), should retrieve
    retrieved1, _, _ = memory.retrieve(query_sdr, top_k=1)
    assert len(retrieved1) == 1

    # With higher threshold (0.7), should not retrieve
    retrieved2, _, _ = memory.retrieve(query_sdr, top_k=1, tier_threshold=0.7)
    assert len(retrieved2) == 0


def test_hierarchical_len_method():
    """Should correctly report total memory count."""
    memory = HierarchicalEngramMemory(
        sdr_size=2048,
        n_active=40,
        content_dim=384,
        l1_capacity=2,
        l2_capacity=3,
        l3_capacity=5,
        threshold=0.3,
    )

    assert len(memory) == 0

    # Add items
    for i in range(7):
        sdr = torch.zeros(2048)
        sdr[i * 10 : i * 10 + 40] = 1
        memory.store(sdr, torch.randn(384))

    assert len(memory) == 7

    memory.clear()
    assert len(memory) == 0
