# neuromanifold_gpt/tests/test_engram.py
"""Tests for SDREngramMemory - breadcrumb trail for infinite context."""
import pytest
import torch
from neuromanifold_gpt.model.memory.engram import SDREngramMemory


def test_engram_store_retrieve():
    """Should store and retrieve by SDR similarity."""
    memory = SDREngramMemory(sdr_size=2048, capacity=100, n_active=40)

    # Create an SDR
    sdr = torch.zeros(2048)
    sdr[:40] = 1
    content = torch.randn(384)

    # Store
    memory.store(sdr, content)

    # Retrieve with same SDR
    retrieved, similarity = memory.retrieve(sdr, top_k=1)

    assert retrieved.shape == (1, 384)
    assert similarity[0] > 0.9


def test_engram_capacity_limit():
    """Old memories should be evicted when at capacity."""
    memory = SDREngramMemory(sdr_size=2048, capacity=5, n_active=40)

    # Store 10 items
    for i in range(10):
        sdr = torch.zeros(2048)
        sdr[i * 4 : (i + 1) * 4 + 36] = 1  # Overlapping SDRs
        memory.store(sdr, torch.randn(384))

    assert len(memory) == 5


def test_engram_similarity_threshold():
    """Low similarity should return nothing."""
    memory = SDREngramMemory(sdr_size=2048, capacity=100, n_active=40, threshold=0.5)

    sdr1 = torch.zeros(2048)
    sdr1[:40] = 1
    memory.store(sdr1, torch.randn(384))

    # Query with non-overlapping SDR
    sdr2 = torch.zeros(2048)
    sdr2[100:140] = 1

    retrieved, similarity = memory.retrieve(sdr2, top_k=1)

    assert len(retrieved) == 0 or similarity[0] < 0.5


def test_engram_partial_overlap_retrieval():
    """Should retrieve with partial SDR overlap."""
    memory = SDREngramMemory(sdr_size=2048, capacity=100, n_active=40, threshold=0.3)

    # Store SDR with bits 0-39 active
    sdr1 = torch.zeros(2048)
    sdr1[:40] = 1
    content1 = torch.randn(384)
    memory.store(sdr1, content1)

    # Query with 50% overlap (bits 20-59 active)
    query_sdr = torch.zeros(2048)
    query_sdr[20:60] = 1

    retrieved, similarity = memory.retrieve(query_sdr, top_k=1)

    # 20 bits overlap out of 40 = 0.5 similarity
    assert len(retrieved) == 1
    assert 0.4 <= similarity[0] <= 0.6  # Allow some tolerance


def test_engram_multiple_retrieval():
    """Should retrieve multiple results sorted by similarity."""
    memory = SDREngramMemory(sdr_size=2048, capacity=100, n_active=40, threshold=0.2)

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

    retrieved, similarity = memory.retrieve(query_sdr, top_k=3)

    # Should be sorted by similarity (descending)
    assert len(retrieved) >= 2
    assert all(similarity[i] >= similarity[i + 1] for i in range(len(similarity) - 1))
    assert similarity[0] > 0.9  # First should be near-exact match


def test_engram_clear():
    """Should clear all memories."""
    memory = SDREngramMemory(sdr_size=2048, capacity=100, n_active=40)

    # Store some items
    for _ in range(5):
        sdr = torch.zeros(2048)
        sdr[:40] = 1
        memory.store(sdr, torch.randn(384))

    assert len(memory) == 5

    memory.clear()

    assert len(memory) == 0


def test_engram_empty_retrieval():
    """Should handle retrieval from empty memory."""
    memory = SDREngramMemory(sdr_size=2048, capacity=100, n_active=40)

    query_sdr = torch.zeros(2048)
    query_sdr[:40] = 1

    retrieved, similarity = memory.retrieve(query_sdr, top_k=5)

    assert len(retrieved) == 0
    assert len(similarity) == 0


def test_engram_device_movement():
    """Memory should move with model to different devices."""
    memory = SDREngramMemory(sdr_size=2048, capacity=100, n_active=40)

    sdr = torch.zeros(2048)
    sdr[:40] = 1
    memory.store(sdr, torch.randn(384))

    # Move to same device (CPU) - should work
    memory = memory.to("cpu")

    # Verify retrieval still works
    retrieved, similarity = memory.retrieve(sdr, top_k=1)
    assert len(retrieved) == 1
