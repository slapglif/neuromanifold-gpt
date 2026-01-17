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


def test_engram_batch_retrieve():
    """Batch retrieval should match sequential retrieve() results."""
    memory = SDREngramMemory(sdr_size=2048, capacity=100, n_active=40, threshold=0.3)

    # Store 5 memories with different SDRs
    stored_sdrs = []
    stored_contents = []
    for i in range(5):
        sdr = torch.zeros(2048)
        sdr[i * 10 : i * 10 + 40] = 1
        content = torch.randn(384)
        stored_sdrs.append(sdr)
        stored_contents.append(content)
        memory.store(sdr, content)

    # Create batch of 3 queries
    query_batch = torch.stack(
        [
            stored_sdrs[0],  # Exact match with first
            stored_sdrs[2],  # Exact match with third
            stored_sdrs[1],  # Exact match with second
        ]
    )

    # Batch retrieval
    batch_contents, batch_sim = memory.retrieve_batch(query_batch, top_k=3)

    # Sequential retrieval for comparison
    for i, query_sdr in enumerate(query_batch):
        seq_contents, seq_sim = memory.retrieve(query_sdr, top_k=3)

        # Compare top result (should be exact match)
        assert batch_sim[i, 0] > 0.9, f"Batch element {i} should have high similarity"
        assert seq_sim[0] > 0.9, f"Sequential retrieval {i} should have high similarity"

        # Check that top similarities match (within small tolerance)
        # Note: We only compare valid (non-zero) similarities
        valid_mask = batch_sim[i] > 0
        num_valid = valid_mask.sum().item()
        if num_valid > 0:
            assert torch.allclose(
                batch_sim[i, :num_valid], seq_sim[:num_valid], atol=1e-5
            ), f"Similarities should match for batch element {i}"


def test_engram_batch_retrieve_empty():
    """Batch retrieval should handle empty memory gracefully."""
    memory = SDREngramMemory(sdr_size=2048, capacity=100, n_active=40)

    # Create batch of queries
    query_batch = torch.zeros(3, 2048)
    query_batch[0, :40] = 1
    query_batch[1, 10:50] = 1
    query_batch[2, 20:60] = 1

    # Retrieve from empty memory
    contents, similarities = memory.retrieve_batch(query_batch, top_k=5)

    # Should return zero-filled tensors
    assert contents.shape == (3, 5, 384)
    assert similarities.shape == (3, 5)
    assert torch.all(contents == 0)
    assert torch.all(similarities == 0)


def test_engram_batch_retrieve_threshold_filtering():
    """Threshold filtering should work per batch element."""
    memory = SDREngramMemory(sdr_size=2048, capacity=100, n_active=40, threshold=0.5)

    # Store one SDR with bits 0-39 active
    sdr = torch.zeros(2048)
    sdr[:40] = 1
    content = torch.randn(384)
    memory.store(sdr, content)

    # Create batch with different overlap levels
    query_batch = torch.zeros(3, 2048)
    query_batch[0, :40] = 1  # 100% overlap (40/40) - should retrieve
    query_batch[1, :25] = 1  # 62.5% overlap (25/40) - should retrieve
    query_batch[1, 100:115] = 1  # Fill rest to maintain 40 active bits
    query_batch[2, :15] = 1  # 37.5% overlap (15/40) - below threshold
    query_batch[2, 100:125] = 1  # Fill rest to maintain 40 active bits

    contents, similarities = memory.retrieve_batch(query_batch, top_k=5)

    # First query: high similarity, should retrieve
    assert similarities[0, 0] > 0.9, "Exact match should have high similarity"
    assert torch.any(contents[0] != 0), "Should have retrieved content"

    # Second query: medium similarity above threshold, should retrieve
    assert 0.5 <= similarities[1, 0] < 0.7, "Partial overlap should be above threshold"
    assert torch.any(contents[1] != 0), "Should have retrieved content"

    # Third query: low similarity below threshold, should not retrieve
    assert similarities[2, 0] < 0.5, "Low overlap should be below threshold"
    # All entries should be zero (threshold filtered out)
    assert torch.all(
        similarities[2] < 0.5
    ), "All similarities should be below threshold"


def test_engram_batch_retrieve_variable_results():
    """Different batch elements can return different numbers of results."""
    memory = SDREngramMemory(sdr_size=2048, capacity=100, n_active=40, threshold=0.3)

    # Store 3 memories
    for i in range(3):
        sdr = torch.zeros(2048)
        sdr[i * 50 : i * 50 + 40] = 1
        memory.store(sdr, torch.randn(384))

    # Create queries with different numbers of expected matches
    query_batch = torch.zeros(2, 2048)

    # First query: overlaps well with first memory only
    query_batch[0, :40] = 1

    # Second query: has no good overlap with any memory
    query_batch[1, 1000:1040] = 1

    contents, similarities = memory.retrieve_batch(query_batch, top_k=5)

    # First batch element should have at least one good match
    assert similarities[0, 0] > 0.8, "First query should find a match"
    num_valid_first = (similarities[0] > 0.3).sum().item()
    assert num_valid_first >= 1, "First query should have valid results"

    # Second batch element should have fewer or no good matches
    num_valid_second = (similarities[1] > 0.3).sum().item()
    assert num_valid_second < num_valid_first, "Second query should have fewer matches"

    # Both should have same shape (zero-padded to top_k)
    assert contents.shape == (2, 5, 384)
    assert similarities.shape == (2, 5)


def test_engram_batch_retrieve_device_compatibility():
    """Batch retrieval should work with device movement."""
    memory = SDREngramMemory(sdr_size=2048, capacity=100, n_active=40)

    # Store some data
    sdr = torch.zeros(2048)
    sdr[:40] = 1
    memory.store(sdr, torch.randn(384))

    # Create batch query
    query_batch = torch.zeros(2, 2048)
    query_batch[0, :40] = 1
    query_batch[1, 10:50] = 1

    # Move memory to CPU (already on CPU, but verifies compatibility)
    memory = memory.to("cpu")

    # Retrieve should work
    contents, similarities = memory.retrieve_batch(query_batch, top_k=3)

    assert contents.shape == (2, 3, 384)
    assert similarities.shape == (2, 3)
    assert similarities[0, 0] > 0.9, "First query should match"


def test_engram_batch_retrieve_padding():
    """Batch retrieval should pad results to top_k size."""
    memory = SDREngramMemory(sdr_size=2048, capacity=5, n_active=40, threshold=0.3)

    # Store only 2 items
    for i in range(2):
        sdr = torch.zeros(2048)
        sdr[i * 10 : i * 10 + 40] = 1
        memory.store(sdr, torch.randn(384))

    # Query batch requesting top_k=5 (more than stored)
    query_batch = torch.zeros(2, 2048)
    query_batch[0, :40] = 1

    contents, similarities = memory.retrieve_batch(query_batch, top_k=5)

    # Should be padded to top_k=5
    assert contents.shape == (2, 5, 384)
    assert similarities.shape == (2, 5)

    # First batch element should have 2 valid results, rest padded with zeros
    num_nonzero = (similarities[0] > 0).sum().item()
    assert num_nonzero <= 2, "Should have at most 2 valid results"
