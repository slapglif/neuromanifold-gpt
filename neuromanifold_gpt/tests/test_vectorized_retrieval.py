"""Integration tests comparing sequential vs vectorized memory retrieval.

These tests verify that the new vectorized retrieve_batch() method produces
identical results to the sequential retrieve() approach, ensuring correctness
of the 10-50x performance optimization.
"""
import pytest
import torch

from neuromanifold_gpt.model.memory.engram import SDREngramMemory


def test_retrieve_batch_vs_sequential_equivalence():
    """Batch retrieval should produce identical results to sequential retrieval."""
    # Create memory with some stored SDRs
    sdr_size = 2048
    capacity = 100
    n_active = 40
    content_dim = 128
    memory = SDREngramMemory(sdr_size, capacity, n_active, content_dim=content_dim)

    # Store some memories
    batch_size = 8
    stored_sdrs = torch.zeros(batch_size, sdr_size)
    stored_contents = torch.randn(batch_size, content_dim)

    # Create sparse SDRs with n_active bits set
    for i in range(batch_size):
        indices = torch.randperm(sdr_size)[:n_active]
        stored_sdrs[i, indices] = 1.0

    memory.store_batch(stored_sdrs, stored_contents)

    # Create query SDRs
    num_queries = 4
    query_sdrs = torch.zeros(num_queries, sdr_size)
    for i in range(num_queries):
        indices = torch.randperm(sdr_size)[:n_active]
        query_sdrs[i, indices] = 1.0

    top_k = 5

    # Sequential retrieval (old approach)
    sequential_contents = []
    sequential_sims = []
    for i in range(num_queries):
        contents, sims = memory.retrieve(query_sdrs[i], top_k=top_k)
        # Pad to top_k if needed
        if len(contents) < top_k:
            pad_size = top_k - len(contents)
            contents = torch.cat(
                [contents, torch.zeros(pad_size, content_dim, device=contents.device)],
                dim=0,
            )
            sims = torch.cat([sims, torch.zeros(pad_size, device=sims.device)], dim=0)
        sequential_contents.append(contents)
        sequential_sims.append(sims)

    sequential_contents = torch.stack(
        sequential_contents
    )  # (num_queries, top_k, content_dim)
    sequential_sims = torch.stack(sequential_sims)  # (num_queries, top_k)

    # Batch retrieval (new approach)
    batch_contents, batch_sims = memory.retrieve_batch(query_sdrs, top_k=top_k)

    # Compare results - should be nearly identical
    assert torch.allclose(
        sequential_contents, batch_contents, atol=1e-6
    ), "Contents from batch retrieval should match sequential retrieval"
    assert torch.allclose(
        sequential_sims, batch_sims, atol=1e-6
    ), "Similarities from batch retrieval should match sequential retrieval"


def test_large_batch_vectorized_retrieval():
    """Test vectorized retrieval with larger batch sizes."""
    sdr_size = 2048
    capacity = 200
    n_active = 40
    content_dim = 128
    memory = SDREngramMemory(sdr_size, capacity, n_active, content_dim=content_dim)

    # Store many memories
    num_stored = 50
    stored_sdrs = torch.zeros(num_stored, sdr_size)
    stored_contents = torch.randn(num_stored, content_dim)

    for i in range(num_stored):
        indices = torch.randperm(sdr_size)[:n_active]
        stored_sdrs[i, indices] = 1.0

    memory.store_batch(stored_sdrs, stored_contents)

    # Test with large batch
    batch_size = 64
    query_sdrs = torch.zeros(batch_size, sdr_size)
    for i in range(batch_size):
        indices = torch.randperm(sdr_size)[:n_active]
        query_sdrs[i, indices] = 1.0

    top_k = 10

    # Should complete without errors
    contents, sims = memory.retrieve_batch(query_sdrs, top_k=top_k)

    # Verify shapes
    assert contents.shape == (batch_size, top_k, content_dim)
    assert sims.shape == (batch_size, top_k)

    # All similarities should be in valid range [0, 1]
    assert torch.all(sims >= 0.0) and torch.all(sims <= 1.0)


def test_retrieval_threshold_filtering():
    """Test that threshold filtering works correctly in batch retrieval."""
    sdr_size = 2048
    capacity = 100
    n_active = 40
    content_dim = 128
    threshold = 0.5

    memory = SDREngramMemory(
        sdr_size, capacity, n_active, content_dim=content_dim, threshold=threshold
    )

    # Store one memory
    stored_sdr = torch.zeros(1, sdr_size)
    indices = torch.randperm(sdr_size)[:n_active]
    stored_sdr[0, indices] = 1.0
    stored_content = torch.randn(1, content_dim)
    memory.store_batch(stored_sdr, stored_content)

    # Create queries with varying overlap
    query_sdrs = torch.zeros(3, sdr_size)

    # Query 0: High overlap (same as stored)
    query_sdrs[0] = stored_sdr[0]

    # Query 1: Medium overlap (50% overlap)
    overlap_indices = indices[: n_active // 2]
    new_indices = torch.randperm(sdr_size)[: n_active // 2]
    new_indices = new_indices[~torch.isin(new_indices, indices)][: n_active // 2]
    query_sdrs[1, overlap_indices] = 1.0
    query_sdrs[1, new_indices] = 1.0

    # Query 2: Low overlap (10% overlap)
    small_overlap = indices[: n_active // 10]
    remaining = torch.randperm(sdr_size)[: n_active - len(small_overlap)]
    remaining = remaining[~torch.isin(remaining, indices)][
        : n_active - len(small_overlap)
    ]
    query_sdrs[2, small_overlap] = 1.0
    query_sdrs[2, remaining] = 1.0

    contents, sims = memory.retrieve_batch(query_sdrs, top_k=5)

    # High overlap query should retrieve (similarity >= threshold)
    assert sims[0, 0] > threshold, "High overlap should exceed threshold"

    # Results below threshold should be zeroed
    for i in range(3):
        for k in range(5):
            if sims[i, k] < threshold:
                assert torch.allclose(
                    contents[i, k], torch.zeros(content_dim)
                ), f"Content should be zero when similarity {sims[i, k]:.3f} < threshold {threshold}"
