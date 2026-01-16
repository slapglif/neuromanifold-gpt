#!/usr/bin/env python3
"""
Verification script for subtask-2-3: Verify cache extension works for longer sequences.

This script:
1. Creates a RamanujanPositionalEmbedding with block_size=256
2. Verifies initial cache is set to 256
3. Calls forward with sequence length 512 (longer than initial block_size)
4. Verifies cached_seq_len extends to 512
5. Verifies output shape is correct (1, 512, embed_dim)
"""

import torch
from neuromanifold_gpt.model.embeddings.ramanujan import RamanujanPositionalEmbedding

def verify_cache_extension():
    print("=" * 60)
    print("Verification: Cache Extension for Longer Sequences")
    print("=" * 60)

    # Step 1: Create RamanujanPositionalEmbedding with block_size=256
    block_size = 256
    embed_dim = 64
    print(f"\n1. Creating RamanujanPositionalEmbedding(block_size={block_size}, embed_dim={embed_dim})")
    emb = RamanujanPositionalEmbedding(block_size, embed_dim)
    print(f"   ✓ Created successfully")

    # Step 2: Verify initial cache
    print(f"\n2. Verifying initial cache state")
    print(f"   Initial cached_seq_len: {emb.cached_seq_len}")
    print(f"   Initial pe.shape: {emb.pe.shape}")
    assert emb.cached_seq_len == block_size, f"Expected cached_seq_len={block_size}, got {emb.cached_seq_len}"
    assert emb.pe.shape == (block_size, embed_dim), f"Expected pe.shape=({block_size}, {embed_dim}), got {emb.pe.shape}"
    print(f"   ✓ Initial cache correctly set to block_size={block_size}")

    # Step 3: Call forward with longer sequence length
    seq_len = 512
    batch_size = 2
    print(f"\n3. Calling forward() with sequence length={seq_len} (longer than block_size)")
    idx = torch.zeros(batch_size, seq_len, dtype=torch.long)
    print(f"   Input idx.shape: {idx.shape}")

    result = emb.forward(idx)

    # Step 4: Verify cache extension
    print(f"\n4. Verifying cache extension")
    print(f"   New cached_seq_len: {emb.cached_seq_len}")
    print(f"   New pe.shape: {emb.pe.shape}")
    assert emb.cached_seq_len == seq_len, f"Expected cached_seq_len={seq_len}, got {emb.cached_seq_len}"
    assert emb.pe.shape == (seq_len, embed_dim), f"Expected pe.shape=({seq_len}, {embed_dim}), got {emb.pe.shape}"
    print(f"   ✓ Cache successfully extended from {block_size} to {seq_len}")

    # Step 5: Verify output shape
    print(f"\n5. Verifying output shape")
    expected_shape = (1, seq_len, embed_dim)
    print(f"   Output shape: {result.shape}")
    print(f"   Expected shape: {expected_shape}")
    assert result.shape == expected_shape, f"Expected output.shape={expected_shape}, got {result.shape}"
    print(f"   ✓ Output shape is correct")

    # Additional verification: Test that subsequent calls don't rebuild cache
    print(f"\n6. Additional verification: Subsequent calls use cached values")
    cached_pe_id_before = id(emb.pe)
    idx2 = torch.zeros(batch_size, seq_len - 10, dtype=torch.long)  # Shorter sequence
    result2 = emb.forward(idx2)
    cached_pe_id_after = id(emb.pe)
    assert cached_pe_id_before == cached_pe_id_after, "Cache should not be rebuilt for shorter sequences"
    assert emb.cached_seq_len == seq_len, f"cached_seq_len should remain {seq_len}"
    print(f"   ✓ Cache not rebuilt for shorter sequence (seq_len={idx2.shape[1]})")

    # Test extending cache further
    print(f"\n7. Testing further cache extension")
    seq_len_extended = 1024
    idx3 = torch.zeros(batch_size, seq_len_extended, dtype=torch.long)
    result3 = emb.forward(idx3)
    assert emb.cached_seq_len == seq_len_extended, f"Expected cached_seq_len={seq_len_extended}, got {emb.cached_seq_len}"
    assert result3.shape == (1, seq_len_extended, embed_dim), f"Expected shape (1, {seq_len_extended}, {embed_dim}), got {result3.shape}"
    print(f"   ✓ Cache successfully extended from {seq_len} to {seq_len_extended}")

    print("\n" + "=" * 60)
    print("✅ ALL VERIFICATIONS PASSED")
    print("=" * 60)
    print("\nSummary:")
    print(f"  - Initial block_size: {block_size}")
    print(f"  - First extension: {block_size} → {seq_len}")
    print(f"  - Second extension: {seq_len} → {seq_len_extended}")
    print(f"  - Cache correctly extends dynamically for longer sequences")
    print(f"  - Output shapes are correct for all sequence lengths")
    print(f"  - Cache is not rebuilt unnecessarily for shorter sequences")

    return True

if __name__ == "__main__":
    try:
        verify_cache_extension()
        print("\n✅ VERIFICATION SUCCESSFUL")
        exit(0)
    except AssertionError as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
